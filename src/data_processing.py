"""Loading and validation helpers for the reverse-logistics DSS."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.priority import calculate_store_priorities


STORE_ID = "Store ID"
STORE_NAME = "Store name"
LATITUDE = "Latitude"
LONGITUDE = "Longitude"
DEMAND = "Current empty bottle crates (demand)"
STORAGE_CAPACITY = "Maximum crate storage capacity"
PICKUP_REQUEST = "Pickup request"
DEPOT_ID = "DEPOT"
EARTH_RADIUS_KM = 6371.0088
REQUIRED_STORE_COLUMNS = {
    STORE_ID,
    STORE_NAME,
    DEMAND,
    STORAGE_CAPACITY,
    PICKUP_REQUEST,
}


def load_default_inputs(base_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load checked-in inputs from the project root's ``data/`` directory."""
    data_dir = Path(base_dir).resolve() / "data"
    stores = pd.read_csv(data_dir / "stores.csv")
    matrix = pd.read_csv(data_dir / "distances.csv").set_index("NodeID")
    with (data_dir / "parameters.json").open(encoding="utf-8") as file:
        parameters = json.load(file)
    return stores, matrix, parameters


def build_geodesic_matrix(
    stores: pd.DataFrame,
    depot_latitude: float,
    depot_longitude: float,
) -> pd.DataFrame:
    """Build a symmetric straight-line km matrix from supplied coordinates."""
    if not {STORE_ID, LATITUDE, LONGITUDE}.issubset(stores.columns):
        raise ValueError("Store IDs and latitude/longitude are required to build distances")
    try:
        depot_latitude = float(depot_latitude)
        depot_longitude = float(depot_longitude)
    except (TypeError, ValueError) as exc:
        raise ValueError("Warehouse coordinates must be numeric") from exc
    if (
        not math.isfinite(depot_latitude)
        or not math.isfinite(depot_longitude)
        or not -90 <= depot_latitude <= 90
        or not -180 <= depot_longitude <= 180
    ):
        raise ValueError("Warehouse coordinates are outside valid geographic bounds")

    try:
        numeric_latitude = pd.to_numeric(stores[LATITUDE], errors="raise").to_numpy(dtype=float)
        numeric_longitude = pd.to_numeric(stores[LONGITUDE], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Store coordinates must be numeric") from exc
    if (
        not np.isfinite(numeric_latitude).all()
        or not np.isfinite(numeric_longitude).all()
        or ((numeric_latitude < -90) | (numeric_latitude > 90)).any()
        or ((numeric_longitude < -180) | (numeric_longitude > 180)).any()
    ):
        raise ValueError("Store coordinates are missing or outside valid geographic bounds")

    nodes = [(DEPOT_ID, depot_latitude, depot_longitude)]
    nodes.extend(
        (
            str(row[STORE_ID]),
            float(row[LATITUDE]),
            float(row[LONGITUDE]),
        )
        for _, row in stores.iterrows()
    )
    if len({node[0] for node in nodes}) != len(nodes):
        raise ValueError("Store IDs must be unique and cannot equal the depot ID")

    def haversine(first: tuple[str, float, float], second: tuple[str, float, float]) -> float:
        lat1, lon1 = math.radians(first[1]), math.radians(first[2])
        lat2, lon2 = math.radians(second[1]), math.radians(second[2])
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        hav = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
        )
        return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(hav))

    ids = [node[0] for node in nodes]
    matrix = pd.DataFrame(0.0, index=ids, columns=ids)
    for first in range(len(nodes)):
        for second in range(first + 1, len(nodes)):
            distance = haversine(nodes[first], nodes[second])
            matrix.iat[first, second] = distance
            matrix.iat[second, first] = distance
    return matrix


def validate_distance_matrix(
    matrix: pd.DataFrame,
    stores: pd.DataFrame,
    depot_id: str = DEPOT_ID,
) -> list[dict[str, Any]]:
    """Return matrix validation findings without altering the input matrix."""
    checks: list[dict[str, Any]] = []
    if not isinstance(matrix, pd.DataFrame):
        return [
            {
                "Check": "Distance matrix format",
                "Status": "Failed",
                "Affected records": 1,
                "Explanation": "Distance matrix must be a Pandas DataFrame.",
                "Recommended action": "Load a labeled square matrix or upload a valid matrix CSV.",
            }
        ]

    normalized = matrix.copy(deep=True)
    if "NodeID" in normalized.columns:
        normalized = normalized.set_index("NodeID")
    required_nodes = {depot_id, *stores[STORE_ID].astype(str)}
    row_nodes = set(map(str, normalized.index))
    col_nodes = set(map(str, normalized.columns))
    missing_rows = required_nodes - row_nodes
    missing_columns = required_nodes - col_nodes
    checks.append(
        {
            "Check": "Distance matrix node coverage",
            "Status": "Passed" if not (missing_rows or missing_columns) else "Failed",
            "Affected records": len(missing_rows | missing_columns),
            "Explanation": (
                "Matrix includes the depot and every store."
                if not (missing_rows or missing_columns)
                else f"Missing row nodes: {sorted(missing_rows)}; missing column nodes: {sorted(missing_columns)}."
            ),
            "Recommended action": (
                "No action required."
                if not (missing_rows or missing_columns)
                else "Supply a complete distance matrix or provide valid coordinates and a warehouse location."
            ),
        }
    )
    try:
        numeric = normalized.astype(float)
        values = numeric.to_numpy()
        valid = (
            np.isfinite(values).all()
            and (values >= 0).all()
            and np.allclose(values, values.T, rtol=0, atol=1e-8)
            and np.allclose(np.diag(values), 0, rtol=0, atol=1e-8)
        )
        checks.append(
            {
                "Check": "Distance matrix values",
                "Status": "Passed" if valid else "Failed",
                "Affected records": 0 if valid else 1,
                "Explanation": (
                    "Matrix is numeric, finite, symmetric, non-negative, and has a zero diagonal."
                    if valid
                    else "Matrix values must be finite/non-negative and symmetric with a zero diagonal."
                ),
                "Recommended action": "No action required." if valid else "Correct or regenerate the matrix.",
            }
        )
    except (TypeError, ValueError):
        checks.append(
            {
                "Check": "Distance matrix values",
                "Status": "Failed",
                "Affected records": 1,
                "Explanation": "Distance matrix contains non-numeric entries.",
                "Recommended action": "Provide a numeric square distance matrix.",
            }
        )
    return checks


def validate_stores(stores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate uploaded rows and return an enriched copy plus a check report.

    Store records are not dropped or repaired. Existing derived priority
    columns are recalculated only on the returned copy.
    """
    if not isinstance(stores, pd.DataFrame):
        raise TypeError("stores must be a Pandas DataFrame")
    report: list[dict[str, Any]] = []
    if stores.empty:
        report.append(
            {
                "Check": "Non-empty dataset",
                "Status": "Failed",
                "Affected records": 0,
                "Explanation": "The input dataset contains no store rows.",
                "Recommended action": "Upload a CSV with at least one store.",
            }
        )
        return stores.copy(deep=True), pd.DataFrame(report)

    missing = REQUIRED_STORE_COLUMNS.difference(stores.columns)
    report.append(
        {
            "Check": "Required columns",
            "Status": "Passed" if not missing else "Failed",
            "Affected records": len(missing),
            "Explanation": "All required store fields are present."
            if not missing
            else f"Missing fields: {sorted(missing)}.",
            "Recommended action": "No action required."
            if not missing
            else "Add the missing required columns and upload again.",
        }
    )
    if missing:
        return stores.copy(deep=True), pd.DataFrame(report)

    validated = stores.copy(deep=True)
    duplicate_count = int(validated[STORE_ID].duplicated(keep=False).sum())
    invalid_ids = int(
        validated[STORE_ID].isna().sum()
        + validated[STORE_ID].map(
            lambda value: not isinstance(value, str) or not value.strip()
        ).sum()
    )
    id_issues = duplicate_count + invalid_ids
    report.append(
        {
            "Check": "Unique store IDs",
            "Status": "Passed" if id_issues == 0 else "Failed",
            "Affected records": id_issues,
            "Explanation": "Store IDs are unique."
            if id_issues == 0
            else "Store IDs must be unique, non-empty strings.",
            "Recommended action": "No action required."
            if id_issues == 0
            else "Resolve duplicate or invalid IDs; records are not automatically removed.",
        }
    )

    invalid_names = int(validated[STORE_NAME].isna().sum())
    report.append(
        {
            "Check": "Store names",
            "Status": "Passed" if invalid_names == 0 else "Warning",
            "Affected records": invalid_names,
            "Explanation": "Store names are present."
            if invalid_names == 0
            else "Some store names are blank; Store IDs remain the unique route keys.",
            "Recommended action": "No action required."
            if invalid_names == 0
            else "Add names for clearer tables and map popups.",
        }
    )

    numeric_validity: dict[str, bool] = {}
    for column in (DEMAND, STORAGE_CAPACITY, PICKUP_REQUEST):
        try:
            validated[column] = pd.to_numeric(validated[column], errors="raise")
            values = validated[column].to_numpy(dtype=float)
            numeric_valid = np.isfinite(values).all()
        except (TypeError, ValueError):
            numeric_valid = False
            values = np.array([])
        if column == DEMAND:
            valid = numeric_valid and (values >= 0).all() and np.equal(values, np.floor(values)).all()
            explanation = "Demand is finite, non-negative, and measured in whole crates."
        elif column == STORAGE_CAPACITY:
            valid = numeric_valid and (values > 0).all() and np.equal(values, np.floor(values)).all()
            explanation = "Store capacities are finite positive whole crate counts."
        else:
            valid = numeric_valid and np.isin(values, [0, 1]).all()
            explanation = "Pickup requests contain only 0 or 1."
        numeric_validity[column] = bool(valid)
        report.append(
            {
                "Check": column,
                "Status": "Passed" if valid else "Failed",
                "Affected records": 0 if valid else len(validated),
                "Explanation": explanation if valid else f"{column} contains invalid values.",
                "Recommended action": "No action required."
                if valid
                else "Correct invalid values; the application will not silently alter records.",
            }
        )

    if numeric_validity.get(DEMAND, False) and numeric_validity.get(STORAGE_CAPACITY, False):
        over_capacity = validated[DEMAND] > validated[STORAGE_CAPACITY]
        count = int(over_capacity.sum())
        report.append(
            {
                "Check": "Demand within store capacity",
                "Status": "Passed" if count == 0 else "Failed",
                "Affected records": count,
                "Explanation": "No store's current demand exceeds its storage capacity."
                if count == 0
                else "Some current demands exceed store storage capacity.",
                "Recommended action": "No action required."
                if count == 0
                else "Review demand/capacity input; routing is blocked until resolved.",
            }
        )

    if LATITUDE in validated.columns and LONGITUDE in validated.columns:
        try:
            lat = pd.to_numeric(validated[LATITUDE], errors="raise").to_numpy(dtype=float)
            lon = pd.to_numeric(validated[LONGITUDE], errors="raise").to_numpy(dtype=float)
            coord_ok = (
                np.isfinite(lat).all()
                and np.isfinite(lon).all()
                and ((lat >= -90) & (lat <= 90)).all()
                and ((lon >= -180) & (lon <= 180)).all()
            )
        except (TypeError, ValueError):
            coord_ok = False
        report.append(
            {
                "Check": "Store coordinates",
                "Status": "Passed" if coord_ok else "Failed",
                "Affected records": 0 if coord_ok else len(validated),
                "Explanation": "Latitude and longitude are valid."
                if coord_ok
                else "Coordinates are missing, non-numeric, or outside geographic bounds.",
                "Recommended action": "No action required."
                if coord_ok
                else "Correct coordinates; route mapping and coordinate-based distances will be unavailable.",
            }
        )
    else:
        report.append(
            {
                "Check": "Store coordinates",
                "Status": "Warning",
                "Affected records": len(validated),
                "Explanation": "Latitude and/or longitude fields are absent.",
                "Recommended action": "Provide coordinates for maps and distance generation.",
            }
        )

    mandatory_checks = [item for item in report if item["Status"] == "Failed"]
    if not mandatory_checks:
        try:
            with np.errstate(divide="raise", invalid="raise"):
                validated = calculate_store_priorities(validated)
            report.append(
                {
                    "Check": "Derived filling rate and priority",
                    "Status": "Passed",
                    "Affected records": len(validated),
                    "Explanation": "Filling rate and 70/30 priority scores calculated.",
                    "Recommended action": "No action required.",
                }
            )
        except (ValueError, TypeError) as exc:
            report.append(
                {
                    "Check": "Derived filling rate and priority",
                    "Status": "Failed",
                    "Affected records": len(validated),
                    "Explanation": str(exc),
                    "Recommended action": "Correct the source input before running optimization.",
                }
            )
    return validated, pd.DataFrame(report)
