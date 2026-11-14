"""Priority-based greedy route construction for reverse logistics."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


STORE_ID_COLUMN = "Store ID"
LATITUDE_COLUMN = "Latitude"
LONGITUDE_COLUMN = "Longitude"
DEMAND_COLUMN = "Current empty bottle crates (demand)"
PRIORITY_SCORE_COLUMN = "Collection priority score"
DISTANCE_NODE_COLUMN = "NodeID"
DEFAULT_DEPOT_ID = "DEPOT"


@dataclass
class RoutingResult:
    """Routes and an explicit record for every store not visited."""

    routes: list[dict[str, Any]]
    unserved_stores: pd.DataFrame


def _validate_number(name: str, value: float, *, allow_zero: bool) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be a finite number")
    if allow_zero:
        invalid = numeric_value < 0
        expected = "non-negative"
    else:
        invalid = numeric_value <= 0
        expected = "greater than zero"
    if invalid:
        raise ValueError(f"{name} must be {expected}")
    return numeric_value


def _normalize_distance_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(distance_matrix, pd.DataFrame):
        raise TypeError("distance_matrix must be a pandas DataFrame")

    matrix = distance_matrix.copy(deep=True)
    if DISTANCE_NODE_COLUMN in matrix.columns:
        if matrix[DISTANCE_NODE_COLUMN].duplicated().any():
            raise ValueError("distance matrix node IDs must be unique")
        matrix = matrix.set_index(DISTANCE_NODE_COLUMN)
    if matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise ValueError("distance matrix node IDs must be unique")
    if set(matrix.index) != set(matrix.columns):
        raise ValueError("distance matrix row and column node IDs must match")
    matrix = matrix.loc[matrix.index, matrix.index]

    try:
        matrix = matrix.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("distance matrix values must be numeric") from exc
    values = matrix.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("distance matrix values must be finite and non-negative")
    if not np.allclose(values, values.T, rtol=0, atol=1e-8):
        raise ValueError("distance matrix must be symmetric")
    if not np.allclose(np.diag(values), 0, rtol=0, atol=1e-8):
        raise ValueError("distance matrix diagonal values must be zero")
    if len(matrix) > 1 and (values[~np.eye(len(matrix), dtype=bool)] <= 0).any():
        raise ValueError("distances between distinct nodes must be positive")
    return matrix


def _route_metrics(
    sequence: list[str],
    store_data: dict[str, dict[str, float]],
    matrix: pd.DataFrame,
    vehicle_speed_kmh: float,
    vehicle_capacity: int,
    fixed_service_minutes: float,
    variable_service_minutes_per_crate: float,
) -> dict[str, float]:
    # Sum all depot/store legs, then convert kilometres to minutes at vehicle speed.
    total_distance = sum(
        float(matrix.loc[origin, destination])
        for origin, destination in zip(sequence, sequence[1:])
    )
    travel_time = total_distance / vehicle_speed_kmh * 60

    # Collection time includes fixed stop time and variable time per crate.
    total_crates = sum(store_data[store_id]["demand"] for store_id in sequence[1:-1])
    collection_time = len(sequence[1:-1]) * fixed_service_minutes
    collection_time += total_crates * variable_service_minutes_per_crate

    return {
        "total_distance_km": total_distance,
        "total_travel_time_minutes": travel_time,
        "total_collection_time_minutes": collection_time,
        "total_crates_collected": int(total_crates),
        "vehicle_utilization": total_crates / vehicle_capacity,
        "vehicle_utilization_percent": 100 * total_crates / vehicle_capacity,
        "total_duration_minutes": travel_time + collection_time,
    }


def generate_routes(
    stores: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    vehicle_capacity: int,
    vehicle_speed_kmh: float,
    fixed_collection_time_minutes: float,
    variable_collection_time_minutes_per_crate: float,
    route_duration_limit_minutes: float,
    depot_id: str = DEFAULT_DEPOT_ID,
) -> RoutingResult:
    """Generate feasible depot-to-depot routes with priority-based insertion.

    Positive-demand stores are required. Zero-demand rows are explicitly
    returned in ``unserved_stores`` as ``not_required``. For each insertion,
    the highest remaining priority score tier with a feasible choice is used;
    within that tier, the insertion with least added travel time is selected.
    Ties in score and added travel time retain input order and then insertion
    order, respectively.

    The distance matrix must be in kilometres. Service time is fixed per stop
    plus a variable amount per crate. No optimization solver is used.
    """
    if not isinstance(stores, pd.DataFrame):
        raise TypeError("stores must be a pandas DataFrame")

    required_columns = {
        STORE_ID_COLUMN,
        LATITUDE_COLUMN,
        LONGITUDE_COLUMN,
        DEMAND_COLUMN,
        PRIORITY_SCORE_COLUMN,
    }
    missing_columns = required_columns.difference(stores.columns)
    if missing_columns:
        raise ValueError(f"Missing required store columns: {sorted(missing_columns)}")
    if stores[STORE_ID_COLUMN].isna().any() or stores[STORE_ID_COLUMN].duplicated().any():
        raise ValueError("store IDs must be present and unique")
    if not stores[STORE_ID_COLUMN].map(lambda value: isinstance(value, str) and value).all():
        raise ValueError("store IDs must be non-empty strings")
    if not isinstance(depot_id, str) or not depot_id:
        raise ValueError("depot_id must be a non-empty string")
    if depot_id in set(stores[STORE_ID_COLUMN]):
        raise ValueError("depot_id cannot also be a store ID")

    capacity_value = _validate_number("vehicle_capacity", vehicle_capacity, allow_zero=False)
    if not capacity_value.is_integer():
        raise ValueError("vehicle_capacity must be a whole crate count")
    capacity = int(capacity_value)
    speed = _validate_number("vehicle_speed_kmh", vehicle_speed_kmh, allow_zero=False)
    fixed_service = _validate_number(
        "fixed_collection_time_minutes", fixed_collection_time_minutes, allow_zero=True
    )
    variable_service = _validate_number(
        "variable_collection_time_minutes_per_crate",
        variable_collection_time_minutes_per_crate,
        allow_zero=True,
    )
    duration_limit = _validate_number(
        "route_duration_limit_minutes", route_duration_limit_minutes, allow_zero=False
    )

    # Parse a private copy so the caller's store data is never changed.
    data = stores.copy(deep=True).reset_index(drop=True)
    for column in (LATITUDE_COLUMN, LONGITUDE_COLUMN, DEMAND_COLUMN, PRIORITY_SCORE_COLUMN):
        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must contain numeric values") from exc
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite values")

    if ((data[LATITUDE_COLUMN] < -90) | (data[LATITUDE_COLUMN] > 90)).any():
        raise ValueError("store latitudes must be between -90 and 90")
    if ((data[LONGITUDE_COLUMN] < -180) | (data[LONGITUDE_COLUMN] > 180)).any():
        raise ValueError("store longitudes must be between -180 and 180")
    if ((data[PRIORITY_SCORE_COLUMN] < 0) | (data[PRIORITY_SCORE_COLUMN] > 1)).any():
        raise ValueError("store priority scores must be between 0 and 1")
    demands = data[DEMAND_COLUMN].to_numpy(dtype=float)
    if (demands < 0).any() or not np.equal(demands, np.floor(demands)).all():
        raise ValueError("store demands must be non-negative whole crate counts")

    matrix = _normalize_distance_matrix(distance_matrix)
    store_ids = data[STORE_ID_COLUMN].tolist()
    missing_nodes = {depot_id, *store_ids}.difference(matrix.index)
    if missing_nodes:
        raise ValueError(f"distance matrix is missing nodes: {sorted(missing_nodes)}")

    store_data: dict[str, dict[str, float]] = {}
    unserved_rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        store_id = row[STORE_ID_COLUMN]
        demand = int(row[DEMAND_COLUMN])
        store_data[store_id] = {
            "demand": demand,
            "priority": float(row[PRIORITY_SCORE_COLUMN]),
        }
        if demand == 0:
            unserved_rows.append(
                {
                    STORE_ID_COLUMN: store_id,
                    "status": "not_required",
                    "reason": "zero_demand_not_required",
                }
            )
        elif demand > capacity:
            unserved_rows.append(
                {
                    STORE_ID_COLUMN: store_id,
                    "status": "infeasible",
                    "reason": "store_demand_exceeds_vehicle_capacity",
                }
            )

    remaining = [
        store_id
        for store_id in store_ids
        if 0 < store_data[store_id]["demand"] <= capacity
    ]
    routes: list[dict[str, Any]] = []

    while remaining:
        sequence = [depot_id, depot_id]
        route_crates = 0

        while True:
            # Stable sorting preserves original input order for equal priorities.
            ordered_remaining = sorted(
                remaining, key=lambda store_id: -store_data[store_id]["priority"]
            )
            priority_tiers: list[list[str]] = []
            for store_id in ordered_remaining:
                score = store_data[store_id]["priority"]
                if (
                    not priority_tiers
                    or store_data[priority_tiers[-1][0]]["priority"] != score
                ):
                    priority_tiers.append([])
                priority_tiers[-1].append(store_id)

            selected_insertion = None
            current_distance = _route_metrics(
                sequence,
                store_data,
                matrix,
                speed,
                capacity,
                fixed_service,
                variable_service,
            )["total_distance_km"]

            # Fall back to the next priority tier only when a higher tier has no
            # feasible insertion due to capacity or the route-duration limit.
            for tier in priority_tiers:
                feasible_insertions = []
                for input_order, store_id in enumerate(tier):
                    demand = int(store_data[store_id]["demand"])
                    if route_crates + demand > capacity:
                        continue
                    for insertion_index in range(1, len(sequence)):
                        candidate_sequence = (
                            sequence[:insertion_index]
                            + [store_id]
                            + sequence[insertion_index:]
                        )
                        metrics = _route_metrics(
                            candidate_sequence,
                            store_data,
                            matrix,
                            speed,
                            capacity,
                            fixed_service,
                            variable_service,
                        )
                        if metrics["total_duration_minutes"] <= duration_limit + 1e-9:
                            added_travel_time = (
                                metrics["total_distance_km"] - current_distance
                            ) / speed * 60
                            feasible_insertions.append(
                                (
                                    added_travel_time,
                                    input_order,
                                    insertion_index,
                                    store_id,
                                    candidate_sequence,
                                )
                            )
                if feasible_insertions:
                    # Minimize added travel time within the highest feasible tier.
                    feasible_insertions.sort(key=lambda choice: choice[:3])
                    selected_insertion = feasible_insertions[0]
                    break

            if selected_insertion is None:
                # Returning to the depot closes the route when no insertion fits.
                break

            _, _, _, selected_store, sequence = selected_insertion
            route_crates += int(store_data[selected_store]["demand"])
            remaining.remove(selected_store)

        if len(sequence) > 2:
            metrics = _route_metrics(
                sequence,
                store_data,
                matrix,
                speed,
                capacity,
                fixed_service,
                variable_service,
            )
            routes.append(
                {
                    "route_id": f"R{len(routes) + 1:02d}",
                    "sequence": sequence,
                    **metrics,
                }
            )
        else:
            # An individually infeasible store is reported, not silently skipped.
            impossible_store = remaining.pop(0)
            unserved_rows.append(
                {
                    STORE_ID_COLUMN: impossible_store,
                    "status": "infeasible",
                    "reason": "single_store_route_exceeds_duration_limit",
                }
            )

    unserved = pd.DataFrame(
        unserved_rows,
        columns=[STORE_ID_COLUMN, "status", "reason"],
    )
    return RoutingResult(routes=routes, unserved_stores=unserved)
