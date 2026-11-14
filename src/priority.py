"""Store priority calculations for the reverse-logistics routing heuristic."""

import warnings

import numpy as np
import pandas as pd


DEMAND_COLUMN = "Current empty bottle crates (demand)"
CAPACITY_COLUMN = "Maximum crate storage capacity"
PICKUP_REQUEST_COLUMN = "Pickup request"
FILLING_RATE_COLUMN = "Filling rate"
PRIORITY_SCORE_COLUMN = "Collection priority score"
PRIORITY_RANK_COLUMN = "Priority rank"
SATURATION_COLUMN = "Approaching saturation"
DEFAULT_SATURATION_WARNING_THRESHOLD = 0.80


def calculate_store_priorities(
    stores: pd.DataFrame,
    saturation_warning_threshold: float = DEFAULT_SATURATION_WARNING_THRESHOLD,
) -> pd.DataFrame:
    """Validate store inputs, calculate priorities, and return a ranked copy.

    Stores with equal priority scores retain their original input order.
    """
    if not isinstance(stores, pd.DataFrame):
        raise TypeError("stores must be a pandas DataFrame")

    if isinstance(saturation_warning_threshold, (bool, np.bool_)):
        raise ValueError("saturation_warning_threshold must be between 0 and 1")
    try:
        threshold = float(saturation_warning_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "saturation_warning_threshold must be a number between 0 and 1"
        ) from exc
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("saturation_warning_threshold must be between 0 and 1")

    required_columns = {DEMAND_COLUMN, CAPACITY_COLUMN, PICKUP_REQUEST_COLUMN}
    missing_columns = required_columns.difference(stores.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    # Work on a copy so calculations never alter the caller's source data.
    result = stores.copy(deep=True)

    # Convert numeric input columns explicitly and reject missing or non-numeric data.
    for column in (DEMAND_COLUMN, CAPACITY_COLUMN, PICKUP_REQUEST_COLUMN):
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must contain numeric values") from exc

        values = result[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain finite, non-missing values")

    demand = result[DEMAND_COLUMN].to_numpy(dtype=float)
    capacity = result[CAPACITY_COLUMN].to_numpy(dtype=float)
    pickup_request = result[PICKUP_REQUEST_COLUMN].to_numpy(dtype=float)

    # Crates are indivisible resources, so demand and storage capacity must be whole crates.
    if not np.equal(demand, np.floor(demand)).all():
        raise ValueError(f"{DEMAND_COLUMN} must contain whole crate counts")
    if not np.equal(capacity, np.floor(capacity)).all():
        raise ValueError(f"{CAPACITY_COLUMN} must contain whole crate counts")
    if (demand < 0).any():
        raise ValueError(f"{DEMAND_COLUMN} cannot be negative")
    if (capacity <= 0).any():
        raise ValueError(f"{CAPACITY_COLUMN} must be greater than zero")
    if (demand > capacity).any():
        raise ValueError(f"{DEMAND_COLUMN} cannot exceed {CAPACITY_COLUMN}")
    if not np.isin(pickup_request, [0, 1]).all():
        raise ValueError(f"{PICKUP_REQUEST_COLUMN} must contain only 0 or 1")

    # Filling rate is the share of the store's storage capacity currently occupied.
    result[FILLING_RATE_COLUMN] = demand / capacity

    # The pickup request is a binary priority signal combined with filling rate.
    result[PRIORITY_SCORE_COLUMN] = (
        0.70 * result[FILLING_RATE_COLUMN] + 0.30 * pickup_request
    )

    # Flag stores at or above the configured filling-rate warning threshold.
    result[SATURATION_COLUMN] = result[FILLING_RATE_COLUMN] >= threshold
    approaching = result.loc[result[SATURATION_COLUMN]]
    if not approaching.empty:
        label_column = next(
            (column for column in ("Store name", "Store ID") if column in result),
            None,
        )
        labels = (
            approaching[label_column].astype(str).tolist()
            if label_column
            else approaching.index.astype(str).tolist()
        )
        warnings.warn(
            f"Stores at or above {threshold:.0%} storage capacity: "
            + ", ".join(labels),
            UserWarning,
            stacklevel=2,
        )

    # A stable descending sort preserves input order when priority scores tie.
    result = result.sort_values(
        PRIORITY_SCORE_COLUMN, ascending=False, kind="mergesort"
    )
    result[PRIORITY_RANK_COLUMN] = np.arange(1, len(result) + 1)
    return result
