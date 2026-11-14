"""Recalculate and compare operational KPIs for heuristic route schedules.

KPI definitions
---------------
* Total distance: sum of matrix distances along every route leg, including
  departure from and return to the depot.
* Transportation time: total distance / average speed * 60 minutes per hour.
* Collection time: fixed service minutes per visited store plus variable
  minutes per collected crate.
* Operational time: transportation time + collection time.
* Crates and bottles: sum of demand at unique visited stores, and crates times
  bottles per crate, respectively.
* Routes and slots: unique route IDs and unique scheduled time-slot IDs.
* Vehicle utilization: collected crates / (number of routes * vehicle
  capacity) * 100. A no-route solution has 0% utilization.
* Stores served: unique positive-demand stores visited / all positive-demand
  stores * 100. Zero-demand stores are not in the denominator.
* High-volume stores not collected: stores with demand / vehicle capacity >=
  0.90 that do not appear in a route.
* Time-slot utilization: recalculated operational minutes / (used slots *
  available minutes per slot) * 100. A no-slot solution has 0% utilization.

All travel and service measures are recalculated from the supplied data.
Potentially stale metrics embedded in route dictionaries or schedule tables
are ignored.
"""

from typing import Any

import numpy as np
import pandas as pd


STORE_ID_COLUMN = "Store ID"
DEMAND_COLUMN = "Current empty bottle crates (demand)"
ROUTE_ID_COLUMN = "route_id"
ROUTE_SEQUENCE_COLUMN = "sequence"
SCHEDULE_SLOT_COLUMN = "Time slot"
SCHEDULE_ROUTE_COLUMN = "Route ID"
DISTANCE_NODE_COLUMN = "NodeID"
HIGH_VOLUME_DEMAND_SHARE = 0.90
PERCENT_KPIS = {
    "Vehicle capacity utilization",
    "Percentage of stores served",
    "Total time-slot utilization",
    "Travel time percentage",
    "Service time percentage",
    "Route feasibility rate",
}


def _finite_number(name: str, value: Any, *, allow_zero: bool) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(numeric_value):
        raise ValueError(f"{name} must be a finite number")
    invalid = numeric_value < 0 if allow_zero else numeric_value <= 0
    if invalid:
        expected = "non-negative" if allow_zero else "greater than zero"
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


def _get_schedule_assignments(
    routes: list[dict[str, Any]],
    schedule: pd.DataFrame,
    matrix: pd.DataFrame,
    demands: dict[str, int],
    vehicle_capacity: int,
    vehicle_speed_kmh: float,
    fixed_service_minutes: float,
    variable_service_minutes_per_crate: float,
    slot_duration_minutes: float,
) -> dict[str, Any]:
    if not isinstance(routes, list):
        raise TypeError("routes must be a list of generated route dictionaries")
    if not isinstance(schedule, pd.DataFrame):
        raise TypeError("schedule must be a pandas DataFrame")
    required_schedule_columns = {SCHEDULE_SLOT_COLUMN, SCHEDULE_ROUTE_COLUMN}
    missing_columns = required_schedule_columns.difference(schedule.columns)
    if missing_columns:
        raise ValueError(f"Schedule is missing columns: {sorted(missing_columns)}")

    route_by_id: dict[str, dict[str, Any]] = {}
    seen_stores: set[str] = set()
    depot_ids: set[str] = set()
    for index, route in enumerate(routes):
        if not isinstance(route, dict):
            raise TypeError(f"routes[{index}] must be a route dictionary")
        route_id = route.get(ROUTE_ID_COLUMN)
        if not isinstance(route_id, str) or not route_id:
            raise ValueError(f"routes[{index}] must have a non-empty route_id")
        if route_id in route_by_id:
            raise ValueError(f"Route IDs must be unique; duplicate: {route_id}")
        sequence = route.get(ROUTE_SEQUENCE_COLUMN)
        if not isinstance(sequence, (list, tuple)) or len(sequence) < 3:
            raise ValueError(f"Route {route_id} must have a depot-to-depot sequence")
        sequence = list(sequence)
        if sequence[0] != sequence[-1]:
            raise ValueError(f"Route {route_id} must start and end at the same depot")
        depot_ids.add(sequence[0])
        store_sequence = sequence[1:-1]
        if len(store_sequence) != len(set(store_sequence)):
            raise ValueError(f"Route {route_id} visits a store more than once")
        repeated_stores = seen_stores.intersection(store_sequence)
        if repeated_stores:
            raise ValueError(
                "A store cannot be counted more than once across routes: "
                + ", ".join(sorted(repeated_stores))
            )
        unknown_stores = set(store_sequence).difference(demands)
        if unknown_stores:
            raise ValueError(
                f"Route {route_id} contains stores missing from the store data: "
                + ", ".join(sorted(unknown_stores))
            )
        missing_nodes = set(sequence).difference(matrix.index)
        if missing_nodes:
            raise ValueError(
                f"Route {route_id} contains nodes missing from distance matrix: "
                + ", ".join(sorted(missing_nodes))
            )
        seen_stores.update(store_sequence)

        # Recalculate metrics from route legs and original store demand.
        distance = sum(
            float(matrix.loc[origin, destination])
            for origin, destination in zip(sequence, sequence[1:])
        )
        travel_time = distance / vehicle_speed_kmh * 60
        crates = sum(demands[store_id] for store_id in store_sequence)
        if crates > vehicle_capacity:
            raise ValueError(f"Route {route_id} exceeds vehicle capacity")
        collection_time = len(store_sequence) * fixed_service_minutes
        collection_time += crates * variable_service_minutes_per_crate
        route_by_id[route_id] = {
            "sequence": sequence,
            "stores": store_sequence,
            "distance": distance,
            "travel_time": travel_time,
            "collection_time": collection_time,
            "duration": travel_time + collection_time,
            "crates": crates,
        }

    if len(depot_ids) > 1:
        raise ValueError("All routes must use the same central depot")

    if schedule[SCHEDULE_ROUTE_COLUMN].isna().any():
        raise ValueError("Schedule route IDs cannot be missing")
    if schedule[SCHEDULE_ROUTE_COLUMN].duplicated().any():
        raise ValueError("Each route must be assigned to exactly one schedule row")
    scheduled_route_ids = set(schedule[SCHEDULE_ROUTE_COLUMN])
    if scheduled_route_ids != set(route_by_id):
        missing_assignments = set(route_by_id).difference(scheduled_route_ids)
        unknown_assignments = scheduled_route_ids.difference(route_by_id)
        raise ValueError(
            "Schedule must assign every route exactly once; "
            f"missing={sorted(missing_assignments)}, "
            f"unknown={sorted(unknown_assignments)}"
        )

    slot_by_route: dict[str, Any] = {}
    for _, row in schedule.iterrows():
        slot = _finite_number("time-slot ID", row[SCHEDULE_SLOT_COLUMN], allow_zero=False)
        if not slot.is_integer():
            raise ValueError("Time-slot IDs must be positive integers")
        slot_by_route[row[SCHEDULE_ROUTE_COLUMN]] = int(slot)

    slot_time: dict[Any, float] = {}
    for route_id, slot in slot_by_route.items():
        slot_time[slot] = slot_time.get(slot, 0.0) + route_by_id[route_id]["duration"]
    exceeded_slots = [
        slot for slot, duration in slot_time.items()
        if duration > slot_duration_minutes + 1e-9
    ]
    if exceeded_slots:
        raise ValueError(
            "Recalculated route durations exceed slot duration for slots: "
            + ", ".join(map(str, sorted(exceeded_slots)))
        )

    return {
        "routes": route_by_id,
        "slot_by_route": slot_by_route,
        "slot_time": slot_time,
        "visited_stores": seen_stores,
    }


def _evaluate_solution(
    routes: list[dict[str, Any]],
    schedule: pd.DataFrame,
    matrix: pd.DataFrame,
    demands: dict[str, int],
    vehicle_capacity: int,
    vehicle_speed_kmh: float,
    fixed_service_minutes: float,
    variable_service_minutes_per_crate: float,
    slot_duration_minutes: float,
    bottles_per_crate: int,
    route_duration_limit_minutes: float,
    number_of_available_time_slots: int | None,
) -> dict[str, float | int]:
    evaluated = _get_schedule_assignments(
        routes,
        schedule,
        matrix,
        demands,
        vehicle_capacity,
        vehicle_speed_kmh,
        fixed_service_minutes,
        variable_service_minutes_per_crate,
        slot_duration_minutes,
    )
    route_metrics = evaluated["routes"]
    visited_stores = evaluated["visited_stores"]
    if number_of_available_time_slots is not None and any(
        slot > number_of_available_time_slots for slot in evaluated["slot_time"]
    ):
        raise ValueError("Scheduled time-slot ID exceeds the available time-slot count")

    total_distance = sum(route["distance"] for route in route_metrics.values())
    transportation_time = sum(route["travel_time"] for route in route_metrics.values())
    collection_time = sum(route["collection_time"] for route in route_metrics.values())
    operational_time = transportation_time + collection_time
    total_crates = sum(route["crates"] for route in route_metrics.values())
    route_count = len(route_metrics)
    slot_count = len(evaluated["slot_time"])

    # Only stores with positive demand are required and belong in the service denominator.
    required_stores = {store_id for store_id, demand in demands.items() if demand > 0}
    served_count = len(visited_stores.intersection(required_stores))
    served_percent = 100 * served_count / len(required_stores) if required_stores else 0.0

    # Apply the requested high-volume definition: store demand is at least 90%
    # of one vehicle's crate capacity; it is not the store-capacity filling rate.
    high_volume_uncollected = sum(
        1
        for store_id, demand in demands.items()
        if demand / vehicle_capacity >= HIGH_VOLUME_DEMAND_SHARE
        and store_id not in visited_stores
    )

    # Aggregate utilization divides collected crates by the capacity of all routes used.
    total_vehicle_capacity = route_count * vehicle_capacity
    vehicle_utilization = (
        100 * total_crates / total_vehicle_capacity if total_vehicle_capacity else 0.0
    )

    # Measure utilization against the configured slot resource, including unused slots.
    slot_capacity_count = (
        slot_count
        if number_of_available_time_slots is None
        else number_of_available_time_slots
    )
    total_slot_capacity = slot_capacity_count * slot_duration_minutes
    slot_utilization = (
        100 * operational_time / total_slot_capacity if total_slot_capacity else 0.0
    )
    vehicle_idle_time = max(0.0, total_slot_capacity - operational_time)
    remaining_crates = sum(demands.values()) - total_crates
    unserved_store_count = len(required_stores - visited_stores)
    route_feasibility_rate = (
        100
        * sum(
            route["duration"] <= route_duration_limit_minutes + 1e-9
            and route["crates"] <= vehicle_capacity
            for route in route_metrics.values()
        )
        / route_count
        if route_count
        else 0.0
    )

    return {
        "Total distance travelled": total_distance,
        "Total transportation time": transportation_time,
        "Total collection time": collection_time,
        "Total operational time": operational_time,
        "Total crates collected": total_crates,
        "Total bottles collected": total_crates * bottles_per_crate,
        "Total crates remaining": remaining_crates,
        "Number of routes": route_count,
        "Number of time slots used": slot_count,
        "Vehicle capacity utilization": vehicle_utilization,
        "Percentage of stores served": served_percent,
        "High-volume stores (>=90% vehicle capacity) not collected": high_volume_uncollected,
        "Total time-slot utilization": slot_utilization,
        "Number of unserved stores": unserved_store_count,
        "Distance per crate collected": total_distance / total_crates if total_crates else 0.0,
        "Average stores per route": (
            sum(len(route["stores"]) for route in route_metrics.values()) / route_count
            if route_count
            else 0.0
        ),
        "Travel time percentage": 100 * transportation_time / operational_time if operational_time else 0.0,
        "Service time percentage": 100 * collection_time / operational_time if operational_time else 0.0,
        "Vehicle idle time": vehicle_idle_time,
        "Route feasibility rate": route_feasibility_rate,
    }


def compare_solutions(
    stores: pd.DataFrame,
    initial_routes: list[dict[str, Any]],
    initial_schedule: pd.DataFrame,
    optimized_routes: list[dict[str, Any]],
    optimized_schedule: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    vehicle_capacity: int,
    vehicle_speed_kmh: float,
    fixed_collection_time_minutes: float,
    variable_collection_time_minutes_per_crate: float,
    slot_duration_minutes: float,
    bottles_per_crate: int = 12,
    route_duration_limit_minutes: float | None = None,
    number_of_available_time_slots: int | None = None,
) -> pd.DataFrame:
    """Return a KPI table comparing initial and optimized heuristic schedules.

    Inputs are the original store table, route sequences, and their schedule
    tables. Embedded route metrics and schedule duration columns are ignored:
    distance, service time, demand, and all KPIs are recalculated independently.

    ``Change`` is optimized minus initial. Percentage KPI changes are reported
    in percentage points; other KPI changes use the KPI's own unit.
    """
    if not isinstance(stores, pd.DataFrame):
        raise TypeError("stores must be a pandas DataFrame")
    missing_store_columns = {STORE_ID_COLUMN, DEMAND_COLUMN}.difference(stores.columns)
    if missing_store_columns:
        raise ValueError(f"Stores are missing columns: {sorted(missing_store_columns)}")
    if stores[STORE_ID_COLUMN].isna().any() or stores[STORE_ID_COLUMN].duplicated().any():
        raise ValueError("Store IDs must be present and unique")

    capacity_value = _finite_number("vehicle_capacity", vehicle_capacity, allow_zero=False)
    if not capacity_value.is_integer():
        raise ValueError("vehicle_capacity must be a whole crate count")
    capacity = int(capacity_value)
    speed = _finite_number("vehicle_speed_kmh", vehicle_speed_kmh, allow_zero=False)
    fixed_service = _finite_number(
        "fixed_collection_time_minutes",
        fixed_collection_time_minutes,
        allow_zero=True,
    )
    variable_service = _finite_number(
        "variable_collection_time_minutes_per_crate",
        variable_collection_time_minutes_per_crate,
        allow_zero=True,
    )
    slot_duration = _finite_number(
        "slot_duration_minutes", slot_duration_minutes, allow_zero=False
    )
    route_duration_limit = _finite_number(
        "route_duration_limit_minutes",
        slot_duration if route_duration_limit_minutes is None else route_duration_limit_minutes,
        allow_zero=False,
    )
    bottles_value = _finite_number("bottles_per_crate", bottles_per_crate, allow_zero=False)
    if not bottles_value.is_integer():
        raise ValueError("bottles_per_crate must be a positive integer")
    bottles = int(bottles_value)
    if number_of_available_time_slots is not None:
        available_slots_value = _finite_number(
            "number_of_available_time_slots",
            number_of_available_time_slots,
            allow_zero=True,
        )
        if not available_slots_value.is_integer():
            raise ValueError("number_of_available_time_slots must be a whole number")
        available_slots = int(available_slots_value)
    else:
        available_slots = None

    try:
        numeric_demands = pd.to_numeric(stores[DEMAND_COLUMN], errors="raise").to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{DEMAND_COLUMN} must contain numeric values") from exc
    if not np.isfinite(numeric_demands).all() or (numeric_demands < 0).any():
        raise ValueError(f"{DEMAND_COLUMN} must contain finite non-negative values")
    if not np.equal(numeric_demands, np.floor(numeric_demands)).all():
        raise ValueError(f"{DEMAND_COLUMN} must contain whole crate counts")
    demands = dict(zip(stores[STORE_ID_COLUMN], numeric_demands.astype(int)))
    if not all(isinstance(store_id, str) and store_id for store_id in demands):
        raise ValueError("Store IDs must be non-empty strings")

    matrix = _normalize_distance_matrix(distance_matrix)
    initial_values = _evaluate_solution(
        initial_routes,
        initial_schedule,
        matrix,
        demands,
        capacity,
        speed,
        fixed_service,
        variable_service,
        slot_duration,
        bottles,
        route_duration_limit,
        available_slots,
    )
    optimized_values = _evaluate_solution(
        optimized_routes,
        optimized_schedule,
        matrix,
        demands,
        capacity,
        speed,
        fixed_service,
        variable_service,
        slot_duration,
        bottles,
        route_duration_limit,
        available_slots,
    )

    # Keep a consistent order and disclose units and how change was calculated.
    units = {
        "Total distance travelled": "km",
        "Total transportation time": "minutes",
        "Total collection time": "minutes",
        "Total operational time": "minutes",
        "Total crates collected": "crates",
        "Total bottles collected": "bottles",
        "Total crates remaining": "crates",
        "Number of routes": "routes",
        "Number of time slots used": "slots",
        "Vehicle capacity utilization": "%",
        "Percentage of stores served": "%",
        "High-volume stores (>=90% vehicle capacity) not collected": "stores",
        "Total time-slot utilization": "%",
        "Number of unserved stores": "stores",
        "Distance per crate collected": "km/crate",
        "Average stores per route": "stores/route",
        "Travel time percentage": "%",
        "Service time percentage": "%",
        "Vehicle idle time": "minutes",
        "Route feasibility rate": "%",
    }
    rows = []
    for kpi, unit in units.items():
        initial_value = initial_values[kpi]
        optimized_value = optimized_values[kpi]
        is_percentage = kpi in PERCENT_KPIS
        rows.append(
            {
                "KPI": kpi,
                "Unit": unit,
                "Initial Greedy": initial_value,
                "Optimized Heuristic": optimized_value,
                "Change (optimized - initial)": optimized_value - initial_value,
                "Change unit": "percentage points" if is_percentage else unit,
            }
        )
    return pd.DataFrame(rows)


def evaluate_solutions(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Alias for :func:`compare_solutions`."""
    return compare_solutions(*args, **kwargs)
