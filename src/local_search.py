"""Feasibility-preserving best-improvement local search for vehicle routes."""

from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd


STORE_ID_COLUMN = "Store ID"
DEMAND_COLUMN = "Current empty bottle crates (demand)"
ROUTE_ID_COLUMN = "route_id"
ROUTE_SEQUENCE_COLUMN = "sequence"
ROUTE_DURATION_COLUMN = "total_duration_minutes"
SCHEDULE_SLOT_COLUMN = "Time slot"
SCHEDULE_ROUTE_COLUMN = "Route ID"
DEPOT_ID = "DEPOT"

SCHEDULE_COLUMNS = [
    "Time slot",
    "Route ID",
    "Assigned stores",
    "Route duration",
    "Remaining time-slot capacity",
]


@dataclass
class LocalSearchResult:
    """Initial and improved schedules, metrics, and search diagnostics."""

    original_routes: list[dict[str, Any]]
    optimized_routes: list[dict[str, Any]]
    original_schedule: pd.DataFrame
    optimized_schedule: pd.DataFrame
    original_total_route_time: float
    optimized_total_route_time: float
    original_total_distance_km: float
    optimized_total_distance_km: float
    distance_saved_km: float
    distance_saved_percent: float
    time_saved_minutes: float
    time_saved_percent: float
    iterations: int
    relocation_moves_accepted: int
    swap_moves_accepted: int
    two_opt_moves_accepted: int
    convergence_history: pd.DataFrame
    feasible: bool
    termination_reason: str


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


def _normalize_matrix(distance_matrix: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(distance_matrix, pd.DataFrame):
        raise TypeError("distance_matrix must be a pandas DataFrame")
    matrix = distance_matrix.copy(deep=True)
    if "NodeID" in matrix.columns:
        if matrix["NodeID"].duplicated().any():
            raise ValueError("distance matrix node IDs must be unique")
        matrix = matrix.set_index("NodeID")
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
    demands: dict[str, int],
    matrix: pd.DataFrame,
    vehicle_speed_kmh: float,
    fixed_service_minutes: float,
    variable_service_minutes_per_crate: float,
) -> dict[str, float]:
    # Travel time is derived from the route's total matrix distance and vehicle speed.
    distance = sum(
        float(matrix.loc[origin, destination])
        for origin, destination in zip(sequence, sequence[1:])
    )
    travel_time = distance / vehicle_speed_kmh * 60

    # Every visited store incurs the fixed stop time plus service time per crate.
    crates = sum(demands[store_id] for store_id in sequence[1:-1])
    collection_time = len(sequence[1:-1]) * fixed_service_minutes
    collection_time += crates * variable_service_minutes_per_crate

    return {
        "total_distance_km": distance,
        "total_travel_time_minutes": travel_time,
        "total_collection_time_minutes": collection_time,
        "total_crates_collected": crates,
        "vehicle_utilization": 0.0,
        "vehicle_utilization_percent": 0.0,
        "total_duration_minutes": travel_time + collection_time,
    }


def _evaluate(
    routes: list[dict[str, Any]],
    slot_by_route: dict[str, int],
    demands: dict[str, int],
    matrix: pd.DataFrame,
    vehicle_capacity: int,
    vehicle_speed_kmh: float,
    fixed_service_minutes: float,
    variable_service_minutes_per_crate: float,
    route_duration_limit: float,
    slot_duration: float,
    max_routes_per_slot: int,
) -> tuple[bool, float, list[dict[str, Any]]]:
    """Recalculate all route metrics and check route and slot feasibility."""
    seen_stores: set[str] = set()
    evaluated_routes = []
    slot_time: dict[int, float] = {}
    slot_counts: dict[int, int] = {}
    total_time = 0.0
    feasible = True

    for route in routes:
        sequence = route[ROUTE_SEQUENCE_COLUMN]
        route_id = route[ROUTE_ID_COLUMN]
        if len(sequence) < 3 or sequence[0] != sequence[-1]:
            feasible = False
            break
        route_stores = sequence[1:-1]
        if len(route_stores) != len(set(route_stores)):
            feasible = False
            break
        if seen_stores.intersection(route_stores):
            feasible = False
            break
        seen_stores.update(route_stores)

        metrics = _route_metrics(
            sequence,
            demands,
            matrix,
            vehicle_speed_kmh,
            fixed_service_minutes,
            variable_service_minutes_per_crate,
        )
        crates = int(metrics["total_crates_collected"])
        if crates > vehicle_capacity or metrics["total_duration_minutes"] > route_duration_limit + 1e-9:
            feasible = False
        route_slot = slot_by_route[route_id]
        slot_time[route_slot] = slot_time.get(route_slot, 0.0) + metrics[
            "total_duration_minutes"
        ]
        slot_counts[route_slot] = slot_counts.get(route_slot, 0) + 1
        total_time += metrics["total_duration_minutes"]
        evaluated_routes.append({**route, **metrics})

    if any(value > slot_duration + 1e-9 for value in slot_time.values()):
        feasible = False
    if any(value > max_routes_per_slot for value in slot_counts.values()):
        feasible = False
    return feasible, total_time, evaluated_routes


def _candidate_key(
    operation: str,
    route_ids: tuple[str, ...],
    store_ids: tuple[str, ...],
    positions: tuple[int, ...],
) -> tuple[Any, ...]:
    """Stable ordering makes equal-improvement choices reproducible."""
    return tuple(sorted(route_ids)), tuple(sorted(store_ids)), operation, positions


def _make_schedule(
    routes: list[dict[str, Any]],
    slot_by_route: dict[str, int],
    slot_duration: float,
) -> pd.DataFrame:
    rows = []
    used_by_slot: dict[int, float] = {}
    for route in sorted(routes, key=lambda item: (slot_by_route[item[ROUTE_ID_COLUMN]], item["route_order"])):
        slot = slot_by_route[route[ROUTE_ID_COLUMN]]
        used = used_by_slot.get(slot, 0.0) + route[ROUTE_DURATION_COLUMN]
        used_by_slot[slot] = used
        rows.append(
            {
                "Time slot": slot,
                "Route ID": route[ROUTE_ID_COLUMN],
                "Assigned stores": ", ".join(route[ROUTE_SEQUENCE_COLUMN][1:-1]),
                "Route duration": route[ROUTE_DURATION_COLUMN],
                "Remaining time-slot capacity": max(0.0, slot_duration - used),
            }
        )
    return pd.DataFrame(rows, columns=SCHEDULE_COLUMNS)


def improve_routes(
    routes: list[dict[str, Any]],
    schedule: pd.DataFrame,
    stores: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    vehicle_capacity: int,
    vehicle_speed_kmh: float,
    fixed_collection_time_minutes: float,
    variable_collection_time_minutes_per_crate: float,
    route_duration_limit_minutes: float,
    slot_duration_minutes: float,
    max_routes_per_slot: int,
    max_iterations: int = 100,
    min_improvement_minutes: float = 0.01,
) -> LocalSearchResult:
    """Improve route order and membership using feasible best-improvement search.

    Slot assignments and route order within each slot remain fixed. Each
    iteration evaluates every feasible relocation, pairwise swap, and 2-opt
    neighbor, then accepts the candidate with the greatest time reduction if
    it saves at least ``min_improvement_minutes``.

    Example: for DEPOT -> A -> B -> C -> D -> DEPOT with leg distances
    2, 8, 1, 7, and 3 km, reversing B-C replaces A-B and C-D with
    A-C = 1 km and B-D = 2 km. The route falls from 21 km to 9 km,
    saving 24 travel minutes at 30 km/h if all feasibility limits still hold.
    """
    if not isinstance(routes, list):
        raise TypeError("routes must be a list of route dictionaries")
    if not isinstance(schedule, pd.DataFrame):
        raise TypeError("schedule must be a pandas DataFrame")
    if not isinstance(stores, pd.DataFrame):
        raise TypeError("stores must be a pandas DataFrame")
    if not {STORE_ID_COLUMN, DEMAND_COLUMN}.issubset(stores.columns):
        raise ValueError(
            f"stores must include {STORE_ID_COLUMN!r} and {DEMAND_COLUMN!r}"
        )
    if not {SCHEDULE_SLOT_COLUMN, SCHEDULE_ROUTE_COLUMN}.issubset(schedule.columns):
        raise ValueError("schedule must include 'Time slot' and 'Route ID' columns")
    if stores[STORE_ID_COLUMN].isna().any() or stores[STORE_ID_COLUMN].duplicated().any():
        raise ValueError("store IDs must be present and unique")

    capacity_value = _finite_number("vehicle_capacity", vehicle_capacity, allow_zero=False)
    if not capacity_value.is_integer():
        raise ValueError("vehicle_capacity must be a whole crate count")
    capacity = int(capacity_value)
    speed = _finite_number("vehicle_speed_kmh", vehicle_speed_kmh, allow_zero=False)
    fixed_service = _finite_number(
        "fixed_collection_time_minutes", fixed_collection_time_minutes, allow_zero=True
    )
    variable_service = _finite_number(
        "variable_collection_time_minutes_per_crate",
        variable_collection_time_minutes_per_crate,
        allow_zero=True,
    )
    route_limit = _finite_number(
        "route_duration_limit_minutes", route_duration_limit_minutes, allow_zero=False
    )
    slot_duration = _finite_number(
        "slot_duration_minutes", slot_duration_minutes, allow_zero=False
    )
    routes_per_slot_value = _finite_number(
        "max_routes_per_slot", max_routes_per_slot, allow_zero=False
    )
    if not routes_per_slot_value.is_integer():
        raise ValueError("max_routes_per_slot must be a positive integer")
    routes_per_slot = int(routes_per_slot_value)
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, Integral):
        raise ValueError("max_iterations must be a non-negative integer")
    if max_iterations < 0:
        raise ValueError("max_iterations must be a non-negative integer")
    min_improvement = _finite_number(
        "min_improvement_minutes", min_improvement_minutes, allow_zero=False
    )

    matrix = _normalize_matrix(distance_matrix)
    demand_values = pd.to_numeric(stores[DEMAND_COLUMN], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(demand_values).all() or (demand_values < 0).any():
        raise ValueError("store demands must be finite and non-negative")
    if not np.equal(demand_values, np.floor(demand_values)).all():
        raise ValueError("store demands must be whole crate counts")
    demands = dict(zip(stores[STORE_ID_COLUMN], demand_values.astype(int)))
    if not all(isinstance(store_id, str) and store_id for store_id in demands):
        raise ValueError("store IDs must be non-empty strings")

    route_ids = [route.get(ROUTE_ID_COLUMN) for route in routes]
    if any(not isinstance(route_id, str) or not route_id for route_id in route_ids):
        raise ValueError("every route must have a non-empty route_id")
    if len(route_ids) != len(set(route_ids)):
        raise ValueError("route IDs must be unique")
    if schedule[SCHEDULE_ROUTE_COLUMN].duplicated().any():
        raise ValueError("schedule must assign each route at most once")
    schedule_routes = set(schedule[SCHEDULE_ROUTE_COLUMN])
    if schedule_routes != set(route_ids):
        raise ValueError("schedule must assign every route exactly once")

    slot_by_route: dict[str, int] = {}
    for _, row in schedule.iterrows():
        slot_value = _finite_number("time slot", row[SCHEDULE_SLOT_COLUMN], allow_zero=False)
        if not slot_value.is_integer():
            raise ValueError("time slot numbers must be positive integers")
        slot_by_route[row[SCHEDULE_ROUTE_COLUMN]] = int(slot_value)
    slot_counts = pd.Series(list(slot_by_route.values())).value_counts()
    if (slot_counts > routes_per_slot).any():
        raise ValueError("initial schedule exceeds max_routes_per_slot")

    original_routes = []
    seen_stores: set[str] = set()
    depot_nodes: set[str] = set()
    for route_order, route in enumerate(routes):
        if not isinstance(route, dict):
            raise TypeError("each route must be a dictionary")
        sequence = route.get(ROUTE_SEQUENCE_COLUMN)
        if not isinstance(sequence, (list, tuple)) or len(sequence) < 3:
            raise ValueError(f"Route {route.get(ROUTE_ID_COLUMN)!r} must have stores")
        sequence = list(sequence)
        if sequence[0] != sequence[-1]:
            raise ValueError(f"Route {route[ROUTE_ID_COLUMN]} must start and end at its depot")
        depot_nodes.add(sequence[0])
        if len(sequence[1:-1]) != len(set(sequence[1:-1])):
            raise ValueError(f"Route {route[ROUTE_ID_COLUMN]} visits a store more than once")
        route_stores = set(sequence[1:-1])
        if not route_stores.issubset(demands):
            raise ValueError(
                f"Route {route[ROUTE_ID_COLUMN]} contains stores missing from the store data"
            )
        if not route_stores.issubset(matrix.index):
            raise ValueError(f"Route {route[ROUTE_ID_COLUMN]} has stores missing from the matrix")
        if sequence[0] not in matrix.index:
            raise ValueError(f"Route {route[ROUTE_ID_COLUMN]} depot is missing from the matrix")
        if seen_stores.intersection(route_stores):
            raise ValueError("a store cannot appear in more than one route")
        seen_stores.update(route_stores)
        original_routes.append(
            {
                **route,
                ROUTE_SEQUENCE_COLUMN: sequence,
                "route_order": route_order,
            }
        )
    if len(depot_nodes) > 1:
        raise ValueError("all routes must use the same central depot")
    required_stores = {store_id for store_id, demand in demands.items() if demand > 0}
    if seen_stores != required_stores:
        missing_stores = sorted(required_stores - seen_stores)
        unexpected_stores = sorted(seen_stores - required_stores)
        raise ValueError(
            "initial routes must visit every positive-demand store exactly once; "
            f"missing={missing_stores}, unexpected={unexpected_stores}"
        )

    # Preserve the input schedule's row order within each slot throughout search.
    schedule_order = {
        route_id: order
        for order, route_id in enumerate(schedule[SCHEDULE_ROUTE_COLUMN].tolist())
    }
    original_routes.sort(key=lambda route: schedule_order[route[ROUTE_ID_COLUMN]])

    initial_feasible, initial_total, evaluated = _evaluate(
        original_routes,
        slot_by_route,
        demands,
        matrix,
        capacity,
        speed,
        fixed_service,
        variable_service,
        route_limit,
        slot_duration,
        routes_per_slot,
    )
    if not initial_feasible:
        raise ValueError("initial route schedule is infeasible")
    original_routes = [dict(route) for route in evaluated]
    original_schedule = _make_schedule(original_routes, slot_by_route, slot_duration)

    current_routes = [dict(route, sequence=list(route[ROUTE_SEQUENCE_COLUMN])) for route in original_routes]
    current_total = initial_total
    accepted_moves = {"relocate": 0, "swap": 0, "2-opt": 0}
    convergence_rows = [
        {
            "Iteration": 0,
            "Total route time (min)": current_total,
            "Total distance (km)": sum(route["total_distance_km"] for route in current_routes),
            "Accepted move": "Initial solution",
            "Improvement (min)": 0.0,
        }
    ]
    iterations = 0
    termination_reason = "no_improving_move"

    while iterations < max_iterations:
        iterations += 1
        candidates: dict[tuple[Any, ...], tuple[tuple[Any, ...], list[dict[str, Any]], str]] = {}
        slot_to_routes: dict[int, list[int]] = {}
        for route_index, route in enumerate(current_routes):
            slot_to_routes.setdefault(slot_by_route[route[ROUTE_ID_COLUMN]], []).append(route_index)

        def add_candidate(
            changed: list[dict[str, Any]],
            operation: str,
            related_routes: tuple[str, ...],
            related_stores: tuple[str, ...],
            positions: tuple[int, ...],
        ) -> None:
            key = tuple(
                (route[ROUTE_ID_COLUMN], tuple(route[ROUTE_SEQUENCE_COLUMN][1:-1]))
                for route in sorted(changed, key=lambda item: item[ROUTE_ID_COLUMN])
            )
            tie_key = _candidate_key(operation, related_routes, related_stores, positions)
            existing = candidates.get(key)
            if existing is None or tie_key < existing[0]:
                candidates[key] = (tie_key, changed, operation)

        for slot_route_indices in slot_to_routes.values():
            # Relocation can reorder a store inside its route or transfer it to
            # another route in the same slot; moving the only stop is disallowed
            # so every originally scheduled route remains an active route.
            for source_index in slot_route_indices:
                source = current_routes[source_index]
                source_stores = source[ROUTE_SEQUENCE_COLUMN][1:-1]
                for source_position, store_id in enumerate(source_stores, start=1):
                    for target_index in slot_route_indices:
                        target = current_routes[target_index]
                        if source_index != target_index and len(source_stores) <= 1:
                            continue
                        target_sequence = target[ROUTE_SEQUENCE_COLUMN]
                        removed_source = list(source[ROUTE_SEQUENCE_COLUMN])
                        removed_source.pop(source_position)
                        if source_index == target_index:
                            for insert_position in range(1, len(removed_source)):
                                candidate_sequence = list(removed_source)
                                candidate_sequence.insert(insert_position, store_id)
                                if candidate_sequence == source[ROUTE_SEQUENCE_COLUMN]:
                                    continue
                                changed = [
                                    dict(route, sequence=list(route[ROUTE_SEQUENCE_COLUMN]))
                                    for route in current_routes
                                ]
                                changed[source_index][ROUTE_SEQUENCE_COLUMN] = candidate_sequence
                                add_candidate(
                                    changed,
                                    "relocate",
                                    (source[ROUTE_ID_COLUMN],),
                                    (store_id,),
                                    (source_position, insert_position),
                                )
                        else:
                            removed_target = list(target_sequence)
                            for insert_position in range(1, len(removed_target)):
                                candidate_source = list(removed_source)
                                candidate_target = list(removed_target)
                                candidate_target.insert(insert_position, store_id)
                                changed = [
                                    dict(route, sequence=list(route[ROUTE_SEQUENCE_COLUMN]))
                                    for route in current_routes
                                ]
                                changed[source_index][ROUTE_SEQUENCE_COLUMN] = candidate_source
                                changed[target_index][ROUTE_SEQUENCE_COLUMN] = candidate_target
                                add_candidate(
                                    changed,
                                    "relocate",
                                    (source[ROUTE_ID_COLUMN], target[ROUTE_ID_COLUMN]),
                                    (store_id,),
                                    (source_position, insert_position),
                                )

            # Pairwise swaps are considered both within a route and between
            # routes in this same slot; no store changes its time-slot assignment.
            for first_index, second_index in combinations(slot_route_indices, 2):
                first = current_routes[first_index]
                second = current_routes[second_index]
                for first_position, first_store in enumerate(
                    first[ROUTE_SEQUENCE_COLUMN][1:-1], start=1
                ):
                    for second_position, second_store in enumerate(
                        second[ROUTE_SEQUENCE_COLUMN][1:-1], start=1
                    ):
                        changed = [
                            dict(route, sequence=list(route[ROUTE_SEQUENCE_COLUMN]))
                            for route in current_routes
                        ]
                        changed[first_index][ROUTE_SEQUENCE_COLUMN][first_position] = second_store
                        changed[second_index][ROUTE_SEQUENCE_COLUMN][second_position] = first_store
                        add_candidate(
                            changed,
                            "swap",
                            (first[ROUTE_ID_COLUMN], second[ROUTE_ID_COLUMN]),
                            (first_store, second_store),
                            (first_position, second_position),
                        )

            # 2-opt reverses every contiguous interior segment of one route.
            for route_index in slot_route_indices:
                route = current_routes[route_index]
                sequence = route[ROUTE_SEQUENCE_COLUMN]
                for start in range(1, len(sequence) - 2):
                    for end in range(start + 1, len(sequence) - 1):
                        reversed_sequence = (
                            sequence[:start] + list(reversed(sequence[start : end + 1])) + sequence[end + 1 :]
                        )
                        changed = [
                            dict(item, sequence=list(item[ROUTE_SEQUENCE_COLUMN]))
                            for item in current_routes
                        ]
                        changed[route_index][ROUTE_SEQUENCE_COLUMN] = reversed_sequence
                        add_candidate(
                            changed,
                            "2-opt",
                            (route[ROUTE_ID_COLUMN],),
                            (sequence[start], sequence[end]),
                            (start, end),
                        )

        best_candidate = None
        best_reduction = min_improvement - 1e-12
        for tie_key, candidate_routes, operation in candidates.values():
            feasible, candidate_total, evaluated_candidate = _evaluate(
                candidate_routes,
                slot_by_route,
                demands,
                matrix,
                capacity,
                speed,
                fixed_service,
                variable_service,
                route_limit,
                slot_duration,
                routes_per_slot,
            )
            if not feasible:
                continue
            reduction = current_total - candidate_total
            if reduction < min_improvement - 1e-12:
                continue
            if (
                best_candidate is None
                or reduction > best_reduction + 1e-12
                or (abs(reduction - best_reduction) <= 1e-12 and tie_key < best_candidate[0])
            ):
                best_reduction = reduction
                best_candidate = (tie_key, evaluated_candidate, operation, candidate_total)

        if best_candidate is None:
            termination_reason = "no_improving_move"
            break

        _, current_routes, operation, next_total = best_candidate
        improvement = current_total - next_total
        current_total = next_total
        accepted_moves[operation] += 1
        convergence_rows.append(
            {
                "Iteration": iterations,
                "Total route time (min)": current_total,
                "Total distance (km)": sum(
                    route["total_distance_km"] for route in current_routes
                ),
                "Accepted move": operation,
                "Improvement (min)": improvement,
            }
        )
    else:
        termination_reason = "max_iterations_reached"

    optimized_routes = [
        {key: value for key, value in route.items() if key != "route_order"}
        for route in current_routes
    ]
    original_routes_output = [
        {key: value for key, value in route.items() if key != "route_order"}
        for route in original_routes
    ]
    optimized_schedule = _make_schedule(current_routes, slot_by_route, slot_duration)
    optimized_total = sum(route[ROUTE_DURATION_COLUMN] for route in current_routes)
    original_distance = sum(route["total_distance_km"] for route in original_routes)
    optimized_distance = sum(route["total_distance_km"] for route in current_routes)
    distance_saved = original_distance - optimized_distance
    distance_saved_percent = (
        100 * distance_saved / original_distance if original_distance else 0.0
    )
    saved = initial_total - optimized_total
    saved_percent = 100 * saved / initial_total if initial_total else 0.0
    final_feasible, _, _ = _evaluate(
        current_routes,
        slot_by_route,
        demands,
        matrix,
        capacity,
        speed,
        fixed_service,
        variable_service,
        route_limit,
        slot_duration,
        routes_per_slot,
    )
    return LocalSearchResult(
        original_routes=original_routes_output,
        optimized_routes=optimized_routes,
        original_schedule=original_schedule,
        optimized_schedule=optimized_schedule,
        original_total_route_time=initial_total,
        optimized_total_route_time=optimized_total,
        original_total_distance_km=original_distance,
        optimized_total_distance_km=optimized_distance,
        distance_saved_km=distance_saved,
        distance_saved_percent=distance_saved_percent,
        time_saved_minutes=saved,
        time_saved_percent=saved_percent,
        iterations=iterations,
        relocation_moves_accepted=accepted_moves["relocate"],
        swap_moves_accepted=accepted_moves["swap"],
        two_opt_moves_accepted=accepted_moves["2-opt"],
        convergence_history=pd.DataFrame(convergence_rows),
        feasible=final_feasible,
        termination_reason=termination_reason,
    )
