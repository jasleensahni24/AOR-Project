"""End-to-end and algorithm-specific tests for the reverse-logistics heuristic."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.evaluation import compare_solutions
from src.local_search import improve_routes
from src.priority import calculate_store_priorities
from src.routing import generate_routes
from src.scheduling import (
    SchedulingInfeasibleError,
    schedule_routes,
)

STORE_ID = "Store ID"
DEMAND = "Current empty bottle crates (demand)"
STORAGE = "Maximum crate storage capacity"
PICKUP = "Pickup request"
DEPOT = "DEPOT"


def make_stores(
    store_ids: list[str],
    demands: list[int],
    *,
    requests: list[int] | None = None,
    capacities: list[int] | None = None,
    latitudes: list[float] | None = None,
    longitudes: list[float] | None = None,
) -> pd.DataFrame:
    count = len(store_ids)
    frame = pd.DataFrame(
        {
            STORE_ID: store_ids,
            "Store name": [f"Test Store {index + 1}" for index in range(count)],
            "Latitude": latitudes if latitudes is not None else [31.5 + index / 1000 for index in range(count)],
            "Longitude": longitudes if longitudes is not None else [74.8 + index / 1000 for index in range(count)],
            DEMAND: demands,
            STORAGE: capacities if capacities is not None else [10] * count,
            PICKUP: requests if requests is not None else [0] * count,
        }
    )
    return calculate_store_priorities(frame, saturation_warning_threshold=1.0)


def make_line_matrix(store_ids: list[str]) -> pd.DataFrame:
    nodes = [DEPOT, *store_ids]
    positions = {node: index for index, node in enumerate(nodes)}
    return pd.DataFrame(
        [
            [
                0.0 if first == second else 1.0 + abs(positions[first] - positions[second])
                for second in nodes
            ]
            for first in nodes
        ],
        index=nodes,
        columns=nodes,
    )


def make_route(route_id: str, duration: float, stores: list[str] | None = None) -> dict[str, Any]:
    assigned = stores if stores is not None else [route_id]
    return {
        "route_id": route_id,
        "sequence": [DEPOT, *assigned, DEPOT],
        "total_duration_minutes": duration,
    }


def route_parameters(**overrides: Any) -> dict[str, Any]:
    return {
        "vehicle_capacity": 5,
        "vehicle_speed_kmh": 30,
        "fixed_collection_time_minutes": 4,
        "variable_collection_time_minutes_per_crate": 0.25,
        "route_duration_limit_minutes": 120,
        **overrides,
    }


def run_routing(
    stores: pd.DataFrame,
    matrix: pd.DataFrame,
    **overrides: Any,
):
    return generate_routes(stores, matrix, **route_parameters(**overrides))


def independent_route_metrics(
    sequence: list[str],
    stores: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    speed: float,
    fixed_service: float,
    variable_service: float,
) -> dict[str, float]:
    demand = stores.set_index(STORE_ID)[DEMAND].to_dict()
    distance = sum(
        float(matrix.loc[first, second])
        for first, second in zip(sequence, sequence[1:])
    )
    crates = sum(int(demand[store]) for store in sequence[1:-1])
    travel = distance / speed * 60
    service = len(sequence[1:-1]) * fixed_service + crates * variable_service
    return {
        "distance": distance,
        "crates": crates,
        "travel": travel,
        "service": service,
        "duration": travel + service,
    }


def assert_operational_invariants(
    stores: pd.DataFrame,
    matrix: pd.DataFrame,
    routes: list[dict[str, Any]],
    schedule: pd.DataFrame,
    unserved: pd.DataFrame,
    *,
    capacity: int,
    route_limit: float,
    slot_duration: float,
    max_routes_per_slot: int | None = None,
    speed: float = 30,
    fixed_service: float = 4,
    variable_service: float = 0.25,
) -> None:
    demand_by_id = stores.set_index(STORE_ID)[DEMAND].astype(int).to_dict()
    visited: list[str] = []
    route_ids = [route["route_id"] for route in routes]
    assert len(route_ids) == len(set(route_ids))
    assert schedule["Route ID"].is_unique
    assert set(schedule["Route ID"]) == set(route_ids)

    per_slot: dict[Any, float] = defaultdict(float)
    slot_route_count: dict[Any, int] = defaultdict(int)
    route_duration_by_id: dict[str, float] = {}
    for route in routes:
        sequence = route["sequence"]
        assert sequence[0] == DEPOT and sequence[-1] == DEPOT
        route_stores = sequence[1:-1]
        assert len(route_stores) == len(set(route_stores))
        visited.extend(route_stores)
        metrics = independent_route_metrics(
            sequence,
            stores,
            matrix,
            speed=speed,
            fixed_service=fixed_service,
            variable_service=variable_service,
        )
        assert metrics["crates"] <= capacity
        assert metrics["duration"] <= route_limit + 1e-9
        assert route["total_distance_km"] == pytest.approx(metrics["distance"])
        assert route["total_travel_time_minutes"] == pytest.approx(metrics["travel"])
        assert route["total_collection_time_minutes"] == pytest.approx(metrics["service"])
        assert route["total_duration_minutes"] == pytest.approx(metrics["duration"])
        assert route["total_duration_minutes"] == pytest.approx(
            route["total_travel_time_minutes"] + route["total_collection_time_minutes"]
        )
        slot = schedule.loc[schedule["Route ID"] == route["route_id"], "Time slot"].iloc[0]
        per_slot[slot] += metrics["duration"]
        slot_route_count[slot] += 1
        route_duration_by_id[route["route_id"]] = metrics["duration"]

    assert len(visited) == len(set(visited))
    eligible = {store_id for store_id, crates in demand_by_id.items() if crates > 0}
    explicitly_unserved = set(unserved[STORE_ID]) if not unserved.empty else set()
    assert set(visited).isdisjoint(explicitly_unserved)
    assert set(visited) | explicitly_unserved == set(demand_by_id)
    assert set(visited) == eligible - explicitly_unserved.intersection(eligible)
    assert all(duration <= slot_duration + 1e-9 for duration in per_slot.values())
    if max_routes_per_slot is not None:
        assert all(count <= max_routes_per_slot for count in slot_route_count.values())
    used_by_slot: dict[Any, float] = defaultdict(float)
    for _, assignment in schedule.iterrows():
        route_id = assignment["Route ID"]
        slot = assignment["Time slot"]
        expected_duration = route_duration_by_id[route_id]
        assert assignment["Route duration"] == pytest.approx(expected_duration)
        used_by_slot[slot] += expected_duration
        assert assignment["Remaining time-slot capacity"] == pytest.approx(
            max(0.0, slot_duration - used_by_slot[slot])
        )


@pytest.fixture
def twenty_store_data():
    store_ids = [f"S{index:02d}" for index in range(1, 21)]
    demands = [index % 3 + 1 for index in range(20)]
    stores = make_stores(store_ids, demands, requests=[index % 2 for index in range(20)])
    return stores, make_line_matrix(store_ids)


@pytest.fixture
def improving_search_data():
    nodes = [DEPOT, "A", "B", "C", "D"]
    matrix = pd.DataFrame(
        [
            [0, 2, 5, 3, 4],
            [2, 0, 8, 2, 4],
            [5, 8, 0, 1, 2],
            [3, 2, 1, 0, 7],
            [4, 4, 2, 7, 0],
        ],
        index=nodes,
        columns=nodes,
    )
    stores = make_stores(["A", "B", "C", "D"], [1, 1, 1, 1])
    routes = [
        make_route("R1", 0, ["A", "B", "C"]),
        make_route("R2", 0, ["D"]),
    ]
    schedule = pd.DataFrame(
        {"Time slot": [1, 1], "Route ID": ["R1", "R2"]}
    )
    return stores, matrix, routes, schedule


@pytest.mark.case(
    id="H-01",
    name="Twenty-store feasible pipeline",
    inputs="20 deterministic stores; capacity 8; route limit 180 min; three 300-min slots.",
    expected="All eligible stores served once, scheduled feasibly, and independently audited.",
)
def test_normal_operation_with_twenty_stores(twenty_store_data):
    stores, matrix = twenty_store_data
    result = run_routing(stores, matrix, vehicle_capacity=8, route_duration_limit_minutes=180)
    schedule = schedule_routes(
        result.routes,
        slot_duration_minutes=300,
        max_routes_per_slot=3,
        number_of_slots=3,
    )
    assert result.unserved_stores.empty
    assert len({store for route in result.routes for store in route["sequence"][1:-1]}) == 20
    assert_operational_invariants(
        stores,
        matrix,
        result.routes,
        schedule,
        result.unserved_stores,
        capacity=8,
        route_limit=180,
        slot_duration=300,
        max_routes_per_slot=3,
    )


@pytest.mark.case(
    id="H-02",
    name="Full storage utilization priority",
    inputs="One store has 4 crates demand and 4 crates storage capacity.",
    expected="Filling rate is 1.0 and the store is flagged at saturation.",
)
def test_store_at_one_hundred_percent_storage_utilization():
    with pytest.warns(UserWarning, match="100% storage capacity"):
        stores = make_stores(["FULL"], [4], capacities=[4])
    assert stores.iloc[0]["Filling rate"] == pytest.approx(1.0)
    assert bool(stores.iloc[0]["Approaching saturation"])


@pytest.mark.case(
    id="H-03",
    name="Explicit pickup request priority",
    inputs="Two otherwise identical one-crate stores; one requests collection.",
    expected="Pickup request raises priority by 0.30 and both stores remain routable.",
)
def test_explicit_pickup_request_changes_priority_and_is_served():
    stores = make_stores(["NO_REQUEST", "REQUEST"], [1, 1], requests=[0, 1])
    priority = stores.set_index(STORE_ID)["Collection priority score"]
    assert priority["REQUEST"] - priority["NO_REQUEST"] == pytest.approx(0.30)
    result = run_routing(stores, make_line_matrix(["NO_REQUEST", "REQUEST"]))
    visited = {store for route in result.routes for store in route["sequence"][1:-1]}
    assert visited == {"NO_REQUEST", "REQUEST"}
    assert result.unserved_stores.empty


@pytest.mark.case(
    id="H-04",
    name="Vehicle reaches exact capacity",
    inputs="Two stores demand 2 and 3 crates; vehicle capacity is 5 crates.",
    expected="A feasible route collects exactly 5 crates without exceeding capacity.",
)
def test_vehicle_can_reach_maximum_capacity_exactly():
    stores = make_stores(["A", "B"], [2, 3], requests=[1, 1])
    matrix = make_line_matrix(["A", "B"])
    result = run_routing(stores, matrix, vehicle_capacity=5)
    route = next(route for route in result.routes if route["total_crates_collected"] == 5)
    assert route["total_crates_collected"] == 5
    assert_operational_invariants(
        stores, matrix, result.routes,
        schedule_routes(result.routes, 120, 2, 2), result.unserved_stores,
        capacity=5, route_limit=120, slot_duration=120,
        max_routes_per_slot=2,
    )


@pytest.mark.case(
    id="H-05",
    name="Route duration limit reporting",
    inputs="One store requires an 8.25-minute visit; route limit is 8 minutes.",
    expected="The store is reported infeasible and no over-duration route is emitted.",
)
def test_route_exceeding_duration_limit_is_reported():
    stores = make_stores(["A"], [1])
    matrix = pd.DataFrame([[0, 1], [1, 0]], index=[DEPOT, "A"], columns=[DEPOT, "A"])
    result = run_routing(stores, matrix, route_duration_limit_minutes=8)
    assert result.routes == []
    assert result.unserved_stores.iloc[0]["Store ID"] == "A"
    assert result.unserved_stores.iloc[0]["status"] == "infeasible"


@pytest.mark.case(
    id="H-06",
    name="No explicit pickup requests",
    inputs="Three positive-demand stores; all pickup request values are 0.",
    expected="Priority depends on filling rate only and all feasible stores are served.",
)
def test_no_pickup_request_from_any_store():
    stores = make_stores(["A", "B", "C"], [1, 2, 3], requests=[0, 0, 0])
    assert (stores[PICKUP] == 0).all()
    assert np.allclose(stores["Collection priority score"], 0.70 * stores["Filling rate"])
    result = run_routing(stores, make_line_matrix(["A", "B", "C"]))
    assert not result.unserved_stores["status"].eq("infeasible").any()
    assert {s for r in result.routes for s in r["sequence"][1:-1]} == {"A", "B", "C"}


@pytest.mark.case(
    id="H-07",
    name="All stores have zero demand",
    inputs="Two stores with zero demand; distance matrix contains depot and both stores.",
    expected="No route is created; each store is marked not_required, not silently omitted.",
)
def test_all_stores_zero_demand():
    stores = make_stores(["A", "B"], [0, 0])
    matrix = make_line_matrix(["A", "B"])
    result = run_routing(stores, matrix)
    assert result.routes == []
    assert set(result.unserved_stores[STORE_ID]) == {"A", "B"}
    assert set(result.unserved_stores["status"]) == {"not_required"}
    schedule = schedule_routes([], 120, 2, 2)
    assert schedule.empty
    assert_operational_invariants(
        stores, matrix, result.routes, schedule, result.unserved_stores,
        capacity=5, route_limit=120, slot_duration=120,
    )


@pytest.mark.case(
    id="H-08",
    name="Demand exceeds vehicle capacity",
    inputs="One store demands 6 crates; vehicle capacity is 5.",
    expected="Store is explicitly reported infeasible and is not placed on a route.",
)
def test_insufficient_vehicle_capacity_is_explicit():
    stores = make_stores(["OVERSIZE"], [6])
    matrix = make_line_matrix(["OVERSIZE"])
    result = run_routing(stores, matrix, vehicle_capacity=5)
    assert result.routes == []
    assert result.unserved_stores.iloc[0]["status"] == "infeasible"
    assert result.unserved_stores.iloc[0]["reason"] == "store_demand_exceeds_vehicle_capacity"


@pytest.mark.case(
    id="H-09",
    name="Insufficient time slots",
    inputs="Two 6-minute routes; one slot has 10 minutes and at most two routes.",
    expected="Typed infeasibility contains feasible=False, assigned partial schedule, and remaining route IDs.",
)
def test_insufficient_time_slots_returns_structured_infeasibility():
    routes = [make_route("R1", 6), make_route("R2", 6)]
    with pytest.raises(SchedulingInfeasibleError) as caught:
        schedule_routes(routes, 10, max_routes_per_slot=2, number_of_slots=1)
    error = caught.value
    assert error.feasible is False
    assert error.partial_schedule["Route ID"].tolist() == ["R1"]
    assert error.unscheduled_route_ids == ["R2"]
    assert error.partial_schedule["Route duration"].sum() <= 10


@pytest.mark.case(
    id="H-10",
    name="Duplicate store IDs rejected",
    inputs="Two input rows share Store ID 'A'.",
    expected="Route generation raises a clear validation error before creating assignments.",
)
def test_duplicate_store_ids_are_rejected():
    stores = make_stores(["A", "A"], [1, 1])
    with pytest.raises(ValueError, match="present and unique"):
        run_routing(stores, make_line_matrix(["A", "B"]))


@pytest.mark.parametrize(
    ("coordinate_field", "bad_value", "message"),
    [
        pytest.param("Latitude", np.nan, "finite", id="missing-coordinate"),
        pytest.param("Latitude", 91.0, "between -90 and 90", id="out-of-range-latitude"),
        pytest.param("Longitude", 181.0, "between -180 and 180", id="out-of-range-longitude"),
        pytest.param("Latitude", "north", "numeric", id="nonnumeric-coordinate"),
    ],
)
@pytest.mark.case(
    id="H-11",
    name="Missing or invalid coordinates",
    inputs="A required store coordinate is missing, nonnumeric, or outside geographic bounds.",
    expected="Route generation rejects the invalid coordinate input explicitly.",
)
def test_missing_or_invalid_coordinates_are_rejected(coordinate_field, bad_value, message):
    stores = make_stores(["A"], [1])
    if isinstance(bad_value, str):
        stores[coordinate_field] = stores[coordinate_field].astype(object)
    stores.loc[0, coordinate_field] = bad_value
    with pytest.raises(ValueError, match=message):
        run_routing(stores, make_line_matrix(["A"]))


@pytest.mark.case(
    id="H-12",
    name="Local search improves feasible route set",
    inputs="Two routes share one slot; route orders include a deliberately long leg.",
    expected="Local search returns a feasible strict time improvement without changing slot assignments.",
)
def test_local_search_improves_valid_initial_solution(improving_search_data):
    stores, matrix, routes, schedule = improving_search_data
    result = improve_routes(
        routes, schedule, stores, matrix,
        vehicle_capacity=4,
        vehicle_speed_kmh=30,
        fixed_collection_time_minutes=0,
        variable_collection_time_minutes_per_crate=0,
        route_duration_limit_minutes=120,
        slot_duration_minutes=120,
        max_routes_per_slot=4,
    )
    assert result.feasible
    assert result.optimized_total_route_time < result.original_total_route_time
    assert result.optimized_total_route_time <= result.original_total_route_time
    assert result.termination_reason == "no_improving_move"
    assert result.original_schedule["Time slot"].tolist() == result.optimized_schedule["Time slot"].tolist()
    assert result.convergence_history["Total route time (min)"].is_monotonic_decreasing
    assert (result.convergence_history["Improvement (min)"].iloc[1:] > 0).all()
    assert _store_slots(result.original_routes, result.original_schedule) == _store_slots(
        result.optimized_routes, result.optimized_schedule
    )
    assert_operational_invariants(
        stores,
        matrix,
        result.optimized_routes,
        result.optimized_schedule,
        pd.DataFrame(columns=[STORE_ID]),
        capacity=4,
        route_limit=120,
        slot_duration=120,
        max_routes_per_slot=4,
        fixed_service=0,
        variable_service=0,
    )


@pytest.mark.case(
    id="H-13",
    name="Local search has no improving move",
    inputs="One-stop route has no alternate ordering or legal relocation.",
    expected="Initial route remains unchanged and search terminates with no_improving_move.",
)
def test_local_search_finds_no_improving_move():
    stores = make_stores(["A"], [1])
    matrix = pd.DataFrame([[0, 1], [1, 0]], index=[DEPOT, "A"], columns=[DEPOT, "A"])
    routes = [make_route("R1", 0, ["A"])]
    schedule = pd.DataFrame({"Time slot": [1], "Route ID": ["R1"]})
    result = improve_routes(
        routes, schedule, stores, matrix,
        vehicle_capacity=5,
        vehicle_speed_kmh=30,
        fixed_collection_time_minutes=4,
        variable_collection_time_minutes_per_crate=0.25,
        route_duration_limit_minutes=120,
        slot_duration_minutes=120,
        max_routes_per_slot=1,
    )
    assert result.termination_reason == "no_improving_move"
    assert result.optimized_total_route_time == pytest.approx(result.original_total_route_time)
    assert result.optimized_routes[0]["sequence"] == ["DEPOT", "A", "DEPOT"]
    assert_operational_invariants(
        stores,
        matrix,
        result.optimized_routes,
        result.optimized_schedule,
        pd.DataFrame(columns=[STORE_ID]),
        capacity=5,
        route_limit=120,
        slot_duration=120,
        max_routes_per_slot=1,
    )


def _store_slots(routes: list[dict[str, Any]], schedule: pd.DataFrame) -> dict[str, int]:
    route_to_slot = schedule.set_index("Route ID")["Time slot"].to_dict()
    return {
        store: int(route_to_slot[route["route_id"]])
        for route in routes
        for store in route["sequence"][1:-1]
    }


def _one_slot_neighborhood_signatures(
    routes: list[dict[str, Any]],
) -> set[tuple[tuple[str, tuple[str, ...]], ...]]:
    """Independently enumerate unique relocate, swap, and 2-opt neighbors."""
    sequences = [list(route["sequence"]) for route in routes]
    neighbors: set[tuple[tuple[str, tuple[str, ...]], ...]] = set()

    def add(candidate_sequences: list[list[str]]) -> None:
        signature = tuple(
            sorted(
                (route["route_id"], tuple(sequence[1:-1]))
                for route, sequence in zip(routes, candidate_sequences, strict=True)
            )
        )
        original = tuple(
            sorted(
                (route["route_id"], tuple(sequence[1:-1]))
                for route, sequence in zip(routes, sequences, strict=True)
            )
        )
        if signature != original:
            neighbors.add(signature)

    for source_index, source_sequence in enumerate(sequences):
        source_stores = source_sequence[1:-1]
        for source_position, store in enumerate(source_stores, start=1):
            removed_source = list(source_sequence)
            removed_source.pop(source_position)
            for target_index, target_sequence in enumerate(sequences):
                if source_index != target_index and len(source_stores) <= 1:
                    continue
                if source_index == target_index:
                    for insert_position in range(1, len(removed_source)):
                        changed = [list(sequence) for sequence in sequences]
                        changed[source_index] = list(removed_source)
                        changed[source_index].insert(insert_position, store)
                        add(changed)
                else:
                    for insert_position in range(1, len(target_sequence)):
                        changed = [list(sequence) for sequence in sequences]
                        changed[source_index] = list(removed_source)
                        changed[target_index].insert(insert_position, store)
                        add(changed)

    for first_index, second_index in combinations(range(len(sequences)), 2):
        for first_position, first_store in enumerate(sequences[first_index][1:-1], start=1):
            for second_position, second_store in enumerate(
                sequences[second_index][1:-1], start=1
            ):
                changed = [list(sequence) for sequence in sequences]
                changed[first_index][first_position] = second_store
                changed[second_index][second_position] = first_store
                add(changed)

    for route_index, sequence in enumerate(sequences):
        for start in range(1, len(sequence) - 2):
            for end in range(start + 1, len(sequence) - 1):
                changed = [list(item) for item in sequences]
                changed[route_index] = (
                    sequence[:start]
                    + list(reversed(sequence[start : end + 1]))
                    + sequence[end + 1 :]
                )
                add(changed)
    return neighbors


@pytest.mark.case(
    id="H-14",
    name="Local search rejects capacity violation",
    inputs="Two full two-crate routes; a cross-route relocation would raise target load above capacity 2.",
    expected="Infeasible candidate is observed/rejected; accepted route loads remain at or below capacity.",
)
def test_local_search_attempting_capacity_violation_is_rejected(
    improving_search_data, monkeypatch
):
    import src.local_search as local_search_module

    stores, matrix, _, _ = improving_search_data
    stores = make_stores(["A", "B", "C", "D"], [1, 1, 1, 1])
    routes = [
        make_route("R1", 0, ["A", "B"]),
        make_route("R2", 0, ["C", "D"]),
    ]
    schedule = pd.DataFrame({"Time slot": [1, 1], "Route ID": ["R1", "R2"]})
    original_evaluate = local_search_module._evaluate
    infeasible_candidate_seen = False

    def observe_evaluate(*args, **kwargs):
        nonlocal infeasible_candidate_seen
        evaluated = original_evaluate(*args, **kwargs)
        if not evaluated[0]:
            infeasible_candidate_seen = True
        return evaluated

    monkeypatch.setattr(local_search_module, "_evaluate", observe_evaluate)
    result = improve_routes(
        routes, schedule, stores, matrix,
        vehicle_capacity=2,
        vehicle_speed_kmh=30,
        fixed_collection_time_minutes=0,
        variable_collection_time_minutes_per_crate=0,
        route_duration_limit_minutes=120,
        slot_duration_minutes=240,
        max_routes_per_slot=2,
        max_iterations=2,
    )
    assert infeasible_candidate_seen
    assert result.feasible
    assert all(route["total_crates_collected"] <= 2 for route in result.optimized_routes)
    assert_operational_invariants(
        stores,
        matrix,
        result.optimized_routes,
        result.optimized_schedule,
        pd.DataFrame(columns=[STORE_ID]),
        capacity=2,
        route_limit=120,
        slot_duration=240,
        max_routes_per_slot=2,
        fixed_service=0,
        variable_service=0,
    )


@pytest.mark.case(
    id="H-15",
    name="Local search preserves time-slot assignments",
    inputs="Routes occupy two separate slots and may improve internally.",
    expected="Every store remains assigned to its original slot; no cross-slot transfer occurs.",
)
def test_local_search_does_not_relocate_or_swap_across_time_slots(improving_search_data):
    stores, matrix, _, _ = improving_search_data
    routes = [
        make_route("R1", 0, ["A", "B"]),
        make_route("R2", 0, ["C", "D"]),
    ]
    schedule = pd.DataFrame({"Time slot": [1, 2], "Route ID": ["R1", "R2"]})
    result = improve_routes(
        routes, schedule, stores, matrix,
        vehicle_capacity=4,
        vehicle_speed_kmh=30,
        fixed_collection_time_minutes=0,
        variable_collection_time_minutes_per_crate=0,
        route_duration_limit_minutes=120,
        slot_duration_minutes=120,
        max_routes_per_slot=1,
    )
    assert _store_slots(result.original_routes, result.original_schedule) == {
        "A": 1, "B": 1, "C": 2, "D": 2
    }
    assert _store_slots(result.optimized_routes, result.optimized_schedule) == {
        "A": 1, "B": 1, "C": 2, "D": 2
    }
    assert_operational_invariants(
        stores,
        matrix,
        result.optimized_routes,
        result.optimized_schedule,
        pd.DataFrame(columns=[STORE_ID]),
        capacity=4,
        route_limit=120,
        slot_duration=120,
        max_routes_per_slot=1,
        fixed_service=0,
        variable_service=0,
    )


@pytest.mark.case(
    id="H-16",
    name="Best-fit scheduling and tie-breaking",
    inputs="Route durations [6, 6, 4, 4], slot length 10, up to 2 routes per slot.",
    expected="Input route order is processed; best fit assigns ties to the earlier slot and opens slots only as needed.",
)
def test_best_fit_order_tie_break_and_new_slot_behavior():
    routes = [
        make_route("R1", 6),
        make_route("R2", 6),
        make_route("R3", 4),
        make_route("R4", 4),
    ]
    schedule = schedule_routes(routes, 10, max_routes_per_slot=2, number_of_slots=2)
    assignments = schedule.set_index("Route ID")["Time slot"].to_dict()
    assert assignments == {"R1": 1, "R2": 2, "R3": 1, "R4": 2}
    assert schedule["Route ID"].tolist() == ["R1", "R3", "R2", "R4"]
    assert schedule["Remaining time-slot capacity"].tolist() == [4, 0, 4, 0]


@pytest.mark.case(
    id="H-17",
    name="Every eligible store served or reported",
    inputs="One feasible store, one over-capacity store, and one zero-demand store.",
    expected="Feasible store is visited; all other stores have explicit infeasible/not_required statuses.",
)
def test_all_eligible_stores_served_or_explicitly_unserved():
    stores = make_stores(["OK", "TOO_LARGE", "ZERO"], [2, 6, 0])
    matrix = make_line_matrix(["OK", "TOO_LARGE", "ZERO"])
    result = run_routing(stores, matrix, vehicle_capacity=5)
    visited = {store for route in result.routes for store in route["sequence"][1:-1]}
    reports = result.unserved_stores.set_index(STORE_ID)["status"].to_dict()
    assert visited == {"OK"}
    assert reports == {"TOO_LARGE": "infeasible", "ZERO": "not_required"}


@pytest.mark.parametrize(
    "stores",
    [
        pytest.param(
            pd.DataFrame(
                columns=[
                    STORE_ID, "Latitude", "Longitude", DEMAND,
                    "Collection priority score",
                ]
            ),
            id="empty-dataframe",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    STORE_ID: ["A"],
                    "Latitude": [31.5],
                    "Longitude": [74.8],
                    DEMAND: [-1],
                    "Collection priority score": [0.5],
                }
            ),
            id="negative-demand",
        ),
    ],
)
@pytest.mark.case(
    id="H-18",
    name="Empty and invalid input data",
    inputs="An empty but correctly shaped store table or a negative crate demand.",
    expected="Empty data safely returns no routes; negative demand is rejected.",
)
def test_empty_dataset_and_invalid_demand(stores):
    matrix = make_line_matrix(["A"]) if not stores.empty else pd.DataFrame(
        [[0.0]], index=[DEPOT], columns=[DEPOT]
    )
    if stores.empty:
        result = run_routing(stores, matrix)
        assert result.routes == []
        assert result.unserved_stores.empty
    else:
        with pytest.raises(ValueError, match="non-negative whole crate"):
            run_routing(stores, matrix)


@pytest.mark.case(
    id="H-19",
    name="Steepest descent selects the best candidate",
    inputs="One iteration of local search examines all feasible route-neighborhood candidates.",
    expected="The accepted candidate has the minimum feasible duration captured during that iteration.",
)
def test_best_improvement_selects_greatest_duration_reduction(
    improving_search_data, monkeypatch
):
    import src.local_search as local_search_module

    stores, matrix, routes, schedule = improving_search_data
    original_evaluate = local_search_module._evaluate
    evaluation_trace: list[
        tuple[tuple[tuple[str, tuple[str, ...]], ...], bool, float]
    ] = []

    def capture_evaluate(*args, **kwargs):
        evaluated = original_evaluate(*args, **kwargs)
        signature = tuple(
            sorted(
                (
                    route["route_id"],
                    tuple(route["sequence"][1:-1]),
                )
                for route in args[0]
            )
        )
        evaluation_trace.append((signature, evaluated[0], evaluated[1]))
        return evaluated

    monkeypatch.setattr(local_search_module, "_evaluate", capture_evaluate)
    result = improve_routes(
        routes, schedule, stores, matrix,
        vehicle_capacity=4,
        vehicle_speed_kmh=30,
        fixed_collection_time_minutes=0,
        variable_collection_time_minutes_per_crate=0,
        route_duration_limit_minutes=120,
        slot_duration_minutes=120,
        max_routes_per_slot=4,
        max_iterations=1,
    )
    assert result.iterations == 1
    assert result.termination_reason == "max_iterations_reached"
    expected_candidates = _one_slot_neighborhood_signatures(routes)
    observed_candidates = [trace[0] for trace in evaluation_trace[1:-1]]
    assert len(observed_candidates) == len(expected_candidates)
    assert set(observed_candidates) == expected_candidates
    feasible_candidate_durations = [
        duration for _, feasible, duration in evaluation_trace[1:-1] if feasible
    ]
    assert result.optimized_total_route_time == pytest.approx(
        min(feasible_candidate_durations)
    )
    assert result.optimized_total_route_time < result.original_total_route_time
    assert result.optimized_total_route_time >= 0


@pytest.mark.case(
    id="H-20",
    name="Fixed-slot priority delay remains constant",
    inputs="Local search changes route structure but keeps every store in its original slot.",
    expected="Store-to-slot mapping and any score based only on store priority and assigned slot are unchanged.",
)
def test_fixed_assignments_keep_priority_delay_score_unchanged(improving_search_data):
    stores, matrix, routes, schedule = improving_search_data
    result = improve_routes(
        routes, schedule, stores, matrix,
        vehicle_capacity=4,
        vehicle_speed_kmh=30,
        fixed_collection_time_minutes=0,
        variable_collection_time_minutes_per_crate=0,
        route_duration_limit_minutes=120,
        slot_duration_minutes=120,
        max_routes_per_slot=4,
    )
    priority = stores.set_index(STORE_ID)["Collection priority score"].to_dict()
    initial_slots = _store_slots(result.original_routes, result.original_schedule)
    final_slots = _store_slots(result.optimized_routes, result.optimized_schedule)
    initial_delay_score = sum(priority[store] * slot for store, slot in initial_slots.items())
    final_delay_score = sum(priority[store] * slot for store, slot in final_slots.items())
    assert final_slots == initial_slots
    assert final_delay_score == pytest.approx(initial_delay_score)


@pytest.mark.case(
    id="H-21",
    name="KPI formulas and high-volume threshold",
    inputs="Demand [9, 1, 0], capacity 10; only the one-crate store is collected.",
    expected="Coverage 50%; missed high-volume 1; vehicle utilization 10%; slot utilization independently recomputed.",
)
def test_kpi_formulas_and_threshold_are_independently_recomputed():
    stores = make_stores(["HIGH", "LOW", "ZERO"], [9, 1, 0])
    matrix = pd.DataFrame(
        [[0, 1, 1, 1], [1, 0, 2, 2], [1, 2, 0, 2], [1, 2, 2, 0]],
        index=[DEPOT, "HIGH", "LOW", "ZERO"],
        columns=[DEPOT, "HIGH", "LOW", "ZERO"],
    )
    routes = [{"route_id": "R1", "sequence": [DEPOT, "LOW", DEPOT]}]
    schedule = pd.DataFrame({"Time slot": [1], "Route ID": ["R1"]})
    table = compare_solutions(
        stores, routes, schedule, routes, schedule, matrix,
        vehicle_capacity=10,
        vehicle_speed_kmh=30,
        fixed_collection_time_minutes=4,
        variable_collection_time_minutes_per_crate=0.25,
        slot_duration_minutes=100,
        route_duration_limit_minutes=100,
        number_of_available_time_slots=2,
    ).set_index("KPI")
    assert table.loc["Percentage of stores served", "Initial Greedy"] == pytest.approx(50.0)
    assert table.loc[
        "High-volume stores (>=90% vehicle capacity) not collected", "Initial Greedy"
    ] == 1
    assert table.loc["Vehicle capacity utilization", "Initial Greedy"] == pytest.approx(10.0)
    expected_duration = (2 / 30 * 60) + 4 + 0.25
    assert table.loc["Total time-slot utilization", "Initial Greedy"] == pytest.approx(
        100 * expected_duration / (2 * 100)
    )


@pytest.mark.case(
    id="H-22",
    name="KPI zero-route and zero-demand edge cases",
    inputs="Two positive-demand stores, no routes, and an empty schedule.",
    expected="Coverage and utilization return safe zero values without division errors.",
)
def test_kpi_zero_routes_empty_schedule_and_zero_demand():
    stores = make_stores(["A", "B"], [1, 2])
    matrix = make_line_matrix(["A", "B"])
    empty_schedule = pd.DataFrame(columns=["Time slot", "Route ID"])
    table = compare_solutions(
        stores, [], empty_schedule, [], empty_schedule, matrix,
        vehicle_capacity=5,
        vehicle_speed_kmh=30,
        fixed_collection_time_minutes=4,
        variable_collection_time_minutes_per_crate=0.25,
        slot_duration_minutes=100,
        number_of_available_time_slots=4,
    ).set_index("KPI")
    assert table.loc["Percentage of stores served", "Initial Greedy"] == 0
    assert table.loc["Vehicle capacity utilization", "Initial Greedy"] == 0
    assert table.loc["Total time-slot utilization", "Initial Greedy"] == 0
    assert table.loc["Number of routes", "Initial Greedy"] == 0

    no_demand = make_stores(["ZERO"], [0])
    zero_matrix = make_line_matrix(["ZERO"])
    zero_table = compare_solutions(
        no_demand, [], empty_schedule, [], empty_schedule, zero_matrix,
        vehicle_capacity=5,
        vehicle_speed_kmh=30,
        fixed_collection_time_minutes=4,
        variable_collection_time_minutes_per_crate=0.25,
        slot_duration_minutes=100,
    ).set_index("KPI")
    assert zero_table.loc["Percentage of stores served", "Initial Greedy"] == 0
    assert zero_table.loc["High-volume stores (>=90% vehicle capacity) not collected", "Initial Greedy"] == 0


@pytest.mark.case(
    id="H-23",
    name="KPI rejects assignment beyond available slots",
    inputs="A scheduled route is assigned to slot 2 while only one slot is configured.",
    expected="KPI evaluation rejects the schedule rather than counting unavailable slot resources.",
)
def test_kpi_rejects_time_slot_beyond_configured_availability():
    stores = make_stores(["A"], [1])
    matrix = make_line_matrix(["A"])
    route = [{"route_id": "R1", "sequence": [DEPOT, "A", DEPOT]}]
    schedule = pd.DataFrame({"Time slot": [2], "Route ID": ["R1"]})
    with pytest.raises(ValueError, match="exceeds the available time-slot count"):
        compare_solutions(
            stores, route, schedule, route, schedule, matrix,
            vehicle_capacity=5,
            vehicle_speed_kmh=30,
            fixed_collection_time_minutes=4,
            variable_collection_time_minutes_per_crate=0.25,
            slot_duration_minutes=120,
            number_of_available_time_slots=1,
        )
