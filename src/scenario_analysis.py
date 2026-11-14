"""Scenario execution using the existing route and scheduling heuristics."""

from typing import Any

import pandas as pd

from src.routing import generate_routes
from src.scheduling import SchedulingInfeasibleError, schedule_routes


def run_scenario(
    stores: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    parameters: dict[str, Any],
    scenario_name: str,
) -> dict[str, Any]:
    """Run a separate greedy route/schedule scenario without changing baseline."""
    route_result = generate_routes(
        stores=stores,
        distance_matrix=distance_matrix,
        vehicle_capacity=parameters["vehicle_capacity"],
        vehicle_speed_kmh=parameters["vehicle_speed_kmh"],
        fixed_collection_time_minutes=parameters["fixed_collection_time_minutes"],
        variable_collection_time_minutes_per_crate=parameters[
            "variable_collection_time_minutes_per_crate"
        ],
        route_duration_limit_minutes=parameters["route_duration_limit_minutes"],
    )
    try:
        schedule = schedule_routes(
            route_result.routes,
            slot_duration_minutes=parameters["slot_duration_minutes"],
            max_routes_per_slot=parameters["max_routes_per_slot"],
            number_of_slots=parameters["number_of_slots"],
        )
    except SchedulingInfeasibleError as exc:
        return {
            "scenario_name": scenario_name,
            "feasible": False,
            "routes": route_result.routes,
            "schedule": exc.partial_schedule,
            "unserved_stores": route_result.unserved_stores,
            "unscheduled_route_ids": exc.unscheduled_route_ids,
            "reason": str(exc),
            "parameters": dict(parameters),
        }

    return {
        "scenario_name": scenario_name,
        "feasible": not (
            route_result.unserved_stores["status"] == "infeasible"
        ).any()
        if not route_result.unserved_stores.empty
        else True,
        "routes": route_result.routes,
        "schedule": schedule,
        "unserved_stores": route_result.unserved_stores,
        "unscheduled_route_ids": [],
        "reason": (
            "All positive-demand stores were routed and scheduled."
            if route_result.unserved_stores.empty
            else (
                "Some stores have zero demand and are not required."
                if not (route_result.unserved_stores["status"] == "infeasible").any()
                else "One or more stores were explicitly reported infeasible by the route heuristic."
            )
        ),
        "parameters": dict(parameters),
    }


def summarize_scenario(
    result: dict[str, Any],
    vehicle_capacity: int,
    slot_duration_minutes: float,
    stores: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Calculate scenario summary from route sequences and schedule assignments."""
    routes = result["routes"]
    schedule = result["schedule"]
    total_crates = sum(int(route["total_crates_collected"]) for route in routes)
    total_distance = sum(float(route["total_distance_km"]) for route in routes)
    total_duration = sum(float(route["total_duration_minutes"]) for route in routes)
    route_count = len(routes)
    used_slots = schedule["Time slot"].nunique() if not schedule.empty else 0
    unserved = result["unserved_stores"]
    unserved_count = (
        int((unserved["status"] == "infeasible").sum())
        if not unserved.empty and "status" in unserved
        else 0
    )
    visited = {
        store_id for route in routes for store_id in route["sequence"][1:-1]
    }
    high_volume_uncollected = 0
    if stores is not None:
        high_volume_uncollected = int(
            (
                (stores["Current empty bottle crates (demand)"] / vehicle_capacity >= 0.90)
                & (~stores["Store ID"].isin(visited))
            ).sum()
        )
    return {
        "Scenario": result["scenario_name"],
        "Feasibility": "Feasible" if result["feasible"] else "Infeasible",
        "Stores served": sum(len(route["sequence"][1:-1]) for route in routes),
        "Crates collected": total_crates,
        "Total distance (km)": total_distance,
        "Total route duration (min)": total_duration,
        "Number of routes": route_count,
        "Used time slots": used_slots,
        "Vehicle utilization (%)": (
            100 * total_crates / (route_count * vehicle_capacity) if route_count else 0.0
        ),
        "Time-slot utilization (%)": (
            100 * total_duration / (used_slots * slot_duration_minutes) if used_slots else 0.0
        ),
        "High-volume stores not collected": high_volume_uncollected,
        "Unserved stores": unserved_count,
        "Reason": result["reason"],
    }
