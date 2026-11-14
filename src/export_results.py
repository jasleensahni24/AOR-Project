"""Consistent CSV and text exports for the reverse-logistics DSS."""

from typing import Any

import pandas as pd


def route_plan_frame(
    routes: list[dict[str, Any]],
    schedule: pd.DataFrame,
    stores: pd.DataFrame,
    vehicle_capacity: int,
    distance_matrix: pd.DataFrame,
    vehicle_speed_kmh: float,
    fixed_collection_time_minutes: float,
    variable_collection_time_minutes_per_crate: float,
) -> pd.DataFrame:
    """Expand routes into accurate elapsed-time rows for each stop."""
    matrix = distance_matrix
    if "NodeID" in matrix.columns:
        matrix = matrix.set_index("NodeID")
    demand = dict(
        zip(stores["Store ID"], stores["Current empty bottle crates (demand)"].astype(int))
    )
    slot_by_route = dict(zip(schedule["Route ID"], schedule["Time slot"]))
    rows = []
    for route in routes:
        route_id = route["route_id"]
        sequence = route["sequence"]
        slot = slot_by_route.get(route_id)
        used = 0
        rows.append({
                "Route ID": route_id,
                "Vehicle ID": "Vehicle 1",
                "Time slot": slot,
                "Stop sequence": 0,
                "Store ID": "DEPOT",
                "Collection quantity (crates)": 0,
                "Arrival time (elapsed min)": 0.0,
                "Departure time (elapsed min)": 0.0,
                "Travel duration (min)": 0.0,
                "Service duration (min)": 0.0,
                "Route duration (min)": route["total_duration_minutes"],
                "Remaining vehicle capacity (crates)": vehicle_capacity,
            })
        elapsed = 0.0
        prior = sequence[0]
        for stop_number, store_id in enumerate(sequence[1:-1], start=1):
            travel = float(matrix.loc[prior, store_id]) / vehicle_speed_kmh * 60
            arrival = elapsed + travel
            service = (
                fixed_collection_time_minutes
                + variable_collection_time_minutes_per_crate * demand[store_id]
            )
            departure = arrival + service
            used += demand[store_id]
            rows.append(
                {
                    "Route ID": route_id,
                    "Vehicle ID": "Vehicle 1",
                    "Time slot": slot,
                    "Stop sequence": stop_number,
                    "Store ID": store_id,
                    "Collection quantity (crates)": demand[store_id],
                    "Arrival time (elapsed min)": arrival,
                    "Departure time (elapsed min)": departure,
                    "Travel duration (min)": travel,
                    "Service duration (min)": service,
                    "Route duration (min)": route["total_duration_minutes"],
                    "Remaining vehicle capacity (crates)": vehicle_capacity - used,
                }
            )
            elapsed = departure
            prior = store_id
        return_travel = float(matrix.loc[prior, sequence[-1]]) / vehicle_speed_kmh * 60
        return_arrival = elapsed + return_travel
        rows.append(
            {
                "Route ID": route_id,
                "Vehicle ID": "Vehicle 1",
                "Time slot": slot,
                "Stop sequence": len(sequence) - 1,
                "Store ID": "DEPOT",
                "Collection quantity (crates)": 0,
                "Arrival time (elapsed min)": return_arrival,
                "Departure time (elapsed min)": return_arrival,
                "Travel duration (min)": return_travel,
                "Service duration (min)": 0.0,
                "Route duration (min)": route["total_duration_minutes"],
                "Remaining vehicle capacity (crates)": vehicle_capacity - used,
            }
        )
    return pd.DataFrame(rows)


def store_collection_frame(
    stores: pd.DataFrame,
    routes: list[dict[str, Any]],
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Return original store records plus route/slot collection assignment."""
    assignments = {}
    slot_by_route = dict(zip(schedule["Route ID"], schedule["Time slot"]))
    for route in routes:
        for sequence_number, store_id in enumerate(route["sequence"][1:-1], start=1):
            assignments[store_id] = {
                "Assigned route": route["route_id"],
                "Time slot": slot_by_route.get(route["route_id"]),
                "Stop sequence": sequence_number,
            }
    result = stores.copy(deep=True)
    result["Assigned route"] = result["Store ID"].map(
        lambda value: assignments.get(value, {}).get("Assigned route")
    )
    result["Time slot"] = result["Store ID"].map(
        lambda value: assignments.get(value, {}).get("Time slot")
    )
    result["Stop sequence"] = result["Store ID"].map(
        lambda value: assignments.get(value, {}).get("Stop sequence")
    )
    result["Collection status"] = result["Store ID"].map(
        lambda value: "Collected" if value in assignments else "Not collected"
    )
    return result


def vehicle_performance_frame(routes: list[dict[str, Any]], vehicle_capacity: int) -> pd.DataFrame:
    """Summarize route activity for the single vehicle supported by the heuristic."""
    return pd.DataFrame(
        [
            {
                "Vehicle ID": "Vehicle 1",
                "Route ID": route["route_id"],
                "Stores visited": len(route["sequence"][1:-1]),
                "Crates collected": route["total_crates_collected"],
                "Distance (km)": route["total_distance_km"],
                "Travel time (min)": route["total_travel_time_minutes"],
                "Collection time (min)": route["total_collection_time_minutes"],
                "Route duration (min)": route["total_duration_minutes"],
                "Capacity utilization (%)": 100
                * route["total_crates_collected"]
                / vehicle_capacity,
            }
            for route in routes
        ]
    )


def slot_utilization_frame(
    schedule: pd.DataFrame,
    slot_duration_minutes: float,
) -> pd.DataFrame:
    """Aggregate route times by slot from the provided schedule table."""
    if schedule.empty:
        return pd.DataFrame(
            columns=[
                "Time slot",
                "Routes",
                "Route duration (min)",
                "Remaining capacity (min)",
                "Utilization (%)",
            ]
        )
    grouped = schedule.groupby("Time slot", as_index=False).agg(
        Routes=("Route ID", "nunique"),
        **{"Route duration (min)": ("Route duration", "sum")},
    )
    grouped["Remaining capacity (min)"] = (
        slot_duration_minutes - grouped["Route duration (min)"]
    ).clip(lower=0)
    grouped["Utilization (%)"] = (
        100 * grouped["Route duration (min)"] / slot_duration_minutes
    )
    return grouped


def executive_report(
    stores: pd.DataFrame,
    initial_kpis: pd.DataFrame,
    optimized_kpis: pd.DataFrame,
    feasible: bool,
) -> str:
    """Compose a concise data-backed executive report as plain text."""
    table = initial_kpis.set_index("KPI")
    improved = optimized_kpis.set_index("KPI")
    distance_before = float(table.loc["Total distance travelled", "Initial Greedy"])
    distance_after = float(improved.loc["Total distance travelled", "Optimized Heuristic"])
    time_before = float(table.loc["Total operational time", "Initial Greedy"])
    time_after = float(improved.loc["Total operational time", "Optimized Heuristic"])
    stores_count = stores["Store ID"].nunique()
    crate_total = int(stores["Current empty bottle crates (demand)"].sum())
    return (
        "Reverse Logistics Collection and Route Optimization System\n"
        "Executive performance report\n\n"
        "Project overview\n"
        "Priority-based greedy route construction followed by best-improvement "
        "local search (relocate, swap, and 2-opt). Flexible time slots are "
        "duration-limited and are not fixed clock-time windows.\n\n"
        f"Dataset summary\nStores: {stores_count}\n"
        f"Available crates: {crate_total}\n\n"
        "Initial versus optimized performance\n"
        f"Distance: {distance_before:.2f} km -> {distance_after:.2f} km "
        f"({distance_after - distance_before:+.2f} km)\n"
        f"Operational time: {time_before:.2f} min -> {time_after:.2f} min "
        f"({time_after - time_before:+.2f} min)\n\n"
        f"Feasibility assessment\n{'Feasible' if feasible else 'Infeasible or incomplete'}\n\n"
        "Managerial recommendations\n"
        "Use the route schedule as a planning aid. Confirm road-network travel "
        "times, store opening hours, and vehicle availability before dispatch; "
        "the sample matrix represents straight-line distance and the current "
        "routing model supports one vehicle making sequential routes.\n"
    )
