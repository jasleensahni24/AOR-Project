"""Reusable route, space-time, and vehicle schedule visualizations."""

from typing import Any

import pandas as pd
import plotly.graph_objects as go


ROUTE_COLORS = [
    "#1769aa",
    "#198754",
    "#d97706",
    "#7c3aed",
    "#dc3545",
    "#0891b2",
    "#a21caf",
    "#475569",
    "#65a30d",
    "#c2410c",
]


def route_color(route_index: int) -> str:
    """Return a consistent route color across charts and maps."""
    return ROUTE_COLORS[route_index % len(ROUTE_COLORS)]


def _route_data(routes: list[dict[str, Any]], stores: pd.DataFrame) -> dict[str, dict[str, int]]:
    return dict(
        zip(
            stores["Store ID"],
            stores["Current empty bottle crates (demand)"].astype(int),
        )
    )


def build_space_time_figure(
    routes: list[dict[str, Any]],
    schedule: pd.DataFrame,
    stores: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    vehicle_speed_kmh: float,
    fixed_service_minutes: float,
    variable_service_minutes_per_crate: float,
) -> go.Figure:
    """Plot actual elapsed route time against cumulative route distance.

    The x-axis resets for each flexible time slot. Several sequential routes
    in one slot are positioned consecutively in elapsed time; distance resets
    to zero whenever the vehicle returns to the depot.
    """
    matrix = distance_matrix
    if "NodeID" in matrix.columns:
        matrix = matrix.set_index("NodeID")
    demand = _route_data(routes, stores)
    route_lookup = {route["route_id"]: route for route in routes}
    ordered = schedule.sort_values(["Time slot", "Route ID"], kind="mergesort")
    fig = go.Figure()

    for route_index, (_, assignment) in enumerate(ordered.iterrows()):
        route_id = assignment["Route ID"]
        route = route_lookup[route_id]
        slot = assignment["Time slot"]
        prior_routes = ordered[
            (ordered["Time slot"] == slot)
            & (ordered["Route ID"].map(
                lambda value: list(route_lookup).index(value)
                < list(route_lookup).index(route_id)
            ))
        ]
        elapsed = 0.0
        if not prior_routes.empty:
            elapsed = sum(
                float(route_lookup[row["Route ID"]]["total_duration_minutes"])
                for _, row in prior_routes.iterrows()
            )

        sequence = route["sequence"]
        x_values = [elapsed]
        y_values = [0.0]
        hover = [f"Route {route_id}<br>Vehicle 1<br>Warehouse departure<br>Slot {slot}"]
        cumulative_distance = 0.0
        current_time = elapsed
        for stop_number, (origin, destination) in enumerate(
            zip(sequence, sequence[1:]), start=1
        ):
            leg_distance = float(matrix.loc[origin, destination])
            travel_duration = leg_distance / vehicle_speed_kmh * 60
            arrival = current_time + travel_duration
            cumulative_distance += leg_distance
            x_values.append(arrival)
            y_values.append(cumulative_distance)
            store_description = (
                "Warehouse return"
                if destination == sequence[-1] and stop_number == len(sequence) - 1
                else f"Arrival at {destination}, stop {stop_number}"
            )
            hover.append(
                f"Route {route_id}<br>Vehicle 1<br>{store_description}"
                f"<br>Travel: {travel_duration:.2f} min"
                f"<br>Distance from depot: {cumulative_distance:.2f} km"
            )
            current_time = arrival

            if destination != sequence[-1]:
                service = fixed_service_minutes + variable_service_minutes_per_crate * demand[destination]
                departure = current_time + service
                # The equal-distance pair is the horizontal service segment.
                x_values.append(departure)
                y_values.append(cumulative_distance)
                hover.append(
                    f"Route {route_id}<br>Vehicle 1<br>Store {destination}"
                    f"<br>Stop {stop_number}<br>Arrival: {arrival:.2f} min"
                    f"<br>Departure: {departure:.2f} min"
                    f"<br>Travel: {travel_duration:.2f} min"
                    f"<br>Service: {service:.2f} min"
                    f"<br>Crates: {demand[destination]}"
                )
                current_time = departure

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name=f"{route_id} · Slot {slot}",
                line={"color": route_color(route_index), "width": 3},
                marker={"size": 6},
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Vehicle space-time diagram (elapsed time by time slot)",
        xaxis_title="Elapsed minutes within flexible time slot",
        yaxis_title="Cumulative distance from warehouse (km)",
        legend_title="Route",
        template="plotly_white",
        hovermode="closest",
    )
    return fig


def build_vehicle_gantt(
    routes: list[dict[str, Any]],
    schedule: pd.DataFrame,
    stores: pd.DataFrame,
    distance_matrix: pd.DataFrame,
    vehicle_speed_kmh: float,
    fixed_service_minutes: float,
    variable_service_minutes_per_crate: float,
    slot_duration_minutes: float,
) -> go.Figure:
    """Show sequential travel/service periods and unused slot time."""
    matrix = distance_matrix
    if "NodeID" in matrix.columns:
        matrix = matrix.set_index("NodeID")
    demand = _route_data(routes, stores)
    route_lookup = {route["route_id"]: route for route in routes}
    figure = go.Figure()
    ordered = schedule.sort_values(["Time slot", "Route ID"], kind="mergesort")
    used_time: dict[Any, float] = {}
    legend_seen: set[str] = set()

    for route_index, (_, assignment) in enumerate(ordered.iterrows()):
        slot = assignment["Time slot"]
        route_id = assignment["Route ID"]
        route = route_lookup[route_id]
        route_start = used_time.get(slot, 0.0)
        route_time = route_start
        sequence = route["sequence"]
        for origin, destination in zip(sequence, sequence[1:]):
            travel = float(matrix.loc[origin, destination]) / vehicle_speed_kmh * 60
            activity_name = "Travel"
            figure.add_trace(
                go.Bar(
                    x=[travel],
                    y=[f"Slot {slot} · {route_id}"],
                    base=[route_time],
                    orientation="h",
                    name=activity_name,
                    marker_color="#1769aa",
                    showlegend=activity_name not in legend_seen,
                    hovertemplate=(
                        f"{origin} → {destination}<br>Travel: {travel:.2f} min"
                        "<br>Elapsed: %{base:.2f}–%{base+x:.2f} min<extra></extra>"
                    ),
                )
            )
            legend_seen.add(activity_name)
            route_time += travel
            if destination != sequence[-1]:
                service = fixed_service_minutes + variable_service_minutes_per_crate * demand[destination]
                activity_name = "Collection / service"
                figure.add_trace(
                    go.Bar(
                        x=[service],
                        y=[f"Slot {slot} · {route_id}"],
                        base=[route_time],
                        orientation="h",
                        name=activity_name,
                        marker_color="#198754",
                        showlegend=activity_name not in legend_seen,
                        hovertemplate=(
                            f"Service at {destination}<br>Duration: {service:.2f} min"
                            f"<br>Crates: {demand[destination]}"
                            "<br>Elapsed: %{base:.2f}–%{base+x:.2f} min<extra></extra>"
                        ),
                    )
                )
                legend_seen.add(activity_name)
                route_time += service
        used_time[slot] = route_time

    # Show unused duration after the last scheduled route as explicit idle time.
    for slot, used in used_time.items():
        idle = max(0.0, slot_duration_minutes - used)
        if idle:
            figure.add_trace(
                go.Bar(
                    x=[idle],
                    y=[f"Slot {slot} · unused capacity"],
                    base=[used],
                    orientation="h",
                    name="Unused slot time",
                    marker_color="#cbd5e1",
                    showlegend="Unused slot time" not in legend_seen,
                    hovertemplate="Unused slot time: %{x:.2f} min<extra></extra>",
                )
            )
            legend_seen.add("Unused slot time")
    figure.update_layout(
        title="Single-vehicle route activity by flexible slot",
        xaxis_title="Elapsed minutes within slot",
        yaxis_title="Slot and route",
        barmode="overlay",
        template="plotly_white",
        legend_title="Activity",
    )
    for slot in sorted(set(schedule["Time slot"])):
        figure.add_vline(
            x=slot_duration_minutes,
            line_dash="dash",
            line_color="#64748b",
            annotation_text=f"Slot {slot} limit",
            annotation_position="top",
        )
    return figure

