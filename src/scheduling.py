"""Greedy best-fit assignment of vehicle routes to flexible time slots."""

from typing import Any

import numpy as np
import pandas as pd


ROUTE_ID_COLUMN = "route_id"
ROUTE_DURATION_COLUMN = "total_duration_minutes"
ROUTE_SEQUENCE_COLUMN = "sequence"


class SchedulingInfeasibleError(ValueError):
    """Raised when not all routes can be assigned to the available time slots."""

    def __init__(
        self,
        message: str,
        partial_schedule: pd.DataFrame,
        unscheduled_route_ids: list[str],
    ) -> None:
        super().__init__(message)
        self.feasible = False
        self.partial_schedule = partial_schedule
        self.unscheduled_route_ids = unscheduled_route_ids


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(numeric_value) or not numeric_value.is_integer() or numeric_value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(numeric_value)


def _get_route_stores(route: dict[str, Any]) -> str:
    """Return displayable store IDs from a generated route sequence."""
    sequence = route.get(ROUTE_SEQUENCE_COLUMN)
    if not isinstance(sequence, (list, tuple)) or len(sequence) < 2:
        raise ValueError(
            f"Route {route.get(ROUTE_ID_COLUMN)!r} must have a depot-to-depot sequence"
        )
    if sequence[0] != sequence[-1]:
        raise ValueError(
            f"Route {route.get(ROUTE_ID_COLUMN)!r} must start and end at the same depot"
        )
    stores = sequence[1:-1]
    if len(stores) != len(set(stores)):
        raise ValueError(
            f"Route {route.get(ROUTE_ID_COLUMN)!r} visits a store more than once"
        )
    return ", ".join(str(store_id) for store_id in stores)


def _make_schedule_table(
    assignments: list[dict[str, Any]],
    slot_duration: float,
) -> pd.DataFrame:
    columns = [
        "Time slot",
        "Route ID",
        "Assigned stores",
        "Route duration",
        "Remaining time-slot capacity",
    ]
    rows = []
    for slot_number in sorted({item["slot_number"] for item in assignments}):
        used_time = 0.0
        # Keep the original route order within each time slot.
        slot_assignments = sorted(
            (item for item in assignments if item["slot_number"] == slot_number),
            key=lambda item: item["route_order"],
        )
        for item in slot_assignments:
            used_time += item["duration"]
            rows.append(
                {
                    "Time slot": slot_number,
                    "Route ID": item["route_id"],
                    "Assigned stores": item["stores"],
                    "Route duration": item["duration"],
                    "Remaining time-slot capacity": max(0.0, slot_duration - used_time),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def schedule_routes(
    routes: list[dict[str, Any]],
    slot_duration_minutes: float,
    max_routes_per_slot: int,
    number_of_slots: int,
) -> pd.DataFrame:
    """Assign routes with greedy best-fit while preserving chronological slots.

    Routes are processed in their supplied order. Each route is assigned to
    the already-open slot that leaves the least remaining duration; equal fits
    prefer the earlier slot. A new slot is opened only when no existing slot
    can accommodate the route. The returned table is ordered by time slot and
    then by the original route order.

    Raises ``SchedulingInfeasibleError`` if any route cannot be scheduled.
    The exception exposes both the partial schedule and all unassigned route
    IDs so no route is silently dropped.
    """
    if not isinstance(routes, list):
        raise TypeError("routes must be a list of generated route dictionaries")
    if isinstance(slot_duration_minutes, (bool, np.bool_)):
        raise ValueError("slot_duration_minutes must be a finite positive number")
    try:
        slot_duration = float(slot_duration_minutes)
    except (TypeError, ValueError) as exc:
        raise ValueError("slot_duration_minutes must be a finite positive number") from exc
    if not np.isfinite(slot_duration) or slot_duration <= 0:
        raise ValueError("slot_duration_minutes must be a finite positive number")

    route_limit = _positive_integer("max_routes_per_slot", max_routes_per_slot)
    slot_limit = _positive_integer("number_of_slots", number_of_slots)

    route_records: list[dict[str, Any]] = []
    route_ids: list[str] = []
    for route_order, route in enumerate(routes):
        if not isinstance(route, dict):
            raise TypeError(f"routes[{route_order}] must be a route dictionary")
        route_id = route.get(ROUTE_ID_COLUMN)
        if not isinstance(route_id, str) or not route_id:
            raise ValueError(f"routes[{route_order}] must have a non-empty route_id")
        if route_id in route_ids:
            raise ValueError(f"route IDs must be unique; duplicate: {route_id}")
        route_ids.append(route_id)

        duration_value = route.get(ROUTE_DURATION_COLUMN)
        if isinstance(duration_value, (bool, np.bool_)):
            raise ValueError(f"Route {route_id} duration must be finite and non-negative")
        try:
            duration = float(duration_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Route {route_id} duration must be finite and non-negative"
            ) from exc
        if not np.isfinite(duration) or duration < 0:
            raise ValueError(f"Route {route_id} duration must be finite and non-negative")

        stores = _get_route_stores(route)
        route_records.append(
            {
                "route_order": route_order,
                "route_id": route_id,
                "duration": duration,
                "stores": stores,
            }
        )

    assignments: list[dict[str, Any]] = []
    slot_used: list[float] = []
    slot_route_counts: list[int] = []

    for route_index, route in enumerate(route_records):
        if route["duration"] > slot_duration + 1e-9:
            partial = _make_schedule_table(assignments, slot_duration)
            raise SchedulingInfeasibleError(
                f"Route {route['route_id']} duration ({route['duration']:.2f} min) "
                f"exceeds slot duration ({slot_duration:.2f} min).",
                partial,
                route_ids[route_index:],
            )

        # Best-fit minimizes leftover slot time while respecting route-count limits.
        eligible_slots = [
            slot_index
            for slot_index, used in enumerate(slot_used)
            if slot_route_counts[slot_index] < route_limit
            and used + route["duration"] <= slot_duration + 1e-9
        ]
        if eligible_slots:
            slot_index = min(
                eligible_slots,
                key=lambda index: (
                    slot_duration - (slot_used[index] + route["duration"]),
                    index,
                ),
            )
        elif len(slot_used) < slot_limit:
            slot_index = len(slot_used)
            slot_used.append(0.0)
            slot_route_counts.append(0)
        else:
            partial = _make_schedule_table(assignments, slot_duration)
            raise SchedulingInfeasibleError(
                f"Unable to assign all routes within {slot_limit} time slots; "
                f"no feasible slot is available for route {route['route_id']}.",
                partial,
                route_ids[route_index:],
            )

        slot_used[slot_index] += route["duration"]
        slot_route_counts[slot_index] += 1
        assignments.append(
            {
                **route,
                "slot_number": slot_index + 1,
            }
        )

    return _make_schedule_table(assignments, slot_duration)
