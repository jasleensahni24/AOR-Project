import unittest

from src.scheduling import SchedulingInfeasibleError, schedule_routes


def route(route_id, duration, stores=None):
    assigned_stores = stores or [route_id]
    return {
        "route_id": route_id,
        "sequence": ["DEPOT", *assigned_stores, "DEPOT"],
        "total_duration_minutes": duration,
    }


class ScheduleRoutesTests(unittest.TestCase):
    def test_best_fit_assigns_routes_and_reports_remaining_capacity(self):
        routes = [
            route("R1", 6, ["A"]),
            route("R2", 6, ["B"]),
            route("R3", 4, ["C"]),
            route("R4", 4, ["D"]),
        ]

        schedule = schedule_routes(
            routes,
            slot_duration_minutes=10,
            max_routes_per_slot=2,
            number_of_slots=2,
        )

        self.assertEqual(schedule["Time slot"].tolist(), [1, 1, 2, 2])
        self.assertEqual(schedule["Route ID"].tolist(), ["R1", "R3", "R2", "R4"])
        self.assertEqual(schedule["Assigned stores"].tolist(), ["A", "C", "B", "D"])
        self.assertEqual(
            schedule["Remaining time-slot capacity"].tolist(),
            [4, 0, 4, 0],
        )

    def test_route_count_limit_opens_another_slot(self):
        schedule = schedule_routes(
            [route("R1", 2), route("R2", 2), route("R3", 2)],
            slot_duration_minutes=10,
            max_routes_per_slot=2,
            number_of_slots=2,
        )

        self.assertEqual(schedule["Time slot"].tolist(), [1, 1, 2])
        self.assertEqual(schedule["Route ID"].tolist(), ["R1", "R2", "R3"])

    def test_infeasibility_exposes_partial_schedule_and_unscheduled_routes(self):
        routes = [route("R1", 6), route("R2", 6), route("R3", 5)]

        with self.assertRaises(SchedulingInfeasibleError) as context:
            schedule_routes(
                routes,
                slot_duration_minutes=10,
                max_routes_per_slot=2,
                number_of_slots=1,
            )

        error = context.exception
        self.assertEqual(error.partial_schedule["Route ID"].tolist(), ["R1"])
        self.assertEqual(error.unscheduled_route_ids, ["R2", "R3"])

    def test_route_longer_than_slot_is_infeasible(self):
        with self.assertRaises(SchedulingInfeasibleError) as context:
            schedule_routes(
                [route("R1", 11), route("R2", 2)],
                slot_duration_minutes=10,
                max_routes_per_slot=2,
                number_of_slots=2,
            )

        self.assertEqual(context.exception.unscheduled_route_ids, ["R1", "R2"])
        self.assertTrue(context.exception.partial_schedule.empty)

    def test_rejects_route_with_duplicate_store(self):
        invalid_route = route("R1", 3, ["A", "A"])

        with self.assertRaisesRegex(ValueError, "more than once"):
            schedule_routes(
                [invalid_route],
                slot_duration_minutes=10,
                max_routes_per_slot=2,
                number_of_slots=1,
            )

    def test_empty_route_list_returns_empty_schedule(self):
        result = schedule_routes(
            [],
            slot_duration_minutes=10,
            max_routes_per_slot=2,
            number_of_slots=1,
        )

        self.assertEqual(
            result.columns.tolist(),
            [
                "Time slot",
                "Route ID",
                "Assigned stores",
                "Route duration",
                "Remaining time-slot capacity",
            ],
        )
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
