import unittest

import pandas as pd

from src.local_search import improve_routes


def build_inputs(routes):
    stores = pd.DataFrame(
        {
            "Store ID": ["A", "B", "C", "D"],
            "Current empty bottle crates (demand)": [1, 1, 1, 1],
        }
    )
    distance_matrix = pd.DataFrame(
        [
            [0, 2, 5, 3, 4],
            [2, 0, 8, 2, 4],
            [5, 8, 0, 1, 2],
            [3, 2, 1, 0, 7],
            [4, 4, 2, 7, 0],
        ],
        index=["DEPOT", "A", "B", "C", "D"],
        columns=["DEPOT", "A", "B", "C", "D"],
    )
    schedule = pd.DataFrame(
        [
            {
                "Time slot": 1,
                "Route ID": route["route_id"],
                "Assigned stores": ", ".join(route["sequence"][1:-1]),
                "Route duration": route["total_duration_minutes"],
                "Remaining time-slot capacity": 120 - route["total_duration_minutes"],
            }
            for route in routes
        ]
    )
    return stores, distance_matrix, schedule


def route(route_id, stores, distance=0, duration=0, crates=None):
    return {
        "route_id": route_id,
        "sequence": ["DEPOT", *stores, "DEPOT"],
        "total_distance_km": distance,
        "total_travel_time_minutes": duration,
        "total_collection_time_minutes": 0,
        "total_crates_collected": crates or len(stores),
        "vehicle_utilization": 0,
        "vehicle_utilization_percent": 0,
        "total_duration_minutes": duration,
    }


def improve(routes, schedule, stores, matrix, **overrides):
    parameters = {
        "vehicle_capacity": 4,
        "vehicle_speed_kmh": 30,
        "fixed_collection_time_minutes": 0,
        "variable_collection_time_minutes_per_crate": 0,
        "route_duration_limit_minutes": 120,
        "slot_duration_minutes": 120,
        "max_routes_per_slot": 4,
    }
    parameters.update(overrides)
    return improve_routes(routes, schedule, stores, matrix, **parameters)


class ImproveRoutesTests(unittest.TestCase):
    def test_best_improvement_reduces_time_and_preserves_slot_assignments(self):
        routes = [route("R1", ["A", "B", "C"]), route("R2", ["D"])]
        stores, matrix, schedule = build_inputs(routes)

        result = improve(routes, schedule, stores, matrix)

        self.assertTrue(result.feasible)
        self.assertLess(result.optimized_total_route_time, result.original_total_route_time)
        self.assertGreater(result.time_saved_minutes, 0)
        self.assertGreater(result.distance_saved_km, 0)
        self.assertEqual(
            result.original_schedule["Time slot"].tolist(),
            result.optimized_schedule["Time slot"].tolist(),
        )
        self.assertEqual(
            result.original_schedule["Route ID"].tolist(),
            result.optimized_schedule["Route ID"].tolist(),
        )
        self.assertEqual(result.termination_reason, "no_improving_move")

    def test_all_neighborhood_move_counters_and_recalculations(self):
        routes = [
            route("R1", ["A", "B", "C"]),
            route("R2", ["D"]),
        ]
        stores, matrix, schedule = build_inputs(routes)

        result = improve(routes, schedule, stores, matrix)

        self.assertGreaterEqual(
            result.relocation_moves_accepted
            + result.swap_moves_accepted
            + result.two_opt_moves_accepted,
            1,
        )
        self.assertEqual(result.swap_moves_accepted, 1)
        for route_result in result.optimized_routes:
            self.assertAlmostEqual(
                route_result["total_duration_minutes"],
                route_result["total_travel_time_minutes"]
                + route_result["total_collection_time_minutes"],
            )

    def test_relocation_neighborhood_can_improve_routes(self):
        nodes = ["DEPOT", "A", "B", "C", "D", "E"]
        matrix = pd.DataFrame(
            [
                [0, 2, 17, 4, 17, 7],
                [2, 0, 13, 12, 17, 10],
                [17, 13, 0, 19, 5, 20],
                [4, 12, 19, 0, 9, 4],
                [17, 17, 5, 9, 0, 9],
                [7, 10, 20, 4, 9, 0],
            ],
            index=nodes,
            columns=nodes,
        )
        stores = pd.DataFrame(
            {
                "Store ID": nodes[1:],
                "Current empty bottle crates (demand)": [1] * 5,
            }
        )
        routes = [route("R1", ["A", "B"]), route("R2", ["C", "D", "E"])]
        schedule = pd.DataFrame(
            {"Time slot": [1, 1], "Route ID": ["R1", "R2"]}
        )

        result = improve(
            routes,
            schedule,
            stores,
            matrix,
            max_iterations=1,
            route_duration_limit_minutes=300,
            slot_duration_minutes=500,
        )

        self.assertEqual(result.relocation_moves_accepted, 1)
        self.assertLess(result.optimized_total_route_time, result.original_total_route_time)

    def test_two_opt_neighborhood_can_improve_route_order(self):
        nodes = ["DEPOT", "A", "B", "C", "D", "E"]
        matrix = pd.DataFrame(
            [
                [0, 15, 13, 15, 4, 9],
                [15, 0, 5, 16, 10, 15],
                [13, 5, 0, 13, 10, 6],
                [15, 16, 13, 0, 15, 15],
                [4, 10, 10, 15, 0, 10],
                [9, 15, 6, 15, 10, 0],
            ],
            index=nodes,
            columns=nodes,
        )
        stores = pd.DataFrame(
            {
                "Store ID": nodes[1:],
                "Current empty bottle crates (demand)": [1] * 5,
            }
        )
        routes = [route("R1", ["A", "B", "C", "D"]), route("R2", ["E"])]
        schedule = pd.DataFrame(
            {"Time slot": [1, 1], "Route ID": ["R1", "R2"]}
        )

        result = improve(
            routes,
            schedule,
            stores,
            matrix,
            max_iterations=1,
            route_duration_limit_minutes=300,
            slot_duration_minutes=500,
        )

        self.assertEqual(result.two_opt_moves_accepted, 1)
        self.assertLess(result.optimized_total_route_time, result.original_total_route_time)

    def test_never_accepts_infeasible_capacity_change(self):
        routes = [route("R1", ["A", "B"]), route("R2", ["C", "D"])]
        stores, matrix, schedule = build_inputs(routes)

        result = improve(
            routes,
            schedule,
            stores,
            matrix,
            vehicle_capacity=2,
            max_routes_per_slot=2,
        )

        self.assertTrue(result.feasible)
        self.assertTrue(
            all(route["total_crates_collected"] <= 2 for route in result.optimized_routes)
        )

    def test_respects_slot_duration_and_route_duration_limits(self):
        routes = [route("R1", ["A", "B"]), route("R2", ["C", "D"])]
        stores, matrix, schedule = build_inputs(routes)

        result = improve(
            routes,
            schedule,
            stores,
            matrix,
            route_duration_limit_minutes=50,
            slot_duration_minutes=65,
        )

        self.assertTrue(result.feasible)
        self.assertTrue(
            all(route["total_duration_minutes"] <= 50 for route in result.optimized_routes)
        )
        self.assertLessEqual(result.optimized_schedule["Route duration"].sum(), 65)

    def test_respects_minimum_improvement_threshold(self):
        routes = [route("R1", ["A", "B", "C", "D"])]
        stores, matrix, schedule = build_inputs(routes)

        result = improve(
            routes,
            schedule,
            stores,
            matrix,
            min_improvement_minutes=1000,
        )

        self.assertEqual(result.optimized_routes[0]["sequence"], routes[0]["sequence"])
        self.assertEqual(result.time_saved_minutes, 0)
        self.assertEqual(result.termination_reason, "no_improving_move")

    def test_maximum_iteration_limit_is_reported(self):
        routes = [route("R1", ["A", "B", "C"]), route("R2", ["D"])]
        stores, matrix, schedule = build_inputs(routes)

        result = improve(routes, schedule, stores, matrix, max_iterations=1)

        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.termination_reason, "max_iterations_reached")

    def test_does_not_modify_inputs(self):
        routes = [route("R1", ["A", "B", "C"]), route("R2", ["D"])]
        stores, matrix, schedule = build_inputs(routes)
        original_routes = [dict(route, sequence=list(route["sequence"])) for route in routes]
        original_stores = stores.copy(deep=True)
        original_matrix = matrix.copy(deep=True)
        original_schedule = schedule.copy(deep=True)

        improve(routes, schedule, stores, matrix)

        self.assertEqual(routes, original_routes)
        pd.testing.assert_frame_equal(stores, original_stores)
        pd.testing.assert_frame_equal(matrix, original_matrix)
        pd.testing.assert_frame_equal(schedule, original_schedule)

    def test_rejects_an_infeasible_initial_schedule(self):
        routes = [route("R1", ["A", "B", "C", "D"])]
        stores, matrix, schedule = build_inputs(routes)

        with self.assertRaisesRegex(ValueError, "initial route schedule is infeasible"):
            improve(routes, schedule, stores, matrix, vehicle_capacity=3)


if __name__ == "__main__":
    unittest.main()
