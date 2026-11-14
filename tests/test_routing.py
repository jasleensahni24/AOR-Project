import unittest

import pandas as pd

from src.routing import generate_routes


def make_stores(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "Store ID",
            "Latitude",
            "Longitude",
            "Current empty bottle crates (demand)",
            "Collection priority score",
        ],
    )


def make_matrix(values):
    return pd.DataFrame(values, index=["DEPOT", "A", "B", "C"], columns=["DEPOT", "A", "B", "C"])


def generate(stores, matrix, **overrides):
    parameters = {
        "vehicle_capacity": 5,
        "vehicle_speed_kmh": 30,
        "fixed_collection_time_minutes": 4,
        "variable_collection_time_minutes_per_crate": 0.25,
        "route_duration_limit_minutes": 60,
    }
    parameters.update(overrides)
    return generate_routes(stores, matrix, **parameters)


class GenerateRoutesTests(unittest.TestCase):
    def setUp(self):
        self.matrix = make_matrix(
            [
                [0, 1, 2, 3],
                [1, 0, 1, 2],
                [2, 1, 0, 1],
                [3, 2, 1, 0],
            ]
        )

    def test_builds_depot_routes_and_reports_route_metrics(self):
        stores = make_stores(
            [
                ["A", 31.0, 74.0, 3, 0.9],
                ["B", 31.1, 74.1, 2, 0.8],
                ["C", 31.2, 74.2, 1, 0.7],
            ]
        )

        result = generate(stores, self.matrix)

        self.assertEqual(len(result.routes), 2)
        route = result.routes[0]
        self.assertEqual(route["sequence"], ["DEPOT", "B", "A", "DEPOT"])
        self.assertEqual(route["total_crates_collected"], 5)
        self.assertAlmostEqual(route["total_distance_km"], 4.0)
        self.assertAlmostEqual(route["total_travel_time_minutes"], 8.0)
        self.assertAlmostEqual(route["total_collection_time_minutes"], 9.25)
        self.assertAlmostEqual(route["vehicle_utilization"], 1.0)
        self.assertEqual(result.routes[1]["sequence"][0], "DEPOT")
        self.assertEqual(result.routes[1]["sequence"][-1], "DEPOT")
        self.assertEqual(result.routes[1]["sequence"][1:-1], ["C"])
        self.assertTrue(result.unserved_stores.empty)

    def test_vehicle_capacity_is_never_exceeded(self):
        stores = make_stores(
            [
                ["A", 31.0, 74.0, 4, 0.9],
                ["B", 31.1, 74.1, 3, 0.8],
                ["C", 31.2, 74.2, 2, 0.7],
            ]
        )

        result = generate(stores, self.matrix, vehicle_capacity=5)

        self.assertTrue(all(route["total_crates_collected"] <= 5 for route in result.routes))
        visited = [store for route in result.routes for store in route["sequence"][1:-1]]
        self.assertCountEqual(visited, ["A", "B", "C"])

    def test_duration_limit_reports_individually_infeasible_store(self):
        stores = make_stores([["A", 31.0, 74.0, 1, 0.9]])

        result = generate(stores, self.matrix, route_duration_limit_minutes=1)

        self.assertEqual(result.routes, [])
        self.assertEqual(result.unserved_stores.iloc[0]["Store ID"], "A")
        self.assertEqual(result.unserved_stores.iloc[0]["status"], "infeasible")
        self.assertEqual(
            result.unserved_stores.iloc[0]["reason"],
            "single_store_route_exceeds_duration_limit",
        )

    def test_zero_and_over_capacity_demands_are_explicitly_reported(self):
        stores = make_stores(
            [
                ["A", 31.0, 74.0, 0, 0.5],
                ["B", 31.1, 74.1, 6, 0.9],
                ["C", 31.2, 74.2, 2, 0.8],
            ]
        )

        result = generate(stores, self.matrix)

        self.assertEqual(
            set(result.unserved_stores["Store ID"]),
            {"A", "B"},
        )
        self.assertEqual(
            result.unserved_stores.set_index("Store ID").loc["A", "status"],
            "not_required",
        )
        self.assertEqual(
            result.unserved_stores.set_index("Store ID").loc["B", "status"],
            "infeasible",
        )
        visited = [store for route in result.routes for store in route["sequence"][1:-1]]
        self.assertEqual(visited, ["C"])

    def test_priority_tier_falls_back_when_top_store_cannot_fit(self):
        stores = make_stores(
            [
                ["A", 31.0, 74.0, 3, 0.99],
                ["B", 31.1, 74.1, 3, 0.9],
                ["C", 31.2, 74.2, 2, 0.8],
            ]
        )

        result = generate(
            stores,
            self.matrix,
            vehicle_capacity=5,
        )

        self.assertEqual(result.routes[0]["sequence"], ["DEPOT", "C", "A", "DEPOT"])
        self.assertEqual(result.routes[1]["sequence"], ["DEPOT", "B", "DEPOT"])
        self.assertTrue(result.unserved_stores.empty)

    def test_input_stores_and_matrix_are_not_modified(self):
        stores = make_stores([["A", 31.0, 74.0, 1, 0.9]])
        original_stores = stores.copy(deep=True)
        original_matrix = self.matrix.copy(deep=True)

        generate(stores, self.matrix)

        pd.testing.assert_frame_equal(stores, original_stores)
        pd.testing.assert_frame_equal(self.matrix, original_matrix)

    def test_rejects_asymmetric_distance_matrix(self):
        stores = make_stores([["A", 31.0, 74.0, 1, 0.9]])
        invalid_matrix = self.matrix.copy()
        invalid_matrix.loc["DEPOT", "A"] = 2

        with self.assertRaisesRegex(ValueError, "symmetric"):
            generate(stores, invalid_matrix)


if __name__ == "__main__":
    unittest.main()
