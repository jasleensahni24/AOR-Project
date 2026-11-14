import unittest

import pandas as pd

from src.evaluation import compare_solutions


class CompareSolutionsTests(unittest.TestCase):
    def setUp(self):
        self.stores = pd.DataFrame(
            {
                "Store ID": ["A", "B", "C", "ZERO"],
                "Current empty bottle crates (demand)": [4, 11, 1, 0],
                "Maximum crate storage capacity": [10, 12, 5, 5],
            }
        )
        self.matrix = pd.DataFrame(
            [
                [0, 1, 2, 1],
                [1, 0, 2, 1],
                [2, 2, 0, 2],
                [1, 1, 2, 0],
            ],
            index=["DEPOT", "A", "B", "C"],
            columns=["DEPOT", "A", "B", "C"],
        )
        self.initial_routes = [
            {
                "route_id": "R1",
                "sequence": ["DEPOT", "A", "C", "DEPOT"],
                "total_distance_km": 9999,
                "total_duration_minutes": 9999,
            }
        ]
        self.initial_schedule = pd.DataFrame(
            [{"Time slot": 1, "Route ID": "R1", "Route duration": 9999}]
        )
        self.optimized_routes = [
            {
                "route_id": "R1",
                "sequence": ["DEPOT", "A", "C", "DEPOT"],
                "total_distance_km": 9999,
                "total_duration_minutes": 9999,
            },
            {
                "route_id": "R2",
                "sequence": ["DEPOT", "B", "DEPOT"],
                "total_distance_km": 9999,
                "total_duration_minutes": 9999,
            },
        ]
        self.optimized_schedule = pd.DataFrame(
            [
                {"Time slot": 1, "Route ID": "R1", "Route duration": 9999},
                {"Time slot": 1, "Route ID": "R2", "Route duration": 9999},
            ]
        )

    def evaluate(self, **overrides):
        parameters = {
            "stores": self.stores,
            "initial_routes": self.initial_routes,
            "initial_schedule": self.initial_schedule,
            "optimized_routes": self.optimized_routes,
            "optimized_schedule": self.optimized_schedule,
            "distance_matrix": self.matrix,
            "vehicle_capacity": 12,
            "vehicle_speed_kmh": 30,
            "fixed_collection_time_minutes": 4,
            "variable_collection_time_minutes_per_crate": 0.25,
            "slot_duration_minutes": 60,
            "bottles_per_crate": 12,
        }
        parameters.update(overrides)
        return compare_solutions(**parameters)

    def value(self, table, kpi, column):
        return table.set_index("KPI").loc[kpi, column]

    def test_recalculates_all_kpis_from_supplied_data(self):
        table = self.evaluate()

        self.assertAlmostEqual(
            self.value(table, "Total distance travelled", "Initial Greedy"), 3
        )
        self.assertAlmostEqual(
            self.value(table, "Total transportation time", "Initial Greedy"), 6
        )
        self.assertAlmostEqual(
            self.value(table, "Total collection time", "Initial Greedy"), 9.25
        )
        self.assertAlmostEqual(
            self.value(table, "Total operational time", "Initial Greedy"), 15.25
        )
        self.assertEqual(self.value(table, "Total crates collected", "Initial Greedy"), 5)
        self.assertEqual(self.value(table, "Total bottles collected", "Initial Greedy"), 60)
        self.assertEqual(self.value(table, "Number of routes", "Initial Greedy"), 1)
        self.assertEqual(self.value(table, "Number of time slots used", "Initial Greedy"), 1)
        self.assertAlmostEqual(
            self.value(table, "Vehicle capacity utilization", "Initial Greedy"),
            100 * 5 / 12,
        )
        self.assertAlmostEqual(
            self.value(table, "Percentage of stores served", "Initial Greedy"),
            100 * 2 / 3,
        )
        self.assertEqual(
            self.value(
                table,
                "High-volume stores (>=90% vehicle capacity) not collected",
                "Initial Greedy",
            ),
            1,
        )
        self.assertAlmostEqual(
            self.value(table, "Total time-slot utilization", "Initial Greedy"),
            100 * 15.25 / 60,
        )
        self.assertEqual(self.value(table, "Total crates remaining", "Initial Greedy"), 11)
        self.assertEqual(self.value(table, "Number of unserved stores", "Initial Greedy"), 1)
        self.assertAlmostEqual(
            self.value(table, "Travel time percentage", "Initial Greedy"),
            100 * 6 / 15.25,
        )
        self.assertAlmostEqual(
            self.value(table, "Service time percentage", "Initial Greedy"),
            100 * 9.25 / 15.25,
        )
        self.assertEqual(self.value(table, "Route feasibility rate", "Initial Greedy"), 100)

    def test_route_feasibility_rate_uses_configured_duration_limit(self):
        table = self.evaluate(route_duration_limit_minutes=10)

        self.assertEqual(
            self.value(table, "Route feasibility rate", "Initial Greedy"), 0
        )
        self.assertEqual(
            self.value(table, "Number of unserved stores", "Initial Greedy"), 1
        )

    def test_reports_optimized_comparison_and_percentage_point_changes(self):
        table = self.evaluate()

        self.assertEqual(
            self.value(table, "Total crates collected", "Optimized Heuristic"), 16
        )
        self.assertEqual(
            self.value(table, "Percentage of stores served", "Optimized Heuristic"), 100
        )
        self.assertEqual(
            self.value(
                table,
                "High-volume stores (>=90% vehicle capacity) not collected",
                "Optimized Heuristic",
            ),
            0,
        )
        served_row = table.set_index("KPI").loc["Percentage of stores served"]
        self.assertAlmostEqual(served_row["Change (optimized - initial)"], 100 / 3)
        self.assertEqual(served_row["Change unit"], "percentage points")

    def test_rejects_duplicate_store_visits_across_routes(self):
        bad_routes = [
            {"route_id": "R1", "sequence": ["DEPOT", "A", "DEPOT"]},
            {"route_id": "R2", "sequence": ["DEPOT", "A", "DEPOT"]},
        ]
        bad_schedule = pd.DataFrame(
            [{"Time slot": 1, "Route ID": "R1"}, {"Time slot": 1, "Route ID": "R2"}]
        )

        with self.assertRaisesRegex(ValueError, "counted more than once"):
            self.evaluate(initial_routes=bad_routes, initial_schedule=bad_schedule)

    def test_handles_zero_routes_and_zero_used_slots_without_division_errors(self):
        empty_schedule = pd.DataFrame(columns=["Time slot", "Route ID"])

        table = self.evaluate(
            initial_routes=[],
            initial_schedule=empty_schedule,
            optimized_routes=[],
            optimized_schedule=empty_schedule,
        )

        self.assertEqual(self.value(table, "Vehicle capacity utilization", "Initial Greedy"), 0)
        self.assertEqual(self.value(table, "Total time-slot utilization", "Initial Greedy"), 0)
        self.assertEqual(self.value(table, "Percentage of stores served", "Initial Greedy"), 0)
        self.assertEqual(self.value(table, "Number of routes", "Initial Greedy"), 0)
        self.assertEqual(self.value(table, "Number of time slots used", "Initial Greedy"), 0)

    def test_does_not_modify_source_data(self):
        stores_before = self.stores.copy(deep=True)
        matrix_before = self.matrix.copy(deep=True)
        self.evaluate()
        pd.testing.assert_frame_equal(self.stores, stores_before)
        pd.testing.assert_frame_equal(self.matrix, matrix_before)

    def test_rejects_schedule_that_exceeds_slot_capacity_after_recalculation(self):
        too_short_schedule = self.optimized_schedule.copy()

        with self.assertRaisesRegex(ValueError, "exceed slot duration"):
            self.evaluate(slot_duration_minutes=10)


if __name__ == "__main__":
    unittest.main()
