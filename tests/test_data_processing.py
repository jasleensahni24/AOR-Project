import unittest

import pandas as pd

from src.data_processing import (
    build_geodesic_matrix,
    validate_distance_matrix,
    validate_stores,
)


def valid_stores():
    return pd.DataFrame(
        {
            "Store ID": ["S1", "S2"],
            "Store name": ["One", "Two"],
            "Latitude": [31.6, 31.7],
            "Longitude": [74.8, 74.9],
            "Current empty bottle crates (demand)": [2, 3],
            "Maximum crate storage capacity": [5, 6],
            "Pickup request": [0, 1],
        }
    )


class ValidateStoresTests(unittest.TestCase):
    def test_valid_dataset_gets_derived_columns(self):
        validated, report = validate_stores(valid_stores())
        self.assertIn("Filling rate", validated)
        self.assertIn("Collection priority score", validated)
        self.assertFalse((report["Status"] == "Failed").any())

    def test_missing_columns_fail_without_dropping_rows(self):
        stores = valid_stores().drop(columns=["Store name"])
        validated, report = validate_stores(stores)
        self.assertEqual(len(validated), len(stores))
        self.assertIn("Failed", report["Status"].tolist())
        self.assertNotIn("Collection priority score", validated)

    def test_empty_dataset_fails(self):
        validated, report = validate_stores(valid_stores().iloc[0:0])
        self.assertTrue(validated.empty)
        self.assertEqual(report.iloc[0]["Status"], "Failed")

    def test_duplicate_ids_and_negative_demand_are_reported(self):
        stores = valid_stores()
        stores.loc[1, "Store ID"] = "S1"
        stores.loc[0, "Current empty bottle crates (demand)"] = -1
        _, report = validate_stores(stores)
        failed = set(report.loc[report["Status"] == "Failed", "Check"])
        self.assertIn("Unique store IDs", failed)
        self.assertIn("Current empty bottle crates (demand)", failed)

    def test_demand_over_capacity_is_reported(self):
        stores = valid_stores()
        stores.loc[0, "Current empty bottle crates (demand)"] = 7
        _, report = validate_stores(stores)
        self.assertIn(
            "Demand within store capacity",
            report.loc[report["Status"] == "Failed", "Check"].tolist(),
        )

    def test_missing_coordinates_warn_and_records_are_retained(self):
        stores = valid_stores().drop(columns=["Latitude", "Longitude"])
        validated, report = validate_stores(stores)
        self.assertEqual(len(validated), 2)
        self.assertEqual(
            report.loc[report["Check"] == "Store coordinates", "Status"].iloc[0],
            "Warning",
        )

    def test_geodesic_matrix_is_symmetric_and_contains_zero_diagonal(self):
        matrix = build_geodesic_matrix(valid_stores(), 31.65, 74.85)
        self.assertTrue((matrix.values.diagonal() == 0).all())
        self.assertTrue((matrix.values == matrix.values.T).all())
        self.assertTrue((matrix.values[~__import__("numpy").eye(3, dtype=bool)] > 0).all())

    def test_geodesic_matrix_rejects_missing_or_out_of_range_coordinates(self):
        stores = valid_stores()
        stores.loc[0, "Latitude"] = float("nan")
        with self.assertRaisesRegex(ValueError, "missing or outside"):
            build_geodesic_matrix(stores, 31.65, 74.85)

        with self.assertRaisesRegex(ValueError, "Warehouse coordinates"):
            build_geodesic_matrix(valid_stores(), 91, 74.85)

    def test_incomplete_distance_matrix_is_reported(self):
        stores = valid_stores()
        matrix = pd.DataFrame(
            [[0, 1], [1, 0]],
            index=["DEPOT", "S1"],
            columns=["DEPOT", "S1"],
        )
        report = validate_distance_matrix(matrix, stores)
        self.assertEqual(report[0]["Status"], "Failed")


if __name__ == "__main__":
    unittest.main()
