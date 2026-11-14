import unittest
import warnings

import pandas as pd

from src.priority import (
    calculate_store_priorities,
    DEFAULT_SATURATION_WARNING_THRESHOLD,
)


DEMAND = "Current empty bottle crates (demand)"
CAPACITY = "Maximum crate storage capacity"
PICKUP = "Pickup request"


def make_store(store_id, demand, capacity, pickup_request):
    return {
        "Store ID": store_id,
        DEMAND: demand,
        CAPACITY: capacity,
        PICKUP: pickup_request,
    }


class CalculateStorePrioritiesTests(unittest.TestCase):
    def test_high_filling_rate_with_pickup_request(self):
        stores = pd.DataFrame([make_store("urgent", 9, 10, 1)])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calculate_store_priorities(stores)

        self.assertAlmostEqual(result.iloc[0]["Filling rate"], 0.9)
        self.assertAlmostEqual(result.iloc[0]["Collection priority score"], 0.93)
        self.assertTrue(result.iloc[0]["Approaching saturation"])
        self.assertTrue(any("urgent" in str(item.message) for item in caught))

    def test_high_filling_rate_without_pickup_request(self):
        stores = pd.DataFrame([make_store("full", 9, 10, 0)])

        with self.assertWarnsRegex(UserWarning, "full"):
            result = calculate_store_priorities(stores)

        self.assertAlmostEqual(result.iloc[0]["Collection priority score"], 0.63)
        self.assertTrue(result.iloc[0]["Approaching saturation"])

    def test_low_filling_rate_with_pickup_request(self):
        stores = pd.DataFrame([make_store("requesting", 2, 10, 1)])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = calculate_store_priorities(stores)

        self.assertAlmostEqual(result.iloc[0]["Collection priority score"], 0.44)
        self.assertFalse(result.iloc[0]["Approaching saturation"])
        self.assertEqual(caught, [])

    def test_equal_priority_scores_keep_original_order(self):
        stores = pd.DataFrame(
            [
                make_store("first", 4, 8, 0),
                make_store("second", 4, 8, 0),
            ]
        )

        result = calculate_store_priorities(stores)

        self.assertEqual(result["Store ID"].tolist(), ["first", "second"])
        self.assertEqual(result["Priority rank"].tolist(), [1, 2])

    def test_invalid_demand_and_capacity_values_raise(self):
        invalid_cases = [
            make_store("negative demand", -1, 10, 0),
            make_store("fractional demand", 1.5, 10, 0),
            make_store("zero capacity", 0, 0, 0),
            make_store("fractional capacity", 1, 10.5, 0),
            make_store("over capacity", 11, 10, 0),
            make_store("negative capacity", 1, -2, 0),
            make_store("invalid pickup request", 1, 10, 2),
        ]
        for store in invalid_cases:
            with self.subTest(store=store["Store ID"]):
                with self.assertRaises(ValueError):
                    calculate_store_priorities(pd.DataFrame([store]))

    def test_input_dataframe_is_not_modified(self):
        stores = pd.DataFrame([make_store("unchanged", 2, 10, 1)])
        original = stores.copy(deep=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calculate_store_priorities(stores)

        pd.testing.assert_frame_equal(stores, original)

    def test_default_saturation_threshold_is_80_percent(self):
        self.assertEqual(DEFAULT_SATURATION_WARNING_THRESHOLD, 0.80)


if __name__ == "__main__":
    unittest.main()
