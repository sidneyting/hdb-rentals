"""Behavioral checks for fold isolation, calendar alignment and reversibility."""

import unittest

import numpy as np
import pandas as pd

from hdb_rentals.rent_adjustment import MarketRentAdjustment


def rentals():
    rows = []
    for month in range(12):
        for size in [60.0, 90.0]:
            for town in ["bedok", "queenstown"]:
                rows.append({"MONTH_INDEX": month, "TOWN": town,
                             "FLAT_TYPE": "3-room" if size == 60 else "4-room",
                             "FLAT_MODEL": "improved", "FLOOR_AREA_SQM": size,
                             "FLAT_AGE": 25.0,
                             "MONTHLY_RENT": np.exp(7 + .02 * month + .003 * size + .1 * (town == "queenstown"))})
    return pd.DataFrame(rows)


class RentAdjustmentTests(unittest.TestCase):
    def test_known_market_growth_and_round_trip(self):
        frame = rentals()
        frame.index = np.repeat(np.arange(len(frame) // 2), 2)
        adj = MarketRentAdjustment("linear").fit(frame)
        np.testing.assert_allclose(adj.log_level_, .02 * np.arange(12), atol=1e-10)
        restored = adj.restore(adj.adjusted_rents(frame), frame["MONTH_INDEX"])
        np.testing.assert_allclose(restored, frame["MONTHLY_RENT"])
        adjusted = adj.adjusted_rents(frame).reshape(12, 4)
        np.testing.assert_allclose(adjusted, np.broadcast_to(adjusted[0], adjusted.shape))

    def test_sparse_future_months_use_calendar_distance_and_preserve_order(self):
        adj = MarketRentAdjustment("linear").fit(rentals())
        months = [15, 11, 13, 15]
        np.testing.assert_allclose(adj.multipliers(months), np.exp(.02 * np.array(months)), atol=1e-10)

    def test_evaluation_labels_do_not_affect_forecast(self):
        frame = rentals()
        fit = frame[frame.MONTH_INDEX < 8]
        future = frame[frame.MONTH_INDEX >= 8].copy()
        adj = MarketRentAdjustment().fit(fit)
        before = adj.multipliers(future.MONTH_INDEX)
        future.MONTHLY_RENT *= 1000
        np.testing.assert_array_equal(before, adj.multipliers(future.MONTH_INDEX))
        np.testing.assert_allclose(adj.log_level_, .02 * np.arange(8), atol=1e-10)

    def test_observed_boundary_month_and_hold_forecast(self):
        adj = MarketRentAdjustment("hold").fit(rentals())
        np.testing.assert_allclose(adj.multipliers([11, 12, 27]), np.exp(.22))
        self.assertEqual(adj.multipliers([]).shape, (0,))
        with self.assertRaises(ValueError):
            adj.multipliers([-1])
        with self.assertRaises(ValueError):
            MarketRentAdjustment("oracle")


if __name__ == "__main__":
    unittest.main()
