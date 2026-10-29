"""Check whole-block exclusion and selection independent of future labels."""

import unittest

import numpy as np
import pandas as pd

from hdb_rentals.block_evaluation import block_time_partitions


def fixture():
    return pd.DataFrame([
        {"block_id": block, "RENT_APPROVAL_DATE": date, "MONTHLY_RENT": 2000}
        for date in ["2022-01-01", "2023-08-01", "2024-02-01"]
        for block in range(10)
    ] + [{"block_id": 99, "RENT_APPROVAL_DATE": "2024-02-01", "MONTHLY_RENT": 3000}])


class BlockEvaluationTests(unittest.TestCase):
    def split(self, frame, **kwargs):
        return block_time_partitions(frame, "2023-07-01", "2024-01-01", **kwargs)

    def test_held_blocks_excluded_from_every_fit_and_tuning_partition(self):
        parts = self.split(fixture())
        held = set(parts["held_block_ids"])
        self.assertEqual(len(held), 2)
        for name in ["inner_fit", "inner_tune", "outer_fit"]:
            self.assertFalse(held.intersection(parts[name].block_id))
        evaluation = parts["outer_compare"]
        self.assertEqual(evaluation.block_slice.value_counts().to_dict(),
                         {"familiar": 8, "held_out": 2, "naturally_unseen": 1})
        self.assertEqual(set(evaluation.loc[evaluation.block_slice.eq("held_out"), "block_id"]), held)

    def test_selection_is_repeatable_and_does_not_use_future_rows_or_prices(self):
        original = fixture()
        changed = original.copy()
        changed.MONTHLY_RENT *= 100
        changed = pd.concat([changed, pd.DataFrame([{
            "block_id": 100, "RENT_APPROVAL_DATE": "2025-01-01", "MONTHLY_RENT": 9000}])])
        a, b = self.split(original), self.split(changed)
        np.testing.assert_array_equal(a["held_block_ids"], b["held_block_ids"])
        self.assertFalse(np.array_equal(a["held_block_ids"], self.split(original, seed=29)["held_block_ids"]))

    def test_zero_fraction_preserves_temporal_control_and_invalid_inputs_fail(self):
        parts = self.split(fixture(), fraction=0)
        self.assertEqual(len(parts["outer_fit"]), 20)
        self.assertEqual(len(parts["held_block_ids"]), 0)
        self.assertNotIn("held_out", set(parts["outer_compare"].block_slice))
        for fraction in [-.1, 1.0]:
            with self.assertRaises(ValueError):
                self.split(fixture(), fraction=fraction)


if __name__ == "__main__":
    unittest.main()
