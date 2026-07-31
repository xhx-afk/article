from __future__ import annotations

import unittest

import torch

from tests.test_unified_evaluator import evaluator


class QueryMaskMetricsTest(unittest.TestCase):
    def test_nonempty_metrics_have_no_smoothing_and_empty_empty_is_excluded(self):
        accumulator = evaluator.AlignmentAccumulator()
        selected = torch.tensor([[True, True, True]])
        truth = torch.tensor(
            [[[[True, False], [False, False]],
              [[True, False], [False, False]],
              [[False, False], [False, False]]]]
        )
        prediction = torch.tensor(
            [[[[True, False], [False, False]],
              [[False, True], [False, False]],
              [[False, False], [False, False]]]]
        )
        probability = prediction.float()
        accumulator._record_positive_mask_group(
            "matched", selected, probability, prediction, truth, torch.zeros((1, 3))
        )
        _, query, _, _ = accumulator.report()
        matched = query["matched"]
        nonempty = matched["nonempty_target"]
        self.assertEqual(matched["nonempty_target_count"], 2)
        self.assertEqual(matched["empty_target_count"], 1)
        self.assertEqual(matched["empty_empty_count"], 1)
        self.assertAlmostEqual(nonempty["micro_precision"], 0.5)
        self.assertAlmostEqual(nonempty["micro_recall"], 0.5)
        self.assertAlmostEqual(nonempty["micro_Dice"], 0.5)
        self.assertAlmostEqual(nonempty["micro_IoU"], 1 / 3)
        self.assertAlmostEqual(nonempty["macro_Dice_nonempty"], 0.5)
        self.assertNotIn("Dice", query["far_background"])

    def test_far_background_reports_false_activation_only(self):
        accumulator = evaluator.AlignmentAccumulator()
        probability = torch.tensor(
            [[[[0.1, 0.1], [0.1, 0.1]], [[0.9, 0.1], [0.1, 0.1]]]]
        )
        prediction = probability >= 0.5
        accumulator._record_far_mask_group(
            torch.tensor([[True, True]]), probability, prediction
        )
        _, query, _, _ = accumulator.report()
        far = query["far_background"]
        self.assertEqual(far["count"], 2)
        self.assertEqual(far["predicted_positive_pixel_count"], 1)
        self.assertAlmostEqual(far["false_positive_pixel_rate"], 1 / 8)
        self.assertAlmostEqual(far["empty_prediction_ratio"], 0.5)
        self.assertNotIn("Dice", far)

    def test_zero_denominator_is_null(self):
        accumulator = evaluator.AlignmentAccumulator()
        _, query, _, _ = accumulator.report()
        self.assertIsNone(query["matched"]["nonempty_target"]["micro_Dice"])
        self.assertIsNone(query["far_background"]["false_positive_pixel_rate"])


if __name__ == "__main__":
    unittest.main()
