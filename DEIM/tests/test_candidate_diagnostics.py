from __future__ import annotations

import unittest

import torch

from tests.test_unified_evaluator import evaluator


class CandidateDiagnosticsTest(unittest.TestCase):
    def test_synthetic_near_candidates_produce_pairs(self):
        outputs = {
            "pred_logits": torch.tensor(
                [[[4.0, -4.0], [3.0, -4.0], [2.0, -4.0], [-4.0, 4.0], [-4.0, 2.0]]]
            ),
            "pred_boxes": torch.tensor(
                [[[0.50, 0.50, 0.40, 0.40],
                  [0.50, 0.50, 0.30, 0.30],
                  [0.50, 0.50, 0.25, 0.25],
                  [0.20, 0.20, 0.20, 0.20],
                  [0.20, 0.20, 0.14, 0.14]]]
            ),
        }
        targets = [{
            "boxes": torch.tensor([[0.50, 0.50, 0.40, 0.40], [0.20, 0.20, 0.20, 0.20]]),
            "labels": torch.tensor([0, 1]),
        }]
        indices = [(torch.tensor([0, 3]), torch.tensor([0, 1]))]
        accumulator = evaluator.AlignmentAccumulator()
        accumulator.update_candidates(outputs, targets, indices)
        _, _, candidate, _ = accumulator.report()
        self.assertEqual(candidate["num_GT"], 2)
        self.assertGreater(candidate["GT_with_at_least_2_candidates"], 0)
        self.assertGreater(candidate["ranking_pair_count"], 0)
        self.assertEqual(candidate["GT_with_anchor"], 2)
        self.assertIn("0", candidate["per_class"])
        self.assertIn("candidate_iou_p50", candidate)


if __name__ == "__main__":
    unittest.main()
