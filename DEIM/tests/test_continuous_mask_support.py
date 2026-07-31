from __future__ import annotations

import unittest

import torch

from tests.sqalign_test_utils import load_alignment_module


alignment = load_alignment_module("sqalign_v3_continuous_support")


class ContinuousMaskSupportTest(unittest.TestCase):
    def test_box_support_is_continuous_and_density_aware(self) -> None:
        mask = torch.zeros((1, 100, 100), dtype=torch.float32)
        mask[0, 30:70, 30:70] = 1
        gt_boxes = torch.tensor([[0.50, 0.50, 0.40, 0.40]])
        pred_boxes = torch.tensor([
            [0.50, 0.50, 0.40, 0.40],  # exact
            [0.40, 0.50, 0.20, 0.40],  # half the mask
            [0.50, 0.50, 0.80, 0.80],  # full coverage, low density
            [0.08, 0.08, 0.08, 0.08],  # background
        ])
        support = alignment.compute_continuous_mask_support(
            mask, pred_boxes, gt_boxes, torch.zeros(4, dtype=torch.long)
        )
        self.assertAlmostEqual(float(support[0]), 1.0, places=4)
        self.assertGreater(float(support[1]), 0.4)
        self.assertLess(float(support[1]), float(support[0]))
        self.assertLess(float(support[2]), float(support[0]))
        self.assertAlmostEqual(float(support[3]), 0.0, places=6)

    def test_thin_mask_with_correct_box_is_not_lost(self) -> None:
        mask = torch.zeros((1, 100, 100), dtype=torch.float32)
        mask[0, 49:51, 20:80] = 1
        box = torch.tensor([[0.50, 0.50, 0.60, 0.02]])
        support = alignment.compute_continuous_mask_support(
            mask, box, box, torch.tensor([0])
        )
        self.assertGreater(float(support[0]), 0.9)

    def test_assignment_prefers_hungarian_gt_and_handles_empty_gt(self) -> None:
        pred = torch.tensor([[[0.25, 0.50, 0.20, 0.20], [0.75, 0.50, 0.20, 0.20]]])
        targets = [{"boxes": torch.tensor([[0.25, 0.50, 0.20, 0.20], [0.75, 0.50, 0.20, 0.20]])}]
        matched = [(torch.tensor([0]), torch.tensor([1]))]
        iou, index = alignment.build_best_gt_assignment(pred, targets, matched)
        self.assertEqual(int(index[0, 0]), 1)
        self.assertAlmostEqual(float(iou[0, 0]), 0.0, places=5)
        empty_iou, empty_index = alignment.build_best_gt_assignment(
            pred, [{"boxes": torch.empty((0, 4))}], [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
        )
        self.assertTrue(torch.equal(empty_iou, torch.zeros_like(empty_iou)))
        self.assertTrue(torch.equal(empty_index, torch.full_like(empty_index, -1)))

    def test_invalid_mask_is_ignored_and_far_background_is_zero(self) -> None:
        pred = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.05, 0.05, 0.05, 0.05]]])
        targets = [{
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
            "masks": torch.ones((1, 32, 32)),
            "mask_valid": torch.tensor([False]),
        }]
        matched = [(torch.tensor([0]), torch.tensor([0]))]
        best_iou, best_index = alignment.build_best_gt_assignment(pred, targets, matched)
        target, valid, is_matched, near, far = alignment.build_semantic_quality_targets(
            pred, targets, best_index, best_iou, matched
        )
        self.assertFalse(bool(valid[0, 0]))
        self.assertTrue(bool(valid[0, 1]))
        self.assertEqual(float(target[0, 1]), 0.0)
        self.assertTrue(bool(is_matched[0, 0]))
        self.assertFalse(bool(near[0, 0]))
        self.assertTrue(bool(far[0, 1]))


if __name__ == "__main__":
    unittest.main()

