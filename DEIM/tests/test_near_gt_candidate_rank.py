from __future__ import annotations

import unittest

import torch

from tests.sqalign_test_utils import load_alignment_module


alignment = load_alignment_module("sqalign_v3_candidate")


class NearGTCandidateRankTest(unittest.TestCase):
    def _groups(self, scores: torch.Tensor, boxes: torch.Tensor):
        targets = [{
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
            "labels": torch.tensor([0]),
        }]
        matched = [(torch.tensor([0]), torch.tensor([0]))]
        return alignment.build_per_gt_candidate_groups(
            scores, boxes, targets, matched, iou_threshold=0.30, max_candidates_per_gt=10
        )

    def test_ordered_scores_have_lower_loss(self) -> None:
        boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.3, 0.3], [0.5, 0.5, 0.24, 0.24]]])
        good = torch.tensor([[[0.90, 0.05], [0.55, 0.05], [0.20, 0.05]]])
        bad = torch.tensor([[[0.20, 0.05], [0.55, 0.05], [0.90, 0.05]]])
        groups = self._groups(good, boxes)
        good_loss = alignment.near_gt_candidate_rank_loss(good, groups)
        bad_loss = alignment.near_gt_candidate_rank_loss(bad, groups)
        self.assertLess(float(good_loss), float(bad_loss))

    def test_wrong_anchor_class_increases_loss(self) -> None:
        boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.3, 0.3]]])
        good = torch.tensor([[[0.90, 0.10], [0.50, 0.10]]])
        confused = torch.tensor([[[0.90, 0.98], [0.50, 0.10]]])
        groups = self._groups(good, boxes)
        self.assertLess(
            float(alignment.near_gt_candidate_rank_loss(good, groups)),
            float(alignment.near_gt_candidate_rank_loss(confused, groups)),
        )

    def test_insufficient_candidates_return_zero_and_boxes_are_detached(self) -> None:
        boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.1, 0.1, 0.1, 0.1]]], requires_grad=True)
        scores = torch.tensor([[[0.9, 0.1], [0.2, 0.1]]], requires_grad=True)
        groups = self._groups(scores, boxes)
        loss = alignment.near_gt_candidate_rank_loss(scores, groups)
        self.assertEqual(float(loss), 0.0)
        (loss + scores.sum() * 0.0).backward()
        self.assertIsNone(boxes.grad)

    def test_rank_loss_updates_scores_not_boxes(self) -> None:
        boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.3, 0.3]]], requires_grad=True)
        scores = torch.tensor([[[0.3, 0.8], [0.8, 0.1]]], requires_grad=True)
        groups = self._groups(scores, boxes)
        loss = alignment.near_gt_candidate_rank_loss(scores, groups)
        loss.backward()
        self.assertIsNotNone(scores.grad)
        self.assertGreater(float(scores.grad.abs().sum()), 0.0)
        self.assertIsNone(boxes.grad)


if __name__ == "__main__":
    unittest.main()

