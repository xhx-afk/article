from __future__ import annotations

import unittest

import torch

from tests.sqalign_test_utils import load_alignment_module


alignment = load_alignment_module("background_ranking_core")


class BackgroundRankingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.best_iou = torch.tensor([[0.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0]])
        self.indices = [(torch.tensor([0]), torch.tensor([0]))]
        self.targets = [{"labels": torch.tensor([0])}]

    def test_candidates_are_unmatched_far_and_topk_uses_final_score(self) -> None:
        scores = torch.tensor([[[0.8], [0.1], [0.9], [0.99], [0.7], [0.6], [0.5]]])
        selected, counts = alignment.select_final_score_background(
            scores, self.best_iou, self.indices, topk=2, iou_threshold=0.1
        )
        self.assertEqual(float(counts[0]), 2.0)
        self.assertTrue(selected[0, 2])
        self.assertTrue(selected[0, 4])
        self.assertFalse(selected[0, 0])
        self.assertFalse(selected[0, 3])
        selected_five, counts_five = alignment.select_final_score_background(
            scores, self.best_iou, self.indices, topk=5, iou_threshold=0.1
        )
        self.assertEqual(float(counts_five[0]), 5.0)
        self.assertEqual(int(selected_five.sum()), 5)

    def test_ranking_loss_order_and_empty_cases(self) -> None:
        selected = torch.tensor([[False, True, False, False, False, False, False]])
        good = torch.tensor([[[0.9], [0.1], [0.0], [0.0], [0.0], [0.0], [0.0]]], requires_grad=True)
        bad = torch.tensor([[[0.2], [0.8], [0.0], [0.0], [0.0], [0.0], [0.0]]], requires_grad=True)
        good_loss = alignment.final_score_ranking_loss(good, self.targets, self.indices, selected)
        bad_loss = alignment.final_score_ranking_loss(bad, self.targets, self.indices, selected)
        self.assertLess(float(good_loss), float(bad_loss))
        bad_loss.backward()
        self.assertGreater(float(bad.grad.abs().sum()), 0.0)
        empty_indices = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
        zero = alignment.final_score_ranking_loss(bad, self.targets, empty_indices, selected)
        self.assertEqual(float(zero), 0.0)
        self.assertTrue(zero.requires_grad)
        no_background = alignment.final_score_ranking_loss(
            bad, self.targets, self.indices, torch.zeros_like(selected)
        )
        self.assertEqual(float(no_background), 0.0)

    def test_gradient_reaches_class_loc_and_semantic_branches(self) -> None:
        class_logits = torch.tensor([[[2.0], [1.0]]], requires_grad=True)
        loc_logits = torch.tensor([[[1.0], [0.5]]], requires_grad=True)
        defect_logit = torch.tensor([[[0.5], [-0.5]]], requires_grad=True)
        semantic = defect_logit.sigmoid()
        final = alignment.compose_dual_quality_scores(
            class_logits.sigmoid(), loc_logits, semantic, 0.25, 0.25
        )
        selected = torch.tensor([[False, True]])
        targets = [{"labels": torch.tensor([0])}]
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        loss = alignment.final_score_ranking_loss(final, targets, indices, selected)
        loss.backward()
        for tensor in (class_logits, loc_logits, defect_logit):
            self.assertGreater(float(tensor.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
