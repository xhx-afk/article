from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from tests.sqalign_test_utils import load_alignment_module


alignment = load_alignment_module("sqalign_v3_gate")


class SemanticSuppressionGateTest(unittest.TestCase):
    def test_gate_limits_and_identity(self) -> None:
        logits = torch.tensor([[[-100.0], [100.0], [0.0]]])
        gate = alignment.semantic_suppression_gate(logits, gate_lambda=0.2, gamma=2.0)
        self.assertAlmostEqual(float(gate[0, 0]), 0.8, places=5)
        self.assertAlmostEqual(float(gate[0, 1]), 1.0, places=5)
        self.assertTrue(bool((gate <= 1.0).all()))
        self.assertTrue(bool((gate >= 0.8).all()))
        identity = alignment.semantic_suppression_gate(logits, gate_lambda=0.0, gamma=3.0)
        self.assertTrue(torch.equal(identity, torch.ones_like(identity)))
        class_scores = torch.rand((1, 3, 4))
        self.assertTrue(bool((class_scores * gate <= class_scores).all()))

    def test_statistics_do_not_reduce_to_raw_top_pixel(self) -> None:
        background = torch.full((1, 1, 7, 7), -10.0)
        isolated = background.clone()
        isolated[..., 3, 3] = 10.0
        foreground = torch.full((1, 1, 7, 7), 10.0)
        stats = alignment.semantic_mask_statistics(torch.cat((background, isolated, foreground), dim=1))
        self.assertLess(float(stats[0, 1, 0]), 0.1)  # ROI mean stays low for one hot pixel.
        self.assertGreater(float(stats[0, 2, 0]), 0.9)
        self.assertGreater(float(stats[0, 1, 1]), 0.9)  # max alone would be misleading.

    def test_learned_head_accepts_soft_targets_and_backpropagates(self) -> None:
        torch.manual_seed(7)
        head = alignment.SemanticQualityHead(query_dim=8, query_projection_dim=4, hidden_dim=16)
        masks = torch.randn((2, 3, 7, 7), requires_grad=True)
        queries = torch.randn((2, 3, 8), requires_grad=True)
        logits = head(masks, queries)
        target = torch.linspace(0, 1, 6).reshape(2, 3, 1)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        self.assertEqual(tuple(logits.shape), (2, 3, 1))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(masks.grad.abs().sum()), 0.0)
        self.assertGreater(float(queries.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

