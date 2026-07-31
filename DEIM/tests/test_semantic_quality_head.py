from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from tests.sqalign_test_utils import load_alignment_module


alignment = load_alignment_module("sqalign_v3_semantic_head_test")


class SemanticQualityHeadTest(unittest.TestCase):
    def test_statistics_distinguish_dense_and_isolated_response(self):
        background = torch.full((1, 1, 7, 7), -8.0)
        foreground = torch.full((1, 1, 7, 7), 8.0)
        isolated = background.clone()
        isolated[..., 0, 0] = 8.0
        stats = alignment.semantic_mask_statistics(torch.cat((background, foreground, isolated), dim=1))
        self.assertLess(stats[0, 0, 0].item(), 0.01)
        self.assertGreater(stats[0, 1, 0].item(), 0.99)
        self.assertLess(stats[0, 2, 0].item(), 0.10)
        self.assertLess(stats[0, 2, 5].item(), 0.10)

    def test_learned_head_and_soft_bce_backward(self):
        head = alignment.SemanticQualityHead(query_dim=8, query_projection_dim=4, hidden_dim=16)
        masks = torch.randn((2, 3, 7, 7), requires_grad=True)
        query = torch.randn((2, 3, 8), requires_grad=True)
        logits = head(masks, query)
        target = torch.tensor([[[0.0], [0.5], [1.0]], [[0.2], [0.7], [0.9]]])
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss.backward()
        self.assertEqual(tuple(logits.shape), (2, 3, 1))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(head.quality_mlp[-1].weight.grad)
        self.assertIsNotNone(masks.grad)


if __name__ == "__main__":
    unittest.main()

