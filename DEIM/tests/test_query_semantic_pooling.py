from __future__ import annotations

import unittest

import torch

from tests.sqalign_test_utils import load_alignment_module


alignment = load_alignment_module("semantic_pooling_core")


class QuerySemanticPoolingTest(unittest.TestCase):
    def test_high_low_thin_evidence_and_ratio(self) -> None:
        logits = torch.full((1, 1, 32, 32), -8.0, requires_grad=True)
        with torch.no_grad():
            logits[:, :, 4:16, 4:16] = 8.0
            logits[:, :, 20, 4:28] = 8.0
        boxes = torch.tensor([[
            [0.3125, 0.3125, 0.25, 0.25],
            [0.80, 0.20, 0.20, 0.20],
            [0.50, 0.64, 0.75, 0.10],
        ]], requires_grad=True)
        topk = alignment.pool_query_semantic_evidence(logits, boxes, output_size=7, topk_ratio=0.10)
        mean_like = alignment.pool_query_semantic_evidence(logits, boxes, output_size=7, topk_ratio=1.0)
        self.assertGreater(float(topk[0, 0]), 0.9)
        self.assertLess(float(topk[0, 1]), 0.1)
        self.assertGreater(float(topk[0, 2]), float(mean_like[0, 2]))
        topk.sum().backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        self.assertIsNone(boxes.grad)

    def test_clamps_out_of_bounds_and_handles_empty_queries(self) -> None:
        logits = torch.zeros((2, 1, 8, 8), requires_grad=True)
        boxes = torch.tensor([
            [[-0.2, -0.2, 1.0, 1.0], [1.2, 1.2, 1.0, 1.0]],
            [[0.5, 0.5, 0.0, 0.0], [0.5, 0.5, 2.0, 2.0]],
        ])
        scores = alignment.pool_query_semantic_evidence(logits, boxes)
        self.assertEqual(tuple(scores.shape), (2, 2, 1))
        self.assertTrue(torch.isfinite(scores).all())
        empty = alignment.pool_query_semantic_evidence(logits[:1], boxes[:1, :0])
        self.assertEqual(tuple(empty.shape), (1, 0, 1))
        empty_batch = alignment.pool_query_semantic_evidence(
            torch.empty((0, 1, 8, 8)), torch.empty((0, 2, 4))
        )
        self.assertEqual(tuple(empty_batch.shape), (0, 2, 1))

    def test_semantic_supervision_ignores_unmatched_near_gt_queries(self) -> None:
        best_iou = torch.tensor([[0.8, 0.2, 0.0]])
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        target, valid, positive, background = alignment.build_query_semantic_supervision(
            best_iou, indices, background_iou_threshold=0.1
        )
        torch.testing.assert_close(target, torch.tensor([[1.0, 0.0, 0.0]]))
        self.assertTrue(positive[0, 0] and valid[0, 0])
        self.assertFalse(valid[0, 1])
        self.assertTrue(background[0, 2] and valid[0, 2])

if __name__ == "__main__":
    unittest.main()
