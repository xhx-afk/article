from __future__ import annotations

import unittest

import torch

from tests.sqalign_test_utils import load_alignment_module


alignment = load_alignment_module("sqalign_v3_query_mask")


class QueryConditionedROIMaskTest(unittest.TestCase):
    def test_query_and_roi_both_change_mask(self) -> None:
        query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        roi = torch.zeros((1, 2, 2, 3, 3))
        roi[:, :, 0] = 1.0
        roi[:, 1, 1, 1:, 1:] = 2.0
        logits = alignment.query_conditioned_mask_logits(query, roi)
        self.assertEqual(tuple(logits.shape), (1, 2, 3, 3))
        self.assertFalse(torch.allclose(logits[:, 0], logits[:, 1]))

    def test_roi_align_shape_gradients_and_detached_boxes(self) -> None:
        torch.manual_seed(3)
        pixel_projection = alignment.SemanticPixelProjection(4, 8, downsample=False)
        query_projection = alignment.QuerySemanticProjection(6, 8)
        pixels = torch.randn((1, 4, 12, 12), requires_grad=True)
        queries = torch.randn((1, 2, 6), requires_grad=True)
        boxes = torch.tensor(
            [[[0.30, 0.30, 0.35, 0.35], [0.70, 0.70, 0.35, 0.35]]],
            requires_grad=True,
        )
        embedded_pixels = pixel_projection(pixels)
        roi = alignment.roi_align_query_pixel_features(
            embedded_pixels, boxes, output_size=7, detach_boxes=True
        )
        embedded_queries = query_projection(queries)
        logits = alignment.query_conditioned_mask_logits(embedded_queries, roi)
        self.assertEqual(tuple(roi.shape), (1, 2, 8, 7, 7))
        self.assertEqual(tuple(logits.shape), (1, 2, 7, 7))
        logits.square().mean().backward()
        self.assertIsNotNone(pixels.grad)
        self.assertIsNotNone(queries.grad)
        self.assertIsNone(boxes.grad)
        self.assertGreater(float(pixels.grad.abs().sum()), 0.0)
        self.assertGreater(float(queries.grad.abs().sum()), 0.0)

    def test_same_query_different_rois_produce_different_masks(self) -> None:
        pixels = torch.zeros((1, 2, 8, 8))
        pixels[:, 0, :4, :4] = 1
        pixels[:, 1, 4:, 4:] = 1
        boxes = torch.tensor([[[0.25, 0.25, 0.50, 0.50], [0.75, 0.75, 0.50, 0.50]]])
        roi = alignment.roi_align_query_pixel_features(pixels, boxes, output_size=3)
        query = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        logits = alignment.query_conditioned_mask_logits(query, roi)
        self.assertFalse(torch.allclose(logits[0, 0], logits[0, 1]))


if __name__ == "__main__":
    unittest.main()

