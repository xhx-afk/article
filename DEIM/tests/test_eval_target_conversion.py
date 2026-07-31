from __future__ import annotations

import unittest

import torch
from torchvision import tv_tensors

from tools.wood.sqalign_v3_target_utils import build_alignment_targets


class EvalTargetConversionTest(unittest.TestCase):
    def test_tv_tensor_conversion_does_not_mutate_raw_target(self):
        raw_boxes = tv_tensors.BoundingBoxes(
            [[96.0, 192.0, 480.0, 672.0]], format="XYXY", canvas_size=(960, 960)
        )
        masks = torch.ones((1, 960, 960), dtype=torch.uint8)
        raw = {"boxes": raw_boxes, "masks": masks, "labels": torch.tensor([2])}
        converted = build_alignment_targets([raw], sample_hw=(32, 32))[0]
        torch.testing.assert_close(
            converted["boxes"], torch.tensor([[0.30, 0.45, 0.40, 0.50]])
        )
        torch.testing.assert_close(raw["boxes"].as_subclass(torch.Tensor), raw_boxes.as_subclass(torch.Tensor))
        self.assertIs(converted["masks"], masks)
        self.assertIsNot(converted, raw)

    def test_plain_tensor_fallback_and_empty_boxes(self):
        target = {"boxes": torch.tensor([[10.0, 20.0, 50.0, 80.0]])}
        converted = build_alignment_targets([target], sample_hw=(100, 100))[0]
        torch.testing.assert_close(converted["boxes"], torch.tensor([[0.30, 0.50, 0.40, 0.60]]))
        empty = build_alignment_targets(
            [{"boxes": torch.empty((0, 4))}], sample_hw=(100, 200)
        )[0]
        self.assertEqual(tuple(empty["boxes"].shape), (0, 4))

    def test_invalid_boxes_abort(self):
        invalid = (
            torch.tensor([[float("nan"), 0.0, 1.0, 1.0]]),
            torch.tensor([[0.0, 0.0, 101.0, 50.0]]),
            torch.tensor([[10.0, 10.0, 10.0, 20.0]]),
        )
        for boxes in invalid:
            with self.subTest(boxes=boxes), self.assertRaises(ValueError):
                build_alignment_targets([{"boxes": boxes}], sample_hw=(100, 100))


if __name__ == "__main__":
    unittest.main()
