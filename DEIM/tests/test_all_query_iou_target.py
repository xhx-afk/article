from __future__ import annotations

import unittest

import torch

from tests.sqalign_test_utils import load_alignment_module


alignment = load_alignment_module("all_query_iou_core")


class AllQueryIoUTargetTest(unittest.TestCase):
    def test_overlap_disjoint_best_of_many_and_batch_variation(self) -> None:
        boxes = torch.tensor([
            [[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]],
            [[0.7, 0.7, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]],
            [[0.3, 0.3, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]],
        ], requires_grad=True)
        targets = [
            {"boxes": torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]])},
            {"boxes": torch.tensor([[0.7, 0.7, 0.2, 0.2]])},
            {"boxes": torch.empty((0, 4))},
        ]
        target = alignment.build_all_query_best_iou_target(boxes, targets)
        self.assertEqual(tuple(target.shape), (3, 2))
        self.assertEqual(float(target[0, 0]), 1.0)
        self.assertEqual(float(target[0, 1]), 0.0)
        self.assertEqual(float(target[1, 0]), 1.0)
        self.assertTrue(torch.equal(target[2], torch.zeros(2)))
        self.assertFalse(target.requires_grad)
        self.assertTrue(torch.isfinite(target).all())

    def test_amp_is_finite_and_target_does_not_backpropagate(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        boxes = torch.rand((2, 8, 4), device=device, requires_grad=True)
        targets = [
            {"boxes": torch.rand((3, 4), device=device)},
            {"boxes": torch.empty((0, 4), device=device)},
        ]
        device_type = "cuda" if device.type == "cuda" else "cpu"
        amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        with torch.autocast(device_type=device_type, dtype=amp_dtype):
            target = alignment.build_all_query_best_iou_target(boxes, targets)
        self.assertTrue(torch.isfinite(target).all())
        self.assertFalse(target.requires_grad)
        self.assertIsNone(boxes.grad)

    def test_empty_query_dimension(self) -> None:
        boxes = torch.empty((2, 0, 4), requires_grad=True)
        targets = [{"boxes": torch.rand((1, 4))}, {"boxes": torch.empty((0, 4))}]
        target = alignment.build_all_query_best_iou_target(boxes, targets)
        self.assertEqual(tuple(target.shape), (2, 0))
        self.assertTrue(torch.isfinite(target).all())


if __name__ == "__main__":
    unittest.main()
