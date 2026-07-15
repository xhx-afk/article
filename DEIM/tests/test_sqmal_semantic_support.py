from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "engine" / "deim" / "sqmal.py"
SPEC = importlib.util.spec_from_file_location("sqmal_standalone", MODULE_PATH)
sqmal = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(sqmal)


class SemanticSupportTest(unittest.TestCase):
    def test_support_behaviour_and_fallback(self) -> None:
        masks = torch.zeros((4, 64, 64), dtype=torch.bool)
        masks[0, 16:48, 16:48] = True
        masks[1, 16:48, 16:48] = True
        masks[2, 16:48, 16:48] = True
        masks[3, 31:33, 8:56] = True
        gt = torch.tensor([
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.75, 2 / 64],
        ])
        pred = torch.tensor([
            [0.5, 0.5, 0.5, 0.5],
            [0.375, 0.5, 0.25, 0.5],
            [0.5, 0.5, 1.0, 1.0],
            [0.5, 0.5, 0.75, 2 / 64],
        ])
        support = sqmal.compute_semantic_support(pred, gt, masks, torch.ones(4, dtype=torch.bool))
        self.assertGreater(float(support[0]), 0.99)
        self.assertLess(float(support[1]), float(support[0]))
        self.assertLess(float(support[2]), float(support[0]))
        self.assertGreater(float(support[3]), 0.95)
        self.assertTrue(torch.isfinite(support).all())

        invalid = sqmal.compute_semantic_support(pred[:1], gt[:1], masks[:1], torch.tensor([False]))
        self.assertEqual(float(invalid[0]), 1.0)

    def test_empty_and_beta_zero(self) -> None:
        boxes = torch.empty((0, 4))
        support = sqmal.compute_semantic_support(
            boxes, boxes, torch.empty((0, 64, 64), dtype=torch.bool), torch.empty((0,), dtype=torch.bool)
        )
        self.assertEqual(tuple(support.shape), (0,))
        iou = torch.tensor([0.2, 0.8])
        target = sqmal.compute_sq_quality_target(iou, torch.tensor([0.1, 0.1]), beta=0.0)
        self.assertTrue(torch.allclose(target, iou))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_amp(self) -> None:
        masks = torch.zeros((1, 64, 64), device="cuda", dtype=torch.bool)
        masks[:, 16:48, 16:48] = True
        boxes = torch.tensor([[0.5, 0.5, 0.5, 0.5]], device="cuda", dtype=torch.float16)
        with torch.autocast("cuda", dtype=torch.float16):
            output = sqmal.compute_semantic_support(boxes, boxes, masks, torch.tensor([True], device="cuda"))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()

