from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch


if "torch.utils.tensorboard" not in sys.modules:
    tensorboard_module = types.ModuleType("torch.utils.tensorboard")
    tensorboard_module.SummaryWriter = object
    sys.modules["torch.utils.tensorboard"] = tensorboard_module
if "calflops" not in sys.modules:
    calflops_module = types.ModuleType("calflops")
    calflops_module.calculate_flops = lambda **_kwargs: ("N/A", "N/A", "N/A")
    sys.modules["calflops"] = calflops_module
if "faster_coco_eval" not in sys.modules:
    from pycocotools.coco import COCO
    from pycocotools import mask as pycoco_mask

    faster_module = types.ModuleType("faster_coco_eval")
    faster_module.__path__ = []
    faster_module.COCO = COCO
    faster_module.COCOeval_faster = object
    faster_module.init_as_pycocotools = lambda: None
    faster_core = types.ModuleType("faster_coco_eval.core")
    faster_core.__path__ = []
    faster_mask = types.ModuleType("faster_coco_eval.core.mask")
    for attribute in dir(pycoco_mask):
        if not attribute.startswith("__"):
            setattr(faster_mask, attribute, getattr(pycoco_mask, attribute))
    sys.modules["faster_coco_eval"] = faster_module
    sys.modules["faster_coco_eval.core"] = faster_core
    sys.modules["faster_coco_eval.core.mask"] = faster_mask

from engine.core import YAMLConfig  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the full HGNetv2 smoke test")
class SQMALForwardTest(unittest.TestCase):
    def _config(self, name: str) -> YAMLConfig:
        cfg = YAMLConfig(
            str(ROOT / "configs" / "deim_dfine" / "ablation_sqmal" / name),
            eval_spatial_size=[128, 128],
            num_classes=3,
        )
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        return cfg

    def test_q0_compatibility_and_q4_backward(self) -> None:
        device = torch.device("cuda")
        torch.manual_seed(7)
        q0_cfg = self._config("q0_baseline.yml")
        q0 = q0_cfg.model.to(device).eval()
        q4_cfg = self._config("q4_sqmal_hbg.yml")
        q4 = q4_cfg.model.to(device).eval()
        missing, unexpected = q4.load_state_dict(q0.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all("dec_quality_head" in key or "defectness_head" in key for key in missing))

        images = torch.randn((2, 3, 128, 128), device=device)
        with torch.no_grad():
            baseline = q0(images)
            extended = q4(images)
        self.assertTrue(torch.equal(baseline["pred_logits"], extended["pred_logits"]))
        self.assertTrue(torch.equal(baseline["pred_boxes"], extended["pred_boxes"]))
        self.assertIn("pred_quality", extended)
        self.assertNotIn("pred_defect_logits", extended)

        q4.train()
        criterion = q4_cfg.criterion.to(device).train()
        criterion.set_epoch(5)
        masks = torch.zeros((1, 128, 128), device=device, dtype=torch.bool)
        masks[:, 32:96, 32:96] = True
        targets = [
            {
                "labels": torch.tensor([index % 3], device=device),
                "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=device),
                "masks": masks.clone(),
                "mask_valid": torch.tensor([True], device=device),
                "orig_size": torch.tensor([128, 128], device=device),
            }
            for index in range(2)
        ]
        outputs = q4(images, targets=targets)
        self.assertEqual(tuple(outputs["pred_quality"].shape), (2, 300, 1))
        self.assertEqual(outputs["pred_defect_logits"].shape[:2], (2, 1))
        losses = criterion(outputs, targets)
        for key in ("loss_sqmal", "loss_quality", "loss_defect", "loss_hard_bg"):
            self.assertIn(key, losses)
            self.assertTrue(torch.isfinite(losses[key]))
        total = sum(value for key, value in losses.items() if key.startswith("loss_"))
        total.backward()
        for name in ("dec_quality_head", "defectness_head", "dec_score_head", "dec_bbox_head"):
            grad = sum(
                parameter.grad.abs().sum()
                for param_name, parameter in q4.named_parameters()
                if name in param_name and parameter.grad is not None
            )
            self.assertGreater(float(grad), 0.0, name)


if __name__ == "__main__":
    unittest.main()
