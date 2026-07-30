from __future__ import annotations

import unittest
from pathlib import Path

import torch

from tests.sqalign_test_utils import ROOT, install_optional_dependency_stubs


install_optional_dependency_stubs()

from engine.core import YAMLConfig  # noqa: E402


class SQAlignForwardBackwardTest(unittest.TestCase):
    def _config(self, name: str, image_size: int = 128, num_classes: int = 3) -> YAMLConfig:
        config = YAMLConfig(
            str(ROOT / "configs" / "deim_dfine" / "ablation_sqalign" / name),
            eval_spatial_size=[image_size, image_size],
            num_classes=num_classes,
        )
        config.yaml_cfg["HGNetv2"]["pretrained"] = False
        return config

    def test_clean_ablation_ladder_builds(self) -> None:
        expected = {
            "r0_baseline.yml": (False, False, False, False),
            "r1_all_query_loc.yml": (True, False, False, False),
            "r2_loc_defect_aux.yml": (True, True, False, False),
            "r3_query_semantic.yml": (True, True, True, False),
            "r4_final_score_rank.yml": (True, True, True, True),
        }
        for filename, flags in expected.items():
            with self.subTest(filename=filename):
                config = self._config(filename, image_size=64)
                model = config.model
                criterion = config.criterion
                actual = (
                    model.decoder.use_loc_quality_head,
                    model.use_defectness_head,
                    model.use_query_semantic,
                    criterion.use_final_bg_rank,
                )
                self.assertEqual(actual, flags)
                self.assertIn("mal", criterion.losses)

    def test_coco_pretrained_checkpoint_has_loadable_shared_weights(self) -> None:
        checkpoint_path = ROOT / "weight" / "deim_dfine_hgnetv2_l_coco_50e.pth"
        if not checkpoint_path.exists():
            self.skipTest("local COCO tuning checkpoint is unavailable")
        config = self._config("r4_final_score_rank.yml", image_size=64, num_classes=9)
        model = config.model
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        pretrained = checkpoint.get("ema", {}).get("module", checkpoint.get("model", checkpoint))
        current = model.state_dict()
        compatible = {
            key: value for key, value in pretrained.items()
            if key in current and current[key].shape == value.shape
        }
        result = model.load_state_dict(compatible, strict=False)
        self.assertGreater(len(compatible), 100)
        self.assertTrue(any("dec_loc_quality_head" in key for key in result.missing_keys))
        self.assertTrue(any("defectness_head" in key for key in result.missing_keys))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the full-model AMP smoke test")
    def test_two_image_amp_forward_backward_and_main_only_losses(self) -> None:
        device = torch.device("cuda")
        torch.manual_seed(11)
        config = self._config("r4_final_score_rank.yml")
        model = config.model.to(device).train()
        criterion = config.criterion.to(device).train()
        criterion.set_epoch(10)
        images = torch.randn((2, 3, 128, 128), device=device)
        targets = []
        for class_index in (0, 1):
            mask = torch.zeros((1, 128, 128), device=device, dtype=torch.bool)
            mask[:, 32:96, 32:96] = True
            targets.append({
                "labels": torch.tensor([class_index], device=device),
                "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=device),
                "masks": mask,
                "mask_valid": torch.tensor([True], device=device),
                "orig_size": torch.tensor([128, 128], device=device),
            })

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images, targets=targets)
            losses = criterion(outputs, targets)
            total = sum(value for key, value in losses.items() if key.startswith("loss_"))

        self.assertEqual(tuple(outputs["pred_logits"].shape), (2, 300, 3))
        self.assertEqual(tuple(outputs["pred_boxes"].shape), (2, 300, 4))
        self.assertEqual(tuple(outputs["pred_loc_quality"].shape), (2, 300, 1))
        self.assertEqual(tuple(outputs["pred_sem_quality"].shape), (2, 300, 1))
        self.assertEqual(outputs["pred_defect_logits"].shape[:2], (2, 1))
        for key in (
            "loss_loc_quality",
            "loss_defect",
            "loss_query_semantic",
            "loss_final_bg_rank",
        ):
            self.assertIn(key, losses)
            self.assertTrue(torch.isfinite(losses[key]), key)
            self.assertFalse(any(name.startswith(key + "_") for name in losses), key)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        for parameter_group in (
            "dec_loc_quality_head",
            "defectness_head",
            "dec_score_head",
            "dec_bbox_head",
            "encoder",
        ):
            gradient_parts = [
                parameter.grad.detach().float().flatten()
                for name, parameter in model.named_parameters()
                if parameter_group in name and parameter.grad is not None
            ]
            self.assertTrue(gradient_parts, parameter_group)
            gradient = torch.cat(gradient_parts)
            finite_gradient = gradient[torch.isfinite(gradient)]
            self.assertGreater(finite_gradient.numel(), 0, parameter_group)
            self.assertGreater(float(finite_gradient.abs().sum()), 0.0, parameter_group)

        model.eval()
        postprocessor = config.postprocessor.to(device).eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            inference_outputs = model(images)
            results = postprocessor(
                inference_outputs,
                torch.tensor([[128, 128], [128, 128]], device=device),
            )
        self.assertIn("pred_loc_quality", inference_outputs)
        self.assertIn("pred_sem_quality", inference_outputs)
        self.assertNotIn("pred_defect_logits", inference_outputs)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["scores"].shape[0], 300)


if __name__ == "__main__":
    unittest.main()
