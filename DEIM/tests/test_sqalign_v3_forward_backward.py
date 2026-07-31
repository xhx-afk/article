from __future__ import annotations

import contextlib
import unittest

import torch

from tests.sqalign_test_utils import ROOT, install_optional_dependency_stubs


install_optional_dependency_stubs()

from engine.core import YAMLConfig  # noqa: E402


class SQAlignV3ForwardBackwardTest(unittest.TestCase):
    def _config(self, name: str, image_size: int = 64, num_classes: int = 3) -> YAMLConfig:
        config = YAMLConfig(
            str(ROOT / "configs" / "deim_dfine" / "ablation_sqalign_v3" / name),
            eval_spatial_size=[image_size, image_size],
            num_classes=num_classes,
        )
        config.yaml_cfg["HGNetv2"]["pretrained"] = False
        return config

    def test_clean_v0_to_v4_ladder_builds(self) -> None:
        expected = {
            "v0_baseline.yml": (False, False, False, False),
            "v1_defect_aux.yml": (True, False, False, False),
            "v2_query_conditioned_mask.yml": (True, True, False, False),
            "v3_continuous_semantic_quality.yml": (True, True, True, False),
            "v4_near_gt_candidate_rank.yml": (True, True, True, True),
        }
        for filename, flags in expected.items():
            with self.subTest(filename=filename):
                config = self._config(filename)
                model, criterion = config.model, config.criterion
                actual = (
                    model.enable_defect_auxiliary,
                    model.enable_query_conditioned_mask,
                    model.enable_semantic_quality,
                    criterion.enable_candidate_rank,
                )
                self.assertEqual(actual, flags)
                self.assertIn("mal", criterion.losses)
                self.assertNotIn("loc_quality", criterion.losses)

    def test_coco_pretrained_weights_remain_loadable(self) -> None:
        checkpoint_path = ROOT / "weight" / "deim_dfine_hgnetv2_l_coco_50e.pth"
        if not checkpoint_path.exists():
            self.skipTest("local COCO tuning checkpoint is unavailable")
        model = self._config("v4_near_gt_candidate_rank.yml", num_classes=9).model
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        pretrained = checkpoint.get("ema", {}).get("module", checkpoint.get("model", checkpoint))
        current = model.state_dict()
        compatible = {
            key: value for key, value in pretrained.items()
            if key in current and current[key].shape == value.shape
        }
        result = model.load_state_dict(compatible, strict=False)
        self.assertGreater(len(compatible), 100)
        self.assertTrue(any("defectness_head" in key for key in result.missing_keys))
        self.assertTrue(any("semantic_pixel_projection" in key for key in result.missing_keys))

    def test_batch_two_forward_backward_and_main_only_losses(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(11)
        image_size = 128
        config = self._config("v4_near_gt_candidate_rank.yml", image_size=image_size)
        model = config.model.to(device).train()
        criterion = config.criterion.to(device).train()
        criterion.set_epoch(20, total_epochs=25)
        images = torch.randn((2, 3, image_size, image_size), device=device)
        targets = []
        for class_index in (0, 1):
            mask = torch.zeros((1, image_size, image_size), device=device, dtype=torch.bool)
            mask[:, 32:96, 32:96] = True
            targets.append({
                "labels": torch.tensor([class_index], device=device),
                "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]], device=device),
                "masks": mask,
                "mask_valid": torch.tensor([True], device=device),
                "orig_size": torch.tensor([image_size, image_size], device=device),
            })

        # Full precision isolates graph correctness; AMP + GradScaler is covered
        # by the bounded smoke script where scaled backward is representative.
        autocast = contextlib.nullcontext()
        with autocast:
            outputs = model(images, targets=targets)
            losses = criterion(outputs, targets)
            total = sum(value for key, value in losses.items() if key.startswith("loss_"))

        self.assertEqual(tuple(outputs["pred_logits"].shape), (2, 300, 3))
        self.assertEqual(tuple(outputs["pred_boxes"].shape), (2, 300, 4))
        self.assertEqual(tuple(outputs["pred_query_features"].shape), (2, 300, 256))
        self.assertEqual(outputs["pred_defect_logits"].shape[:2], (2, 1))
        self.assertEqual(tuple(outputs["pred_query_mask_logits"].shape), (2, 300, 7, 7))
        self.assertEqual(tuple(outputs["pred_sem_quality"].shape), (2, 300, 1))
        self.assertNotIn("pred_loc_quality", outputs)
        for key in (
            "loss_mal", "loss_bbox", "loss_giou", "loss_fgl",
            "loss_defect", "loss_query_mask", "loss_semantic_quality",
            "loss_semantic_rank", "loss_candidate_rank",
        ):
            self.assertIn(key, losses)
            self.assertTrue(torch.isfinite(losses[key]), key)
        # DDF is conditional in the upstream D-FINE criterion: it is omitted
        # when teacher and student corner distributions are exactly equal.
        if "loss_ddf" in losses:
            self.assertTrue(torch.isfinite(losses["loss_ddf"]))
        for main_only in (
            "loss_defect", "loss_query_mask", "loss_semantic_quality",
            "loss_semantic_rank", "loss_candidate_rank",
        ):
            self.assertFalse(any(key.startswith(main_only + "_") for key in losses), main_only)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        for parameter_group in (
            "defectness_head", "semantic_pixel_projection", "query_semantic_projection",
            "semantic_quality_head", "dec_score_head", "encoder",
        ):
            gradients = [
                parameter.grad.detach().float().flatten()
                for name, parameter in model.named_parameters()
                if parameter_group in name and parameter.grad is not None
            ]
            self.assertTrue(gradients, parameter_group)
            gradient = torch.cat(gradients)
            self.assertTrue(bool(torch.isfinite(gradient).all()), parameter_group)
            self.assertGreater(float(gradient.abs().sum()), 0.0, parameter_group)

        model.eval()
        postprocessor = config.postprocessor.to(device).eval()
        postprocessor.semantic_gate_enabled = True
        with torch.no_grad():
            inference_outputs = model(images)
            results = postprocessor(
                inference_outputs,
                torch.tensor([[image_size, image_size], [image_size, image_size]], device=device),
            )
        self.assertIn("pred_query_features", inference_outputs)
        self.assertIn("pred_query_mask_logits", inference_outputs)
        self.assertIn("pred_sem_quality", inference_outputs)
        self.assertNotIn("pred_defect_logits", inference_outputs)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["scores"].shape[0], 300)


if __name__ == "__main__":
    unittest.main()
