from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_criterion():
    root_package = types.ModuleType("criterion_testpkg")
    root_package.__path__ = []
    deim_package = types.ModuleType("criterion_testpkg.deim")
    deim_package.__path__ = []
    misc_package = types.ModuleType("criterion_testpkg.misc")
    misc_package.__path__ = []
    core = types.ModuleType("criterion_testpkg.core")
    core.register = lambda obj=None, **_kwargs: obj if obj is not None else (lambda value: value)
    dist_utils = types.ModuleType("criterion_testpkg.misc.dist_utils")
    dist_utils.get_world_size = lambda: 1
    dist_utils.is_dist_available_and_initialized = lambda: False
    dfine_utils = types.ModuleType("criterion_testpkg.deim.dfine_utils")
    dfine_utils.bbox2distance = lambda *args, **kwargs: None
    box_ops = types.ModuleType("criterion_testpkg.deim.box_ops")

    def cxcywh_to_xyxy(boxes):
        center, size = boxes[..., :2], boxes[..., 2:]
        return torch.cat((center - size / 2, center + size / 2), dim=-1)

    def box_iou(boxes1, boxes2):
        lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        inter = (rb - lt).clamp_min(0).prod(-1)
        area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp_min(0).prod(-1)
        area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp_min(0).prod(-1)
        union = area1[:, None] + area2[None] - inter
        return inter / union.clamp_min(1e-6), union

    box_ops.box_cxcywh_to_xyxy = cxcywh_to_xyxy
    box_ops.box_iou = box_iou
    box_ops.generalized_box_iou = lambda a, b: box_iou(a, b)[0]
    modules = {
        root_package.__name__: root_package,
        deim_package.__name__: deim_package,
        misc_package.__name__: misc_package,
        core.__name__: core,
        dist_utils.__name__: dist_utils,
        dfine_utils.__name__: dfine_utils,
        box_ops.__name__: box_ops,
    }
    sys.modules.update(modules)
    sqmal_spec = importlib.util.spec_from_file_location(
        "criterion_testpkg.deim.sqmal", ROOT / "engine" / "deim" / "sqmal.py"
    )
    sqmal_module = importlib.util.module_from_spec(sqmal_spec)
    sys.modules[sqmal_spec.name] = sqmal_module
    assert sqmal_spec.loader is not None
    sqmal_spec.loader.exec_module(sqmal_module)
    criterion_spec = importlib.util.spec_from_file_location(
        "criterion_testpkg.deim.deim_criterion", ROOT / "engine" / "deim" / "deim_criterion.py"
    )
    criterion_module = importlib.util.module_from_spec(criterion_spec)
    sys.modules[criterion_spec.name] = criterion_module
    assert criterion_spec.loader is not None
    criterion_spec.loader.exec_module(criterion_module)
    return criterion_module.DEIMCriterion


DEIMCriterion = _load_criterion()


class SQMALLossTest(unittest.TestCase):
    def _criterion(self):
        return DEIMCriterion(
            matcher=None,
            weight_dict={
                "loss_sqmal": 1.0,
                "loss_quality": 1.0,
                "loss_defect": 1.0,
                "loss_hard_bg": 1.0,
            },
            losses=["sqmal", "quality", "defect", "hard_bg"],
            num_classes=2,
            use_sqmal=True,
            semantic_beta=0.5,
            quality_pos_weight=1.0,
            quality_neg_weight=0.25,
            use_hard_bg=True,
            hard_bg_topk=1,
            hard_bg_iou_threshold=0.1,
        )

    def test_all_new_losses_backward(self) -> None:
        criterion = self._criterion()
        pred_logits = torch.tensor([[[1.0, -1.0], [2.0, -2.0], [-1.0, 1.0]]], requires_grad=True)
        pred_quality = torch.tensor([[[0.2], [-0.5], [0.5]]], requires_grad=True)
        pred_boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.5], [0.05, 0.05, 0.05, 0.05], [0.9, 0.9, 0.05, 0.05]]], requires_grad=True)
        pred_defect = torch.zeros((1, 1, 8, 8), requires_grad=True)
        mask = torch.zeros((1, 64, 64), dtype=torch.bool)
        mask[:, 16:48, 16:48] = True
        targets = [{
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
            "masks": mask,
            "mask_valid": torch.tensor([True]),
            "orig_size": torch.tensor([64, 64]),
        }]
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        outputs = {
            "pred_logits": pred_logits,
            "pred_quality": pred_quality,
            "pred_boxes": pred_boxes,
            "pred_defect_logits": pred_defect,
        }
        criterion._clear_cache()
        losses = {}
        for name in criterion.losses:
            losses.update(criterion.get_loss(name, outputs, targets, indices, 1.0))
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertGreater(float(pred_logits.grad.abs().sum()), 0.0)
        self.assertGreater(float(pred_quality.grad.abs().sum()), 0.0)
        self.assertGreater(float(pred_defect.grad.abs().sum()), 0.0)

    def test_quality_without_positive_or_negative(self) -> None:
        criterion = self._criterion()
        outputs = {
            "pred_logits": torch.empty((1, 0, 2), requires_grad=True),
            "pred_quality": torch.empty((1, 0, 1), requires_grad=True),
            "pred_boxes": torch.empty((1, 0, 4), requires_grad=True),
        }
        targets = [{"labels": torch.empty(0, dtype=torch.long), "boxes": torch.empty((0, 4))}]
        indices = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
        criterion._clear_cache()
        loss = criterion.loss_quality(outputs, targets, indices, 1.0)["loss_quality"]
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()

