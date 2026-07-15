from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_postprocessor_class():
    package = types.ModuleType("sqmal_testpkg")
    package.__path__ = []
    deim_package = types.ModuleType("sqmal_testpkg.deim")
    deim_package.__path__ = []
    core = types.ModuleType("sqmal_testpkg.core")

    def register(obj=None, **_kwargs):
        return obj if obj is not None else (lambda value: value)

    core.register = register
    sys.modules[package.__name__] = package
    sys.modules[deim_package.__name__] = deim_package
    sys.modules[core.__name__] = core

    sqmal_spec = importlib.util.spec_from_file_location(
        "sqmal_testpkg.deim.sqmal", ROOT / "engine" / "deim" / "sqmal.py"
    )
    sqmal_module = importlib.util.module_from_spec(sqmal_spec)
    sys.modules[sqmal_spec.name] = sqmal_module
    assert sqmal_spec.loader is not None
    sqmal_spec.loader.exec_module(sqmal_module)

    post_spec = importlib.util.spec_from_file_location(
        "sqmal_testpkg.deim.postprocessor", ROOT / "engine" / "deim" / "postprocessor.py"
    )
    post_module = importlib.util.module_from_spec(post_spec)
    sys.modules[post_spec.name] = post_module
    assert post_spec.loader is not None
    post_spec.loader.exec_module(post_module)
    return post_module.PostProcessor


PostProcessor = _load_postprocessor_class()


class PostProcessorQualityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.boxes = torch.tensor([[[0.25, 0.5, 0.2, 0.2], [0.75, 0.5, 0.2, 0.2]]])
        self.sizes = torch.tensor([[100.0, 100.0]])
        self.quality = torch.tensor([[[-10.0], [10.0]]])

    def test_focal_rerank_before_flatten_topk(self) -> None:
        logits = torch.tensor([[[4.0, -4.0], [3.0, -4.0]]])
        outputs = {"pred_logits": logits, "pred_boxes": self.boxes, "pred_quality": self.quality}
        baseline = PostProcessor(num_classes=2, use_focal_loss=True, num_top_queries=1)
        reranked = PostProcessor(num_classes=2, use_focal_loss=True, num_top_queries=1, quality_rerank=True)
        base_box = baseline(outputs, self.sizes)[0]["boxes"][0]
        quality_box = reranked(outputs, self.sizes)[0]["boxes"][0]
        self.assertLess(float(base_box[0]), 50.0)
        self.assertGreater(float(quality_box[0]), 50.0)

        identity = PostProcessor(
            num_classes=2, use_focal_loss=True, num_top_queries=1,
            quality_rerank=True, quality_power=0.0,
        )
        self.assertTrue(torch.equal(identity(outputs, self.sizes)[0]["boxes"], baseline(outputs, self.sizes)[0]["boxes"]))

    def test_softmax_rerank(self) -> None:
        logits = torch.tensor([[[4.0, -4.0, -5.0], [3.0, -4.0, -5.0]]])
        outputs = {"pred_logits": logits, "pred_boxes": self.boxes, "pred_quality": self.quality}
        reranked = PostProcessor(num_classes=2, use_focal_loss=False, num_top_queries=1, quality_rerank=True)
        result = reranked(outputs, self.sizes)[0]
        # 基线 softmax 分支返回 normalized boxes；x=0.75 代表第二个 query。
        self.assertGreater(float(result["boxes"][0, 0]), 0.5)

    def test_missing_quality_is_backward_compatible(self) -> None:
        outputs = {"pred_logits": torch.tensor([[[4.0, -4.0], [3.0, -4.0]]]), "pred_boxes": self.boxes}
        baseline = PostProcessor(num_classes=2, use_focal_loss=True, num_top_queries=1)
        reranked = PostProcessor(num_classes=2, use_focal_loss=True, num_top_queries=1, quality_rerank=True)
        self.assertTrue(torch.equal(
            baseline(outputs, self.sizes)[0]["scores"], reranked(outputs, self.sizes)[0]["scores"]
        ))


if __name__ == "__main__":
    unittest.main()
