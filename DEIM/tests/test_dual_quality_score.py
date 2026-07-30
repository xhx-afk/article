from __future__ import annotations

import importlib.util
import sys
import types
import unittest

import torch

from tests.sqalign_test_utils import ROOT, load_alignment_module


alignment = load_alignment_module("dual_score_core")


def load_postprocessor():
    package = types.ModuleType("dual_score_pkg")
    package.__path__ = []
    deim_package = types.ModuleType("dual_score_pkg.deim")
    deim_package.__path__ = []
    core = types.ModuleType("dual_score_pkg.core")
    core.register = lambda obj=None, **_kwargs: obj if obj is not None else (lambda value: value)
    sys.modules[package.__name__] = package
    sys.modules[deim_package.__name__] = deim_package
    sys.modules[core.__name__] = core
    sys.modules["dual_score_pkg.deim.semantic_query_alignment"] = alignment
    spec = importlib.util.spec_from_file_location(
        "dual_score_pkg.deim.postprocessor", ROOT / "engine" / "deim" / "postprocessor.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.PostProcessor


PostProcessor = load_postprocessor()


class DualQualityScoreTest(unittest.TestCase):
    def test_each_quality_branch_changes_score_and_zero_power_is_identity(self) -> None:
        class_scores = torch.full((1, 2, 1), 0.8)
        loc_logits = torch.tensor([[[-4.0], [4.0]]])
        semantic = torch.tensor([[[0.1], [0.9]]])
        loc_scores = alignment.compose_dual_quality_scores(class_scores, loc_logits, None)
        sem_scores = alignment.compose_dual_quality_scores(class_scores, None, semantic)
        self.assertGreater(float(loc_scores[0, 1]), float(loc_scores[0, 0]))
        self.assertGreater(float(sem_scores[0, 1]), float(sem_scores[0, 0]))
        identity = alignment.compose_dual_quality_scores(
            class_scores, loc_logits, semantic, loc_quality_power=0, semantic_quality_power=0
        )
        torch.testing.assert_close(identity, class_scores)

    def test_postprocessor_fuses_before_flatten_topk(self) -> None:
        outputs = {
            "pred_logits": torch.tensor([[[4.0], [3.0]]]),
            "pred_boxes": torch.tensor([[[0.25, 0.5, 0.2, 0.2], [0.75, 0.5, 0.2, 0.2]]]),
            "pred_loc_quality": torch.tensor([[[-10.0], [10.0]]]),
            "pred_sem_quality": torch.tensor([[[0.01], [0.99]]]),
        }
        sizes = torch.tensor([[100.0, 100.0]])
        baseline = PostProcessor(num_classes=1, use_focal_loss=True, num_top_queries=1)
        fused = PostProcessor(
            num_classes=1,
            use_focal_loss=True,
            num_top_queries=1,
            loc_quality_rerank=True,
            semantic_quality_rerank=True,
        )
        self.assertLess(float(baseline(outputs, sizes)[0]["boxes"][0, 0]), 50.0)
        self.assertGreater(float(fused(outputs, sizes)[0]["boxes"][0, 0]), 50.0)


if __name__ == "__main__":
    unittest.main()

