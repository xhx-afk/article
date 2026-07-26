"""CPU-only unit tests for the AxMamba checkpoint evaluation helpers."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from engine.core.yaml_utils import load_config
from output.tools import evaluate_axmamba_checkpoints as evaluation


def test_iou_handles_overlap_disjoint_and_degenerate_boxes():
    assert evaluation._iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)
    assert evaluation._iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0
    assert evaluation._iou([0, 0, 0, 1], [0, 0, 1, 1]) == 0.0


def test_label_maps_support_contiguous_and_sparse_coco_ids():
    direct, inverse = evaluation._label_maps([0, 1], 2, None)
    assert direct == {0: 0, 1: 1}
    assert inverse == {0: 0, 1: 1}

    sparse, inverse_sparse = evaluation._label_maps([1, 3], 2, None)
    assert sparse == {0: 1, 1: 3}
    assert inverse_sparse == {1: 0, 3: 1}

    with pytest.raises(ValueError, match="缺少 COCO 类别"):
        evaluation._label_maps([1, 3], 2, {0: 1})


def test_calibration_bins_compute_ece_and_laece():
    rows = [
        {"score": 0.1, "is_tp": 0, "matched_iou": 0.0},
        {"score": 0.9, "is_tp": 1, "matched_iou": 0.8},
    ]
    summary, bins = evaluation._calibration_bins(rows, 2)
    assert summary["count"] == 2
    assert summary["ECE"] == pytest.approx(0.1)
    assert summary["LaECE"] == pytest.approx(0.1)
    assert [row["count"] for row in bins] == [1, 1]


def test_safe_spearman_reports_perfect_rank_and_constant_unavailable():
    result = evaluation._safe_spearman([1, 2, 3], [10, 20, 30])
    assert result["rho"] == pytest.approx(1.0)
    assert result["count"] == 3

    constant = evaluation._safe_spearman([1, 1, 1], [1, 2, 3])
    assert constant == {"rho": None, "p_value": None, "count": 3}


def test_distribution_filters_non_finite_values():
    result = evaluation._distribution([0.1, float("nan"), float("inf"), 0.9])
    assert result["count"] == 2
    assert result["mean"] == pytest.approx(0.5)
    assert result["median"] == pytest.approx(0.5)
    assert math.isfinite(result["std"])


def test_required_outputs_cover_user_requested_artifacts():
    required = set(evaluation.REQUIRED_OUTPUTS)
    expected = {
        "class_and_all_metrics.json",
        "class_and_all_metrics.png",
        "all_metrics_summary.json",
        "per_class_ap.json",
        "per_class_ap.png",
        "tide_summary.txt",
        "tide_summary.png",
        "tide_metrics.json",
        "confusion_matrix.png",
        "confusion_matrix_with_fp_fn.png",
        "ece_summary.json",
        "ece_summary.png",
        "reliability_bins.csv",
        "recall_by_iou_threshold.json",
        "recall_by_iou_threshold.png",
        "quality_diagnostics.json",
        "quality_diagnostics.png",
    }
    assert expected <= required


def test_parse_tide_summary_returns_structured_objective_metrics():
    text = """
-- predictions --

bbox AP @ [50-95]: 50.02
  Thresh       50       55       60
    AP      80.52    78.55    75.85

                         Main Errors
  Type      Cls      Loc     Both     Dupe      Bkg     Miss
   dAP     5.28     2.55     0.29     0.24     3.69     0.47

        Special Error
  Type   FalsePos   FalseNeg
   dAP      17.06       1.65
"""
    result = evaluation._parse_tide_summary(text)
    assert result["value_unit"] == "percentage_points"
    assert result["bbox_AP_50_95"] == pytest.approx(50.02)
    assert result["AP_by_IoU_threshold"] == [
        {"iou_threshold": 0.5, "AP": 80.52},
        {"iou_threshold": 0.55, "AP": 78.55},
        {"iou_threshold": 0.6, "AP": 75.85},
    ]
    assert result["main_errors_dAP"] == {
        "Cls": 5.28,
        "Loc": 2.55,
        "Both": 0.29,
        "Dupe": 0.24,
        "Bkg": 3.69,
        "Miss": 0.47,
    }
    assert result["special_errors_dAP"] == {
        "FalsePos": 17.06,
        "FalseNeg": 1.65,
    }


class _FakeLQE(nn.Module):
    def __init__(self):
        super().__init__()
        self.reg_conf = nn.Linear(4, 1)


class _FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lqe_layers = nn.ModuleList([_FakeLQE()])


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _FakeDecoder()


def test_lqe_hook_captures_independent_quality_logit():
    model = _FakeModel()
    state, handles = evaluation._install_lqe_quality_hooks(model)
    assert state["candidates"] == ["decoder.lqe_layers.0.reg_conf"]

    output = model.decoder.lqe_layers[0].reg_conf(torch.randn(2, 3, 4))
    assert state["source"] == "decoder.lqe_layers.0.reg_conf"
    assert torch.equal(state["tensor"], output.detach())
    for handle in handles:
        handle.remove()


def test_yaml_loader_does_not_leak_between_top_level_loads(tmp_path):
    first = tmp_path / "first.yml"
    second = tmp_path / "second.yml"
    first.write_text("only_first: 1\n", encoding="utf-8")
    second.write_text("only_second: 2\n", encoding="utf-8")

    assert load_config(first) == {"only_first": 1}
    assert load_config(second) == {"only_second": 2}
