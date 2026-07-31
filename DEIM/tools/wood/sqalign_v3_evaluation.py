"""Reusable single-JSON diagnostics for the SQ-Align V3 evaluator.

This module contains no model-loading side effects and is intentionally easy to
unit-test on synthetic predictions.  The CLI driver is ``evaluate_sqalign_v3``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


TOP_LEVEL_KEYS = (
    "meta", "checkpoint_load", "dataset", "coco", "tide",
    "fixed_thresholds", "error_taxonomy", "confusion", "recall_by_iou",
    "calibration", "defect_map", "query_mask", "semantic_quality",
    "candidate_ranking", "score_analysis", "runtime", "automatic_diagnosis",
)


def empty_report() -> Dict[str, object]:
    """Return the mandatory V3 result structure."""
    return {key: {} for key in TOP_LEVEL_KEYS}


def box_iou_xyxy(box: Sequence[float], boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float64)
    box_array = np.asarray(box, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64)
    intersection_min = np.maximum(box_array[:2], boxes[:, :2])
    intersection_max = np.minimum(box_array[2:], boxes[:, 2:])
    intersection = np.clip(intersection_max - intersection_min, 0.0, None).prod(1)
    box_area = np.clip(box_array[2:] - box_array[:2], 0.0, None).prod()
    boxes_area = np.clip(boxes[:, 2:] - boxes[:, :2], 0.0, None).prod(1)
    return intersection / np.maximum(box_area + boxes_area - intersection, 1e-12)


def classify_detections(
    predictions: Sequence[Mapping[str, object]],
    ground_truth: Sequence[Mapping[str, object]],
    score_threshold: float,
    iou_threshold: float = 0.50,
    far_background_iou: float = 0.10,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Assign the fixed-threshold V3 error taxonomy in descending score order."""
    gt_boxes = np.asarray([gt["box"] for gt in ground_truth], dtype=np.float64).reshape(-1, 4)
    gt_labels = np.asarray([int(gt["label"]) for gt in ground_truth], dtype=np.int64)
    used = np.zeros((len(ground_truth),), dtype=bool)
    rows: List[Dict[str, object]] = []
    counts: MutableMapping[str, int] = defaultdict(int)
    filtered = sorted(
        (prediction for prediction in predictions if float(prediction["score"]) >= score_threshold),
        key=lambda prediction: float(prediction["score"]),
        reverse=True,
    )
    for prediction in filtered:
        label = int(prediction["label"])
        ious = box_iou_xyxy(prediction["box"], gt_boxes)
        best_any = int(ious.argmax()) if ious.size else -1
        best_iou = float(ious[best_any]) if best_any >= 0 else 0.0
        same_class = np.flatnonzero(gt_labels == label)
        best_same = same_class[np.argmax(ious[same_class])] if same_class.size else -1
        best_same_iou = float(ious[best_same]) if best_same >= 0 else 0.0
        matched_gt = -1
        if best_same_iou >= iou_threshold and not used[best_same]:
            error_type = "TP"
            used[best_same] = True
            matched_gt = int(best_same)
        elif best_same_iou >= iou_threshold and used[best_same]:
            error_type = "duplicate_same_class_FP"
        elif best_iou >= iou_threshold:
            error_type = "class_confusion_FP"
        elif best_iou < far_background_iou:
            error_type = "far_background_FP"
        else:
            error_type = "localization_FP"
        counts[error_type] += 1
        row = dict(prediction)
        row.update({"error_type": error_type, "best_iou": best_iou, "matched_gt": matched_gt})
        rows.append(row)
    counts["FN"] = int((~used).sum())
    counts["predictions"] = len(filtered)
    counts["TP"] += 0
    for name in (
        "far_background_FP", "duplicate_same_class_FP",
        "class_confusion_FP", "localization_FP",
    ):
        counts[name] += 0
    return rows, dict(counts)


def precision_recall_f1(counts: Mapping[str, int]) -> Dict[str, float]:
    tp, fn = counts.get("TP", 0), counts.get("FN", 0)
    fp = sum(
        counts.get(key, 0) for key in (
            "far_background_FP", "duplicate_same_class_FP",
            "class_confusion_FP", "localization_FP",
        )
    )
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn),
        "precision": precision, "recall": recall,
        "F1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def distribution(values: Iterable[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0, "mean": 0.0, "std": 0.0, "p10": 0.0,
            "p25": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0,
            "ratio_gt_0.90": 0.0, "ratio_gt_0.95": 0.0, "ratio_lt_0.10": 0.0,
        }
    return {
        "count": int(array.size), "mean": float(array.mean()), "std": float(array.std()),
        "p10": float(np.percentile(array, 10)), "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)), "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "ratio_gt_0.90": float((array > 0.90).mean()),
        "ratio_gt_0.95": float((array > 0.95).mean()),
        "ratio_lt_0.10": float((array < 0.10).mean()),
    }


def calibration_metrics(rows: Sequence[Mapping[str, object]], bins: int = 15) -> Dict[str, object]:
    """Compute ECE, localization-aware ECE, Brier and reliability bins."""
    if not rows:
        return {"ECE": 0.0, "LaECE": 0.0, "Brier_score": 0.0, "bins": []}
    confidence = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    correct = np.asarray([row["error_type"] == "TP" for row in rows], dtype=np.float64)
    matched_iou = np.asarray(
        [float(row.get("best_iou", 0.0)) if row["error_type"] == "TP" else 0.0 for row in rows],
        dtype=np.float64,
    )
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    reliability = []
    ece = laece = 0.0
    for index in range(bins):
        selected = (confidence >= boundaries[index]) & (
            confidence <= boundaries[index + 1] if index == bins - 1 else confidence < boundaries[index + 1]
        )
        count = int(selected.sum())
        bin_confidence = float(confidence[selected].mean()) if count else 0.0
        bin_accuracy = float(correct[selected].mean()) if count else 0.0
        bin_iou = float(matched_iou[selected].mean()) if count else 0.0
        weight = count / confidence.size
        ece += weight * abs(bin_confidence - bin_accuracy)
        laece += weight * abs(bin_confidence - bin_iou)
        reliability.append({
            "lower": float(boundaries[index]), "upper": float(boundaries[index + 1]),
            "count": count, "mean_confidence": bin_confidence,
            "accuracy": bin_accuracy, "mean_matched_IoU": bin_iou,
        })
    return {
        "ECE": float(ece), "LaECE": float(laece),
        "Brier_score": float(np.square(confidence - correct).mean()),
        "mean_confidence": float(confidence.mean()), "accuracy": float(correct.mean()),
        "mean_matched_IoU": float(matched_iou[correct.astype(bool)].mean()) if correct.any() else 0.0,
        "bins": reliability,
    }


def binary_auc(positives: Iterable[float], negatives: Iterable[float]) -> Dict[str, object]:
    positive = np.asarray(list(positives), dtype=np.float64)
    negative = np.asarray(list(negatives), dtype=np.float64)
    if positive.size == 0 or negative.size == 0:
        return {"available": False, "reason": "both classes are required"}
    scores = np.concatenate((positive, negative))
    labels = np.concatenate((np.ones_like(positive), np.zeros_like(negative)))
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    true_positive = np.cumsum(labels)
    false_positive = np.cumsum(1.0 - labels)
    tpr = np.concatenate(([0.0], true_positive / positive.size, [1.0]))
    fpr = np.concatenate(([0.0], false_positive / negative.size, [1.0]))
    roc_auc = float(np.trapz(tpr, fpr))
    precision = true_positive / np.arange(1, labels.size + 1)
    recall = true_positive / positive.size
    pr_auc = float(np.sum((recall - np.concatenate(([0.0], recall[:-1]))) * precision))
    return {"available": True, "ROC_AUC": roc_auc, "PR_AUC": pr_auc}


def finite_json_value(value):
    """Recursively convert NumPy/Tensor-like scalars and forbid NaN/Inf."""
    if isinstance(value, Mapping):
        return {str(key): finite_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def automatic_diagnosis(report: Mapping[str, object]) -> Dict[str, List[str]]:
    """Centralized, deterministic V3 diagnosis rules."""
    warnings: List[str] = []
    summary: List[str] = []
    semantic = report.get("semantic_quality", {})
    groups = semantic.get("groups", {}) if isinstance(semantic, Mapping) else {}
    background = groups.get("far_background_FP", {}) if isinstance(groups, Mapping) else {}
    if isinstance(background, Mapping) and float(background.get("ratio_gt_0.95", 0.0)) > 0.30:
        warnings.append("semantic_quality_saturated")
    query_mask = report.get("query_mask", {})
    matched = query_mask.get("matched", {}) if isinstance(query_mask, Mapping) else {}
    if isinstance(matched, Mapping) and int(matched.get("count", 0)) and float(matched.get("Dice", 0.0)) < 0.50:
        warnings.append("query_mask_not_learning")
    candidate = report.get("candidate_ranking", {})
    if isinstance(candidate, Mapping) and float(candidate.get("ranking_violation_ratio", 0.0)) > 0.30:
        warnings.append("candidate_rank_violation_high")
    taxonomy = report.get("error_taxonomy", {})
    if isinstance(taxonomy, Mapping):
        near = int(taxonomy.get("duplicate_same_class_FP", 0)) + int(taxonomy.get("class_confusion_FP", 0)) + int(taxonomy.get("localization_FP", 0))
        far = int(taxonomy.get("far_background_FP", 0))
        if near > far:
            warnings.append("near_gt_fp_dominates")
    score_analysis = report.get("score_analysis", {})
    if isinstance(score_analysis, Mapping):
        class_only = score_analysis.get("class_score", {})
        gated = score_analysis.get("class_score_x_gate", {})
        if isinstance(class_only, Mapping) and isinstance(gated, Mapping):
            if gated.get("AP", 0.0) < class_only.get("AP", 0.0) and gated.get("background_FP", 0) < class_only.get("background_FP", 0):
                warnings.append("semantic_gate_reduces_fp_but_hurts_ap")
    summary.append("SQ-Align V3 diagnostics completed; compare warnings with the validation gate sweep.")
    return {"warnings": warnings, "summary": summary}

