"""Unified post-training evaluator for SQ-Align V3.

All scalar diagnostics are written to exactly one user-selected JSON.  Optional
TIDE uses a temporary result file which is deleted before the command exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.wood.optional_dependency_stubs import install as install_optional_dependency_stubs  # noqa: E402


install_optional_dependency_stubs()

from engine.core import YAMLConfig  # noqa: E402
from engine.deim.continuous_semantic_alignment import (  # noqa: E402
    build_best_gt_assignment,
    build_per_gt_candidate_groups,
    build_semantic_quality_targets,
    build_union_defect_target,
    pool_gt_instance_roi_masks,
    semantic_suppression_gate,
)
from tools.wood.sqalign_v3_evaluation import (  # noqa: E402
    TOP_LEVEL_KEYS,
    automatic_diagnosis,
    binary_auc,
    box_iou_xyxy,
    calibration_metrics,
    classify_detections,
    distribution,
    empty_report,
    finite_json_value,
    precision_recall_f1,
)
from tools.wood.sqalign_v3_target_utils import (  # noqa: E402
    AlignmentSanityAccumulator,
    build_alignment_targets,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(arguments: Sequence[str], default: object) -> object:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=ROOT, capture_output=True, text=True,
            check=False, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else default
    except (OSError, subprocess.SubprocessError):
        return default


def load_checkpoint(model: torch.nn.Module, path: Path) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema", {}).get("module", checkpoint.get("model", checkpoint))
    current = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    shape_mismatch = sorted(
        key for key, value in state.items()
        if key in current and current[key].shape != value.shape
    )
    result = model.load_state_dict(compatible, strict=False)
    return {
        "source_keys": len(state), "matched_keys": len(compatible),
        "missing_keys": sorted(result.missing_keys),
        "unexpected_keys": sorted(key for key in state if key not in current),
        "shape_mismatch_keys": shape_mismatch,
        "strict": False,
    }


def xywh_to_xyxy(box: Sequence[float]) -> List[float]:
    x, y, width, height = map(float, box)
    return [x, y, x + width, y + height]


def xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)]


def raw_dataset(annotation: Mapping[str, object]):
    images = {int(item["id"]): item for item in annotation.get("images", [])}
    categories = sorted(annotation.get("categories", []), key=lambda item: int(item["id"]))
    names = {int(item["id"]): str(item.get("name", item["id"])) for item in categories}
    # Keep annotation-free images in the taxonomy.  Omitting them silently
    # drops exactly the far-background false positives this evaluator is
    # intended to measure.
    ground_truth: Dict[int, List[Dict[str, object]]] = {
        image_id: [] for image_id in images
    }
    per_class = defaultdict(int)
    valid_masks = 0
    for item in annotation.get("annotations", []):
        if int(item.get("iscrowd", 0)):
            continue
        label = int(item["category_id"])
        valid = bool(item.get("mask_valid", bool(item.get("segmentation"))))
        valid_masks += int(valid)
        ground_truth[int(item["image_id"])].append({
            "box": xywh_to_xyxy(item["bbox"]), "label": label,
            "annotation_id": int(item.get("id", -1)), "mask_valid": valid,
        })
        per_class[label] += 1
    return images, names, ground_truth, per_class, valid_masks


def model_predictions(
    outputs: Mapping[str, torch.Tensor],
    image_ids: Sequence[int],
    image_sizes: Sequence[Tuple[int, int]],
    max_dets: int,
    gate_lambda: float,
    gate_gamma: float,
    gate_enabled: bool = False,
    label_to_category: Mapping[int, int] | None = None,
) -> Tuple[Dict[int, List[Dict[str, object]]], Dict[int, List[Dict[str, object]]]]:
    class_scores = outputs["pred_logits"].float().sigmoid()
    quality = outputs.get("pred_sem_quality")
    comparison_gate = (
        semantic_suppression_gate(quality, gate_lambda, gate_gamma).to(class_scores.dtype)
        if quality is not None else torch.ones_like(class_scores[..., :1])
    )
    resolved_gate = comparison_gate if gate_enabled else torch.ones_like(comparison_gate)
    gated_scores = class_scores * comparison_gate
    boxes = torchvision.ops.box_convert(outputs["pred_boxes"].float(), "cxcywh", "xyxy")
    semantic_probability = quality.float().sigmoid().squeeze(-1) if quality is not None else None
    class_rows: Dict[int, List[Dict[str, object]]] = {}
    gated_rows: Dict[int, List[Dict[str, object]]] = {}
    for batch_index, (image_id, (width, height)) in enumerate(zip(image_ids, image_sizes)):
        scaled_boxes = boxes[batch_index] * boxes.new_tensor([width, height, width, height])
        for score_name, score_tensor, destination in (
            ("class_score", class_scores, class_rows),
            ("class_score_x_gate", gated_scores, gated_rows),
        ):
            flat = score_tensor[batch_index].flatten()
            count = min(int(max_dets), flat.numel())
            values, flat_indices = flat.topk(count)
            query_indices = torch.div(flat_indices, score_tensor.shape[-1], rounding_mode="floor")
            labels = flat_indices % score_tensor.shape[-1]
            rows = []
            for score, query_index, label in zip(values, query_indices, labels):
                query = int(query_index)
                model_label = int(label)
                category_label = (
                    int(label_to_category.get(model_label, model_label))
                    if label_to_category is not None else model_label
                )
                rows.append({
                    "image_id": int(image_id), "query_index": query,
                    "label": category_label, "model_label": model_label,
                    "score": float(score),
                    "class_score": float(class_scores[batch_index, query, label]),
                    "semantic_quality": (
                        float(semantic_probability[batch_index, query])
                        if semantic_probability is not None else None
                    ),
                    "gate": float(comparison_gate[batch_index, query, 0]),
                    "resolved_gate": float(resolved_gate[batch_index, query, 0]),
                    "box": scaled_boxes[query].detach().cpu().tolist(),
                    "score_type": score_name,
                })
            destination[int(image_id)] = rows
    return class_rows, gated_rows


def resolve_semantic_gate(args, yaml_cfg: Mapping[str, object]) -> Tuple[bool, float, float, str]:
    """Resolve the tri-state gate with CLI > config > safe default precedence."""
    post = yaml_cfg.get("PostProcessor", {})
    post = post if isinstance(post, Mapping) else {}
    if args.semantic_gate_enabled is not None:
        enabled = bool(args.semantic_gate_enabled)
        source = "command_line"
    elif "semantic_gate_enabled" in post:
        enabled = bool(post["semantic_gate_enabled"])
        source = "config"
    else:
        enabled = False
        source = "default"
    gate_lambda = (
        float(args.semantic_gate_lambda)
        if args.semantic_gate_lambda is not None
        else float(post.get("semantic_gate_lambda", 0.10))
    )
    gate_gamma = (
        float(args.semantic_gate_gamma)
        if args.semantic_gate_gamma is not None
        else float(post.get("semantic_gate_gamma", 2.0))
    )
    if not 0.0 <= gate_lambda <= 1.0:
        raise ValueError("semantic gate lambda must be in [0,1]")
    if gate_gamma <= 0.0:
        raise ValueError("semantic gate gamma must be positive")
    return enabled, gate_lambda, gate_gamma, source


def coco_detections(rows_by_image: Mapping[int, Sequence[Mapping[str, object]]]) -> List[Dict[str, object]]:
    return [
        {
            "image_id": int(row["image_id"]), "category_id": int(row["label"]),
            "bbox": xyxy_to_xywh(row["box"]), "score": float(row["score"]),
        }
        for rows in rows_by_image.values() for row in rows
    ]


def _mean_valid(array: np.ndarray) -> float:
    array = np.asarray(array)
    valid = array[array > -1]
    return float(valid.mean()) if valid.size else 0.0


def coco_metrics(
    annotation_path: Path,
    detections: Sequence[Mapping[str, object]],
    max_dets: int,
    image_ids: Sequence[int] | None = None,
):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ground_truth = COCO(str(annotation_path))
    if detections:
        detected = ground_truth.loadRes(list(detections))
    else:
        detected = COCO()
        detected.dataset = {
            "images": ground_truth.dataset.get("images", []),
            "categories": ground_truth.dataset.get("categories", []), "annotations": [],
        }
        detected.createIndex()
    evaluator = COCOeval(ground_truth, detected, "bbox")
    # COCO's published AR metric is AR100.  ``max_dets`` controls how many
    # model detections are supplied, while official accumulation remains at
    # [1, 10, 100].
    evaluator.params.maxDets = [1, 10, 100]
    if image_ids is not None:
        evaluator.params.imgIds = sorted(int(image_id) for image_id in image_ids)
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]  # T,R,K,A,M
    recall = evaluator.eval["recall"]  # T,K,A,M
    thresholds = evaluator.params.iouThrs
    output: Dict[str, object] = {"AP@[0.50:0.95]": _mean_valid(precision[:, :, :, 0, -1])}
    for threshold in thresholds:
        index = int(np.argmin(np.abs(thresholds - threshold)))
        output[f"AP{int(round(threshold * 100))}"] = _mean_valid(precision[index, :, :, 0, -1])
    output.update({
        "APS": _mean_valid(precision[:, :, :, 1, -1]),
        "APM": _mean_valid(precision[:, :, :, 2, -1]),
        "APL": _mean_valid(precision[:, :, :, 3, -1]),
        "AR1": _mean_valid(recall[:, :, 0, 0]),
        "AR10": _mean_valid(recall[:, :, 0, 1]),
        "AR100": _mean_valid(recall[:, :, 0, 2]),
        "ARS": _mean_valid(recall[:, :, 1, 2]),
        "ARM": _mean_valid(recall[:, :, 2, 2]),
        "ARL": _mean_valid(recall[:, :, 3, 2]),
    })
    per_class = {}
    category_ids = evaluator.params.catIds
    category_names = {item["id"]: item.get("name", item["id"]) for item in ground_truth.dataset["categories"]}
    annotation_counts = defaultdict(int)
    for item in ground_truth.dataset["annotations"]:
        if not item.get("iscrowd", 0):
            annotation_counts[item["category_id"]] += 1
    for class_index, category_id in enumerate(category_ids):
        class_precision = precision[:, :, class_index, 0, -1]
        values = {"AP": _mean_valid(class_precision), "GT_count": annotation_counts[category_id]}
        for threshold in (0.50, 0.75, 0.80, 0.90):
            threshold_index = int(np.argmin(np.abs(thresholds - threshold)))
            values[f"AP{int(threshold * 100)}"] = _mean_valid(class_precision[threshold_index])
        per_class[str(category_names.get(category_id, category_id))] = values
    output["per_class"] = per_class
    return output


def build_tide_bbox_ground_truth(annotation: Mapping[str, object]):
    """Build a bbox-only TIDE Data object without touching segmentation."""
    from tidecv.data import Data

    data = Data("wood_gt_bbox", max_dets=100)
    for category in annotation.get("categories", []):
        data.add_class(int(category["id"]), str(category.get("name", category["id"])))
    for image in annotation.get("images", []):
        data.add_image(int(image["id"]), str(image.get("file_name", image["id"])))
    for item in annotation.get("annotations", []):
        image_id = int(item["image_id"])
        class_id = int(item["category_id"])
        box = [float(value) for value in item["bbox"]]
        if int(item.get("iscrowd", 0)):
            data.add_ignore_region(image_id, class_id, box=box)
        else:
            data.add_ground_truth(image_id, class_id, box=box)
    return data


def build_tide_bbox_predictions(detections: Sequence[Mapping[str, object]]):
    """Build bbox-only TIDE predictions."""
    from tidecv.data import Data

    data = Data("wood_predictions", max_dets=100)
    for detection in detections:
        data.add_detection(
            int(detection["image_id"]),
            int(detection["category_id"]),
            float(detection["score"]),
            box=[float(value) for value in detection["bbox"]],
        )
    return data


def tide_metrics(annotation_path: Path, detections: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    try:
        from tidecv import TIDE
    except ImportError:
        return {"available": False, "reason": "tidecv is not installed"}
    try:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        gt_data = build_tide_bbox_ground_truth(annotation)
        prediction_data = build_tide_bbox_predictions(detections)
        previous_tide_path = os.environ.get("TIDE_PATH")
        with tempfile.TemporaryDirectory(prefix="sqalign_tide_") as tide_directory:
            os.environ["TIDE_PATH"] = tide_directory
            try:
                tide = TIDE()
                run_name = "sqalign_v3_bbox"
                tide.evaluate(gt_data, prediction_data, mode=TIDE.BOX, name=run_name)
                errors = tide.get_all_errors()
            finally:
                if previous_tide_path is None:
                    os.environ.pop("TIDE_PATH", None)
                else:
                    os.environ["TIDE_PATH"] = previous_tide_path
        main = errors.get("main", {}).get(run_name, {}) if isinstance(errors, Mapping) else {}
        special = errors.get("special", {}).get(run_name, {}) if isinstance(errors, Mapping) else {}

        def numeric(values, name):
            value = values.get(name) if isinstance(values, Mapping) else None
            return float(value) if isinstance(value, (int, float, np.number)) else None

        return {
            "available": True,
            "run_name": run_name,
            "Cls_dAP": numeric(main, "Cls"),
            "Loc_dAP": numeric(main, "Loc"),
            "Both_dAP": numeric(main, "Both"),
            "Dupe_dAP": numeric(main, "Dupe"),
            "Bkg_dAP": numeric(main, "Bkg"),
            "Miss_dAP": numeric(main, "Miss"),
            "FalsePos_dAP": numeric(special, "FalsePos"),
            "FalseNeg_dAP": numeric(special, "FalseNeg"),
        }
    except Exception as error:  # optional dependency/API versions must not abort evaluation
        return {"available": False, "reason": f"tidecv evaluation failed: {error}"}


def taxonomy_report(
    rows_by_image: Mapping[int, Sequence[Mapping[str, object]]],
    ground_truth: Mapping[int, Sequence[Mapping[str, object]]],
    threshold: float,
):
    all_rows: List[Dict[str, object]] = []
    totals = defaultdict(int)
    per_class = defaultdict(lambda: defaultdict(int))
    image_ids = sorted(set(ground_truth) | set(rows_by_image))
    for image_id in image_ids:
        rows, counts = classify_detections(
            rows_by_image.get(image_id, []), ground_truth.get(image_id, []), threshold
        )
        all_rows.extend(rows)
        for key, value in counts.items():
            totals[key] += value
        for row in rows:
            per_class[int(row["label"])][str(row["error_type"])] += 1
        matched_gt = {int(row["matched_gt"]) for row in rows if int(row["matched_gt"]) >= 0}
        for gt_index, gt in enumerate(ground_truth.get(image_id, [])):
            if gt_index not in matched_gt:
                per_class[int(gt["label"])]["FN"] += 1
    summary = {**precision_recall_f1(totals), **dict(totals)}
    summary["background_FP"] = int(totals.get("far_background_FP", 0))
    summary["near_GT_FP"] = int(
        totals.get("duplicate_same_class_FP", 0)
        + totals.get("class_confusion_FP", 0)
        + totals.get("localization_FP", 0)
    )
    summary["per_class"] = {str(label): dict(values) for label, values in per_class.items()}
    return all_rows, summary


def confusion_report(
    classified: Sequence[Mapping[str, object]],
    ground_truth: Mapping[int, Sequence[Mapping[str, object]]],
    class_ids: Sequence[int],
) -> Dict[str, object]:
    labels = [str(label) for label in class_ids] + ["background"]
    index = {label: position for position, label in enumerate(class_ids)}
    background = len(class_ids)
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)  # GT rows, pred cols
    matched = defaultdict(set)
    for row in classified:
        image_id, prediction = int(row["image_id"]), int(row["label"])
        gt_index = int(row.get("matched_gt", -1))
        if row["error_type"] == "TP" and gt_index >= 0:
            gt_label = int(ground_truth[image_id][gt_index]["label"])
            matrix[index[gt_label], index.get(prediction, background)] += 1
            matched[image_id].add(gt_index)
        elif row["error_type"] == "class_confusion_FP":
            boxes = np.asarray([gt["box"] for gt in ground_truth[image_id]])
            best = int(box_iou_xyxy(row["box"], boxes).argmax())
            gt_label = int(ground_truth[image_id][best]["label"])
            matrix[index[gt_label], index.get(prediction, background)] += 1
        else:
            matrix[background, index.get(prediction, background)] += 1
    for image_id, items in ground_truth.items():
        for gt_index, gt in enumerate(items):
            if gt_index not in matched[image_id]:
                matrix[index[int(gt["label"])], background] += 1
    row_sum = matrix.sum(1, keepdims=True)
    column_sum = matrix.sum(0, keepdims=True)
    return {
        "labels": labels, "count_matrix": matrix.tolist(),
        "row_normalized_matrix": (matrix / np.maximum(row_sum, 1)).tolist(),
        "column_normalized_matrix": (matrix / np.maximum(column_sum, 1)).tolist(),
    }


def recall_report(rows_by_image, ground_truth, thresholds: Sequence[float]):
    output = {}
    class_ids = sorted({int(gt["label"]) for items in ground_truth.values() for gt in items})
    for threshold in thresholds:
        found = defaultdict(int)
        total = defaultdict(int)
        for image_id, items in ground_truth.items():
            predictions = rows_by_image.get(image_id, [])
            for gt in items:
                label = int(gt["label"])
                total[label] += 1
                same = [row for row in predictions if int(row["label"]) == label]
                best = max((float(box_iou_xyxy(row["box"], np.asarray([gt["box"]]))[0]) for row in same), default=0.0)
                found[label] += int(best >= threshold)
        output[f"{threshold:.2f}"] = {
            "overall": sum(found.values()) / max(sum(total.values()), 1),
            "per_class": {str(label): found[label] / max(total[label], 1) for label in class_ids},
        }
    return output


def tie_aware_spearman(values: torch.Tensor, targets: torch.Tensor) -> float | None:
    """Compute Spearman correlation with average ranks for tied values."""
    values_np = values.detach().float().cpu().numpy().reshape(-1)
    targets_np = targets.detach().float().cpu().numpy().reshape(-1)
    valid = np.isfinite(values_np) & np.isfinite(targets_np)
    values_np, targets_np = values_np[valid], targets_np[valid]
    if values_np.size < 3:
        return None
    if np.all(values_np == values_np[0]) or np.all(targets_np == targets_np[0]):
        return None
    try:
        from scipy.stats import spearmanr

        result = spearmanr(values_np, targets_np)
        statistic = float(result.statistic)
    except ImportError:
        # Correct average-rank fallback; never use argsort().argsort(), which
        # assigns arbitrary distinct ranks to tied semantic targets.
        def average_rank(array: np.ndarray) -> np.ndarray:
            order = np.argsort(array, kind="mergesort")
            sorted_values = array[order]
            ranks = np.empty(array.size, dtype=np.float64)
            start = 0
            while start < array.size:
                end = start + 1
                while end < array.size and sorted_values[end] == sorted_values[start]:
                    end += 1
                ranks[order[start:end]] = (start + end - 1) / 2.0
                start = end
            return ranks

        statistic = float(np.corrcoef(average_rank(values_np), average_rank(targets_np))[0, 1])
    return statistic if np.isfinite(statistic) else None


class AlignmentAccumulator:
    _PROBABILITY_BINS = 1001

    def __init__(
        self,
        gate_lambda: float = 0.10,
        gate_gamma: float = 2.0,
        gate_enabled: bool = False,
    ) -> None:
        self.defect = defaultdict(float)
        self.defect_class = defaultdict(lambda: [0.0, 0.0])
        self.mask = defaultdict(lambda: defaultdict(float))
        self.mask_histogram = defaultdict(
            lambda: np.zeros(self._PROBABILITY_BINS, dtype=np.int64)
        )
        self.mask_class = defaultdict(lambda: defaultdict(float))
        self.mask_seen = False
        self.correlations = defaultdict(self._new_correlation_state)
        self.candidate = self._new_candidate_state()
        self.candidate_class = defaultdict(self._new_candidate_state)
        self.gate_lambda = float(gate_lambda)
        self.gate_gamma = float(gate_gamma)
        self.gate_enabled = bool(gate_enabled)

    @staticmethod
    def _new_correlation_state():
        return {
            "image_count": 0,
            "query_count": 0,
            "constant_target_image_count": 0,
            "per_image": [],
            "values": [],
            "targets": [],
        }

    @staticmethod
    def _new_candidate_state():
        return {
            "num_GT": 0,
            "GT_with_any_candidate": 0,
            "GT_with_at_least_2_candidates": 0,
            "GT_with_anchor": 0,
            "GT_without_anchor": 0,
            "candidate_counts": [],
            "candidate_ious": [],
            "anchor_scores": [],
            "hard_scores": [],
            "margins": [],
            "ranking_pair_count": 0,
            "ranking_violation_count": 0,
            "highest_score_is_highest_iou": [],
        }

    @staticmethod
    def _boundary(binary: torch.Tensor) -> torch.Tensor:
        expanded = binary.float()
        dilation = F.max_pool2d(expanded, 3, 1, 1)
        erosion = -F.max_pool2d(-expanded, 3, 1, 1)
        return (dilation - erosion) > 0

    def update_defect(self, outputs, targets) -> None:
        logits = outputs.get("pred_defect_logits")
        if logits is None:
            return
        low_target = build_union_defect_target(targets, logits.shape[-2:], logits.device, logits.dtype)
        low_probability = logits.float().sigmoid()
        low_prediction = low_probability >= 0.5
        low_truth = low_target >= 0.5
        self._binary_counts("low", low_prediction, low_truth)
        for batch_index, target in enumerate(targets):
            masks = target.get("masks")
            valid = target.get("mask_valid")
            if masks is None:
                continue
            valid = valid.bool() if valid is not None else torch.ones(masks.shape[0], device=masks.device, dtype=torch.bool)
            union = masks[valid].bool().any(0, keepdim=True).unsqueeze(0) if valid.any() else torch.zeros((1, 1, *masks.shape[-2:]), device=masks.device, dtype=torch.bool)
            probability = F.interpolate(low_probability[batch_index:batch_index + 1], size=masks.shape[-2:], mode="bilinear", align_corners=False)
            prediction = probability >= 0.5
            self._binary_counts("eval_input", prediction, union)
            pred_boundary, true_boundary = self._boundary(prediction), self._boundary(union)
            tolerance_pred = F.max_pool2d(pred_boundary.float(), 3, 1, 1) > 0
            tolerance_true = F.max_pool2d(true_boundary.float(), 3, 1, 1) > 0
            self.defect["boundary_matched_pred"] += float((pred_boundary & tolerance_true).sum())
            self.defect["boundary_pred"] += float(pred_boundary.sum())
            self.defect["boundary_matched_true"] += float((true_boundary & tolerance_pred).sum())
            self.defect["boundary_true"] += float(true_boundary.sum())
            self.defect["foreground_saturated"] += float(((probability > 0.95) & union).sum())
            self.defect["foreground_pixels"] += float(union.sum())
            self.defect["background_probability"] += float((probability * (~union)).sum())
            self.defect["background_pixels"] += float((~union).sum())
            labels = target.get("labels", torch.zeros(masks.shape[0], device=masks.device, dtype=torch.long))
            for instance_mask, is_valid, label in zip(masks, valid, labels):
                if is_valid and instance_mask.any():
                    self.defect_class[int(label)][0] += float((prediction[0, 0] & instance_mask.bool()).sum())
                    self.defect_class[int(label)][1] += float(instance_mask.sum())

    def _binary_counts(self, prefix, prediction, target):
        self.defect[f"{prefix}_intersection"] += float((prediction & target).sum())
        self.defect[f"{prefix}_prediction"] += float(prediction.sum())
        self.defect[f"{prefix}_target"] += float(target.sum())

    def _record_correlation(
        self,
        name: str,
        values: torch.Tensor,
        targets: torch.Tensor,
        selected: torch.Tensor,
    ) -> None:
        state = self.correlations[name]
        for batch_index in range(values.shape[0]):
            keep = selected[batch_index] & torch.isfinite(values[batch_index]) & torch.isfinite(targets[batch_index])
            selected_values = values[batch_index, keep]
            selected_targets = targets[batch_index, keep]
            count = int(selected_values.numel())
            if not count:
                continue
            state["image_count"] += 1
            state["query_count"] += count
            if torch.all(selected_targets == selected_targets[0]):
                state["constant_target_image_count"] += 1
            value = tie_aware_spearman(selected_values, selected_targets)
            if value is not None:
                state["per_image"].append(value)
            state["values"].append(selected_values.detach().cpu())
            state["targets"].append(selected_targets.detach().cpu())

    def _record_positive_mask_group(
        self,
        name: str,
        selected: torch.Tensor,
        probability: torch.Tensor,
        prediction: torch.Tensor,
        truth: torch.Tensor,
        bce: torch.Tensor,
    ) -> None:
        state = self.mask[name]
        pred_sum = prediction.sum((-2, -1))
        true_sum = truth.sum((-2, -1))
        nonempty_target = selected & (true_sum > 0)
        empty_target = selected & ~nonempty_target
        nonempty_prediction = selected & (pred_sum > 0)
        state["count"] += int(selected.sum())
        state["nonempty_target_count"] += int(nonempty_target.sum())
        state["empty_target_count"] += int(empty_target.sum())
        state["nonempty_prediction_count"] += int(nonempty_prediction.sum())
        state["empty_prediction_count"] += int((selected & ~nonempty_prediction).sum())
        state["empty_empty_count"] += int((empty_target & ~nonempty_prediction).sum())

        if nonempty_target.any():
            tp = (prediction & truth).sum((-2, -1)).float()
            fp = (prediction & ~truth).sum((-2, -1)).float()
            fn = (~prediction & truth).sum((-2, -1)).float()
            denominator_dice = 2 * tp + fp + fn
            denominator_iou = tp + fp + fn
            state["micro_TP_pixels"] += float(tp[nonempty_target].sum())
            state["micro_FP_pixels"] += float(fp[nonempty_target].sum())
            state["micro_FN_pixels"] += float(fn[nonempty_target].sum())
            state["macro_Dice_sum"] += float((2 * tp[nonempty_target] / denominator_dice[nonempty_target]).sum())
            state["macro_IoU_sum"] += float((tp[nonempty_target] / denominator_iou[nonempty_target]).sum())
            state["BCE_sum"] += float(bce[nonempty_target].sum())
            state["nonempty_target_empty_prediction_count"] += int(
                (nonempty_target & ~nonempty_prediction).sum()
            )

        if empty_target.any():
            state["empty_target_positive_pixels"] += float(pred_sum[empty_target].sum())
            state["empty_target_total_pixels"] += int(
                empty_target.sum()
            ) * int(prediction.shape[-2] * prediction.shape[-1])

    def _record_far_mask_group(
        self,
        selected: torch.Tensor,
        probability: torch.Tensor,
        prediction: torch.Tensor,
    ) -> None:
        if not selected.any():
            return
        state = self.mask["far_background"]
        values = probability[selected].detach().float()
        pred_sum = prediction[selected].sum((-2, -1))
        state["count"] += int(selected.sum())
        state["predicted_positive_pixel_count"] += float(pred_sum.sum())
        state["total_pixel_count"] += int(values.numel())
        state["probability_sum"] += float(values.sum())
        state["empty_prediction_count"] += int((pred_sum == 0).sum())
        histogram = torch.histc(
            values, bins=self._PROBABILITY_BINS, min=0.0, max=1.0
        ).to(torch.int64).cpu().numpy()
        self.mask_histogram["far_background"] += histogram

    def update_query(self, outputs, targets, indices) -> None:
        quality_logits = outputs.get("pred_sem_quality")
        mask_logits = outputs.get("pred_query_mask_logits")
        best_iou, best_gt = build_best_gt_assignment(outputs["pred_boxes"], targets, indices)
        quality_target, valid, matched, near, far = build_semantic_quality_targets(
            outputs["pred_boxes"], targets, best_gt, best_iou, indices
        )
        if quality_logits is not None:
            quality = quality_logits.float().sigmoid().squeeze(-1)
            associated = valid & (matched | near)
            self._record_correlation("matched", quality, quality_target, valid & matched)
            self._record_correlation("near_unmatched", quality, quality_target, valid & near)
            self._record_correlation(
                "associated_nonzero_target",
                quality,
                quality_target,
                associated & (quality_target > 1e-6),
            )
            self._record_correlation(
                "all_valid_including_far", quality, quality_target, valid
            )
            class_score = outputs["pred_logits"].float().sigmoid().amax(-1)
            for topk in (10, 30):
                top_selected = torch.zeros_like(valid)
                for batch_index in range(quality.shape[0]):
                    candidates = torch.nonzero(associated[batch_index], as_tuple=False).flatten()
                    if candidates.numel():
                        chosen = candidates[
                            class_score[batch_index, candidates]
                            .topk(min(topk, candidates.numel()))
                            .indices
                        ]
                        top_selected[batch_index, chosen] = True
                self._record_correlation(
                    f"top{topk}_associated_per_image",
                    quality,
                    quality_target,
                    top_selected,
                )
        if mask_logits is not None:
            self.mask_seen = True
            roi_target, roi_valid = pool_gt_instance_roi_masks(
                targets, outputs["pred_boxes"], best_gt, best_iou,
                matched_mask=matched, output_size=mask_logits.shape[-1],
            )
            probability = mask_logits.float().sigmoid()
            prediction = probability >= 0.5
            truth = roi_target >= 0.5
            bce = F.binary_cross_entropy_with_logits(mask_logits.float(), roi_target, reduction="none").mean((-2, -1))
            self._record_positive_mask_group(
                "matched", matched & roi_valid, probability, prediction, truth, bce
            )
            self._record_positive_mask_group(
                "near_unmatched", near & roi_valid, probability, prediction, truth, bce
            )
            self._record_far_mask_group(far & roi_valid, probability, prediction)
            true_sum = truth.sum((-2, -1))
            tp = (prediction & truth).sum((-2, -1)).float()
            fp = (prediction & ~truth).sum((-2, -1)).float()
            fn = (~prediction & truth).sum((-2, -1)).float()
            for batch_index, (source, gt_index) in enumerate(indices):
                source, gt_index = source.to(mask_logits.device), gt_index.to(mask_logits.device)
                if source.numel():
                    keep = roi_valid[batch_index, source] & (true_sum[batch_index, source] > 0)
                    source, gt_index = source[keep], gt_index[keep]
                    labels = targets[batch_index]["labels"][gt_index]
                    for query_index, label in zip(source, labels):
                        state = self.mask_class[int(label)]
                        query_tp = float(tp[batch_index, query_index])
                        query_fp = float(fp[batch_index, query_index])
                        query_fn = float(fn[batch_index, query_index])
                        state["count"] += 1
                        state["TP"] += query_tp
                        state["FP"] += query_fp
                        state["FN"] += query_fn
                        state["macro_Dice_sum"] += 2 * query_tp / (
                            2 * query_tp + query_fp + query_fn
                        )
        self.update_candidates(outputs, targets, indices)

    def update_candidates(self, outputs, targets, indices):
        class_scores = outputs["pred_logits"].float().sigmoid()
        quality = outputs.get("pred_sem_quality")
        if self.gate_enabled and quality is not None:
            class_scores = class_scores * semantic_suppression_gate(
                quality, self.gate_lambda, self.gate_gamma
            ).to(class_scores.dtype)
        groups = build_per_gt_candidate_groups(class_scores, outputs["pred_boxes"], targets, indices)
        for batch_index, (image_groups, target) in enumerate(zip(groups, targets)):
            labels = target.get("labels")
            labels = labels if isinstance(labels, torch.Tensor) else torch.zeros(0, dtype=torch.long)
            by_gt = {int(group["gt_index"]): group for group in image_groups}
            for gt_index, label_tensor in enumerate(labels):
                label = int(label_tensor)
                states = (self.candidate, self.candidate_class[label])
                for state in states:
                    state["num_GT"] += 1
                group = by_gt.get(gt_index)
                if group is None:
                    for state in states:
                        state["GT_without_anchor"] += 1
                    continue
                candidates = group["candidate_indices"]
                ious = group["candidate_ious"].float()
                anchor = int(group["anchor_index"])
                scores = class_scores[batch_index, candidates, label]
                pair_mask = ious[:, None] > ious[None, :] + 0.10
                higher, lower = torch.nonzero(pair_mask, as_tuple=True)
                violations = int((scores[higher] <= scores[lower]).sum()) if higher.numel() else 0
                top_position = int(scores.argmax())
                for state in states:
                    state["GT_with_any_candidate"] += 1
                    state["GT_with_at_least_2_candidates"] += int(candidates.numel() >= 2)
                    state["candidate_counts"].append(float(candidates.numel()))
                    state["candidate_ious"].extend(ious.detach().cpu().tolist())
                    state["ranking_pair_count"] += int(higher.numel())
                    state["ranking_violation_count"] += violations
                    state["highest_score_is_highest_iou"].append(
                        float(ious[top_position] == ious.max())
                    )
                    if anchor >= 0:
                        state["GT_with_anchor"] += 1
                    else:
                        state["GT_without_anchor"] += 1
                if anchor < 0:
                    continue
                anchor_score = float(class_scores[batch_index, anchor, label])
                non_anchor = candidates[candidates != anchor]
                hard_score = (
                    float(class_scores[batch_index, non_anchor, label].max())
                    if non_anchor.numel() else None
                )
                for state in states:
                    state["anchor_scores"].append(anchor_score)
                    if hard_score is not None:
                        state["hard_scores"].append(hard_score)
                        state["margins"].append(anchor_score - hard_score)

    def report(self):
        def binary(prefix):
            intersection = self.defect[f"{prefix}_intersection"]
            prediction, target = self.defect[f"{prefix}_prediction"], self.defect[f"{prefix}_target"]
            return {
                "Dice": 2 * intersection / max(prediction + target, 1.0),
                "precision": intersection / max(prediction, 1.0),
                "recall": intersection / max(target, 1.0),
            }
        boundary_precision = self.defect["boundary_matched_pred"] / max(self.defect["boundary_pred"], 1.0)
        boundary_recall = self.defect["boundary_matched_true"] / max(self.defect["boundary_true"], 1.0)
        defect_available = self.defect["low_prediction"] + self.defect["low_target"] > 0
        defect = ({
            "available": True, "low_resolution": binary("low"),
            "eval_input_resolution": binary("eval_input"),
            "boundary_at_eval_input_resolution": {
                "precision": boundary_precision,
                "recall": boundary_recall,
                "F1": 2 * boundary_precision * boundary_recall / max(boundary_precision + boundary_recall, 1e-12),
            },
            "note": "eval_input_resolution is after evaluation resize, not original image resolution",
            "per_class_instance_pixel_recall": {
                str(label): value[0] / max(value[1], 1.0) for label, value in self.defect_class.items()
            },
            "foreground_saturation_ratio": self.defect["foreground_saturated"] / max(self.defect["foreground_pixels"], 1.0),
            "background_activation_mean": self.defect["background_probability"] / max(self.defect["background_pixels"], 1.0),
        } if defect_available else {"available": False, "reason": "checkpoint/config has no defect map output"})
        query = {"available": self.mask_seen}

        def ratio(numerator, denominator):
            return numerator / denominator if denominator else None

        for name in ("matched", "near_unmatched"):
            values = self.mask[name]
            tp, fp, fn = (
                values["micro_TP_pixels"],
                values["micro_FP_pixels"],
                values["micro_FN_pixels"],
            )
            nonempty_count = int(values["nonempty_target_count"])
            empty_count = int(values["empty_target_count"])
            query[name] = {
                "count": int(values["count"]),
                "nonempty_target_count": nonempty_count,
                "empty_target_count": empty_count,
                "nonempty_prediction_count": int(values["nonempty_prediction_count"]),
                "empty_prediction_count": int(values["empty_prediction_count"]),
                "empty_empty_count": int(values["empty_empty_count"]),
                "nonempty_target": {
                    "count": nonempty_count,
                    "empty_prediction_count": int(values["nonempty_target_empty_prediction_count"]),
                    "empty_prediction_ratio": ratio(
                        values["nonempty_target_empty_prediction_count"], nonempty_count
                    ),
                    "micro_TP_pixels": int(tp),
                    "micro_FP_pixels": int(fp),
                    "micro_FN_pixels": int(fn),
                    "micro_precision": ratio(tp, tp + fp),
                    "micro_recall": ratio(tp, tp + fn),
                    "micro_Dice": ratio(2 * tp, 2 * tp + fp + fn),
                    "micro_IoU": ratio(tp, tp + fp + fn),
                    "macro_Dice_nonempty": ratio(values["macro_Dice_sum"], nonempty_count),
                    "macro_IoU_nonempty": ratio(values["macro_IoU_sum"], nonempty_count),
                    "BCE_nonempty": ratio(values["BCE_sum"], nonempty_count),
                },
                "empty_target": {
                    "count": empty_count,
                    "empty_empty_count": int(values["empty_empty_count"]),
                    "predicted_positive_pixel_count": int(values["empty_target_positive_pixels"]),
                    "total_pixel_count": int(values["empty_target_total_pixels"]),
                    "false_positive_pixel_rate": ratio(
                        values["empty_target_positive_pixels"], values["empty_target_total_pixels"]
                    ),
                },
            }
        far = self.mask["far_background"]
        histogram = self.mask_histogram["far_background"]
        cumulative = np.cumsum(histogram)
        if cumulative.size and cumulative[-1]:
            p95_index = int(np.searchsorted(cumulative, 0.95 * cumulative[-1], side="left"))
            p95_probability = p95_index / (self._PROBABILITY_BINS - 1)
        else:
            p95_probability = None
        query["far_background"] = {
            "count": int(far["count"]),
            "predicted_positive_pixel_count": int(far["predicted_positive_pixel_count"]),
            "total_pixel_count": int(far["total_pixel_count"]),
            "false_positive_pixel_rate": ratio(
                far["predicted_positive_pixel_count"], far["total_pixel_count"]
            ),
            "mean_probability": ratio(far["probability_sum"], far["total_pixel_count"]),
            "p95_probability": p95_probability,
            "empty_prediction_ratio": ratio(far["empty_prediction_count"], far["count"]),
        }
        query["per_class_matched_nonempty"] = {}
        for label, values in self.mask_class.items():
            tp, fp, fn, count = values["TP"], values["FP"], values["FN"], int(values["count"])
            query["per_class_matched_nonempty"][str(label)] = {
                "count": count,
                "micro_Dice": ratio(2 * tp, 2 * tp + fp + fn),
                "micro_IoU": ratio(tp, tp + fp + fn),
                "micro_precision": ratio(tp, tp + fp),
                "micro_recall": ratio(tp, tp + fn),
                "macro_Dice_nonempty": ratio(values["macro_Dice_sum"], count),
            }
        if not query["available"]:
            query["reason"] = "checkpoint/config has no query-mask output"

        def mean_or_none(values):
            return float(np.mean(values)) if values else None

        def candidate_report(state):
            pair_count = int(state["ranking_pair_count"])
            return {
                "num_GT": int(state["num_GT"]),
                "GT_with_any_candidate": int(state["GT_with_any_candidate"]),
                "GT_with_at_least_2_candidates": int(state["GT_with_at_least_2_candidates"]),
                "GT_with_anchor": int(state["GT_with_anchor"]),
                "GT_without_anchor": int(state["GT_without_anchor"]),
                "candidate_count_mean": mean_or_none(state["candidate_counts"]),
                "candidate_count_median": float(np.median(state["candidate_counts"])) if state["candidate_counts"] else None,
                "candidate_count_p90": float(np.percentile(state["candidate_counts"], 90)) if state["candidate_counts"] else None,
                "candidate_iou_mean": mean_or_none(state["candidate_ious"]),
                "candidate_iou_p50": float(np.median(state["candidate_ious"])) if state["candidate_ious"] else None,
                "anchor_true_class_score_mean": mean_or_none(state["anchor_scores"]),
                "hard_candidate_score_mean": mean_or_none(state["hard_scores"]),
                "anchor_hard_margin_mean": mean_or_none(state["margins"]),
                "ranking_pair_count": pair_count,
                "ranking_violation_count": int(state["ranking_violation_count"]),
                "ranking_violation_ratio": ratio(state["ranking_violation_count"], pair_count),
                "highest_score_is_highest_iou_ratio": mean_or_none(
                    state["highest_score_is_highest_iou"]
                ),
            }

        candidate = candidate_report(self.candidate)
        candidate["per_class"] = {
            str(label): candidate_report(state)
            for label, state in sorted(self.candidate_class.items())
        }
        correlations = {}
        for name in (
            "matched",
            "near_unmatched",
            "associated_nonzero_target",
            "top10_associated_per_image",
            "top30_associated_per_image",
            "all_valid_including_far",
        ):
            state = self.correlations[name]
            values = torch.cat(state["values"]) if state["values"] else torch.zeros(0)
            targets = torch.cat(state["targets"]) if state["targets"] else torch.zeros(0)
            correlations[name] = {
                "image_count": int(state["image_count"]),
                "query_count": int(state["query_count"]),
                "mean_per_image_spearman": mean_or_none(state["per_image"]),
                "global_spearman": tie_aware_spearman(values, targets),
                "constant_target_image_count": int(state["constant_target_image_count"]),
                "quality_std": float(values.std(unbiased=False)) if values.numel() else None,
                "target_std": float(targets.std(unbiased=False)) if targets.numel() else None,
            }
        return defect, query, candidate, correlations


def semantic_quality_report(classified: Sequence[Mapping[str, object]], correlations):
    names = ("TP", "far_background_FP", "duplicate_same_class_FP", "class_confusion_FP", "localization_FP")
    groups = {
        name: distribution(
            row["semantic_quality"] for row in classified
            if row["error_type"] == name and row.get("semantic_quality") is not None
        ) for name in names
    }
    groups["all_FP"] = distribution(
        row["semantic_quality"] for row in classified
        if row["error_type"] != "TP" and row.get("semantic_quality") is not None
    )
    positives = [row["semantic_quality"] for row in classified if row["error_type"] == "TP" and row.get("semantic_quality") is not None]
    auc = {}
    for name in (*names[1:], "all_FP"):
        negatives = [row["semantic_quality"] for row in classified if (row["error_type"] == name if name != "all_FP" else row["error_type"] != "TP") and row.get("semantic_quality") is not None]
        auc[f"TP_vs_{name}"] = binary_auc(positives, negatives)
    return {"available": bool(positives), "groups": groups, "AUC": auc, "target_correlation": correlations}


def score_analysis(
    annotation_path, ground_truth, rows_by_type, max_dets, threshold,
    image_ids: Sequence[int] | None = None,
):
    output = {}
    for name, rows_by_image in rows_by_type.items():
        classified, taxonomy = taxonomy_report(rows_by_image, ground_truth, threshold)
        coco = coco_metrics(
            annotation_path, coco_detections(rows_by_image), max_dets,
            image_ids=image_ids,
        )
        auc = binary_auc(
            [row["score"] for row in classified if row["error_type"] == "TP"],
            [row["score"] for row in classified if row["error_type"] != "TP"],
        )
        output[name] = {
            "TP_vs_all_FP_ROC_AUC": auc.get("ROC_AUC"),
            "TP_vs_all_FP_PR_AUC": auc.get("PR_AUC"),
            "AP": coco["AP@[0.50:0.95]"], "AP50": coco["AP50"], "AP75": coco["AP75"],
            "background_FP": taxonomy.get("far_background_FP", 0),
            "near_GT_FP": taxonomy.get("duplicate_same_class_FP", 0) + taxonomy.get("class_confusion_FP", 0) + taxonomy.get("localization_FP", 0),
        }
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="SQ-Align V3 unified single-JSON evaluator")
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("-r", "--checkpoint", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-dets", type=int, default=300)
    parser.add_argument("--score-thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.25, 0.50])
    gate_group = parser.add_mutually_exclusive_group()
    gate_group.add_argument(
        "--semantic-gate-enabled", dest="semantic_gate_enabled", action="store_true"
    )
    gate_group.add_argument(
        "--no-semantic-gate", dest="semantic_gate_enabled", action="store_false"
    )
    parser.set_defaults(semantic_gate_enabled=None)
    parser.add_argument("--semantic-gate-lambda", type=float, default=None)
    parser.add_argument("--semantic-gate-gamma", type=float, default=None)
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--timed-iterations", type=int, default=50)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.config, args.checkpoint, args.images_dir, args.ann_file):
        if not path.exists():
            raise FileNotFoundError(path)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    annotation = json.loads(args.ann_file.read_text(encoding="utf-8"))
    images, category_names, ground_truth, per_class_gt, valid_masks = raw_dataset(annotation)

    config = YAMLConfig(str(args.config))
    gate_enabled, gate_lambda, gate_gamma, gate_source = resolve_semantic_gate(
        args, config.yaml_cfg
    )
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    validation = config.yaml_cfg["val_dataloader"]
    validation["total_batch_size"] = args.batch_size
    validation.pop("batch_size", None)
    validation["num_workers"] = args.num_workers
    validation["dataset"].update({
        "img_folder": str(args.images_dir), "ann_file": str(args.ann_file), "return_masks": True,
    })
    config.yaml_cfg["DEIM"]["return_alignment_maps_in_eval"] = True
    config.yaml_cfg["PostProcessor"]["num_top_queries"] = args.max_dets
    model = config.model
    checkpoint_load = load_checkpoint(model, args.checkpoint)
    device = torch.device(args.device)
    model = model.to(device).eval()
    criterion = config.criterion.to(device).eval()
    loader = config.val_dataloader
    dataset = loader.dataset
    label_to_category = (
        dataset.label2category
        if getattr(dataset, "remap_mscoco_category", False) else None
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    class_rows: Dict[int, List[Dict[str, object]]] = {}
    gated_rows: Dict[int, List[Dict[str, object]]] = {}
    accumulator = AlignmentAccumulator(
        gate_lambda=gate_lambda,
        gate_gamma=gate_gamma,
        gate_enabled=gate_enabled,
    )
    alignment_sanity = AlignmentSanityAccumulator()
    quality_available = False
    latency: List[float] = []
    timed_batches = 0
    seen = 0
    with torch.inference_mode():
        for iteration, (samples, targets) in enumerate(loader):
            if args.max_images is not None and seen >= args.max_images:
                break
            if args.max_images is not None and seen + samples.shape[0] > args.max_images:
                keep = args.max_images - seen
                samples, targets = samples[:keep], targets[:keep]
            samples = samples.to(device)
            raw_targets = [
                {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in target.items()
                }
                for target in targets
            ]
            alignment_targets = build_alignment_targets(
                raw_targets,
                sample_hw=(int(samples.shape[-2]), int(samples.shape[-1])),
            )
            alignment_sanity.update(alignment_targets)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(samples)
            quality_available = quality_available or outputs.get("pred_sem_quality") is not None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if args.warmup_iterations <= iteration < args.warmup_iterations + args.timed_iterations:
                latency.extend([elapsed * 1000 / samples.shape[0]] * samples.shape[0])
                timed_batches += 1
            image_ids = [int(target["image_id"].flatten()[0]) for target in raw_targets]
            image_sizes = [
                (int(images[image_id]["width"]), int(images[image_id]["height"])) for image_id in image_ids
            ]
            batch_class, batch_gated = model_predictions(
                outputs, image_ids, image_sizes, args.max_dets,
                gate_lambda, gate_gamma, gate_enabled=gate_enabled,
                label_to_category=label_to_category,
            )
            class_rows.update(batch_class)
            gated_rows.update(batch_gated)
            match_outputs = {"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]}
            indices = criterion.matcher(match_outputs, alignment_targets)["indices"]
            accumulator.update_defect(outputs, raw_targets)
            accumulator.update_query(outputs, alignment_targets, indices)
            seen += samples.shape[0]

    main_rows = gated_rows if gate_enabled else class_rows
    evaluated_image_ids = sorted(main_rows)
    evaluated_ground_truth = {
        image_id: ground_truth.get(image_id, []) for image_id in evaluated_image_ids
    }
    main_detections = coco_detections(main_rows)
    lowest_threshold = min(args.score_thresholds)
    main_classified, main_taxonomy = taxonomy_report(
        main_rows, evaluated_ground_truth, lowest_threshold
    )
    fixed = {}
    for threshold in args.score_thresholds:
        rows, summary = taxonomy_report(main_rows, evaluated_ground_truth, threshold)
        fixed[f"{threshold:.2f}"] = summary
    defect, query_mask, candidate, correlations = accumulator.report()
    class_ids = sorted(category_names)
    report = empty_report()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    report["meta"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_branch": git_value(["branch", "--show-current"], None),
        "git_commit": git_value(["rev-parse", "HEAD"], None),
        "dirty_worktree": bool(git_value(["status", "--porcelain"], "")),
        "config_path": str(args.config.resolve()), "config_sha256": sha256_file(args.config),
        "checkpoint_path": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint),
        "annotation_path": str(args.ann_file.resolve()), "annotation_sha256": sha256_file(args.ann_file),
        "image_dir": str(args.images_dir.resolve()), "seed": args.seed, "device": str(device),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "num_parameters": parameters, "trainable_parameters": trainable,
        "semantic_gate": {
            "enabled": gate_enabled,
            "source": gate_source,
            "lambda": gate_lambda,
            "gamma": gate_gamma,
            "quality_available": quality_available,
        },
    }
    report["checkpoint_load"] = checkpoint_load
    report["alignment_sanity"] = alignment_sanity.report()
    annotation_count = sum(per_class_gt.values())
    report["dataset"] = {
        "num_images": len(images), "evaluated_images": seen,
        "num_annotations": annotation_count, "num_classes": len(category_names),
        "per_class_GT": {str(category_names.get(label, label)): count for label, count in per_class_gt.items()},
        "valid_mask_count": valid_masks, "invalid_mask_count": annotation_count - valid_masks,
        "valid_mask_ratio": valid_masks / max(annotation_count, 1),
    }
    report["coco"] = coco_metrics(
        args.ann_file, main_detections, args.max_dets,
        image_ids=evaluated_image_ids,
    )
    report["tide"] = (
        tide_metrics(args.ann_file, main_detections)
        if len(evaluated_image_ids) == len(images)
        else {
            "available": False,
            "reason": "TIDE is skipped for --max-images partial evaluation",
        }
    )
    report["fixed_thresholds"] = fixed
    report["error_taxonomy"] = main_taxonomy
    report["confusion"] = confusion_report(
        main_classified, evaluated_ground_truth, class_ids
    )
    report["recall_by_iou"] = recall_report(
        main_rows, evaluated_ground_truth,
        [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    )
    report["calibration"] = {
        "overall": calibration_metrics(main_classified),
        "per_class": {
            str(category_names.get(label, label)): calibration_metrics(
                [row for row in main_classified if int(row["label"]) == label]
            ) for label in class_ids
        },
    }
    report["defect_map"] = defect
    report["query_mask"] = query_mask
    report["semantic_quality"] = semantic_quality_report(main_classified, correlations)
    report["candidate_ranking"] = candidate
    report["score_analysis"] = score_analysis(
        args.ann_file, evaluated_ground_truth,
        {"class_score": class_rows, "class_score_x_gate": gated_rows},
        args.max_dets, lowest_threshold, image_ids=evaluated_image_ids,
    )
    latency_array = np.asarray(latency, dtype=np.float64)
    mean_latency = float(latency_array.mean()) if latency_array.size else 0.0
    report["runtime"] = {
        "warmup_iterations": args.warmup_iterations,
        "timed_iterations": timed_batches,
        "batch_size": args.batch_size, "mean_latency_ms_per_image": mean_latency,
        "p50_latency_ms_per_image": float(np.percentile(latency_array, 50)) if latency_array.size else 0.0,
        "p95_latency_ms_per_image": float(np.percentile(latency_array, 95)) if latency_array.size else 0.0,
        "FPS": 1000.0 / mean_latency if mean_latency > 0 else 0.0,
        "peak_CUDA_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    report["automatic_diagnosis"] = automatic_diagnosis(report)
    if "candidate_diagnostics_impossible_zero" in report["automatic_diagnosis"].get("errors", []):
        warnings.warn(
            "candidate_diagnostics_impossible_zero: full V4 evaluation produced no usable candidate pairs",
            RuntimeWarning,
        )
    assert tuple(report) == TOP_LEVEL_KEYS
    report = finite_json_value(report)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Unified SQ-Align V3 report: {args.output_json}")


if __name__ == "__main__":
    main()
