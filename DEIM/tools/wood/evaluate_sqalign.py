"""Evaluate one SQ-Align checkpoint with COCO and query-alignment diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from optional_dependency_stubs import install  # noqa: E402

install()

from engine.core import YAMLConfig  # noqa: E402
from engine.deim.box_ops import box_cxcywh_to_xyxy, box_iou  # noqa: E402
from engine.deim.semantic_query_alignment import (  # noqa: E402
    build_all_query_best_iou_target,
    build_union_defect_target,
    compose_dual_quality_scores,
)


def _checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "ema" in checkpoint:
        return checkpoint["ema"].get("module", checkpoint["ema"])
    return checkpoint.get("model", checkpoint)


def _load_matching_weights(model: torch.nn.Module, path: Path) -> Dict[str, Any]:
    current = model.state_dict()
    checkpoint = _checkpoint_state(path)
    compatible = {
        key: value for key, value in checkpoint.items()
        if key in current and current[key].shape == value.shape
    }
    result = model.load_state_dict(compatible, strict=False)
    return {
        "loaded": len(compatible),
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
        "shape_mismatch_or_unknown": sorted(set(checkpoint) - set(compatible)),
    }


def _iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if not boxes1.numel() or not boxes2.numel():
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    return box_iou(boxes1, boxes2)[0]


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 2:
        return None
    x_array = np.asarray([pair[0] for pair in pairs])
    y_array = np.asarray([pair[1] for pair in pairs])
    if np.unique(x_array).size < 2 or np.unique(y_array).size < 2:
        return None
    from scipy.stats import spearmanr

    return float(spearmanr(x_array, y_array).statistic)


def _distribution(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray([float(value) for value in values if math.isfinite(float(value))])
    if not array.size:
        return {"count": 0, "mean": None, "std": None, "p25": None, "median": None, "p75": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
    }


def _roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    if not positive or not negative:
        return None
    values = np.asarray(list(positive) + list(negative), dtype=float)
    labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=int)
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    positive_rank_sum = ranks[labels == 1].sum()
    return float((positive_rank_sum - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def _average_precision(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    if not positive or not negative:
        return None
    scored = sorted(
        [(float(score), 1) for score in positive] + [(float(score), 0) for score in negative],
        reverse=True,
    )
    true_positive = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(scored, start=1):
        if label:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / len(positive)


def _category_mapping(dataset: Any, labels: Iterable[int]) -> Dict[int, int]:
    category_ids = sorted(int(category_id) for category_id in dataset.coco.cats)
    observed = set(int(label) for label in labels)
    if observed.issubset(set(category_ids)):
        return {category_id: category_id for category_id in category_ids}
    return {index: category_id for index, category_id in enumerate(category_ids)}


def _compose_scores(outputs: Dict[str, torch.Tensor], postprocessor: Any) -> torch.Tensor:
    if postprocessor.use_focal_loss:
        class_scores = outputs["pred_logits"].sigmoid()
    else:
        class_scores = outputs["pred_logits"].softmax(-1)[..., :-1]
    loc = outputs.get("pred_loc_quality") if postprocessor.loc_quality_rerank else None
    semantic = outputs.get("pred_sem_quality") if postprocessor.semantic_quality_rerank else None
    return compose_dual_quality_scores(
        class_scores,
        loc,
        semantic,
        postprocessor.loc_quality_power,
        postprocessor.semantic_quality_power,
    )


def _collect_inference(
    config: YAMLConfig,
    checkpoint: Path,
    device: torch.device,
    max_dets: int,
) -> Dict[str, Any]:
    model = config.model
    load_result = _load_matching_weights(model, checkpoint)
    model = model.to(device).eval()
    if model.defectness_head is not None:
        model.return_defect_map_in_eval = True
    criterion = config.criterion.to(device).eval()
    postprocessor = config.postprocessor.to(device).eval()
    dataloader = config.val_dataloader

    predictions: List[Dict[str, Any]] = []
    detections: List[Dict[str, Any]] = []
    query_records: List[Dict[str, Any]] = []
    target_labels: List[int] = []
    defect_counts = dict(intersection=0.0, prediction=0.0, target=0.0)

    with torch.no_grad():
        for batch_index, (samples, targets) in enumerate(dataloader):
            samples = samples.to(device)
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            outputs = model(samples)
            matching = criterion.matcher(
                {"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]}, targets
            )["indices"]
            best_iou = build_all_query_best_iou_target(outputs["pred_boxes"], targets)
            final_scores = _compose_scores(outputs, postprocessor)
            loc_quality = outputs.get("pred_loc_quality")
            loc_quality = loc_quality.sigmoid().squeeze(-1) if loc_quality is not None else None
            semantic_quality = outputs.get("pred_sem_quality")
            semantic_quality = semantic_quality.squeeze(-1) if semantic_quality is not None else None
            query_final = final_scores.amax(-1)

            if "pred_defect_logits" in outputs:
                defect_target = build_union_defect_target(
                    targets,
                    outputs["pred_defect_logits"].shape[-2:],
                    device,
                    outputs["pred_defect_logits"].dtype,
                )
                defect_prediction = outputs["pred_defect_logits"].sigmoid() >= 0.5
                defect_binary_target = defect_target >= 0.5
                defect_counts["intersection"] += float((defect_prediction & defect_binary_target).sum())
                defect_counts["prediction"] += float(defect_prediction.sum())
                defect_counts["target"] += float(defect_binary_target.sum())

            for image_index, target in enumerate(targets):
                image_id = int(target["image_id"].reshape(-1)[0])
                original_width, original_height = (float(value) for value in target["orig_size"])
                scale = outputs["pred_boxes"].new_tensor(
                    [original_width, original_height, original_width, original_height]
                )
                pixel_boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"][image_index]) * scale
                gt_pixel_boxes = box_cxcywh_to_xyxy(target["boxes"]) * scale
                target_labels.extend(int(label) for label in target["labels"])
                source_indices, target_indices = matching[image_index]
                matched_mask = torch.zeros(outputs["pred_boxes"].shape[1], device=device, dtype=torch.bool)
                matched_mask[source_indices] = True
                matched_iou = torch.zeros_like(best_iou[image_index])
                if source_indices.numel():
                    matched_iou[source_indices] = _iou_xyxy(
                        pixel_boxes[source_indices], gt_pixel_boxes[target_indices]
                    ).diag()

                for query_index in range(outputs["pred_boxes"].shape[1]):
                    query_records.append({
                        "image_id": image_id,
                        "query_index": query_index,
                        "best_iou": float(best_iou[image_index, query_index]),
                        "matched_iou": float(matched_iou[query_index]),
                        "matched": bool(matched_mask[query_index]),
                        "loc_quality": None if loc_quality is None else float(loc_quality[image_index, query_index]),
                        "semantic_quality": None if semantic_quality is None else float(semantic_quality[image_index, query_index]),
                        "final_query_score": float(query_final[image_index, query_index]),
                    })

                topk = min(int(max_dets), final_scores.shape[1] * final_scores.shape[2])
                scores, flat_indices = final_scores[image_index].flatten().topk(topk)
                labels = flat_indices % final_scores.shape[2]
                query_indices = flat_indices // final_scores.shape[2]
                for score, label, query_index in zip(scores, labels, query_indices):
                    query_index_int = int(query_index)
                    box = pixel_boxes[query_index_int]
                    x1, y1, x2, y2 = (float(value) for value in box)
                    record = {
                        "image_id": image_id,
                        "model_label": int(label),
                        "score": float(score),
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "query_index": query_index_int,
                        "best_iou": float(best_iou[image_index, query_index_int]),
                        "loc_quality": None if loc_quality is None else float(loc_quality[image_index, query_index_int]),
                        "semantic_quality": None if semantic_quality is None else float(semantic_quality[image_index, query_index_int]),
                    }
                    detections.append(record)
            print(f"inference batch {batch_index + 1}/{len(dataloader)}")

    label_to_category = _category_mapping(dataloader.dataset, target_labels)
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        predictions.append({
            "image_id": detection["image_id"],
            "category_id": label_to_category[detection["model_label"]],
            "bbox": [x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)],
            "score": detection["score"],
        })
        detection["category_id"] = label_to_category[detection["model_label"]]
    return {
        "predictions": predictions,
        "detections": detections,
        "queries": query_records,
        "load": load_result,
        "defect_counts": defect_counts,
        "dataset": dataloader.dataset,
    }


def _coco_metrics(dataset: Any, predictions: List[Dict[str, Any]], max_dets: int) -> Dict[str, Any]:
    from pycocotools.cocoeval import COCOeval

    coco_gt = dataset.coco
    coco_dt = coco_gt.loadRes(predictions) if predictions else coco_gt.loadRes([])

    def evaluate(category_ids: List[int] | None = None) -> List[float]:
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.maxDets = [1, min(10, max_dets), max_dets]
        if category_ids is not None:
            evaluator.params.catIds = category_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return [float(value) for value in evaluator.stats]

    overall = evaluate()
    names = coco_gt.cats
    per_class = {}
    for category_id in sorted(names):
        stats = evaluate([category_id])
        per_class[names[category_id]["name"]] = {"AP": stats[0], "AP50": stats[1], "AP75": stats[2]}
    return {
        "AP": overall[0], "AP50": overall[1], "AP75": overall[2],
        "APM": overall[4], "APL": overall[5], "per_class": per_class,
    }


def _ground_truth(dataset: Any) -> Dict[int, List[Dict[str, Any]]]:
    result: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for annotation in dataset.coco.dataset["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        x, y, width, height = annotation["bbox"]
        result[int(annotation["image_id"])].append({
            "category_id": int(annotation["category_id"]),
            "bbox_xyxy": [x, y, x + width, y + height],
        })
    return result


def _box_iou_list(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    return intersection / max(area1 + area2 - intersection, 1e-12)


def _annotate_detections(
    detections: List[Dict[str, Any]],
    ground_truth: Dict[int, List[Dict[str, Any]]],
    score_threshold: float,
    tp_iou: float = 0.5,
    background_iou: float = 0.1,
) -> List[Dict[str, Any]]:
    annotated = []
    by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        if detection["score"] >= score_threshold:
            by_image[detection["image_id"]].append(detection)
    for image_id, rows in by_image.items():
        gt_rows = ground_truth.get(image_id, [])
        used = set()
        for row in sorted(rows, key=lambda item: item["score"], reverse=True):
            all_ious = [_box_iou_list(row["bbox_xyxy"], gt["bbox_xyxy"]) for gt in gt_rows]
            same = [
                (index, all_ious[index]) for index, gt in enumerate(gt_rows)
                if gt["category_id"] == row["category_id"] and index not in used
            ]
            best_same = max(same, key=lambda item: item[1]) if same else (-1, 0.0)
            is_tp = best_same[1] >= tp_iou
            if is_tp:
                used.add(best_same[0])
            annotated.append({
                **row,
                "is_tp": is_tp,
                "matched_iou": best_same[1] if is_tp else 0.0,
                "best_any_iou": max(all_ious, default=0.0),
                "is_background_fp": (not is_tp) and max(all_ious, default=0.0) < background_iou,
                "is_near_duplicate": (not is_tp) and max(all_ious, default=0.0) >= background_iou,
            })
    return annotated


def _confusion_matrix(
    detections: List[Dict[str, Any]],
    ground_truth: Dict[int, List[Dict[str, Any]]],
    category_ids: Sequence[int],
    score_threshold: float,
    iou_threshold: float,
) -> Dict[str, Any]:
    category_to_index = {category_id: index for index, category_id in enumerate(category_ids)}
    background_index = len(category_ids)
    matrix = np.zeros((background_index + 1, background_index + 1), dtype=int)
    detections_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        if detection["score"] >= score_threshold:
            detections_by_image[detection["image_id"]].append(detection)
    image_ids = set(ground_truth) | set(detections_by_image)
    for image_id in image_ids:
        gt_rows = ground_truth.get(image_id, [])
        predicted_rows = sorted(
            detections_by_image.get(image_id, []), key=lambda row: row["score"], reverse=True
        )
        used_gt = set()
        for predicted in predicted_rows:
            candidates = [
                (index, _box_iou_list(predicted["bbox_xyxy"], gt["bbox_xyxy"]))
                for index, gt in enumerate(gt_rows) if index not in used_gt
            ]
            best = max(candidates, key=lambda pair: pair[1]) if candidates else (-1, 0.0)
            predicted_index = category_to_index[predicted["category_id"]]
            if best[1] >= iou_threshold:
                used_gt.add(best[0])
                actual_index = category_to_index[gt_rows[best[0]]["category_id"]]
                matrix[actual_index, predicted_index] += 1
            else:
                matrix[background_index, predicted_index] += 1
        for gt_index, gt in enumerate(gt_rows):
            if gt_index not in used_gt:
                matrix[category_to_index[gt["category_id"]], background_index] += 1
    return {
        "orientation": "rows=ground_truth, columns=prediction",
        "labels": [*category_ids, "background"],
        "matrix": matrix.tolist(),
    }


def _recall_by_iou(
    detections: List[Dict[str, Any]], ground_truth: Dict[int, List[Dict[str, Any]]], thresholds: Sequence[float]
) -> Dict[str, float]:
    total = sum(len(rows) for rows in ground_truth.values())
    result = {}
    for threshold in thresholds:
        matched = 0
        for image_id, gt_rows in ground_truth.items():
            rows = sorted(
                [row for row in detections if row["image_id"] == image_id],
                key=lambda row: row["score"], reverse=True,
            )
            used = set()
            for row in rows:
                candidates = [
                    (index, _box_iou_list(row["bbox_xyxy"], gt["bbox_xyxy"]))
                    for index, gt in enumerate(gt_rows)
                    if index not in used and gt["category_id"] == row["category_id"]
                ]
                if candidates:
                    best = max(candidates, key=lambda item: item[1])
                    if best[1] >= threshold:
                        used.add(best[0])
            matched += len(used)
        result[f"Recall@IoU{threshold:.2f}"] = matched / max(total, 1)
    return result


def _ece(rows: List[Dict[str, Any]], bins: int = 15) -> Dict[str, float]:
    if not rows:
        return {"ECE": 0.0, "LaECE": 0.0}
    scores = np.asarray([row["score"] for row in rows])
    correct = np.asarray([float(row["is_tp"]) for row in rows])
    localization = np.asarray([row["matched_iou"] if row["is_tp"] else 0.0 for row in rows])
    ece = laece = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (scores >= lower) & (scores < upper if upper < 1 else scores <= upper)
        if not mask.any():
            continue
        fraction = mask.mean()
        ece += fraction * abs(scores[mask].mean() - correct[mask].mean())
        laece += fraction * abs(scores[mask].mean() - localization[mask].mean())
    return {"ECE": float(ece), "LaECE": float(laece)}


def _query_diagnostics(queries: List[Dict[str, Any]], detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    loc_rows = [row for row in queries if row["loc_quality"] is not None]
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in loc_rows:
        grouped[row["image_id"]].append(row)

    def top_spearman(limit: int | None) -> float | None:
        selected = []
        for rows in grouped.values():
            ordered = sorted(rows, key=lambda row: row["final_query_score"], reverse=True)
            selected.extend(ordered if limit is None else ordered[:limit])
        return _safe_spearman(
            [row["loc_quality"] for row in selected], [row["best_iou"] for row in selected]
        )

    tp = [row for row in detections if row["is_tp"]]
    background = [row for row in detections if row["is_background_fp"]]
    duplicate = [row for row in detections if row["is_near_duplicate"]]
    return {
        "all_query_loc_quality_best_iou_spearman": top_spearman(None),
        "top100_loc_quality_best_iou_spearman": top_spearman(100),
        "top300_loc_quality_best_iou_spearman": top_spearman(300),
        "loc_quality": {
            "TP": _distribution(row["loc_quality"] for row in tp if row["loc_quality"] is not None),
            "background_FP": _distribution(row["loc_quality"] for row in background if row["loc_quality"] is not None),
            "duplicate_near_GT": _distribution(row["loc_quality"] for row in duplicate if row["loc_quality"] is not None),
        },
        "semantic_quality": {
            "TP": _distribution(row["semantic_quality"] for row in tp if row["semantic_quality"] is not None),
            "background_FP": _distribution(row["semantic_quality"] for row in background if row["semantic_quality"] is not None),
        },
        "semantic_TP_vs_background_ROC_AUC": _roc_auc(
            [row["semantic_quality"] for row in tp if row["semantic_quality"] is not None],
            [row["semantic_quality"] for row in background if row["semantic_quality"] is not None],
        ),
        "semantic_TP_vs_background_PR_AUC": _average_precision(
            [row["semantic_quality"] for row in tp if row["semantic_quality"] is not None],
            [row["semantic_quality"] for row in background if row["semantic_quality"] is not None],
        ),
        "final_score_TP_vs_background_ROC_AUC": _roc_auc(
            [row["score"] for row in tp], [row["score"] for row in background]
        ),
    }


def _save_confusion_plots(confusion: Dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    matrix = np.asarray(confusion["matrix"], dtype=float)
    labels = [str(label) for label in confusion["labels"]]
    variants = [
        (matrix, "confusion_matrix.png", "Confusion matrix (counts)"),
        (
            matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1.0),
            "confusion_matrix_with_fp_fn.png",
            "Confusion matrix with background FP/FN (row-normalized)",
        ),
    ]
    for values, filename, title in variants:
        figure, axis = plt.subplots(figsize=(10, 9))
        image = axis.imshow(values, cmap="Blues", vmin=0)
        axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("prediction")
        axis.set_ylabel("ground truth")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    config = YAMLConfig(args.config)
    if args.loc_power is not None:
        config.yaml_cfg["PostProcessor"]["loc_quality_rerank"] = args.loc_power > 0
        config.yaml_cfg["PostProcessor"]["loc_quality_power"] = args.loc_power
    if args.sem_power is not None:
        config.yaml_cfg["PostProcessor"]["semantic_quality_rerank"] = args.sem_power > 0
        config.yaml_cfg["PostProcessor"]["semantic_quality_power"] = args.sem_power
    if args.images_dir:
        config.yaml_cfg["val_dataloader"]["dataset"]["img_folder"] = args.images_dir
    if args.ann_file:
        config.yaml_cfg["val_dataloader"]["dataset"]["ann_file"] = args.ann_file
    if args.batch_size:
        config.yaml_cfg["val_dataloader"]["total_batch_size"] = args.batch_size
    if args.num_workers is not None:
        config.yaml_cfg["val_dataloader"]["num_workers"] = args.num_workers
    device = torch.device(args.device)
    inference = _collect_inference(config, Path(args.checkpoint), device, args.max_dets)
    ground_truth = _ground_truth(inference["dataset"])
    annotated = _annotate_detections(
        inference["detections"], ground_truth, args.score_threshold, args.tp_iou, args.background_iou
    )
    filtered = [row for row in annotated if row["score"] >= args.score_threshold]
    true_positive = sum(row["is_tp"] for row in filtered)
    false_positive = len(filtered) - true_positive
    ground_truth_count = sum(len(rows) for rows in ground_truth.values())
    false_negative = max(ground_truth_count - true_positive, 0)
    category_ids = sorted(int(category_id) for category_id in inference["dataset"].coco.cats)
    defect = inference["defect_counts"]
    defect_metrics = {
        "Dice": 2 * defect["intersection"] / max(defect["prediction"] + defect["target"], 1.0),
        "pixel_precision": defect["intersection"] / max(defect["prediction"], 1.0),
        "pixel_recall": defect["intersection"] / max(defect["target"], 1.0),
    }
    metrics = {
        "coco": _coco_metrics(inference["dataset"], inference["predictions"], args.max_dets),
        "fixed_threshold": {
            "threshold": args.score_threshold,
            "TP": true_positive,
            "FP": false_positive,
            "FN": false_negative,
            "precision": true_positive / max(true_positive + false_positive, 1),
            "recall": true_positive / max(true_positive + false_negative, 1),
            "background_FP": sum(row["is_background_fp"] for row in filtered),
            "near_GT_duplicate_or_class_FP": sum(row["is_near_duplicate"] for row in filtered),
            "confusion_matrix": _confusion_matrix(
                inference["detections"], ground_truth, category_ids, args.score_threshold, args.tp_iou
            ),
        },
        "recall": _recall_by_iou(
            [row for row in inference["detections"] if row["score"] >= args.recall_score_threshold],
            ground_truth,
            args.recall_ious,
        ),
        "calibration": _ece(filtered),
        "query_alignment": _query_diagnostics(inference["queries"], annotated),
        "defect_map": defect_metrics,
        "checkpoint_load": inference["load"],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.json").write_text(
        json.dumps(inference["predictions"], ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "sqalign_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    split_outputs = {
        "result.json": metrics,
        "per_class_metrics.json": metrics["coco"]["per_class"],
        "background_fp.json": metrics["fixed_threshold"],
        "recall_by_iou.json": metrics["recall"],
        "calibration.json": metrics["calibration"],
        "query_quality.json": {
            key: value for key, value in metrics["query_alignment"].items()
            if "loc_quality" in key or "spearman" in key
        },
        "semantic_query.json": {
            key: value for key, value in metrics["query_alignment"].items()
            if "semantic" in key or "final_score" in key
        },
        "defectness_metrics.json": metrics["defect_map"],
        "confusion_matrix.json": metrics["fixed_threshold"]["confusion_matrix"],
    }
    for filename, payload in split_outputs.items():
        (output_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    _save_confusion_plots(metrics["fixed_threshold"]["confusion_matrix"], output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--images-dir")
    parser.add_argument("--ann-file")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-dets", type=int, default=300)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--recall-score-threshold", type=float, default=0.05)
    parser.add_argument("--tp-iou", type=float, default=0.50)
    parser.add_argument("--background-iou", type=float, default=0.10)
    parser.add_argument("--loc-power", type=float)
    parser.add_argument("--sem-power", type=float)
    parser.add_argument(
        "--recall-ious", type=float, nargs="+", default=[0.30, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
