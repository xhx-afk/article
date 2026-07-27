"""统一评估 LABS-Mamba V3 S0～S5 checkpoint。

每个启用的实验使用完全相同的 COCO 数据、阈值和推理参数，输出 COCO
AP/AP50/AP75、逐类 AP/PR、TIDE、混淆矩阵、FP/FN、ECE/LaECE、IoU
分段召回、逐类背景 FP，以及 D-FINE LQE query-quality 诊断。
每个实验目录另写 all_metrics_summary.json，集中保存该实验的全部客观指标。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ========================= 用户配置区 =========================
CONFIG: Dict[str, Any] = {
    # 统一评估数据。评估验证集时将 test 改为 val；所有实验共用这两个路径。
    "images_dir": "/home/zxw4090/hjw/D-FINE/data/WoodDefect/wood_coco_all_only_defect_quick_balanced_4000/images/test",
    "ann_file": "/home/zxw4090/hjw/D-FINE/data/WoodDefect/wood_coco_all_only_defect_quick_balanced_4000/annotations/instances_test.json",

    # 每个实验会写入 output_root/<experiment name>/，根目录另有跨实验汇总。
    "output_root": "./deim_outputs/labs_mamba_v3_test_evaluation",
    "device": "cuda:0",
    "batch_size": 1,
    "num_workers": 4,
    "max_dets": 100,
    # 默认禁用缓存，优先保证正式测试结果来自本次 checkpoint 推理。
    # 确认配置、权重和数据未改变时，可改为 True 或直接复用已生成目录。
    "reuse_inference_cache": False,
    "continue_on_error": True,
    "strict_checkpoint_structure": True,
    "save_query_records_csv": True,
    # 所有检测指标必须使用相同阈值，才能公平比较 S0～S5。
    "confusion_score_threshold": 0.25,
    "confusion_iou_threshold": 0.50,
    "ece_score_threshold": 0.05,
    "ece_iou_threshold": 0.50,
    "ece_num_bins": 15,
    "recall_score_threshold": 0.05,
    "recall_iou_thresholds": [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95],
    "background_fp_score_threshold": 0.05,
    "background_fp_iou_threshold": 0.10,
    "pr_iou_threshold": 0.50,
    "enable_tide": True,
    # None=自动。也可显式写成 {0: 1, 1: 2, ...}，键为模型 label，值为 COCO category_id。
    "model_label_to_category_id": None,
    # 按需启用实验并填写实际 checkpoint。enabled=False 的实验会跳过。
    # config 与 checkpoint 必须来自同一个实验，严禁交叉加载。
    "experiments": [
        {
            "name": "s0_local_anchor",
            "enabled": False,
            "config": "./configs/deim_dfine/labs_mamba_v3/s0_local_anchor.yml",
            "checkpoint": "./deim_outputs/labs_mamba_v3_s0_local_anchor/best_stg2.pth",
        },
        {
            "name": "s1_hv_sidecar_b64",
            "enabled": False,
            "config": "./configs/deim_dfine/labs_mamba_v3/s1_hv_sidecar_b64.yml",
            "checkpoint": "./deim_outputs/labs_mamba_v3_s1_hv_b64/best_stg2.pth",
        },
        {
            "name": "s2_vh_sidecar_b64",
            "enabled": False,
            "config": "./configs/deim_dfine/labs_mamba_v3/s2_vh_sidecar_b64.yml",
            "checkpoint": "./deim_outputs/labs_mamba_v3_s2_vh_b64/best_stg2.pth",
        },
        {
            "name": "s3_hv_sidecar_b128",
            "enabled": False,
            "config": "./configs/deim_dfine/labs_mamba_v3/s3_hv_sidecar_b128.yml",
            "checkpoint": "./deim_outputs/labs_mamba_v3_s3_hv_b128/best_stg2.pth",
        },
        {
            "name": "s4_vh_sidecar_b128",
            "enabled": False,
            "config": "./configs/deim_dfine/labs_mamba_v3/s4_vh_sidecar_b128.yml",
            "checkpoint": "./deim_outputs/labs_mamba_v3_s4_vh_b128/best_stg2.pth",
        },
        {
            "name": "s5_best_path_center_ablation",
            "enabled": False,
            "config": "./configs/deim_dfine/labs_mamba_v3/s5_best_path_center_ablation.yml",
            "checkpoint": "./deim_outputs/labs_mamba_v3_s5_best_no_center/best_stg2.pth",
        },
    ],
}


REQUIRED_OUTPUTS: Tuple[str, ...] = (
    "predictions.json",
    "all_metrics_summary.json",
    "class_and_all_metrics.json",
    "class_and_all_metrics.png",
    "per_class_ap.json",
    "per_class_ap.png",
    "tide_summary.txt",
    "tide_summary.png",
    "tide_metrics.json",
    "confusion_matrix.csv",
    "confusion_matrix.png",
    "confusion_matrix_with_fp_fn.csv",
    "confusion_matrix_with_fp_fn.png",
    "fp_fn_matrix.json",
    "fp_fn_matrix.csv",
    "fp_fn_matrix.png",
    "per_class_pr_curves.png",
    "ece_summary.json",
    "ece_summary.csv",
    "ece_summary.png",
    "reliability_bins.csv",
    "reliability_diagram.png",
    "recall_by_iou_threshold.json",
    "recall_by_iou_threshold.csv",
    "recall_by_iou_threshold.png",
    "per_class_background_fp.json",
    "per_class_background_fp.csv",
    "per_class_background_fp.png",
    "quality_diagnostics.json",
    "quality_diagnostics.png",
    "quality_distribution_summary.json",
    "quality_distribution_summary.png",
    "quality_spearman_summary.json",
    "quality_spearman_summary.csv",
    "quality_spearman_summary.png",
    "quality_positive_negative_ecdf.png",
    "quality_tp_background_fp_ecdf.png",
    "predictions_bbox_summary.json",
    "predictions_bbox_summary.csv",
    "predictions_bbox_summary.png",
    "raw_query_class_argmax_distribution.json",
    "evaluation_config.json",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _config_tree_digest() -> str:
    """Hash YAML contents so inherited-config edits invalidate inference cache."""
    digest = hashlib.sha256()
    config_root = ROOT / "configs"
    for path in sorted(config_root.rglob("*.yml")):
        digest.update(path.relative_to(config_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value)


def _load_coco(path: Path) -> Dict[str, Any]:
    raw = _load_json(path)
    categories = sorted(raw.get("categories", []), key=lambda item: int(item["id"]))
    images = {int(item["id"]): item for item in raw.get("images", [])}
    annotations = []
    by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for index, ann in enumerate(raw.get("annotations", [])):
        if int(ann.get("iscrowd", 0)):
            continue
        x, y, width, height = (float(v) for v in ann["bbox"])
        if width <= 0 or height <= 0:
            continue
        item = {
            **ann,
            "id": int(ann.get("id", index + 1)),
            "image_id": int(ann["image_id"]),
            "category_id": int(ann["category_id"]),
            "bbox_xywh": [x, y, width, height],
            "bbox_xyxy": [x, y, x + width, y + height],
        }
        annotations.append(item)
        by_image[item["image_id"]].append(item)
    return {
        "path": path,
        "raw": raw,
        "categories": categories,
        "cat_ids": [int(item["id"]) for item in categories],
        "id_to_name": {int(item["id"]): str(item.get("name", item["id"])) for item in categories},
        "images": images,
        "annotations": annotations,
        "by_image": by_image,
    }


def _checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        ema = checkpoint["ema"]
        state = ema.get("module", ema) if isinstance(ema, dict) else ema
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    return {key.removeprefix("module."): value for key, value in state.items()}


def _build_cfg(
    config_path: Path, cfg: Dict[str, Any], experiment: Optional[Dict[str, Any]] = None
) -> Any:
    from engine.core import YAMLConfig
    from engine.core.yaml_utils import merge_dict

    update = {
        "val_dataloader": {
            "total_batch_size": int(cfg["batch_size"]),
            "num_workers": int(cfg["num_workers"]),
            "dataset": {
                "img_folder": str(_resolve(cfg["images_dir"])),
                "ann_file": str(_resolve(cfg["ann_file"])),
            },
        }
    }
    if experiment and experiment.get("config_overrides"):
        update = merge_dict(update, experiment["config_overrides"], inplace=False)
    result = YAMLConfig(str(config_path), **update)
    if "HGNetv2" in result.yaml_cfg:
        result.yaml_cfg["HGNetv2"]["pretrained"] = False
    return result


def _validation_settings(
    experiment: Dict[str, Any], cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Return only settings that must be identical for a fair evaluation suite."""
    cfg_obj = _build_cfg(_resolve(experiment["config"]), cfg, experiment)
    yaml_cfg = cfg_obj.yaml_cfg
    keys = (
        "task",
        "num_classes",
        "remap_mscoco_category",
        "use_focal_loss",
        "eval_spatial_size",
        "val_dataloader",
        "evaluator",
        "PostProcessor",
    )
    return {key: yaml_cfg.get(key) for key in keys}


def _assert_shared_validation_settings(
    experiments: Sequence[Dict[str, Any]], cfg: Dict[str, Any], output_root: Path
) -> None:
    snapshots = {
        str(experiment["name"]): _validation_settings(experiment, cfg)
        for experiment in experiments
    }
    _write_json(output_root / "validation_settings.json", snapshots)
    canonical = {
        name: json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        for name, value in snapshots.items()
    }
    if len(set(canonical.values())) > 1:
        raise RuntimeError(
            "启用实验的有效验证参数不一致；已写入 validation_settings.json。"
            "请统一 val_dataloader、eval_spatial_size、类别数和 PostProcessor 后重试。"
        )


def _label_maps(cat_ids: Sequence[int], num_classes: int, explicit: Any) -> Tuple[Dict[int, int], Dict[int, int]]:
    if explicit is not None:
        label_to_cat = {int(key): int(value) for key, value in explicit.items()}
    elif cat_ids and min(cat_ids) >= 0 and max(cat_ids) < num_classes:
        label_to_cat = {int(cat_id): int(cat_id) for cat_id in cat_ids}
    elif len(cat_ids) == num_classes:
        label_to_cat = {index: int(cat_id) for index, cat_id in enumerate(cat_ids)}
    else:
        raise ValueError(
            f"无法自动映射 model labels 与 COCO category IDs：cat_ids={list(cat_ids)}, "
            f"num_classes={num_classes}。请设置 model_label_to_category_id。"
        )
    cat_to_label = {cat_id: label for label, cat_id in label_to_cat.items()}
    missing = sorted(set(cat_ids) - set(cat_to_label))
    if missing:
        raise ValueError(f"model_label_to_category_id 缺少 COCO 类别：{missing}")
    return label_to_cat, cat_to_label


def _target_for_matcher(
    annotations: Sequence[Dict[str, Any]], image: Dict[str, Any], cat_to_label: Dict[int, int], device: torch.device
) -> Dict[str, torch.Tensor]:
    width, height = float(image["width"]), float(image["height"])
    boxes = []
    labels = []
    for ann in annotations:
        x, y, box_width, box_height = ann["bbox_xywh"]
        boxes.append([(x + box_width / 2) / width, (y + box_height / 2) / height, box_width / width, box_height / height])
        labels.append(cat_to_label[int(ann["category_id"])])
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64, device=device),
    }


def _postprocess_with_metadata(
    outputs: Dict[str, torch.Tensor], original_sizes: torch.Tensor, postprocessor: Any
) -> List[List[Dict[str, Any]]]:
    from engine.deim.box_ops import box_cxcywh_to_xyxy

    logits = outputs["pred_logits"]
    boxes = outputs["pred_boxes"]
    quality = torch.sigmoid(outputs["pred_quality"].squeeze(-1)) if "pred_quality" in outputs else None
    xyxy = box_cxcywh_to_xyxy(boxes)
    scale = original_sizes[:, None, :].repeat(1, 1, 2)
    xyxy = xyxy * scale
    limit = int(postprocessor.num_top_queries)
    rows: List[List[Dict[str, Any]]] = []

    if postprocessor.use_focal_loss:
        class_scores = torch.sigmoid(logits)
        final_scores = class_scores
        k = min(limit, final_scores.shape[1] * final_scores.shape[2])
        scores, indices = torch.topk(final_scores.flatten(1), k=k, dim=1)
        labels = indices % final_scores.shape[2]
        query_indices = indices // final_scores.shape[2]
    else:
        class_scores = torch.softmax(logits, dim=-1)[..., :-1]
        raw_scores, labels = class_scores.max(-1)
        final_scores = raw_scores
        k = min(limit, final_scores.shape[1])
        scores, query_indices = torch.topk(final_scores, k=k, dim=1)
        labels = torch.gather(labels, 1, query_indices)

    for batch_index in range(logits.shape[0]):
        image_rows = []
        for score, label, query_index in zip(scores[batch_index], labels[batch_index], query_indices[batch_index]):
            q = int(query_index.item())
            c = int(label.item())
            image_rows.append({
                "query_index": q,
                "model_label": c,
                "score": float(score.item()),
                "class_score": float(class_scores[batch_index, q, c].item()),
                "quality": None if quality is None else float(quality[batch_index, q].item()),
                "bbox_xyxy": [float(v) for v in xyxy[batch_index, q].tolist()],
            })
        rows.append(image_rows)
    return rows


def _pair_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    from engine.deim.box_ops import box_iou

    return box_iou(boxes1, boxes2)[0] if boxes1.numel() and boxes2.numel() else boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))


def _install_lqe_quality_hooks(
    model: torch.nn.Module,
) -> Tuple[Dict[str, Any], List[Any]]:
    """捕获 D-FINE 最终活动 LQE 的独立质量 logit，不改变模型前向。"""
    state: Dict[str, Any] = {
        "tensor": None,
        "source": None,
        "candidates": [],
    }
    handles = []

    def make_hook(name: str):
        def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
            if isinstance(output, torch.Tensor) and output.ndim == 3 and output.shape[-1] == 1:
                state["tensor"] = output.detach()
                state["source"] = name
        return hook

    for name, module in model.named_modules():
        if ".lqe_layers." not in name or not name.endswith(".reg_conf"):
            continue
        state["candidates"].append(name)
        handles.append(module.register_forward_hook(make_hook(name)))
    return state, handles


@torch.no_grad()
def _run_inference(
    experiment: Dict[str, Any], cfg: Dict[str, Any], coco: Dict[str, Any], output_dir: Path
) -> Dict[str, Any]:
    from engine.deim.box_ops import box_cxcywh_to_xyxy

    config_path = _resolve(experiment["config"])
    checkpoint_path = _resolve(experiment["checkpoint"])
    signature = {
        "cache_version": 4,
        "config": str(config_path),
        "config_mtime_ns": config_path.stat().st_mtime_ns if config_path.exists() else None,
        "config_tree_sha256": _config_tree_digest(),
        "config_overrides": experiment.get("config_overrides", {}),
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": checkpoint_path.stat().st_size if checkpoint_path.exists() else None,
        "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns if checkpoint_path.exists() else None,
        "ann_file": str(_resolve(cfg["ann_file"])),
        "ann_file_size": _resolve(cfg["ann_file"]).stat().st_size,
        "ann_file_mtime_ns": _resolve(cfg["ann_file"]).stat().st_mtime_ns,
        "images_dir": str(_resolve(cfg["images_dir"])),
        "quality_extractor": "dfine_lqe_reg_conf_sigmoid_v1",
    }
    cache_path = output_dir / "inference_cache.pt"
    if bool(cfg["reuse_inference_cache"]) and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if cached.get("cache_signature") == signature:
            print(f"[CACHE] {experiment['name']}: {cache_path}")
            return cached
        print(f"[CACHE] {experiment['name']}: 配置/checkpoint 已变化，重新推理。")
    if not config_path.exists():
        raise FileNotFoundError(f"配置不存在：{config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{checkpoint_path}")

    cfg_obj = _build_cfg(config_path, cfg, experiment)
    device_name = str(cfg["device"])
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("配置请求 CUDA，但 torch.cuda.is_available() 为 False。")
    device = torch.device(device_name)
    model = cfg_obj.model
    missing, unexpected = model.load_state_dict(_checkpoint_state(checkpoint_path), strict=False)
    # Evaluation requires a checkpoint from the exact selected experiment.
    # Warm-start exceptions belong only to load_s0_for_sidecar.py, not testing.
    allowed_missing: List[str] = []
    illegal_missing = list(missing)
    if missing or unexpected:
        print(f"[LOAD] missing={len(missing)}, unexpected={len(unexpected)}")
        if cfg.get("strict_checkpoint_structure"):
            if illegal_missing or unexpected:
                raise RuntimeError(
                    "checkpoint 与实验配置结构不一致："
                    f"illegal_missing={list(illegal_missing)[:20]}, "
                    f"unexpected={list(unexpected)[:20]}"
                )
    model = model.to(device).eval()
    postprocessor = cfg_obj.postprocessor.to(device).eval()
    criterion = cfg_obj.criterion.to(device).eval()
    matcher = criterion.matcher
    dataloader = cfg_obj.val_dataloader
    quality_state, quality_handles = _install_lqe_quality_hooks(model)
    if not quality_handles:
        print("[WARN] 未找到 D-FINE LQE reg_conf；将生成 unavailable quality 占位结果。")

    num_classes = int(getattr(postprocessor, "num_classes", cfg_obj.yaml_cfg.get("num_classes", 0)))
    label_to_cat, cat_to_label = _label_maps(
        coco["cat_ids"], num_classes, cfg.get("model_label_to_category_id")
    )
    predictions: List[Dict[str, Any]] = []
    detections: List[Dict[str, Any]] = []
    query_records: List[Dict[str, Any]] = []
    query_argmax_counts: Dict[int, int] = defaultdict(int)
    joint_target_power = float(getattr(criterion, "gamma", 1.0))

    for batch_index, (samples, targets) in enumerate(dataloader):
        samples = samples.to(device)
        quality_state["tensor"] = None
        quality_state["source"] = None
        outputs = model(samples)
        captured_quality = quality_state.get("tensor")
        if (
            isinstance(captured_quality, torch.Tensor)
            and captured_quality.shape[:2] == outputs["pred_logits"].shape[:2]
        ):
            outputs = dict(outputs)
            outputs["pred_quality"] = captured_quality
        original_sizes = torch.stack([target["orig_size"] for target in targets]).to(device).float()
        processed = _postprocess_with_metadata(outputs, original_sizes, postprocessor)
        image_ids = [int(target["image_id"].reshape(-1)[0].item()) for target in targets]
        match_targets = [
            _target_for_matcher(coco["by_image"].get(image_id, []), coco["images"][image_id], cat_to_label, device)
            for image_id in image_ids
        ]
        if sum(target["labels"].numel() for target in match_targets):
            match_result = matcher(
                {"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]},
                match_targets,
            )
            indices = match_result["indices"]
        else:
            indices = [
                (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
                for _ in image_ids
            ]

        quality_tensor = torch.sigmoid(outputs["pred_quality"].squeeze(-1)) if "pred_quality" in outputs else None
        if postprocessor.use_focal_loss:
            class_scores = torch.sigmoid(outputs["pred_logits"])
        else:
            class_scores = torch.softmax(outputs["pred_logits"], dim=-1)[..., :-1]
        class_confidence, class_argmax = class_scores.max(-1)
        for model_label in class_argmax.reshape(-1).tolist():
            query_argmax_counts[int(model_label)] += 1

        for local_index, image_id in enumerate(image_ids):
            image = coco["images"][image_id]
            annotations = coco["by_image"].get(image_id, [])
            for row in processed[local_index]:
                cat_id = label_to_cat.get(int(row["model_label"]))
                if cat_id not in coco["id_to_name"]:
                    continue
                x1, y1, x2, y2 = row["bbox_xyxy"]
                record = {
                    "image_id": image_id,
                    "category_id": int(cat_id),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": row["score"],
                }
                predictions.append(record)
                detections.append({**record, **row, "category_id": int(cat_id)})

            if quality_tensor is None:
                continue
            gt_boxes = match_targets[local_index]["boxes"]
            query_boxes = outputs["pred_boxes"][local_index]
            best_iou = _pair_iou_xyxy(box_cxcywh_to_xyxy(query_boxes), box_cxcywh_to_xyxy(gt_boxes)).amax(-1) \
                if gt_boxes.numel() else query_boxes.new_zeros((query_boxes.shape[0],))
            src_indices = indices[local_index][0].to(device)
            gt_indices = indices[local_index][1].to(device)
            positive = torch.zeros(query_boxes.shape[0], dtype=torch.bool, device=device)
            positive[src_indices] = True
            matched_iou = query_boxes.new_zeros((query_boxes.shape[0],))
            joint_target = query_boxes.new_zeros((query_boxes.shape[0],))
            matched_gt_ids = [-1] * query_boxes.shape[0]
            if src_indices.numel():
                pair_iou = torch.diag(_pair_iou_xyxy(
                    box_cxcywh_to_xyxy(query_boxes[src_indices]),
                    box_cxcywh_to_xyxy(gt_boxes[gt_indices]),
                ))
                # DEIM MAL 的正 query 分类 target 是匹配 IoU 的 gamma 次幂。
                # It is the current LABS-Mamba experiment's assignment joint target.
                target = pair_iou.clamp(min=0.0, max=1.0).pow(joint_target_power)
                matched_iou[src_indices] = pair_iou
                joint_target[src_indices] = target
                for src, gt in zip(src_indices.tolist(), gt_indices.tolist()):
                    matched_gt_ids[src] = int(annotations[gt]["id"])

            for query_index in range(query_boxes.shape[0]):
                query_records.append({
                    "image_id": image_id,
                    "query_index": query_index,
                    "quality": float(quality_tensor[local_index, query_index].item()),
                    "class_confidence": float(class_confidence[local_index, query_index].item()),
                    "class_argmax_model_label": int(class_argmax[local_index, query_index].item()),
                    "class_argmax_category_id": label_to_cat.get(
                        int(class_argmax[local_index, query_index].item())
                    ),
                    "is_positive": int(positive[query_index].item()),
                    "matched_gt_id": matched_gt_ids[query_index],
                    "matched_iou": float(matched_iou[query_index].item()),
                    "best_iou": float(best_iou[query_index].item()),
                    "joint_target": float(joint_target[query_index].item()),
                })
        print(f"[INFER] {experiment['name']} batch={batch_index + 1}/{len(dataloader)}")

    for handle in quality_handles:
        handle.remove()

    query_argmax_total = sum(query_argmax_counts.values())
    query_argmax_distribution = {
        "total_queries": query_argmax_total,
        "rows": [
            {
                "model_label": model_label,
                "category_id": label_to_cat.get(model_label),
                "class_name": coco["id_to_name"].get(
                    label_to_cat.get(model_label), "UNKNOWN_{}".format(model_label)
                ),
                "count": int(query_argmax_counts.get(model_label, 0)),
                "share": (
                    query_argmax_counts.get(model_label, 0) / query_argmax_total
                    if query_argmax_total else None
                ),
            }
            for model_label in sorted(label_to_cat)
        ],
    }
    result = {
        "cache_signature": signature,
        "predictions": predictions,
        "detections": detections,
        "query_records": query_records,
        "raw_query_class_argmax_distribution": query_argmax_distribution,
        "quality_available": bool(query_records),
        "quality_source": (
            "sigmoid(final active D-FINE LQE reg_conf additive logit)"
            if query_records else None
        ),
        "quality_hook_module": quality_state.get("source"),
        "quality_hook_candidates": quality_state.get("candidates", []),
        "joint_target_definition": "Hungarian-matched IoU ** criterion.gamma",
        "joint_target_power": joint_target_power,
        "label_to_category_id": label_to_cat,
        "missing_keys": list(missing),
        "allowed_missing_keys": allowed_missing,
        "illegal_missing_keys": illegal_missing,
        "unexpected_keys": list(unexpected),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result, cache_path)
    _write_json(output_dir / "predictions.json", predictions)
    _write_json(
        output_dir / "raw_query_class_argmax_distribution.json",
        query_argmax_distribution,
    )
    if cfg.get("save_query_records_csv") and query_records:
        _write_csv(output_dir / "quality_query_records.csv", query_records)
    return result


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    union += max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - intersection
    return 0.0 if union <= 0 else intersection / union


def _prediction_objects(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for index, row in enumerate(rows):
        x, y, width, height = (float(v) for v in row["bbox"])
        result.append({
            **row,
            "id": index,
            "image_id": int(row["image_id"]),
            "category_id": int(row["category_id"]),
            "bbox_xyxy": [x, y, x + width, y + height],
            "score": float(row["score"]),
        })
    return result


def _group_by_image(rows: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    result: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[int(row["image_id"])].append(row)
    return result


def _annotate_detections(
    predictions: List[Dict[str, Any]], gts: List[Dict[str, Any]], score_threshold: float,
    tp_iou: float, background_iou: float,
) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    gt_by_image = _group_by_image(gts)
    pred_by_image = _group_by_image(predictions)
    rows = []
    fn_by_class: Dict[int, int] = defaultdict(int)
    for image_id in sorted(set(gt_by_image) | set(pred_by_image)):
        image_gts = gt_by_image.get(image_id, [])
        used: set[int] = set()
        for pred in sorted(pred_by_image.get(image_id, []), key=lambda item: item["score"], reverse=True):
            if pred["score"] < score_threshold:
                continue
            same_class = [
                (index, gt) for index, gt in enumerate(image_gts)
                if gt["category_id"] == pred["category_id"] and index not in used
            ]
            best_index, best_gt, same_iou = -1, None, 0.0
            for index, gt in same_class:
                overlap = _iou(pred["bbox_xyxy"], gt["bbox_xyxy"])
                if overlap > same_iou:
                    best_index, best_gt, same_iou = index, gt, overlap
            is_tp = best_gt is not None and same_iou >= tp_iou
            if is_tp:
                used.add(best_index)
            max_any_iou = max((_iou(pred["bbox_xyxy"], gt["bbox_xyxy"]) for gt in image_gts), default=0.0)
            rows.append({
                **pred,
                "is_tp": int(is_tp),
                "matched_iou": same_iou if is_tp else 0.0,
                "max_any_iou": max_any_iou,
                "is_background_fp": int(not is_tp and max_any_iou < background_iou),
            })
        for index, gt in enumerate(image_gts):
            if index not in used:
                fn_by_class[int(gt["category_id"])] += 1
    return rows, dict(fn_by_class)


def _require_matplotlib():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _table_image(path: Path, rows: Sequence[Dict[str, Any]], title: str, fields: Sequence[str]) -> None:
    plt = _require_matplotlib()

    def format_value(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value)

    display = []
    for row in rows:
        display.append([format_value(row.get(field)) for field in fields])
    height = max(3.0, 0.36 * (len(display) + 2))
    fig, axis = plt.subplots(figsize=(max(8, 1.35 * len(fields)), height))
    axis.axis("off")
    axis.set_title(title)
    table = axis.table(cellText=display, colLabels=list(fields), loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    for (row_index, _column_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#365f91")
            cell.set_text_props(color="white", weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#edf3f8")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _predictions_bbox_summary(
    predictions: Sequence[Dict[str, Any]], coco: Dict[str, Any], output_dir: Path
) -> List[Dict[str, Any]]:
    """Write per-class prediction bbox statistics as machine-readable data and a table image."""
    rows = []
    groups: List[Tuple[Any, str, List[Dict[str, Any]]]] = [
        ("all", "ALL_CLASSES", list(predictions))
    ]
    groups.extend(
        (
            cat_id,
            coco["id_to_name"][cat_id],
            [row for row in predictions if int(row["category_id"]) == cat_id],
        )
        for cat_id in coco["cat_ids"]
    )
    for cat_id, class_name, selected in groups:
        widths = np.asarray([float(row["bbox"][2]) for row in selected], dtype=float)
        heights = np.asarray([float(row["bbox"][3]) for row in selected], dtype=float)
        areas = widths * heights
        aspects = np.divide(widths, heights, out=np.zeros_like(widths), where=heights > 0)

        def mean(values: np.ndarray) -> Optional[float]:
            return float(values.mean()) if values.size else None

        def median(values: np.ndarray) -> Optional[float]:
            return float(np.median(values)) if values.size else None

        rows.append({
            "category_id": cat_id,
            "class_name": class_name,
            "detections": len(selected),
            "width_mean": mean(widths),
            "width_median": median(widths),
            "height_mean": mean(heights),
            "height_median": median(heights),
            "area_mean": mean(areas),
            "area_median": median(areas),
            "aspect_mean": mean(aspects),
            "aspect_median": median(aspects),
        })

    _write_json(output_dir / "predictions_bbox_summary.json", rows)
    _write_csv(output_dir / "predictions_bbox_summary.csv", rows)
    _table_image(
        output_dir / "predictions_bbox_summary.png",
        rows,
        "Prediction bbox summary (original-image pixels)",
        [
            "class_name", "detections", "width_mean", "width_median",
            "height_mean", "height_median", "area_mean", "area_median",
            "aspect_mean", "aspect_median",
        ],
    )
    return rows


def _run_coco_metrics(
    ann_file: Path, predictions: List[Dict[str, Any]], coco: Dict[str, Any], cfg: Dict[str, Any], output_dir: Path
) -> Dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not predictions:
        raise RuntimeError("预测结果为空，无法运行 COCOeval。")
    coco_gt = COCO(str(ann_file))
    valid_image_ids = set(coco_gt.getImgIds())
    records = [
        {key: row[key] for key in ("image_id", "category_id", "bbox", "score")}
        for row in predictions if int(row["image_id"]) in valid_image_ids
    ]
    coco_dt = coco_gt.loadRes(records)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.catIds = coco["cat_ids"]
    evaluator.params.imgIds = sorted(valid_image_ids)
    evaluator.params.maxDets[-1] = int(cfg["max_dets"])
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    names = ["AP", "AP50", "AP75", "APS", "APM", "APL", "AR1", "AR10", "AR100", "ARS", "ARM", "ARL"]
    overall = {name: float(value) for name, value in zip(names, evaluator.stats.tolist())}
    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    params = evaluator.params
    area_index = list(params.areaRngLbl).index("all")
    max_det_index = len(params.maxDets) - 1
    iou50 = int(np.argmin(np.abs(params.iouThrs - 0.50)))
    iou75 = int(np.argmin(np.abs(params.iouThrs - 0.75)))
    pr_iou = int(np.argmin(np.abs(params.iouThrs - float(cfg["pr_iou_threshold"]))))

    def mean_valid(value: Any) -> Optional[float]:
        array = np.asarray(value, dtype=float)
        array = array[array >= 0]
        return None if array.size == 0 else float(array.mean())

    per_class = []
    pr_rows: Dict[int, List[Dict[str, float]]] = {}
    for class_index, cat_id in enumerate(coco["cat_ids"]):
        row = {
            "category_id": cat_id,
            "class_name": coco["id_to_name"][cat_id],
            "AP": mean_valid(precision[:, :, class_index, area_index, max_det_index]),
            "AP50": mean_valid(precision[iou50, :, class_index, area_index, max_det_index]),
            "AP75": mean_valid(precision[iou75, :, class_index, area_index, max_det_index]),
            "AR100": mean_valid(recall[:, class_index, area_index, max_det_index]),
        }
        per_class.append(row)
        values = precision[pr_iou, :, class_index, area_index, max_det_index]
        pr_rows[cat_id] = [
            {"recall": float(rec), "precision": float(pre)}
            for rec, pre in zip(params.recThrs, values) if pre >= 0
        ]

    combined = {
        "meta": {"ann_file": str(ann_file), "max_dets": int(cfg["max_dets"])},
        "overall": overall,
        "per_class": per_class,
    }
    _write_json(output_dir / "class_and_all_metrics.json", combined)
    _write_json(output_dir / "per_class_ap.json", per_class)
    table_rows = [{"class_name": "ALL", **overall}] + per_class
    _table_image(
        output_dir / "class_and_all_metrics.png", table_rows, "COCO overall and per-class metrics",
        ["class_name", "AP", "AP50", "AP75", "AR100"],
    )

    _table_image(
        output_dir / "per_class_ap.png",
        per_class,
        "Per-class COCO AP (higher is better)",
        ["category_id", "class_name", "AP", "AP50", "AP75", "AR100"],
    )

    plt = _require_matplotlib()
    pr_dir = output_dir / "per_class_pr"
    pr_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(7, 6))
    for cat_id in coco["cat_ids"]:
        rows = pr_rows[cat_id]
        _write_csv(pr_dir / f"{cat_id}_{_safe_name(coco['id_to_name'][cat_id])}.csv", rows)
        class_fig, class_axis = plt.subplots(figsize=(6, 5))
        class_axis.plot([row["recall"] for row in rows], [row["precision"] for row in rows], linewidth=2)
        class_axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
        class_axis.set_title(f"{coco['id_to_name'][cat_id]} PR @ IoU={params.iouThrs[pr_iou]:.2f}")
        class_axis.grid(alpha=0.2)
        class_fig.tight_layout()
        class_fig.savefig(pr_dir / f"{cat_id}_{_safe_name(coco['id_to_name'][cat_id])}.png", dpi=220)
        plt.close(class_fig)
        axis.plot([row["recall"] for row in rows], [row["precision"] for row in rows], label=coco["id_to_name"][cat_id])
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
    axis.set_title(f"Per-class PR curves @ IoU={params.iouThrs[pr_iou]:.2f}")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "per_class_pr_curves.png", dpi=240)
    plt.close(fig)
    per_class_pr = [
        {
            "category_id": cat_id,
            "class_name": coco["id_to_name"][cat_id],
            "iou_threshold": float(params.iouThrs[pr_iou]),
            "points": pr_rows[cat_id],
        }
        for cat_id in coco["cat_ids"]
    ]
    return {
        "evaluator": evaluator,
        "overall": overall,
        "per_class": per_class,
        "per_class_pr": per_class_pr,
    }


def _matrix_csv(path: Path, matrix: np.ndarray, row_names: Sequence[str], col_names: Sequence[str]) -> None:
    rows = []
    for name, values in zip(row_names, matrix.tolist()):
        rows.append({"GT\\Prediction": name, **{column: value for column, value in zip(col_names, values)}})
    _write_csv(path, rows)


def _plot_matrix(path: Path, matrix: np.ndarray, rows: Sequence[str], cols: Sequence[str], title: str) -> None:
    plt = _require_matplotlib()
    fig, axis = plt.subplots(figsize=(max(8, len(cols) * 0.85), max(7, len(rows) * 0.7)))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    axis.set_yticks(range(len(rows)), rows)
    axis.set(xlabel="Prediction", ylabel="Ground truth", title=title)
    threshold = float(matrix.max()) * 0.55 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(int(matrix[row, col])), ha="center", va="center",
                      color="white" if matrix[row, col] > threshold else "black", fontsize=8)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=230)
    plt.close(fig)


def _confusion_analysis(
    gts: List[Dict[str, Any]], predictions: List[Dict[str, Any]], coco: Dict[str, Any], cfg: Dict[str, Any], output_dir: Path
) -> Dict[str, Any]:
    cat_ids = coco["cat_ids"]
    index = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
    count = len(cat_ids)
    core = np.zeros((count, count), dtype=np.int64)
    full = np.zeros((count + 1, count + 1), dtype=np.int64)
    summary = {cat_id: {"TP": 0, "FP": 0, "FN": 0, "background_FP": 0, "class_confusion_FP": 0, "class_confusion_FN": 0} for cat_id in cat_ids}
    gt_by_image = _group_by_image(gts)
    pred_by_image = _group_by_image(predictions)
    score_threshold = float(cfg["confusion_score_threshold"])
    iou_threshold = float(cfg["confusion_iou_threshold"])

    for image_id in sorted(set(gt_by_image) | set(pred_by_image)):
        image_gts = gt_by_image.get(image_id, [])
        image_preds = [row for row in pred_by_image.get(image_id, []) if row["score"] >= score_threshold]
        pairs = []
        for gt_index, gt in enumerate(image_gts):
            for pred_index, pred in enumerate(image_preds):
                overlap = _iou(gt["bbox_xyxy"], pred["bbox_xyxy"])
                if overlap >= iou_threshold:
                    pairs.append((overlap, pred["score"], gt_index, pred_index))
        pairs.sort(reverse=True)
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        for _, _, gt_index, pred_index in pairs:
            if gt_index in used_gt or pred_index in used_pred:
                continue
            used_gt.add(gt_index)
            used_pred.add(pred_index)
            gt_cat = image_gts[gt_index]["category_id"]
            pred_cat = image_preds[pred_index]["category_id"]
            row, col = index[gt_cat], index[pred_cat]
            core[row, col] += 1
            full[row, col] += 1
            if gt_cat == pred_cat:
                summary[gt_cat]["TP"] += 1
            else:
                summary[gt_cat]["FN"] += 1
                summary[gt_cat]["class_confusion_FN"] += 1
                summary[pred_cat]["FP"] += 1
                summary[pred_cat]["class_confusion_FP"] += 1
        for gt_index, gt in enumerate(image_gts):
            if gt_index not in used_gt:
                full[index[gt["category_id"]], count] += 1
                summary[gt["category_id"]]["FN"] += 1
        for pred_index, pred in enumerate(image_preds):
            if pred_index not in used_pred:
                full[count, index[pred["category_id"]]] += 1
                summary[pred["category_id"]]["FP"] += 1
                summary[pred["category_id"]]["background_FP"] += 1

    names = [coco["id_to_name"][cat_id] for cat_id in cat_ids]
    _matrix_csv(output_dir / "confusion_matrix.csv", core, names, names)
    _matrix_csv(output_dir / "confusion_matrix_with_fp_fn.csv", full, names + ["BACKGROUND_FP"], names + ["BACKGROUND_FN"])
    _plot_matrix(output_dir / "confusion_matrix.png", core, names, names, "Class confusion matrix")
    _plot_matrix(
        output_dir / "confusion_matrix_with_fp_fn.png", full,
        names + ["BACKGROUND_FP"], names + ["BACKGROUND_FN"], "Confusion matrix with FP/FN",
    )
    rows = [{"category_id": cat_id, "class_name": coco["id_to_name"][cat_id], **summary[cat_id]} for cat_id in cat_ids]
    _write_json(output_dir / "fp_fn_matrix.json", rows)
    _write_csv(output_dir / "fp_fn_matrix.csv", rows)
    _table_image(
        output_dir / "fp_fn_matrix.png",
        rows,
        "Per-class TP / FP / FN",
        [
            "category_id", "class_name", "TP", "FP", "FN",
            "background_FP", "class_confusion_FP", "class_confusion_FN",
        ],
    )
    return {"core": core.tolist(), "full": full.tolist(), "per_class": rows}


def _calibration_bins(rows: Sequence[Dict[str, Any]], num_bins: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not rows:
        return {"count": 0, "ECE": None, "LaECE": None, "accuracy": None, "mean_confidence": None}, []
    result = []
    ece = 0.0
    laece = 0.0
    total = len(rows)
    for index in range(num_bins):
        lower, upper = index / num_bins, (index + 1) / num_bins
        selected = [row for row in rows if lower <= row["score"] < upper or (index == num_bins - 1 and row["score"] == 1.0)]
        count = len(selected)
        confidence = float(np.mean([row["score"] for row in selected])) if count else None
        accuracy = float(np.mean([row["is_tp"] for row in selected])) if count else None
        localization = float(np.mean([row["matched_iou"] for row in selected])) if count else None
        if count:
            ece += count / total * abs(confidence - accuracy)
            laece += count / total * abs(confidence - localization)
        result.append({
            "bin_index": index, "lower": lower, "upper": upper, "count": count,
            "mean_confidence": confidence, "accuracy": accuracy, "localization_accuracy": localization,
        })
    summary = {
        "count": total,
        "ECE": ece,
        "LaECE": laece,
        "accuracy": float(np.mean([row["is_tp"] for row in rows])),
        "mean_confidence": float(np.mean([row["score"] for row in rows])),
    }
    return summary, result


def _ece_analysis(
    annotated: List[Dict[str, Any]], coco: Dict[str, Any], cfg: Dict[str, Any], output_dir: Path
) -> Dict[str, Any]:
    selected = [row for row in annotated if row["score"] >= float(cfg["ece_score_threshold"])]
    summaries = []
    bin_rows = []
    groups: List[Tuple[Any, str, List[Dict[str, Any]]]] = [("all", "ALL_CLASSES", selected)]
    groups.extend((cat_id, coco["id_to_name"][cat_id], [row for row in selected if row["category_id"] == cat_id]) for cat_id in coco["cat_ids"])
    for cat_id, name, rows in groups:
        summary, bins = _calibration_bins(rows, int(cfg["ece_num_bins"]))
        summaries.append({"category_id": cat_id, "class_name": name, **summary})
        bin_rows.extend({"category_id": cat_id, "class_name": name, **row} for row in bins)
    payload = {
        "definition": "prediction-level ECE and localization-aware ECE",
        "rows": summaries,
        "bins": bin_rows,
    }
    _write_json(output_dir / "ece_summary.json", payload)
    _write_csv(output_dir / "ece_summary.csv", summaries)
    _write_csv(output_dir / "reliability_bins.csv", bin_rows)

    _table_image(
        output_dir / "ece_summary.png",
        summaries,
        "Calibration error (ECE/LaECE lower is better)",
        ["category_id", "class_name", "count", "ECE", "LaECE", "accuracy", "mean_confidence"],
    )

    plt = _require_matplotlib()
    overall_bins = [row for row in bin_rows if row["category_id"] == "all" and row["count"]]
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axis.plot([row["mean_confidence"] for row in overall_bins], [row["accuracy"] for row in overall_bins], "o-", label="Accuracy")
    axis.plot([row["mean_confidence"] for row in overall_bins], [row["localization_accuracy"] for row in overall_bins], "s-", label="Localization accuracy")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Observed quality", title="Reliability diagram")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "reliability_diagram.png", dpi=230)
    plt.close(fig)
    return payload


def _count_recall_tp(
    gts: Sequence[Dict[str, Any]], predictions: Sequence[Dict[str, Any]], iou_threshold: float, category_id: Optional[int]
) -> Tuple[int, int]:
    selected_gts = [row for row in gts if category_id is None or row["category_id"] == category_id]
    selected_preds = [row for row in predictions if category_id is None or row["category_id"] == category_id]
    gt_by_image = _group_by_image(selected_gts)
    pred_by_image = _group_by_image(selected_preds)
    tp = 0
    for image_id, image_gts in gt_by_image.items():
        used: set[int] = set()
        for pred in sorted(pred_by_image.get(image_id, []), key=lambda item: item["score"], reverse=True):
            candidates = [
                (index, gt) for index, gt in enumerate(image_gts)
                if index not in used and gt["category_id"] == pred["category_id"]
            ]
            best_index, best_iou = -1, 0.0
            for index, gt in candidates:
                overlap = _iou(pred["bbox_xyxy"], gt["bbox_xyxy"])
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
            if best_index >= 0 and best_iou >= iou_threshold:
                used.add(best_index)
                tp += 1
    return tp, len(selected_gts)


def _recall_analysis(
    gts: List[Dict[str, Any]], predictions: List[Dict[str, Any]], coco: Dict[str, Any], cfg: Dict[str, Any], output_dir: Path
) -> List[Dict[str, Any]]:
    selected = [row for row in predictions if row["score"] >= float(cfg["recall_score_threshold"])]
    rows = []
    for threshold in cfg["recall_iou_thresholds"]:
        for cat_id in [None] + coco["cat_ids"]:
            tp, total = _count_recall_tp(gts, selected, float(threshold), cat_id)
            rows.append({
                "category_id": "all" if cat_id is None else cat_id,
                "class_name": "ALL_CLASSES" if cat_id is None else coco["id_to_name"][cat_id],
                "iou_threshold": float(threshold), "gt_instances": total, "true_positives": tp,
                "recall": tp / total if total else None,
            })
    _write_json(output_dir / "recall_by_iou_threshold.json", rows)
    _write_csv(output_dir / "recall_by_iou_threshold.csv", rows)
    plt = _require_matplotlib()
    fig, axis = plt.subplots(figsize=(8, 6))
    for cat_id in ["all"] + coco["cat_ids"]:
        values = [row for row in rows if row["category_id"] == cat_id]
        axis.plot([row["iou_threshold"] for row in values], [row["recall"] or 0.0 for row in values], marker="o", label=values[0]["class_name"])
    axis.set(xlabel="IoU threshold", ylabel="Recall", ylim=(0, 1), title="Recall by IoU threshold")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "recall_by_iou_threshold.png", dpi=230)
    plt.close(fig)
    return rows


def _background_fp_analysis(
    annotated: List[Dict[str, Any]], coco: Dict[str, Any], cfg: Dict[str, Any], output_dir: Path
) -> List[Dict[str, Any]]:
    threshold = float(cfg["background_fp_score_threshold"])
    rows = []
    for cat_id in coco["cat_ids"]:
        class_rows = [row for row in annotated if row["category_id"] == cat_id and row["score"] >= threshold]
        background = [row for row in class_rows if row["is_background_fp"]]
        rows.append({
            "category_id": cat_id,
            "class_name": coco["id_to_name"][cat_id],
            "detections": len(class_rows),
            "background_fp": len(background),
            "background_fp_rate": len(background) / len(class_rows) if class_rows else None,
        })
    _write_json(output_dir / "per_class_background_fp.json", rows)
    _write_csv(output_dir / "per_class_background_fp.csv", rows)
    _table_image(
        output_dir / "per_class_background_fp.png",
        rows,
        f"Background FP per class (score >= {threshold:.2f}; lower is better)",
        ["category_id", "class_name", "detections", "background_fp", "background_fp_rate"],
    )
    return rows


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    from scipy.stats import spearmanr
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 2 or len({a for a, _ in pairs}) < 2 or len({b for _, b in pairs}) < 2:
        return {"rho": None, "p_value": None, "count": len(pairs)}
    result = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    # SciPy < 1.10 使用 correlation，新版本使用 statistic；二者数值相同。
    statistic = getattr(result, "statistic", getattr(result, "correlation", result[0]))
    p_value = getattr(result, "pvalue", result[1])
    return {"rho": float(statistic), "p_value": float(p_value), "count": len(pairs)}


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray([value for value in values if value is not None and math.isfinite(float(value))], dtype=float)
    if not array.size:
        return {"count": 0, "mean": None, "std": None, "p25": None, "median": None, "p75": None}
    return {
        "count": int(array.size), "mean": float(array.mean()), "std": float(array.std()),
        "p25": float(np.percentile(array, 25)), "median": float(np.median(array)), "p75": float(np.percentile(array, 75)),
    }


def _placeholder_image(path: Path, title: str, message: str) -> None:
    plt = _require_matplotlib()
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _ecdf_image(
    path: Path, series: Sequence[Tuple[str, Sequence[float]]], title: str
) -> None:
    plt = _require_matplotlib()
    fig, axis = plt.subplots(figsize=(7, 5.5))
    has_values = False
    for label, values in series:
        array = np.sort(
            np.asarray(
                [value for value in values if value is not None and math.isfinite(float(value))],
                dtype=float,
            )
        )
        if not array.size:
            continue
        has_values = True
        cumulative = np.arange(1, array.size + 1, dtype=float) / array.size
        axis.step(array, cumulative, where="post", linewidth=2, label=f"{label} (n={array.size})")
    axis.set(
        title=title,
        xlabel="Quality",
        ylabel="Cumulative fraction",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.2)
    if has_values:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "No available samples", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=230)
    plt.close(fig)


def _quality_analysis(
    inference: Dict[str, Any], annotated: List[Dict[str, Any]], output_dir: Path
) -> Dict[str, Any]:
    query_rows = inference["query_records"]
    quality_detections = [row for row in annotated if row.get("quality") is not None]
    if not query_rows:
        payload = {
            "available": False,
            "reason": "未捕获到 D-FINE LQE reg_conf 输出；未使用分类置信度冒充 quality。",
        }
        _write_json(output_dir / "quality_diagnostics.json", payload)
        _placeholder_image(
            output_dir / "quality_diagnostics.png",
            "Quality diagnostics",
            "D-FINE LQE quality signal is unavailable for this experiment.",
        )
        _write_json(output_dir / "quality_distribution_summary.json", [])
        _placeholder_image(
            output_dir / "quality_distribution_summary.png",
            "LQE quality distribution summary",
            "D-FINE LQE quality signal is unavailable for this experiment.",
        )
        unavailable_rows = [
            {"metric": name, "rho": None, "p_value": None, "count": 0}
            for name in (
                "quality_vs_true_iou",
                "quality_vs_joint_target",
                "all_query_quality_vs_best_iou",
                "all_query_quality_vs_assignment_target",
            )
        ]
        _write_json(output_dir / "quality_spearman_summary.json", unavailable_rows)
        _write_csv(output_dir / "quality_spearman_summary.csv", unavailable_rows)
        _table_image(
            output_dir / "quality_spearman_summary.png",
            unavailable_rows,
            "Quality Spearman correlations (LQE unavailable)",
            ["metric", "rho", "p_value", "count"],
        )
        _placeholder_image(
            output_dir / "quality_positive_negative_ecdf.png",
            "Positive / negative query quality",
            "D-FINE LQE quality signal is unavailable for this experiment.",
        )
        _placeholder_image(
            output_dir / "quality_tp_background_fp_ecdf.png",
            "TP / background FP quality",
            "D-FINE LQE quality signal is unavailable for this experiment.",
        )
        return payload

    positives = [row for row in query_rows if row["is_positive"]]
    negatives = [row for row in query_rows if not row["is_positive"]]
    tp_rows = [row for row in quality_detections if row["is_tp"]]
    background_rows = [row for row in quality_detections if row["is_background_fp"]]
    payload = {
        "available": True,
        "quality_vs_true_iou_spearman": _safe_spearman(
            [row["quality"] for row in positives], [row["matched_iou"] for row in positives]
        ),
        "quality_vs_joint_target_spearman": _safe_spearman(
            [row["quality"] for row in positives], [row["joint_target"] for row in positives]
        ),
        "all_query_quality_vs_best_iou_spearman": _safe_spearman(
            [row["quality"] for row in query_rows], [row["best_iou"] for row in query_rows]
        ),
        "all_query_quality_vs_assignment_target_spearman": _safe_spearman(
            [row["quality"] for row in query_rows], [row["joint_target"] for row in query_rows]
        ),
        "distributions": {
            "positive_queries": _distribution([row["quality"] for row in positives]),
            "negative_queries": _distribution([row["quality"] for row in negatives]),
            "true_positive_detections": _distribution([row["quality"] for row in tp_rows]),
            "background_false_positives": _distribution([row["quality"] for row in background_rows]),
        },
        "quality_source": inference.get("quality_source"),
        "quality_hook_module": inference.get("quality_hook_module"),
        "joint_target_definition": inference.get("joint_target_definition"),
        "joint_target_power": inference.get("joint_target_power"),
        "definitions": {
            "positive_query": "训练 Hungarian matcher 选中的 query",
            "negative_query": "未被训练 matcher 选中的 query",
            "TP": "后处理检测经同类一对一匹配且 IoU 达到 ECE 阈值",
            "background_FP": "非 TP 且与任意 GT 的最大 IoU 小于 background FP 阈值",
            "quality": "最终活动 D-FINE LQE reg_conf 的独立加性 logit 经 sigmoid",
            "true_IoU": "Hungarian 正 query 与分配 GT 的 bbox IoU",
            "joint_target": "DEIM MAL assignment target：matched IoU ** criterion.gamma",
        },
    }
    _write_json(output_dir / "quality_diagnostics.json", payload)
    _write_csv(output_dir / "quality_positive_queries.csv", positives)
    _write_csv(output_dir / "quality_tp_and_background_fp.csv", tp_rows + background_rows)
    distribution_rows = [
        {"group": name, **values}
        for name, values in payload["distributions"].items()
    ]
    _write_json(output_dir / "quality_distribution_summary.json", distribution_rows)
    _table_image(
        output_dir / "quality_distribution_summary.png",
        distribution_rows,
        "LQE quality distribution summary",
        ["group", "count", "mean", "std", "p25", "median", "p75"],
    )

    spearman_rows = [
        {"metric": name, **payload[key]}
        for name, key in (
            ("quality_vs_true_iou", "quality_vs_true_iou_spearman"),
            ("quality_vs_joint_target", "quality_vs_joint_target_spearman"),
            ("all_query_quality_vs_best_iou", "all_query_quality_vs_best_iou_spearman"),
            (
                "all_query_quality_vs_assignment_target",
                "all_query_quality_vs_assignment_target_spearman",
            ),
        )
    ]
    _write_json(output_dir / "quality_spearman_summary.json", spearman_rows)
    _write_csv(output_dir / "quality_spearman_summary.csv", spearman_rows)
    _table_image(
        output_dir / "quality_spearman_summary.png",
        spearman_rows,
        "Quality Spearman correlations",
        ["metric", "rho", "p_value", "count"],
    )
    _ecdf_image(
        output_dir / "quality_positive_negative_ecdf.png",
        [
            ("Positive queries", [row["quality"] for row in positives]),
            ("Negative queries", [row["quality"] for row in negatives]),
        ],
        "Positive / negative query quality",
    )
    _ecdf_image(
        output_dir / "quality_tp_background_fp_ecdf.png",
        [
            ("TP", [row["quality"] for row in tp_rows]),
            ("Background FP", [row["quality"] for row in background_rows]),
        ],
        "TP / background FP quality",
    )

    plt = _require_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def draw_ecdf(
        axis: Any, series: Sequence[Tuple[str, Sequence[float]]], title: str
    ) -> None:
        for label, values in series:
            array = np.sort(np.asarray(values, dtype=float))
            if not array.size:
                continue
            cumulative = np.arange(1, array.size + 1, dtype=float) / array.size
            axis.step(array, cumulative, where="post", linewidth=2, label=f"{label} (n={len(values)})")
        axis.set(title=title, xlabel="Quality", ylabel="Cumulative fraction", xlim=(0, 1), ylim=(0, 1))
        axis.legend()

    draw_ecdf(
        axes[0, 0],
        [
            ("Positive queries", [row["quality"] for row in positives]),
            ("Negative queries", [row["quality"] for row in negatives]),
        ],
        "Positive / negative query quality",
    )
    draw_ecdf(
        axes[0, 1],
        [
            ("TP", [row["quality"] for row in tp_rows]),
            ("Background FP", [row["quality"] for row in background_rows]),
        ],
        "TP / background FP quality",
    )
    axes[1, 0].scatter([row["matched_iou"] for row in positives], [row["quality"] for row in positives], s=8, alpha=0.25)
    rho_iou = payload["quality_vs_true_iou_spearman"]["rho"]
    axes[1, 0].set(title=f"Quality vs true IoU (Spearman={rho_iou})", xlabel="Matched IoU", ylabel="Quality", xlim=(0, 1), ylim=(0, 1))
    axes[1, 1].scatter([row["joint_target"] for row in positives], [row["quality"] for row in positives], s=8, alpha=0.25)
    rho_joint = payload["quality_vs_joint_target_spearman"]["rho"]
    rho_assignment = payload["all_query_quality_vs_assignment_target_spearman"]["rho"]
    axes[1, 1].set(
        title=(
            f"Positive quality vs joint target (Spearman={rho_joint})\n"
            f"All-query assignment-target Spearman={rho_assignment}"
        ),
        xlabel="Joint target",
        ylabel="Quality",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "quality_diagnostics.png", dpi=240)
    plt.close(fig)
    return payload


def _tide_ground_truth(coco: Dict[str, Any]) -> Dict[str, Any]:
    annotations = []
    for ann in coco["annotations"]:
        x, y, width, height = ann["bbox_xywh"]
        x2, y2 = x + width, y + height
        annotations.append({
            "id": ann["id"], "image_id": ann["image_id"], "category_id": ann["category_id"],
            "bbox": [x, y, width, height], "area": float(ann.get("area", width * height)), "iscrowd": 0,
            "segmentation": [[x, y, x2, y, x2, y2, x, y2]],
        })
    return {
        "info": coco["raw"].get("info", {}), "licenses": coco["raw"].get("licenses", []),
        "images": list(coco["images"].values()), "categories": coco["categories"], "annotations": annotations,
    }


def _text_image(path: Path, title: str, text: str) -> None:
    plt = _require_matplotlib()
    lines = text.splitlines() or ["No output"]
    fig, axis = plt.subplots(figsize=(12, max(4, 0.28 * len(lines) + 1.5)))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.01, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=9, transform=axis.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _parse_tide_summary(text: str) -> Dict[str, Any]:
    """Convert tidecv's deterministic console tables into objective JSON values."""
    metrics: Dict[str, Any] = {
        "value_unit": "percentage_points",
        "bbox_AP_50_95": None,
        "AP_by_IoU_threshold": [],
        "main_errors_dAP": {},
        "special_errors_dAP": {},
    }
    match = re.search(
        r"bbox\s+AP\s*@\s*\[50-95\]\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        metrics["bbox_AP_50_95"] = float(match.group(1))

    thresholds: Optional[List[float]] = None
    section: Optional[str] = None
    section_headers: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lowered = line.lower()
        if "main errors" in lowered:
            section = "main"
            section_headers = []
            continue
        if "special error" in lowered:
            section = "special"
            section_headers = []
            continue
        tokens = line.split()
        if tokens and tokens[0].lower() == "thresh":
            try:
                thresholds = [float(value) / 100.0 for value in tokens[1:]]
            except ValueError:
                thresholds = None
            continue
        if tokens and tokens[0] == "AP" and thresholds is not None:
            try:
                values = [float(value) for value in tokens[1:]]
            except ValueError:
                values = []
            metrics["AP_by_IoU_threshold"] = [
                {"iou_threshold": threshold, "AP": value}
                for threshold, value in zip(thresholds, values)
            ]
            thresholds = None
            continue
        if tokens and tokens[0].lower() == "type" and section in ("main", "special"):
            section_headers = tokens[1:]
            continue
        if tokens and tokens[0].lower() == "dap" and section_headers:
            try:
                values = [float(value) for value in tokens[1:]]
            except ValueError:
                values = []
            target = (
                metrics["main_errors_dAP"]
                if section == "main"
                else metrics["special_errors_dAP"]
            )
            target.update(
                {name: value for name, value in zip(section_headers, values)}
            )
    return metrics


def _run_tide(
    coco: Dict[str, Any], predictions_path: Path, cfg: Dict[str, Any], output_dir: Path
) -> Dict[str, Any]:
    summary_path = output_dir / "tide_summary.txt"
    if not cfg.get("enable_tide"):
        text = "TIDE disabled by CONFIG.enable_tide=False"
        summary_path.write_text(text, encoding="utf-8")
        _text_image(output_dir / "tide_summary.png", "TIDE summary", text)
        result = {"available": False, "metrics": _parse_tide_summary(text)}
        _write_json(output_dir / "tide_metrics.json", result)
        return result
    try:
        from tidecv import TIDE, datasets
    except Exception as exc:
        if isinstance(exc, ImportError):
            text = "tidecv is not installed. Run: python -m pip install tidecv"
        else:
            text = traceback.format_exc()
        summary_path.write_text(text, encoding="utf-8")
        _text_image(output_dir / "tide_summary.png", "TIDE summary", text)
        result = {"available": False, "metrics": _parse_tide_summary(text)}
        _write_json(output_dir / "tide_metrics.json", result)
        return result
    gt_path = output_dir / "tide_ground_truth.json"
    _write_json(gt_path, _tide_ground_truth(coco))
    tide_plot_dir = output_dir / "tide_plots"
    tide_plot_dir.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            tide = TIDE()
            tide.evaluate_range(datasets.COCO(str(gt_path)), datasets.COCOResult(str(predictions_path)), mode=TIDE.BOX)
            tide.summarize()
            tide.plot(str(tide_plot_dir))
        text = buffer.getvalue()
    except Exception:
        text = buffer.getvalue() + "\n" + traceback.format_exc()
    summary_path.write_text(text, encoding="utf-8")
    _text_image(output_dir / "tide_summary.png", "TIDE summary", text)
    metrics = _parse_tide_summary(text)
    result = {
        "available": (
            "Traceback (most recent call last)" not in text
            and metrics["bbox_AP_50_95"] is not None
        ),
        "metrics": metrics,
    }
    _write_json(output_dir / "tide_metrics.json", result)
    return result


def _write_all_metrics_summary(
    experiment: Dict[str, Any],
    cfg: Dict[str, Any],
    coco: Dict[str, Any],
    inference: Dict[str, Any],
    bbox_summary: Sequence[Dict[str, Any]],
    coco_metrics: Dict[str, Any],
    confusion: Dict[str, Any],
    calibration: Dict[str, Any],
    recall: Sequence[Dict[str, Any]],
    background_fp: Sequence[Dict[str, Any]],
    quality: Dict[str, Any],
    tide: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Write one objective, machine-readable metric document per experiment."""
    payload = {
        "experiment": str(experiment["name"]),
        "counts": {
            "images": len(coco["images"]),
            "ground_truth_instances": len(coco["annotations"]),
            "predictions": len(inference["predictions"]),
            "classes": len(coco["cat_ids"]),
        },
        "evaluation_parameters": {
            "max_dets": int(cfg["max_dets"]),
            "pr_iou_threshold": float(cfg["pr_iou_threshold"]),
            "confusion_score_threshold": float(cfg["confusion_score_threshold"]),
            "confusion_iou_threshold": float(cfg["confusion_iou_threshold"]),
            "ece_score_threshold": float(cfg["ece_score_threshold"]),
            "ece_iou_threshold": float(cfg["ece_iou_threshold"]),
            "ece_num_bins": int(cfg["ece_num_bins"]),
            "recall_score_threshold": float(cfg["recall_score_threshold"]),
            "recall_iou_thresholds": [
                float(value) for value in cfg["recall_iou_thresholds"]
            ],
            "background_fp_score_threshold": float(
                cfg["background_fp_score_threshold"]
            ),
            "background_fp_iou_threshold": float(
                cfg["background_fp_iou_threshold"]
            ),
        },
        "COCO_bbox": {
            "value_unit": "fraction_0_to_1",
            "overall": coco_metrics["overall"],
            "per_class": coco_metrics["per_class"],
            "per_class_PR": coco_metrics["per_class_pr"],
        },
        "TIDE_bbox": tide,
        "confusion_and_FP_FN": {
            "row_labels": [
                coco["id_to_name"][cat_id] for cat_id in coco["cat_ids"]
            ],
            "column_labels": [
                coco["id_to_name"][cat_id] for cat_id in coco["cat_ids"]
            ],
            "confusion_matrix": confusion["core"],
            "extended_row_labels": [
                *[coco["id_to_name"][cat_id] for cat_id in coco["cat_ids"]],
                "BACKGROUND_FP",
            ],
            "extended_column_labels": [
                *[coco["id_to_name"][cat_id] for cat_id in coco["cat_ids"]],
                "BACKGROUND_FN",
            ],
            "confusion_matrix_with_FP_FN": confusion["full"],
            "per_class": confusion["per_class"],
        },
        "calibration": calibration,
        "recall_by_IoU_threshold": list(recall),
        "per_class_background_FP": list(background_fp),
        "quality_diagnostics": quality,
        "prediction_bbox_statistics": list(bbox_summary),
        "raw_query_class_argmax_distribution": inference.get(
            "raw_query_class_argmax_distribution", {}
        ),
    }
    _write_json(output_dir / "all_metrics_summary.json", payload)
    return payload


def _comparison_plot(rows: Sequence[Dict[str, Any]], output_root: Path) -> None:
    if not rows:
        return
    _write_json(output_root / "comparison_summary.json", list(rows))
    _write_csv(output_root / "comparison_summary.csv", rows)
    _table_image(
        output_root / "comparison_summary.png",
        rows,
        "LABS-Mamba V3 S0-S5 unified evaluation",
        [
            "experiment", "AP", "AP50", "AP75", "Recall@0.50",
            "ECE", "LaECE", "background_FP",
            "quality_IoU_spearman", "quality_joint_spearman",
        ],
    )


def _evaluate_experiment(
    experiment: Dict[str, Any], cfg: Dict[str, Any], coco: Dict[str, Any], output_dir: Path
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 24} {experiment['name']} {'=' * 24}")
    inference = _run_inference(experiment, cfg, coco, output_dir)
    _write_json(output_dir / "predictions.json", inference["predictions"])
    bbox_summary = _predictions_bbox_summary(
        inference["predictions"], coco, output_dir
    )
    prediction_objects = _prediction_objects(inference["predictions"])
    detection_objects = [{**row, "id": index} for index, row in enumerate(inference["detections"])]
    metric_result = _run_coco_metrics(_resolve(cfg["ann_file"]), inference["predictions"], coco, cfg, output_dir)
    confusion = _confusion_analysis(
        coco["annotations"], prediction_objects, coco, cfg, output_dir
    )
    annotated, _ = _annotate_detections(
        detection_objects,
        coco["annotations"],
        score_threshold=min(float(cfg["ece_score_threshold"]), float(cfg["background_fp_score_threshold"])),
        tp_iou=float(cfg["ece_iou_threshold"]),
        background_iou=float(cfg["background_fp_iou_threshold"]),
    )
    ece = _ece_analysis(annotated, coco, cfg, output_dir)
    recall = _recall_analysis(coco["annotations"], prediction_objects, coco, cfg, output_dir)
    background = _background_fp_analysis(annotated, coco, cfg, output_dir)
    quality = _quality_analysis(inference, annotated, output_dir)
    tide = _run_tide(coco, output_dir / "predictions.json", cfg, output_dir)
    tide_available = bool(tide["available"])
    _write_all_metrics_summary(
        experiment=experiment,
        cfg=cfg,
        coco=coco,
        inference=inference,
        bbox_summary=bbox_summary,
        coco_metrics=metric_result,
        confusion=confusion,
        calibration=ece,
        recall=recall,
        background_fp=background,
        quality=quality,
        tide=tide,
        output_dir=output_dir,
    )
    _write_json(output_dir / "evaluation_config.json", {**cfg, "experiment": experiment})
    required_outputs = list(REQUIRED_OUTPUTS)
    for cat_id in coco["cat_ids"]:
        stem = f"{cat_id}_{_safe_name(coco['id_to_name'][cat_id])}"
        required_outputs.extend(
            [f"per_class_pr/{stem}.csv", f"per_class_pr/{stem}.png"]
        )
    missing_outputs = [
        name for name in required_outputs if not (output_dir / name).is_file()
    ]
    _write_json(
        output_dir / "output_manifest.json",
        {
            "experiment": experiment["name"],
            "required_outputs": required_outputs,
            "missing_outputs": missing_outputs,
            "tide_available": tide_available,
            "quality_available": bool(quality.get("available")),
            # Optional dependency/signal availability remains an objective
            # status field and does not invalidate all other metric outputs.
            "complete": not missing_outputs,
        },
    )
    if missing_outputs:
        raise RuntimeError(
            f"评估输出不完整：missing={missing_outputs}, "
            f"TIDE_available={tide_available}, quality_available={quality.get('available')}。"
        )

    overall_ece = next(row for row in ece["rows"] if row["category_id"] == "all")
    recall50 = next(
        (row["recall"] for row in recall if row["category_id"] == "all" and abs(row["iou_threshold"] - 0.5) < 1e-8),
        None,
    )
    return {
        "experiment": experiment["name"],
        "AP": metric_result["overall"]["AP"],
        "AP50": metric_result["overall"]["AP50"],
        "AP75": metric_result["overall"]["AP75"],
        "Recall@0.50": recall50,
        "ECE": overall_ece["ECE"],
        "LaECE": overall_ece["LaECE"],
        "background_FP": sum(row["background_fp"] for row in background),
        "quality_IoU_spearman": quality.get("quality_vs_true_iou_spearman", {}).get("rho"),
        "quality_joint_spearman": quality.get("quality_vs_joint_target_spearman", {}).get("rho"),
        "TIDE_available": tide_available,
        "quality_available": bool(quality.get("available")),
        "quality_all_query_best_iou_spearman": quality.get(
            "all_query_quality_vs_best_iou_spearman", {}
        ).get("rho"),
        "quality_all_query_assignment_spearman": quality.get(
            "all_query_quality_vs_assignment_target_spearman", {}
        ).get("rho"),
    }


def _validate_config(cfg: Dict[str, Any]) -> None:
    ann_file = _resolve(cfg["ann_file"])
    images_dir = _resolve(cfg["images_dir"])
    if not ann_file.exists():
        raise FileNotFoundError(f"ann_file 不存在：{ann_file}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images_dir 不存在：{images_dir}")
    if int(cfg["batch_size"]) <= 0 or int(cfg["num_workers"]) < 0:
        raise ValueError("batch_size 必须 > 0，num_workers 必须 >= 0。")
    if int(cfg["max_dets"]) <= 0 or int(cfg["ece_num_bins"]) <= 0:
        raise ValueError("max_dets 和 ece_num_bins 必须 > 0。")
    for key in (
        "confusion_score_threshold",
        "confusion_iou_threshold",
        "ece_score_threshold",
        "ece_iou_threshold",
        "recall_score_threshold",
        "background_fp_score_threshold",
        "background_fp_iou_threshold",
        "pr_iou_threshold",
    ):
        if not 0.0 <= float(cfg[key]) <= 1.0:
            raise ValueError(f"{key} 必须位于 [0, 1]，当前为 {cfg[key]}。")
    if not cfg["recall_iou_thresholds"] or any(
        not 0.0 <= float(value) <= 1.0 for value in cfg["recall_iou_thresholds"]
    ):
        raise ValueError("recall_iou_thresholds 必须是 [0, 1] 内的非空序列。")
    names = [str(item["name"]) for item in cfg["experiments"]]
    if len(names) != len(set(names)):
        raise ValueError(f"experiment name 重复：{names}")


def run(cfg: Dict[str, Any], selected_names: Optional[Sequence[str]] = None) -> None:
    _validate_config(cfg)
    output_root = _resolve(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    selected = set(selected_names or [])
    known_names = {str(item["name"]) for item in cfg["experiments"]}
    unknown_names = sorted(selected - known_names)
    if unknown_names:
        raise ValueError(f"--experiments 包含未知实验：{unknown_names}")
    active_experiments = [
        experiment
        for experiment in cfg["experiments"]
        if experiment.get("enabled", True)
        and (not selected or experiment["name"] in selected)
    ]
    if not active_experiments:
        raise RuntimeError("没有待评估实验：请在 CONFIG 中启用实验，或检查 --experiments。")
    _assert_shared_validation_settings(active_experiments, cfg, output_root)
    coco = _load_coco(_resolve(cfg["ann_file"]))
    if len(coco["cat_ids"]) != 9:
        print(f"[WARN] test JSON 包含 {len(coco['cat_ids'])} 类，不是预期九类。")
    comparison = []
    failures = []
    attempted = 0
    for experiment in cfg["experiments"]:
        if selected:
            # An explicit CLI selection intentionally overrides the editable
            # enabled flag so one experiment can be evaluated without changing
            # source code.
            if experiment["name"] not in selected:
                continue
        elif not experiment.get("enabled", True):
            continue
        attempted += 1
        try:
            comparison.append(_evaluate_experiment(experiment, cfg, coco, output_root / experiment["name"]))
        except Exception:
            details = traceback.format_exc()
            failure = output_root / experiment["name"] / "FAILED.txt"
            failure.parent.mkdir(parents=True, exist_ok=True)
            failure.write_text(details, encoding="utf-8")
            failures.append({"experiment": experiment["name"], "details": details})
            print(f"[ERROR] {experiment['name']} 失败，详情见 {failure}")
            if not cfg.get("continue_on_error"):
                raise
    assert attempted > 0
    _comparison_plot(comparison, output_root)
    _write_json(output_root / "suite_config.json", cfg)
    _write_json(output_root / "failed_experiments.json", failures)
    if failures:
        failed_names = [row["experiment"] for row in failures]
        raise RuntimeError(f"以下实验评估失败：{failed_names}；请检查各实验 FAILED.txt。")
    print(f"\n[DONE] 统一评估输出：{output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一评估 LABS-Mamba V3 S0～S5")
    parser.add_argument(
        "--experiments",
        default=None,
        help="显式选择实验并覆盖 enabled 标志，逗号分隔，如 s0_local_anchor,s1_hv_sidecar_b64,s2_vh_sidecar_b64",
    )
    parser.add_argument("--ann-file", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--no-tide",
        action="store_true",
        help="跳过 TIDE 依赖并在输出中记录 TIDE_available=False",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = dict(CONFIG)
    cfg["experiments"] = [dict(item) for item in CONFIG["experiments"]]
    for key in ("ann_file", "images_dir", "output_root", "device", "batch_size", "num_workers"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.no_cache:
        cfg["reuse_inference_cache"] = False
    if args.no_tide:
        cfg["enable_tide"] = False
    selected = [item.strip() for item in args.experiments.split(",") if item.strip()] if args.experiments else None
    run(cfg, selected)


if __name__ == "__main__":
    main()
