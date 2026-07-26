"""
DEIM 检测错误分析脚本。

生成内容：
1. 九类混淆矩阵：9x9 核心矩阵，以及带 background(FP/FN) 的完整矩阵；
2. 每类 PR 曲线；
3. 每类置信度直方图，区分 TP / FP；
4. FP / FN 可视化；
5. 按目标面积、长宽比分组的 AP。

默认会加载 checkpoint 重新推理；如果已有 COCO detections JSON，可用
--predictions-json 跳过模型推理。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig  # noqa: E402


# =========================
# 用户配置区
# =========================
CONFIG = {
    # DEIM 配置文件。
    "config_yml": "configs/deim_dfine/deim_hgnetv2_l_wood.yml",

    # 训练好的 checkpoint，通常使用 best_stg2.pth。
    "checkpoint": "output/dfine_hgnetv2_l_wood_multiple_960_fiter50/best_stg2.pth",

    # 测试/验证图片目录和 COCO 标注。None 表示使用 config_yml 中的 val_dataloader。
    "img_folder": None,
    "ann_file": None,

    # 已有 COCO detections JSON 时填这里，可跳过模型推理。
    "predictions_json": None,

    # 错误分析结果输出目录。
    "output_dir": "output/dfine_hgnetv2_l_wood_multiple_960_fiter50/error_analysis",

    # 推理设备、batch size 和 dataloader worker 数。
    "device": "cuda:0",
    "batch_size": 12,
    "num_workers": 4,

    # min_score 控制保存/参与分析的最低预测分数。
    "min_score": 0.001,

    # confusion_score_threshold 控制混淆矩阵和 FP/FN 可视化的预测框阈值。
    "confusion_score_threshold": 0.25,

    # TP/FP/FN 匹配使用的 IoU 阈值；0.5 对应 AP50/PR50。
    "iou_threshold": 0.5,

    # 最多保存多少张 FP/FN 可视化图片。
    "max_error_visualizations": 100,

    # 只分析指定类别；None 表示全部类别，例如 ["Death_Kont", "Live_Kont"]。
    "selected_classes": None,

    # 面积分组，单位为像素面积 w*h。
    "area_bins": [
        {"name": "small_lt_32^2", "min": 0.0, "max": 32.0 * 32.0},
        {"name": "medium_32^2_96^2", "min": 32.0 * 32.0, "max": 96.0 * 96.0},
        {"name": "large_ge_96^2", "min": 96.0 * 96.0, "max": None},
    ],

    # 长宽比分组，aspect = bbox_width / bbox_height。
    "aspect_bins": [
        {"name": "tall_lt_0.5", "min": 0.0, "max": 0.5},
        {"name": "normal_0.5_2", "min": 0.5, "max": 2.0},
        {"name": "wide_ge_2", "min": 2.0, "max": None},
    ],
}


def _remove_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def _load_checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if "ema" in checkpoint:
        ema = checkpoint["ema"]
        state = ema["module"] if isinstance(ema, dict) and "module" in ema else ema
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    return _remove_module_prefix(state)


def _build_cfg(cfg_dict: Dict[str, Any]) -> YAMLConfig:
    update = {
        "val_dataloader": {
            "total_batch_size": int(cfg_dict["batch_size"]),
            "num_workers": int(cfg_dict["num_workers"]),
        }
    }
    dataset_update = {}
    if cfg_dict.get("img_folder"):
        dataset_update["img_folder"] = cfg_dict["img_folder"]
    if cfg_dict.get("ann_file"):
        dataset_update["ann_file"] = cfg_dict["ann_file"]
    if dataset_update:
        update["val_dataloader"]["dataset"] = dataset_update

    cfg = YAMLConfig(cfg_dict["config_yml"], **update)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    return cfg


def _tensor_to_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    if hasattr(value, "numel"):
        return int(value.reshape(-1)[0].item())
    return int(value)


def _xywh_to_xyxy(box: Sequence[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def _xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def _load_coco(ann_file: Path, img_folder: Path) -> Dict[str, Any]:
    with ann_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = sorted(data.get("categories", []), key=lambda item: int(item["id"]))
    cat_ids = [int(cat["id"]) for cat in categories]
    id_to_name = {int(cat["id"]): cat.get("name", str(cat["id"])) for cat in categories}
    name_to_ids: Dict[str, Set[int]] = {}
    for cid, name in id_to_name.items():
        name_to_ids.setdefault(name, set()).add(cid)
        name_to_ids.setdefault(name.lower(), set()).add(cid)

    images = {int(img["id"]): img for img in data.get("images", [])}
    gts = []
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        box = _xywh_to_xyxy(ann["bbox"])
        w, h = float(ann["bbox"][2]), float(ann["bbox"][3])
        if w <= 0 or h <= 0:
            continue
        gts.append(
            {
                "ann_id": int(ann.get("id", len(gts))),
                "image_id": int(ann["image_id"]),
                "category_id": int(ann["category_id"]),
                "bbox": box,
                "area": float(ann.get("area", w * h)),
                "aspect": w / h,
            }
        )

    gt_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for gt in gts:
        gt_by_image.setdefault(gt["image_id"], []).append(gt)

    return {
        "categories": categories,
        "cat_ids": cat_ids,
        "id_to_name": id_to_name,
        "name_to_ids": name_to_ids,
        "images": images,
        "img_folder": img_folder,
        "gts": gts,
        "gt_by_image": gt_by_image,
    }


def _selected_cat_ids(selected_names: Optional[Sequence[str]], name_to_ids: Dict[str, Set[int]]) -> Optional[Set[int]]:
    if not selected_names:
        return None
    selected: Set[int] = set()
    missing = []
    for name in selected_names:
        ids = name_to_ids.get(name) or name_to_ids.get(name.lower())
        if ids:
            selected.update(ids)
        else:
            missing.append(name)
    if missing:
        print(f"[WARN] selected class names not found: {missing}")
    return selected


def _image_path(coco: Dict[str, Any], image_id: int) -> Path:
    return Path(coco["img_folder"]) / coco["images"][int(image_id)]["file_name"]


@torch.no_grad()
def _run_inference(cfg_obj: YAMLConfig, cfg_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    device = torch.device(cfg_dict["device"] if torch.cuda.is_available() else "cpu")
    model = cfg_obj.model
    state = _load_checkpoint_state(Path(cfg_dict["checkpoint"]))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    model = model.to(device).eval()
    postprocessor = cfg_obj.postprocessor.to(device).eval()
    dataloader = cfg_obj.val_dataloader

    predictions: List[Dict[str, Any]] = []
    min_score = float(cfg_dict["min_score"])
    for samples, targets in dataloader:
        samples = samples.to(device)
        targets_on_device = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}
            for target in targets
        ]
        outputs = model(samples)
        orig_target_sizes = torch.stack([target["orig_size"] for target in targets_on_device], dim=0)
        results = postprocessor(outputs, orig_target_sizes)
        for result, target in zip(results, targets):
            image_id = _tensor_to_int(target["image_id"])
            boxes = result["boxes"].detach().cpu().tolist()
            labels = result["labels"].detach().cpu().tolist()
            scores = result["scores"].detach().cpu().tolist()
            for box, label, score in zip(boxes, labels, scores):
                if float(score) < min_score:
                    continue
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(v) for v in box],
                        "score": float(score),
                    }
                )
    return predictions


def _load_predictions(path: Path, min_score: float) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    preds = []
    for rec in records:
        score = float(rec.get("score", 1.0))
        if score < min_score:
            continue
        box = rec["bbox"]
        # COCO detections JSON 使用 xywh；如果显式给了 bbox_xyxy，则优先使用。
        xyxy = rec.get("bbox_xyxy") or _xywh_to_xyxy(box)
        preds.append(
            {
                "image_id": int(rec["image_id"]),
                "category_id": int(rec["category_id"]),
                "bbox": [float(v) for v in xyxy],
                "score": score,
            }
        )
    return preds


def _save_predictions_json(path: Path, predictions: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "image_id": pred["image_id"],
            "category_id": pred["category_id"],
            "bbox": _xyxy_to_xywh(pred["bbox"]),
            "score": pred["score"],
        }
        for pred in predictions
    ]
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _by_image(items: Iterable[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    data: Dict[int, List[Dict[str, Any]]] = {}
    for item in items:
        data.setdefault(int(item["image_id"]), []).append(item)
    return data


def _compute_pr(
    cat_id: int,
    predictions: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    iou_thr: float,
    gt_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Dict[str, Any]:
    """计算某类别 AP/PR；gt_filter 用于面积/长宽比分组。"""
    class_gts = [gt for gt in gts if gt["category_id"] == cat_id]
    positive_gts = [gt for gt in class_gts if gt_filter is None or gt_filter(gt)]
    ignore_gts = [gt for gt in class_gts if gt_filter is not None and not gt_filter(gt)]
    preds = [pred for pred in predictions if pred["category_id"] == cat_id]
    preds = sorted(preds, key=lambda item: -float(item["score"]))

    pos_by_image = _by_image(positive_gts)
    ign_by_image = _by_image(ignore_gts)
    matched: Set[int] = set()
    rows = []

    for pred in preds:
        image_id = int(pred["image_id"])
        best_iou, best_idx = 0.0, None
        for idx, gt in enumerate(pos_by_image.get(image_id, [])):
            if id(gt) in matched:
                continue
            iou = _box_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou, best_idx = iou, idx

        if best_idx is not None and best_iou >= iou_thr:
            matched.add(id(pos_by_image[image_id][best_idx]))
            rows.append({"score": float(pred["score"]), "tp": 1, "fp": 0, "status": "TP"})
        else:
            ignore = any(_box_iou(pred["bbox"], gt["bbox"]) >= iou_thr for gt in ign_by_image.get(image_id, []))
            if ignore:
                continue
            rows.append({"score": float(pred["score"]), "tp": 0, "fp": 1, "status": "FP"})

    npos = len(positive_gts)
    if not rows:
        return {"ap": None, "npos": npos, "points": [], "pred_rows": rows}

    s_tp = s_fp = 0
    points = []
    for row in rows:
        t, f = int(row["tp"]), int(row["fp"])
        s_tp += t
        s_fp += f
        recall = 0.0 if npos == 0 else s_tp / npos
        precision = 0.0 if (s_tp + s_fp) == 0 else s_tp / (s_tp + s_fp)
        points.append({"score": float(row["score"]), "precision": precision, "recall": recall})

    ap = _voc_ap([p["recall"] for p in points], [p["precision"] for p in points]) if npos > 0 else None
    return {"ap": ap, "npos": npos, "points": points, "pred_rows": rows}


def _voc_ap(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def _match_for_confusion(
    preds_by_image: Dict[int, List[Dict[str, Any]]],
    gt_by_image: Dict[int, List[Dict[str, Any]]],
    cat_ids: List[int],
    iou_thr: float,
    score_thr: float,
) -> Tuple[List[List[int]], List[List[int]]]:
    idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    n = len(cat_ids)
    core = [[0 for _ in range(n)] for _ in range(n)]
    full = [[0 for _ in range(n + 1)] for _ in range(n + 1)]

    image_ids = sorted(set(gt_by_image.keys()) | set(preds_by_image.keys()))
    for image_id in image_ids:
        gts = [gt for gt in gt_by_image.get(image_id, []) if gt["category_id"] in idx]
        preds = [
            pred for pred in preds_by_image.get(image_id, [])
            if pred["category_id"] in idx and float(pred["score"]) >= score_thr
        ]
        pairs = []
        for gi, gt in enumerate(gts):
            for pi, pred in enumerate(preds):
                iou = _box_iou(gt["bbox"], pred["bbox"])
                if iou >= iou_thr:
                    same_class = int(gt["category_id"] == pred["category_id"])
                    pairs.append((iou, same_class, float(pred["score"]), gi, pi))
        pairs.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        used_g, used_p = set(), set()
        for _iou, _same_class, _score, gi, pi in pairs:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            r = idx[gts[gi]["category_id"]]
            c = idx[preds[pi]["category_id"]]
            core[r][c] += 1
            full[r][c] += 1
        for gi, gt in enumerate(gts):
            if gi not in used_g:
                full[idx[gt["category_id"]]][n] += 1  # FN
        for pi, pred in enumerate(preds):
            if pi not in used_p:
                full[n][idx[pred["category_id"]]] += 1  # FP
    return core, full


def _write_matrix_csv(path: Path, row_names: List[str], col_names: List[str], matrix: List[List[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred"] + col_names)
        for name, row in zip(row_names, matrix):
            writer.writerow([name] + row)


def _plot_matrix(path: Path, row_names: List[str], col_names: List[str], matrix: List[List[int]], title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skip confusion matrix png: {exc}")
        return
    arr = np.asarray(matrix)
    fig_w = max(8, 0.7 * len(col_names))
    fig_h = max(6, 0.6 * len(row_names))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(col_names)))
    ax.set_yticks(range(len(row_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right")
    ax.set_yticklabels(row_names)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(int(arr[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _write_pr_and_hist(
    output_dir: Path,
    cat_ids: List[int],
    id_to_name: Dict[int, str],
    predictions: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    iou_thr: float,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for cat_id in cat_ids:
        name = id_to_name.get(cat_id, str(cat_id))
        safe_name = _safe_filename(name)
        pr = _compute_pr(cat_id, predictions, gts, iou_thr)
        ap_percent = None if pr["ap"] is None else round(pr["ap"] * 100.0, 4)
        summaries.append({"category_id": cat_id, "class_name": name, "gt_instances": pr["npos"], "AP50": ap_percent})

        curve_csv = output_dir / "pr_curves" / f"{cat_id}_{safe_name}.csv"
        curve_csv.parent.mkdir(parents=True, exist_ok=True)
        with curve_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["score", "precision", "recall"])
            writer.writeheader()
            writer.writerows(pr["points"])

        hist_csv = output_dir / "confidence_histograms" / f"{cat_id}_{safe_name}.csv"
        hist_csv.parent.mkdir(parents=True, exist_ok=True)
        with hist_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["score", "tp", "fp", "status"])
            writer.writeheader()
            writer.writerows(pr["pred_rows"])

        _plot_pr_curve(output_dir / "pr_curves" / f"{cat_id}_{safe_name}.png", pr["points"], name, ap_percent)
        _plot_conf_hist(output_dir / "confidence_histograms" / f"{cat_id}_{safe_name}.png", pr["pred_rows"], name)
    return summaries


def _safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name))


def _plot_pr_curve(path: Path, points: List[Dict[str, float]], name: str, ap_percent: Optional[float]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skip PR png: {exc}")
        return
    recalls = [p["recall"] for p in points]
    precisions = [p["precision"] for p in points]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls, precisions, linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    title = f"PR - {name}" if ap_percent is None else f"PR - {name} (AP50={ap_percent:.2f})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_conf_hist(path: Path, rows: List[Dict[str, Any]], name: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skip confidence histogram png: {exc}")
        return
    tp_scores = [r["score"] for r in rows if r["tp"] == 1]
    fp_scores = [r["score"] for r in rows if r["fp"] == 1]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = [i / 20 for i in range(21)]
    ax.hist(fp_scores, bins=bins, alpha=0.65, label="FP", color="tab:red")
    ax.hist(tp_scores, bins=bins, alpha=0.65, label="TP", color="tab:blue")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title(f"Confidence histogram - {name}")
    ax.legend()
    ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _group_ap(
    bins: List[Dict[str, Any]],
    group_name: str,
    cat_ids: List[int],
    id_to_name: Dict[int, str],
    predictions: List[Dict[str, Any]],
    gts: List[Dict[str, Any]],
    iou_thr: float,
    output_dir: Path,
) -> None:
    rows = []
    for bin_cfg in bins:
        min_v = bin_cfg.get("min")
        max_v = bin_cfg.get("max")
        name = str(bin_cfg["name"])

        def filt(gt: Dict[str, Any], min_v=min_v, max_v=max_v, group_name=group_name) -> bool:
            value = gt["area"] if group_name == "area" else gt["aspect"]
            if min_v is not None and value < float(min_v):
                return False
            if max_v is not None and value >= float(max_v):
                return False
            return True

        aps = []
        for cat_id in cat_ids:
            pr = _compute_pr(cat_id, predictions, gts, iou_thr, gt_filter=filt)
            ap_percent = None if pr["ap"] is None else round(pr["ap"] * 100.0, 4)
            if ap_percent is not None:
                aps.append(ap_percent)
            rows.append(
                {
                    "group_type": group_name,
                    "group": name,
                    "category_id": cat_id,
                    "class_name": id_to_name.get(cat_id, str(cat_id)),
                    "gt_instances": pr["npos"],
                    "AP50": ap_percent,
                }
            )
        rows.append(
            {
                "group_type": group_name,
                "group": name,
                "category_id": "all",
                "class_name": "mAP",
                "gt_instances": sum(r["gt_instances"] for r in rows if r["group"] == name and isinstance(r["category_id"], int)),
                "AP50": None if not aps else round(sum(aps) / len(aps), 4),
            }
        )
    _write_dicts(output_dir / f"ap_by_{group_name}.csv", rows)
    (output_dir / f"ap_by_{group_name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_dicts(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _match_errors_for_image(gts: List[Dict[str, Any]], preds: List[Dict[str, Any]], iou_thr: float, score_thr: float) -> Tuple[List[Any], List[Any], List[Any]]:
    preds = [p for p in preds if float(p["score"]) >= score_thr]
    pairs = []
    for gi, gt in enumerate(gts):
        for pi, pred in enumerate(preds):
            if gt["category_id"] != pred["category_id"]:
                continue
            iou = _box_iou(gt["bbox"], pred["bbox"])
            if iou >= iou_thr:
                pairs.append((iou, float(pred["score"]), gi, pi))
    pairs.sort(key=lambda item: (item[0], item[1]), reverse=True)
    used_g, used_p, tps = set(), set(), []
    for iou, _score, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        tps.append((gts[gi], preds[pi], iou))
    fns = [gt for i, gt in enumerate(gts) if i not in used_g]
    fps = [pred for i, pred in enumerate(preds) if i not in used_p]
    return tps, fps, fns


def _draw_error_visualizations(
    output_dir: Path,
    coco: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    cat_ids: List[int],
    iou_thr: float,
    score_thr: float,
    max_images: int,
) -> None:
    out = output_dir / "fp_fn_visualizations"
    out.mkdir(parents=True, exist_ok=True)
    pred_by_img = _by_image([p for p in predictions if p["category_id"] in cat_ids])
    gt_by_img = {img: [gt for gt in gts if gt["category_id"] in cat_ids] for img, gts in coco["gt_by_image"].items()}
    saved = 0
    for image_id in sorted(set(gt_by_img.keys()) | set(pred_by_img.keys())):
        tps, fps, fns = _match_errors_for_image(gt_by_img.get(image_id, []), pred_by_img.get(image_id, []), iou_thr, score_thr)
        if not fps and not fns:
            continue
        image = Image.open(_image_path(coco, image_id)).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for gt, pred, _iou in tps:
            _draw_box(draw, pred["bbox"], f"TP {coco['id_to_name'].get(pred['category_id'], pred['category_id'])} {pred['score']:.2f}", "blue", font)
        for pred in fps:
            _draw_box(draw, pred["bbox"], f"FP {coco['id_to_name'].get(pred['category_id'], pred['category_id'])} {pred['score']:.2f}", "red", font)
        for gt in fns:
            _draw_box(draw, gt["bbox"], f"FN {coco['id_to_name'].get(gt['category_id'], gt['category_id'])}", "orange", font)
        image.save(out / f"error_{saved + 1:06d}_{image_id}.jpg")
        saved += 1
        if saved >= max_images:
            break
    print(f"Saved FP/FN visualizations: {out} ({saved})")


def _draw_box(draw: ImageDraw.ImageDraw, box: Sequence[float], text: str, color: str, font: ImageFont.ImageFont) -> None:
    x1, y1, x2, y2 = [float(v) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((int(x1), int(y1)), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        tw, th = draw.textsize(text, font=font)
    tx, ty = int(x1), max(0, int(y1) - th - 4)
    draw.rectangle([tx, ty, tx + tw + 6, ty + th + 4], fill="white", outline=color)
    draw.text((tx + 3, ty + 2), text, fill=color, font=font)


def _parse_class_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def run(cfg_dict: Dict[str, Any]) -> None:
    cfg_obj = _build_cfg(cfg_dict)
    ann_file = Path(cfg_obj.yaml_cfg["val_dataloader"]["dataset"]["ann_file"])
    img_folder = Path(cfg_obj.yaml_cfg["val_dataloader"]["dataset"]["img_folder"])
    coco = _load_coco(ann_file, img_folder)
    selected = _selected_cat_ids(cfg_dict.get("selected_classes"), coco["name_to_ids"])
    cat_ids = [cid for cid in coco["cat_ids"] if selected is None or cid in selected]

    output_dir = Path(cfg_dict["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg_dict.get("predictions_json"):
        predictions = _load_predictions(Path(cfg_dict["predictions_json"]), float(cfg_dict["min_score"]))
    else:
        predictions = _run_inference(cfg_obj, cfg_dict)
        _save_predictions_json(output_dir / "predictions.json", predictions)

    predictions = [pred for pred in predictions if selected is None or pred["category_id"] in selected]
    pred_by_img = _by_image(predictions)
    gt_by_img = {img: [gt for gt in gts if gt["category_id"] in cat_ids] for img, gts in coco["gt_by_image"].items()}

    names = [coco["id_to_name"].get(cid, str(cid)) for cid in cat_ids]
    core, full = _match_for_confusion(pred_by_img, gt_by_img, cat_ids, float(cfg_dict["iou_threshold"]), float(cfg_dict["confusion_score_threshold"]))
    matrix_dir = output_dir / "confusion_matrix"
    _write_matrix_csv(matrix_dir / "confusion_matrix_9cls.csv", names, names, core)
    _write_matrix_csv(matrix_dir / "confusion_matrix_with_background.csv", names + ["__background_fp__"], names + ["__background_fn__"], full)
    _plot_matrix(matrix_dir / "confusion_matrix_9cls.png", names, names, core, "9-class confusion matrix")
    _plot_matrix(matrix_dir / "confusion_matrix_with_background.png", names + ["BG(FP)"], names + ["BG(FN)"], full, "Confusion matrix with FP/FN")

    pr_dir = output_dir / "per_class_pr"
    summaries = _write_pr_and_hist(pr_dir, cat_ids, coco["id_to_name"], predictions, coco["gts"], float(cfg_dict["iou_threshold"]))
    _write_dicts(pr_dir / "per_class_ap50.csv", summaries)
    (pr_dir / "per_class_ap50.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    group_dir = output_dir / "grouped_ap"
    group_dir.mkdir(parents=True, exist_ok=True)
    _group_ap(cfg_dict["area_bins"], "area", cat_ids, coco["id_to_name"], predictions, coco["gts"], float(cfg_dict["iou_threshold"]), group_dir)
    _group_ap(cfg_dict["aspect_bins"], "aspect", cat_ids, coco["id_to_name"], predictions, coco["gts"], float(cfg_dict["iou_threshold"]), group_dir)

    _draw_error_visualizations(output_dir, coco, predictions, cat_ids, float(cfg_dict["iou_threshold"]), float(cfg_dict["confusion_score_threshold"]), int(cfg_dict["max_error_visualizations"]))
    print(f"Saved error analysis: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DEIM detection error analysis")
    parser.add_argument("--config-yml", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--img-folder", default=None)
    parser.add_argument("--ann-file", default=None)
    parser.add_argument("--predictions-json", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--confusion-score-threshold", type=float, default=None)
    parser.add_argument("--iou-threshold", type=float, default=None)
    parser.add_argument("--max-error-visualizations", type=int, default=None)
    parser.add_argument("--selected-classes", default=None, help="comma separated class names; empty string means all classes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = dict(CONFIG)
    updates = {
        "config_yml": args.config_yml,
        "checkpoint": args.checkpoint,
        "img_folder": args.img_folder,
        "ann_file": args.ann_file,
        "predictions_json": args.predictions_json,
        "output_dir": args.output_dir,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "min_score": args.min_score,
        "confusion_score_threshold": args.confusion_score_threshold,
        "iou_threshold": args.iou_threshold,
        "max_error_visualizations": args.max_error_visualizations,
    }
    for key, value in updates.items():
        if value is not None:
            cfg[key] = value
    selected = _parse_class_list(args.selected_classes)
    if selected is not None:
        cfg["selected_classes"] = selected
    run(cfg)


if __name__ == "__main__":
    main()
