"""
加载训练好的 DEIM 模型，在 COCO 格式数据集上重新推理，并输出：

1. COCO bbox 总体评估结果；
2. 每个类别的 AP/AR 指标；
3. 全类别 overall AP/AR 指标；
4. 每张图片的 GT/预测对比可视化结果；
5. COCO detections JSON，便于后续复查或二次分析。

本脚本默认指向：
    output/dfine_hgnetv2_l_wood_multiple_960_fiter50/best_stg2.pth

运行示例：
    python output/tools/evaluate_and_visualize.py

如需评估 test 集，可通过命令行覆盖：
    python output/tools/evaluate_and_visualize.py \
        --img-folder ./data/xxx/images/test \
        --ann-file ./data/xxx/annotations/instances_test.json \
        --output-dir output/dfine_hgnetv2_l_wood_multiple_960_fiter50/test_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig  # noqa: E402


# =========================
# 用户配置区
# =========================
CONFIG = {
    # 训练使用的 DEIM 配置文件。类别数、模型结构、输入尺寸等从这里读取。
    "config_yml": "configs/deim_dfine/deim_hgnetv2_l_wood.yml",

    # 已训练好的 checkpoint。通常使用 best_stg2.pth 或 best_stg1.pth。
    "checkpoint": "output/dfine_hgnetv2_l_wood_multiple_960_fiter50/best_stg2.pth",

    # 新数据集图片目录。None 表示使用 config_yml 中 val_dataloader.dataset.img_folder。
    "img_folder": None,

    # 新数据集 COCO 标注 JSON。None 表示使用 config_yml 中 val_dataloader.dataset.ann_file。
    "ann_file": None,

    # 输出目录。会保存图片、预测 JSON、eval/latest.pth 和指标表。
    "output_dir": "output/dfine_hgnetv2_l_wood_multiple_960_fiter50/analysis_infer",

    # 推理设备，例如 "cuda:0" 或 "cpu"。
    "device": "cuda:0",

    # 单进程推理 batch size。显存不足时调小。
    "batch_size": 12,

    # dataloader worker 数。Windows 本地跑可先改成 0。
    "num_workers": 4,

    # 可视化时保留的预测框置信度阈值。
    "score_threshold": 0.1,

    # 最多保存多少张可视化图片；None 表示保存全部。
    "max_visualize_images": 50,

    # 是否在左侧图片绘制 GT 框。左侧绿色为 GT，右侧红色为预测。
    "draw_gt": True,

    # 是否保存所有预测为 COCO detections JSON。
    "save_predictions_json": True,

    # 指标排序字段。None 表示保持类别顺序；可设为 "AP"、"AP50"、"AR" 等。
    "sort_by": None,

    # 是否按排序字段降序排列。
    "sort_desc": True,

    # 指标保留小数位数。
    "digits": 4,

    # COCOeval maxDets 档位；None 表示使用最后一档，标准 COCO 通常为 100。
    "max_det": None,
}


def _remove_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """兼容 DDP 保存的 module.xxx 权重名。"""
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _load_checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    """读取 checkpoint，优先使用 EMA 权重。"""
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
    """构建 YAMLConfig，并按用户配置覆盖验证数据路径、batch size 和 worker。"""
    update: Dict[str, Any] = {
        "val_dataloader": {
            "total_batch_size": int(cfg_dict["batch_size"]),
            "num_workers": int(cfg_dict["num_workers"]),
        }
    }

    dataset_update: Dict[str, Any] = {}
    if cfg_dict.get("img_folder"):
        dataset_update["img_folder"] = cfg_dict["img_folder"]
    if cfg_dict.get("ann_file"):
        dataset_update["ann_file"] = cfg_dict["ann_file"]
    if dataset_update:
        update["val_dataloader"]["dataset"] = dataset_update

    cfg = YAMLConfig(cfg_dict["config_yml"], **update)

    # 加载自己训练好的 checkpoint 时，不再自动加载/下载 HGNetv2 预训练权重。
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    return cfg


def _tensor_to_int(value: Any) -> int:
    """兼容 int 和 0 维/1 维 tensor。"""
    if hasattr(value, "item"):
        return int(value.item())
    if hasattr(value, "numel"):
        return int(value.reshape(-1)[0].item())
    return int(value)


def _category_maps(ann_file: Path) -> Dict[str, Dict[int, Any]]:
    """读取类别 id 到类别名的映射。"""
    with ann_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    id_to_name = {int(cat["id"]): cat.get("name", str(cat["id"])) for cat in data.get("categories", [])}
    return {"id_to_name": id_to_name}


def _image_path_from_target(target: Dict[str, Any], dataloader: Any) -> Path:
    """DEIM target 默认没有 image_path 时，从 COCO 数据集按 image_id 反查原图路径。"""
    if "image_path" in target:
        return Path(target["image_path"])

    image_id = _tensor_to_int(target["image_id"])
    dataset = dataloader.dataset
    img_info = dataset.coco.loadImgs(image_id)[0]
    return Path(dataset.img_folder) / img_info["file_name"]


def _scale_boxes_to_original(boxes: torch.Tensor, orig_size: torch.Tensor, resized_hw: List[int]) -> torch.Tensor:
    """将验证 dataloader 中 resize 后的 GT 框缩放回原图坐标。"""
    boxes = boxes.detach().cpu().clone().float()
    orig_w = float(orig_size[0].item())
    orig_h = float(orig_size[1].item())
    resized_h, resized_w = resized_hw
    boxes[:, 0] *= orig_w / resized_w
    boxes[:, 2] *= orig_w / resized_w
    boxes[:, 1] *= orig_h / resized_h
    boxes[:, 3] *= orig_h / resized_h
    return boxes


def _draw_one_image(
    image_path: Path,
    save_path: Path,
    result: Dict[str, torch.Tensor],
    target: Dict[str, Any],
    id_to_name: Dict[int, str],
    score_threshold: float,
    draw_gt: bool,
    resized_hw: List[int],
) -> None:
    """保存单张图片的左右对比可视化结果：左侧为 GT，右侧为预测。"""
    image = Image.open(image_path).convert("RGB")
    gt_image = image.copy()
    pred_image = image.copy()
    gt_draw = ImageDraw.Draw(gt_image)
    pred_draw = ImageDraw.Draw(pred_image)
    font = ImageFont.load_default()

    if draw_gt and "boxes" in target:
        gt_boxes = _scale_boxes_to_original(target["boxes"], target["orig_size"], resized_hw)
        gt_labels = target["labels"].detach().cpu().reshape(-1).tolist()
        for box, label in zip(gt_boxes.tolist(), gt_labels):
            x1, y1, x2, y2 = box
            name = id_to_name.get(int(label), str(int(label)))
            gt_draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            gt_draw.text((x1, max(0, y1 - 12)), f"GT {name}", fill="green", font=font)

    boxes = result["boxes"].detach().cpu()
    labels = result["labels"].detach().cpu()
    scores = result["scores"].detach().cpu()
    keep = scores >= score_threshold

    for box, label, score in zip(boxes[keep].tolist(), labels[keep].tolist(), scores[keep].tolist()):
        x1, y1, x2, y2 = box
        name = id_to_name.get(int(label), str(int(label)))
        pred_draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        pred_draw.text((x1, y1), f"{name} {score:.2f}", fill="red", font=font)

    width, height = image.size
    title_height = 24
    combined = Image.new("RGB", (width * 2, height + title_height), "white")
    header = ImageDraw.Draw(combined)
    header.rectangle([0, 0, width * 2, title_height], fill=(255, 255, 255))
    header.text((8, 6), "Ground Truth", fill="green", font=font)
    header.text((width + 8, 6), f"Prediction score>={score_threshold:.2f}", fill="red", font=font)
    combined.paste(gt_image, (0, title_height))
    combined.paste(pred_image, (width, title_height))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(save_path)


def _result_to_coco_json(image_id: int, result: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    """将单张图预测结果转成 COCO detections JSON 格式。"""
    boxes = result["boxes"].detach().cpu()
    labels = result["labels"].detach().cpu()
    scores = result["scores"].detach().cpu()

    records = []
    for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        x1, y1, x2, y2 = box
        records.append(
            {
                "image_id": int(image_id),
                "category_id": int(label),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            }
        )
    return records


def _to_numpy(value: Any):
    """将 torch.Tensor、numpy array 或普通序列统一转为 numpy array。"""
    import numpy as np

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.cpu().numpy()
    return np.asarray(value)


def _attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    """同时兼容 COCOeval params 对象和 dict。"""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _mean_valid(values: Any) -> Optional[float]:
    """计算 COCOeval 有效位置的均值；无效位置通常为 -1。"""
    import numpy as np

    arr = _to_numpy(values).astype(float)
    arr = arr[arr >= 0]
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _find_index(values: Sequence[Any], target: Any) -> Optional[int]:
    """在列表中查找目标值；浮点数用近似比较。"""
    for idx, value in enumerate(values):
        if isinstance(value, float) or isinstance(target, float):
            if math.isclose(float(value), float(target), rel_tol=1e-6, abs_tol=1e-6):
                return idx
        elif value == target:
            return idx
    return None


def _area_index(area_labels: Sequence[str], name: str) -> Optional[int]:
    """查找 all/small/medium/large 等面积范围标签对应的索引。"""
    lowered = [str(label).lower() for label in area_labels]
    return _find_index(lowered, name.lower())


def _metric_percent(value: Optional[float], digits: int) -> Optional[float]:
    """COCOeval 内部为 0~1，这里转成百分制显示。"""
    if value is None:
        return None
    return round(value * 100.0, digits)


def _load_categories(ann_file: Path) -> Dict[str, Any]:
    """读取 COCO 标注中的类别信息，并统计每类 GT 图片数和实例数。"""
    with ann_file.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = sorted(coco.get("categories", []), key=lambda item: item["id"])
    cat_by_id = {int(cat["id"]): cat for cat in categories}

    instance_count = {int(cat["id"]): 0 for cat in categories}
    image_ids_by_cat = {int(cat["id"]): set() for cat in categories}
    all_image_ids = set()
    all_instances = 0

    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        cat_id = int(ann["category_id"])
        if cat_id not in instance_count:
            instance_count[cat_id] = 0
            image_ids_by_cat[cat_id] = set()
        instance_count[cat_id] += 1
        image_ids_by_cat[cat_id].add(int(ann["image_id"]))
        all_instances += 1
        all_image_ids.add(int(ann["image_id"]))

    image_count = {cat_id: len(ids) for cat_id, ids in image_ids_by_cat.items()}
    return {
        "categories": categories,
        "cat_by_id": cat_by_id,
        "instance_count": instance_count,
        "image_count": image_count,
        "all_gt_images": len(all_image_ids),
        "all_gt_instances": all_instances,
    }


def _select_eval_axes(eval_data: Dict[str, Any], max_det: Any) -> Tuple[Any, Any, Dict[str, Any]]:
    """读取 COCOeval 张量和 params，并确定常用索引。"""
    import numpy as np

    precision = _to_numpy(eval_data["precision"])
    recall = _to_numpy(eval_data["recall"])
    params = eval_data.get("params")

    iou_thrs = list(_attr_or_key(params, "iouThrs", np.linspace(0.5, 0.95, 10)))
    cat_ids = list(_attr_or_key(params, "catIds", []))
    area_labels = list(_attr_or_key(params, "areaRngLbl", ["all", "small", "medium", "large"]))
    max_dets = list(_attr_or_key(params, "maxDets", []))

    all_area_idx = _area_index(area_labels, "all")
    if all_area_idx is None:
        all_area_idx = 0

    if max_det is None:
        max_det_idx = len(max_dets) - 1 if max_dets else precision.shape[-1] - 1
        selected_max_det = max_dets[max_det_idx] if max_dets else None
    else:
        max_det_idx = _find_index(max_dets, int(max_det))
        if max_det_idx is None:
            raise ValueError(f"max_det={max_det} not found in COCOeval maxDets={max_dets}")
        selected_max_det = int(max_det)

    axes = {
        "iou_thrs": iou_thrs,
        "cat_ids": cat_ids,
        "area_labels": area_labels,
        "max_dets": max_dets,
        "selected_max_det": selected_max_det,
        "max_det_idx": max_det_idx,
        "all_area_idx": all_area_idx,
        "iou50_idx": _find_index(iou_thrs, 0.5),
        "iou75_idx": _find_index(iou_thrs, 0.75),
        "small_idx": _area_index(area_labels, "small"),
        "medium_idx": _area_index(area_labels, "medium"),
        "large_idx": _area_index(area_labels, "large"),
    }
    return precision, recall, axes


def _metrics_from_slices(
    precision_slice: Any,
    recall_slice: Any,
    axes: Dict[str, Any],
    digits: int,
) -> Dict[str, Optional[float]]:
    """从已选 maxDets 的 precision/recall 切片中提取 AP/AR 指标。"""
    all_area_idx = axes["all_area_idx"]
    iou50_idx = axes["iou50_idx"]
    iou75_idx = axes["iou75_idx"]
    small_idx = axes["small_idx"]
    medium_idx = axes["medium_idx"]
    large_idx = axes["large_idx"]

    # precision_slice 的最后一维是面积范围，前面维度可以是 [T,R,K] 或 [T,R]。
    ap = _mean_valid(precision_slice[..., all_area_idx])
    ap50 = _mean_valid(precision_slice[iou50_idx, ..., all_area_idx]) if iou50_idx is not None else None
    ap75 = _mean_valid(precision_slice[iou75_idx, ..., all_area_idx]) if iou75_idx is not None else None
    aps = _mean_valid(precision_slice[..., small_idx]) if small_idx is not None else None
    apm = _mean_valid(precision_slice[..., medium_idx]) if medium_idx is not None else None
    apl = _mean_valid(precision_slice[..., large_idx]) if large_idx is not None else None

    # recall_slice 的最后一维是面积范围，前面维度可以是 [T,K] 或 [T]。
    ar = _mean_valid(recall_slice[..., all_area_idx])
    ar50 = _mean_valid(recall_slice[iou50_idx, ..., all_area_idx]) if iou50_idx is not None else None
    ar75 = _mean_valid(recall_slice[iou75_idx, ..., all_area_idx]) if iou75_idx is not None else None
    ars = _mean_valid(recall_slice[..., small_idx]) if small_idx is not None else None
    arm = _mean_valid(recall_slice[..., medium_idx]) if medium_idx is not None else None
    arl = _mean_valid(recall_slice[..., large_idx]) if large_idx is not None else None

    return {
        "AP": _metric_percent(ap, digits),
        "AP50": _metric_percent(ap50, digits),
        "AP75": _metric_percent(ap75, digits),
        "AP_small": _metric_percent(aps, digits),
        "AP_medium": _metric_percent(apm, digits),
        "AP_large": _metric_percent(apl, digits),
        "AR": _metric_percent(ar, digits),
        "AR50": _metric_percent(ar50, digits),
        "AR75": _metric_percent(ar75, digits),
        "AR_small": _metric_percent(ars, digits),
        "AR_medium": _metric_percent(arm, digits),
        "AR_large": _metric_percent(arl, digits),
    }


def _build_metric_rows(
    eval_data: Dict[str, Any],
    ann_info: Dict[str, Any],
    max_det: Any,
    digits: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """从 COCOeval precision/recall 张量中提取逐类别和全类别 AP/AR。"""
    precision, recall, axes = _select_eval_axes(eval_data, max_det)
    cat_ids = axes["cat_ids"] or [cat["id"] for cat in ann_info["categories"]]
    max_det_idx = axes["max_det_idx"]

    rows = []
    for k, cat_id in enumerate(cat_ids):
        cat_id = int(cat_id)
        cat = ann_info["cat_by_id"].get(cat_id, {"id": cat_id, "name": str(cat_id)})

        cls_precision = precision[:, :, k, :, max_det_idx]
        cls_recall = recall[:, k, :, max_det_idx]
        metric_values = _metrics_from_slices(cls_precision, cls_recall, axes, digits)

        row = {
            "category_id": cat_id,
            "class_name": cat.get("name", str(cat_id)),
            "gt_images": ann_info["image_count"].get(cat_id, 0),
            "gt_instances": ann_info["instance_count"].get(cat_id, 0),
        }
        row.update(metric_values)
        rows.append(row)

    all_precision = precision[:, :, :, :, max_det_idx]
    all_recall = recall[:, :, :, max_det_idx]
    all_metric_values = _metrics_from_slices(all_precision, all_recall, axes, digits)
    overall = {
        "category_id": "all",
        "class_name": "ALL_CLASSES",
        "gt_images": ann_info["all_gt_images"],
        "gt_instances": ann_info["all_gt_instances"],
    }
    overall.update(all_metric_values)

    meta = {
        "iou_thresholds": [float(x) for x in axes["iou_thrs"]],
        "area_labels": [str(x) for x in axes["area_labels"]],
        "max_dets": [int(x) for x in axes["max_dets"]] if axes["max_dets"] else [],
        "selected_max_det": axes["selected_max_det"],
        "num_categories": len(cat_ids),
    }
    return rows, overall, meta


def _sort_rows(rows: List[Dict[str, Any]], sort_by: Optional[str], descending: bool) -> List[Dict[str, Any]]:
    """按指定指标排序；没有该指标或值为空时排在最后。"""
    if not sort_by:
        return rows
    return sorted(
        rows,
        key=lambda row: float("-inf") if row.get(sort_by) is None else row.get(sort_by),
        reverse=descending,
    )


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """写出 CSV，便于用 Excel/WPS 或 pandas 继续分析。"""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int) -> str:
    """统一格式化输出到 Markdown 表格。"""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(rows: List[Dict[str, Any]], digits: int) -> str:
    """生成 Markdown 表格。"""
    if not rows:
        return ""
    fields = list(rows[0].keys())
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(field), digits) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def _write_markdown(path: Path, rows: List[Dict[str, Any]], meta: Dict[str, Any], digits: int, title: str) -> None:
    """写出 Markdown 报告。"""
    lines = [
        f"# {title}",
        "",
        f"- selected_max_det: {meta.get('selected_max_det')}",
        f"- num_categories: {meta.get('num_categories')}",
        "",
        _markdown_table(rows, digits),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _save_metrics_table_image(
    json_path: Path,
    output_path: Path,
    title: str,
    display_cols: Optional[Iterable[str]] = None,
) -> Optional[Path]:
    """可选：如果安装了 pandas/matplotlib，则把指标 JSON 渲染为 PNG 表格。"""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"[WARN] pandas/matplotlib unavailable, skip metrics table image: {exc}")
        return None

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows", [])
    if not rows:
        return None

    default_cols = ["class_name", "gt_images", "gt_instances", "AP", "AP50", "AP75", "AR", "AP_small", "AP_medium", "AP_large"]
    cols = list(display_cols or default_cols)
    existing = set().union(*(row.keys() for row in rows))
    cols = [col for col in cols if col in existing] or sorted(existing)

    df = pd.DataFrame(rows).reindex(columns=cols)
    for col in df.columns:
        if col == "class_name":
            df[col] = df[col].fillna("-").astype(str)
        else:
            df[col] = df[col].apply(lambda value: round(float(value), 2) if pd.notnull(value) else "-")

    fig_width = max(12.0, 1.35 * len(df.columns))
    fig_height = max(3.0, 0.42 * len(df.index) + 1.1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colColours=["#40466e"] * len(df.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)

    for (row_idx, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dce3")
        if row_idx == 0:
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f8f9fa" if row_idx % 2 == 0 else "white")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _save_metric_files(
    metrics_dir: Path,
    rows: List[Dict[str, Any]],
    overall: Dict[str, Any],
    meta: Dict[str, Any],
    digits: int,
) -> None:
    """保存 per-class、all-class 和合并指标文件。"""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    combined_rows = [overall] + rows

    per_class_json = metrics_dir / "per_class_metrics.json"
    _write_csv(metrics_dir / "per_class_metrics.csv", rows)
    _write_markdown(metrics_dir / "per_class_metrics.md", rows, meta, digits, "Per-class COCO Metrics")
    per_class_json.write_text(
        json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    all_class_json = metrics_dir / "all_class_metrics.json"
    _write_csv(metrics_dir / "all_class_metrics.csv", [overall])
    _write_markdown(metrics_dir / "all_class_metrics.md", [overall], meta, digits, "All-class COCO Metrics")
    all_class_json.write_text(
        json.dumps({"meta": meta, "row": overall, "rows": [overall]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    combined_json = metrics_dir / "class_and_all_metrics.json"
    _write_csv(metrics_dir / "class_and_all_metrics.csv", combined_rows)
    _write_markdown(metrics_dir / "class_and_all_metrics.md", combined_rows, meta, digits, "All-class + Per-class COCO Metrics")
    combined_json.write_text(
        json.dumps(
            {"meta": meta, "overall": overall, "rows": combined_rows, "per_class_rows": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _save_metrics_table_image(
        per_class_json,
        metrics_dir / "per_class_metrics_table.png",
        "Per-class COCO metrics",
    )
    _save_metrics_table_image(
        combined_json,
        metrics_dir / "class_and_all_metrics_table.png",
        "All-class + per-class COCO metrics",
    )


@torch.no_grad()
def evaluate_and_visualize(cfg_obj: YAMLConfig, cfg_dict: Dict[str, Any]) -> None:
    """执行模型推理、COCO 评估、指标统计和图片可视化。"""
    output_dir = Path(cfg_dict["output_dir"])
    vis_dir = output_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

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
    evaluator = cfg_obj.evaluator
    evaluator.cleanup()

    ann_file = Path(cfg_obj.yaml_cfg["val_dataloader"]["dataset"]["ann_file"])
    id_to_name = _category_maps(ann_file)["id_to_name"]

    predictions_json: List[Dict[str, Any]] = []
    saved_images = 0
    max_visualize = cfg_dict.get("max_visualize_images")
    score_threshold = float(cfg_dict["score_threshold"])

    for samples, targets in dataloader:
        samples = samples.to(device)
        targets_on_device = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}
            for target in targets
        ]

        outputs = model(samples)
        orig_target_sizes = torch.stack([target["orig_size"] for target in targets_on_device], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        coco_results = {
            target["image_id"].item(): result
            for target, result in zip(targets_on_device, results)
        }
        evaluator.update(coco_results)

        resized_hw = [int(samples.shape[-2]), int(samples.shape[-1])]
        for result, target in zip(results, targets):
            image_id = _tensor_to_int(target["image_id"])
            predictions_json.extend(_result_to_coco_json(image_id, result))

            should_save = max_visualize is None or saved_images < int(max_visualize)
            if should_save:
                image_path = _image_path_from_target(target, dataloader)
                save_path = vis_dir / f"{image_id}_{image_path.stem}.jpg"
                _draw_one_image(
                    image_path=image_path,
                    save_path=save_path,
                    result=result,
                    target=target,
                    id_to_name=id_to_name,
                    score_threshold=score_threshold,
                    draw_gt=bool(cfg_dict["draw_gt"]),
                    resized_hw=resized_hw,
                )
                saved_images += 1

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_data = evaluator.coco_eval["bbox"].eval
    torch.save(eval_data, eval_dir / "latest.pth")

    if cfg_dict.get("save_predictions_json"):
        with (output_dir / "predictions.json").open("w", encoding="utf-8") as f:
            json.dump(predictions_json, f, ensure_ascii=False, indent=2)

    ann_info = _load_categories(ann_file)
    rows, overall, meta = _build_metric_rows(eval_data, ann_info, cfg_dict.get("max_det"), int(cfg_dict["digits"]))
    rows = _sort_rows(rows, cfg_dict.get("sort_by"), bool(cfg_dict["sort_desc"]))

    metrics_dir = output_dir / "metrics"
    _save_metric_files(metrics_dir, rows, overall, meta, int(cfg_dict["digits"]))

    print(f"Saved visualizations: {vis_dir}")
    print(f"Saved eval dict: {eval_dir / 'latest.pth'}")
    if cfg_dict.get("save_predictions_json"):
        print(f"Saved predictions: {output_dir / 'predictions.json'}")
    print(f"Saved metrics: {metrics_dir}")


def parse_args() -> argparse.Namespace:
    """命令行参数会覆盖 CONFIG 配置区。"""
    parser = argparse.ArgumentParser(description="DEIM 推理、可视化和 COCO 指标统计")
    parser.add_argument("--config-yml", default=None, help="DEIM 配置文件")
    parser.add_argument("--checkpoint", default=None, help="训练好的 checkpoint")
    parser.add_argument("--img-folder", default=None, help="COCO 图片目录")
    parser.add_argument("--ann-file", default=None, help="COCO 标注 JSON")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--device", default=None, help="推理设备，例如 cuda:0 或 cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="推理 batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="dataloader worker 数")
    parser.add_argument("--score-threshold", type=float, default=None, help="可视化置信度阈值")
    parser.add_argument("--max-visualize-images", type=int, default=None, help="最多保存多少张可视化图")
    parser.add_argument("--no-draw-gt", action="store_true", help="可视化图片中不绘制 GT 框")
    parser.add_argument("--save-predictions-json", action="store_true", help="保存 COCO detections JSON")
    parser.add_argument("--sort-by", default=None, help="指标排序字段，例如 AP、AP50、AR")
    parser.add_argument("--max-det", type=int, default=None, help="使用哪个 COCO maxDets 档位")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dict = dict(CONFIG)

    cli_updates = {
        "config_yml": args.config_yml,
        "checkpoint": args.checkpoint,
        "img_folder": args.img_folder,
        "ann_file": args.ann_file,
        "output_dir": args.output_dir,
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "score_threshold": args.score_threshold,
        "max_visualize_images": args.max_visualize_images,
        "sort_by": args.sort_by,
        "max_det": args.max_det,
    }
    for key, value in cli_updates.items():
        if value is not None:
            cfg_dict[key] = value
    if args.no_draw_gt:
        cfg_dict["draw_gt"] = False
    if args.save_predictions_json:
        cfg_dict["save_predictions_json"] = True

    cfg_obj = _build_cfg(cfg_dict)
    evaluate_and_visualize(cfg_obj, cfg_dict)


if __name__ == "__main__":
    main()
