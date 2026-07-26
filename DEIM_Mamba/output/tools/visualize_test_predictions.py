"""
DEIM 测试集预测可视化脚本。

输出图片为左右拼接：
    左图：真实 GT 标注框
    右图：模型预测框

支持：
1. 在 CONFIG["selected_pred_classes"] 中指定只显示哪些预测类别；
2. 在 CONFIG["max_visualize_images"] 中指定最多保存多少张图片；
3. 预测框先画框、后画文字，文字带背景并做简单避让，减少重叠框影响类别文字的问题；
4. 可选让左侧 GT 也按同样类别过滤，便于只检查少数类别。

运行示例：
    python output/tools/visualize_test_predictions.py

评估 test 集时覆盖路径：
    python output/tools/visualize_test_predictions.py \
        --img-folder ./data/xxx/images/test \
        --ann-file ./data/xxx/annotations/instances_test.json \
        --output-dir output/dfine_hgnetv2_l_wood_multiple_960_fiter50/test_visualizations
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS_DIR))

from evaluate_and_visualize import (  # noqa: E402
    _build_cfg,
    _category_maps,
    _image_path_from_target,
    _load_checkpoint_state,
    _scale_boxes_to_original,
    _tensor_to_int,
)


# =========================
# 用户配置区
# =========================
CONFIG = {
    # DEIM 配置文件。
    "config_yml": "configs/deim_dfine/deim_hgnetv2_l_wood.yml",

    # 训练好的 checkpoint。
    "checkpoint": "output/dfine_hgnetv2_l_wood_multiple_960_fiter50/best_stg2.pth",

    # 测试/验证图片目录。None 表示使用 config_yml 中 val_dataloader.dataset.img_folder。
    "img_folder": None,

    # 测试/验证 COCO 标注 JSON。None 表示使用 config_yml 中 val_dataloader.dataset.ann_file。
    "ann_file": None,

    # 可视化结果输出目录。
    "output_dir": "output/dfine_hgnetv2_l_wood_multiple_960_fiter50/test_visualizations",

    # 推理设备，例如 "cuda:0" 或 "cpu"。
    "device": "cuda:0",

    # 单进程推理 batch size。
    "batch_size": 12,

    # dataloader worker 数。Windows 本地跑可先改成 0。
    "num_workers": 4,

    # 预测框置信度阈值。
    "score_threshold": 0.1,

    # 最多保存多少张图片；None 表示保存全部。
    "max_visualize_images": 50,

    # 只显示这些预测类别。空列表或 None 表示显示所有预测类别。
    # 示例：["Death_Kont", "Live_Kont"]
    "selected_pred_classes": ["Death_Kont", "Live_Kont"],

    # 左侧 GT 是否也按 selected_pred_classes 过滤。
    # False：左侧显示全部 GT；True：左侧只显示同名类别 GT。
    "filter_gt_to_selected_classes": False,

    # True：只保存至少有一个可见预测框的图片；False：即使右侧没有可见预测也保存。
    "save_only_images_with_visible_predictions": False,

    # 单张图最多显示多少个预测框；None 表示不限制。
    "max_predictions_per_image": None,

    # 输出文件名前缀。
    "file_prefix": "vis",
}


def _load_categories(ann_file: Path) -> Tuple[Dict[int, str], Dict[str, Set[int]]]:
    """读取 COCO categories，返回 id->name 和 name->ids。"""
    with ann_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_name = {
        int(cat["id"]): cat.get("name", str(cat["id"]))
        for cat in data.get("categories", [])
    }
    name_to_ids: Dict[str, Set[int]] = {}
    for cat_id, name in id_to_name.items():
        name_to_ids.setdefault(name, set()).add(cat_id)
        name_to_ids.setdefault(name.lower(), set()).add(cat_id)
    return id_to_name, name_to_ids


def _selected_class_ids(
    selected_names: Optional[Sequence[str]],
    name_to_ids: Dict[str, Set[int]],
) -> Optional[Set[int]]:
    """把类别名白名单转成 category_id 集合。None 表示不过滤。"""
    if not selected_names:
        return None

    selected_ids: Set[int] = set()
    missing = []
    for name in selected_names:
        ids = name_to_ids.get(name) or name_to_ids.get(name.lower())
        if not ids:
            missing.append(name)
            continue
        selected_ids.update(ids)

    if missing:
        print(f"[WARN] These class names were not found in COCO categories: {missing}")
    return selected_ids


def _color_for_label(label: int) -> Tuple[int, int, int]:
    """为类别生成稳定颜色。"""
    palette = [
        (230, 57, 70),
        (29, 53, 87),
        (42, 157, 143),
        (233, 196, 106),
        (244, 162, 97),
        (131, 56, 236),
        (58, 134, 255),
        (255, 0, 110),
        (38, 70, 83),
        (0, 150, 136),
    ]
    return palette[int(label) % len(palette)]


def _text_bbox(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.ImageFont) -> Tuple[int, int, int, int]:
    """兼容不同 Pillow 版本的文字 bbox。"""
    if hasattr(draw, "textbbox"):
        return tuple(int(v) for v in draw.textbbox(xy, text, font=font))
    width, height = draw.textsize(text, font=font)
    x, y = xy
    return x, y, x + int(width), y + int(height)


def _clamp_label_origin(
    x: int,
    y: int,
    label_w: int,
    label_h: int,
    image_w: int,
    image_h: int,
) -> Tuple[int, int]:
    """把文字背景限制在图片内部。"""
    x = max(0, min(int(x), max(0, image_w - label_w)))
    y = max(0, min(int(y), max(0, image_h - label_h)))
    return x, y


def _intersects(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    """判断两个矩形是否相交。"""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _place_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    box: Sequence[float],
    occupied: List[Tuple[int, int, int, int]],
    image_size: Tuple[int, int],
    pad: int = 3,
) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
    """
    为类别文字寻找位置。

    策略：
    1. 优先放在框外上方，其次框内、下方、左右侧；
    2. 所有候选位置都和已有文字冲突时，向下逐行寻找空位；
    3. 文字背景最后绘制，确保不会被后续框线盖住。
    """
    image_w, image_h = image_size
    x1, y1, x2, y2 = [int(round(v)) for v in box]

    raw_bbox = _text_bbox(draw, (0, 0), text, font)
    text_w = max(1, raw_bbox[2] - raw_bbox[0])
    text_h = max(1, raw_bbox[3] - raw_bbox[1])
    label_w = text_w + pad * 2
    label_h = text_h + pad * 2

    candidates = [
        (x1, y1 - label_h - 2),           # 框外上方左对齐
        (x1, y1 + 2),                     # 框内左上
        (x1, y2 + 2),                     # 框外下方
        (x2 - label_w, y1 - label_h - 2), # 框外上方右对齐
        (x2 + 2, y1),                     # 右侧
        (x1 - label_w - 2, y1),           # 左侧
    ]

    for cx, cy in candidates:
        cx, cy = _clamp_label_origin(cx, cy, label_w, label_h, image_w, image_h)
        rect = (cx, cy, cx + label_w, cy + label_h)
        if not any(_intersects(rect, old) for old in occupied):
            return (cx + pad, cy + pad), rect

    # 候选位置都冲突时，从框上方附近开始逐行寻找，尽量减少文字互相覆盖。
    start_y = max(0, min(y1, image_h - label_h))
    for step in range(0, image_h, label_h + 2):
        for sign in (1, -1):
            cy = start_y + sign * step
            cx, cy = _clamp_label_origin(x1, cy, label_w, label_h, image_w, image_h)
            rect = (cx, cy, cx + label_w, cy + label_h)
            if not any(_intersects(rect, old) for old in occupied):
                return (cx + pad, cy + pad), rect

    # 极端拥挤时只能放在框内左上，但仍最后绘制文字，避免被框线盖住。
    cx, cy = _clamp_label_origin(x1, y1, label_w, label_h, image_w, image_h)
    rect = (cx, cy, cx + label_w, cy + label_h)
    return (cx + pad, cy + pad), rect


def _draw_boxes_with_labels(
    image: Image.Image,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    id_to_name: Dict[int, str],
    *,
    scores: Optional[Sequence[float]] = None,
    selected_ids: Optional[Set[int]] = None,
    score_threshold: Optional[float] = None,
    max_items: Optional[int] = None,
    is_prediction: bool = False,
) -> int:
    """先画所有框，再统一画带背景的文字，避免重叠框线盖住类别文字。"""
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    items = []
    for idx, (box, label) in enumerate(zip(boxes, labels)):
        label = int(label)
        score = None if scores is None else float(scores[idx])
        if selected_ids is not None and label not in selected_ids:
            continue
        if score_threshold is not None and score is not None and score < score_threshold:
            continue
        items.append((box, label, score))

    if scores is not None:
        items.sort(key=lambda item: -float(item[2] or 0.0))
    if max_items is not None:
        items = items[: int(max_items)]

    label_payloads = []
    for box, label, score in items:
        x1, y1, x2, y2 = [float(v) for v in box]
        color = _color_for_label(label)
        width = 3 if is_prediction else 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        name = id_to_name.get(label, str(label))
        text = f"{name} {score:.2f}" if score is not None else f"GT {name}"
        label_payloads.append((box, label, text, color))

    occupied: List[Tuple[int, int, int, int]] = []
    for box, _label, text, color in label_payloads:
        text_xy, bg_rect = _place_label(draw, text, font, box, occupied, image.size)
        occupied.append(bg_rect)
        draw.rectangle(bg_rect, fill=(255, 255, 255), outline=color, width=1)
        draw.text(text_xy, text, fill=color, font=font)

    return len(items)


def _combined_canvas(gt_image: Image.Image, pred_image: Image.Image, score_threshold: float) -> Image.Image:
    """拼成左右对比图。"""
    width, height = gt_image.size
    title_height = 28
    combined = Image.new("RGB", (width * 2, height + title_height), "white")
    draw = ImageDraw.Draw(combined)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, width * 2, title_height], fill=(255, 255, 255))
    draw.text((8, 8), "Ground Truth", fill=(0, 128, 0), font=font)
    draw.text((width + 8, 8), f"Prediction score>={score_threshold:.2f}", fill=(200, 0, 0), font=font)
    combined.paste(gt_image, (0, title_height))
    combined.paste(pred_image, (width, title_height))
    return combined


def _visualize_one(
    image_path: Path,
    save_path: Path,
    result: Dict[str, torch.Tensor],
    target: Dict[str, Any],
    id_to_name: Dict[int, str],
    pred_selected_ids: Optional[Set[int]],
    gt_selected_ids: Optional[Set[int]],
    score_threshold: float,
    max_predictions_per_image: Optional[int],
    resized_hw: List[int],
) -> int:
    """保存单张可视化图片，返回可见预测框数量。"""
    image = Image.open(image_path).convert("RGB")
    gt_image = image.copy()
    pred_image = image.copy()

    gt_boxes = _scale_boxes_to_original(target["boxes"], target["orig_size"], resized_hw)
    gt_labels = target["labels"].detach().cpu().reshape(-1).tolist()
    _draw_boxes_with_labels(
        gt_image,
        gt_boxes.tolist(),
        gt_labels,
        id_to_name,
        selected_ids=gt_selected_ids,
        is_prediction=False,
    )

    pred_boxes = result["boxes"].detach().cpu().tolist()
    pred_labels = result["labels"].detach().cpu().reshape(-1).tolist()
    pred_scores = result["scores"].detach().cpu().reshape(-1).tolist()
    visible_pred_count = _draw_boxes_with_labels(
        pred_image,
        pred_boxes,
        pred_labels,
        id_to_name,
        scores=pred_scores,
        selected_ids=pred_selected_ids,
        score_threshold=score_threshold,
        max_items=max_predictions_per_image,
        is_prediction=True,
    )

    combined = _combined_canvas(gt_image, pred_image, score_threshold)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(save_path)
    return visible_pred_count


@torch.no_grad()
def visualize_predictions(cfg_obj: Any, cfg_dict: Dict[str, Any]) -> None:
    """执行推理并保存测试集可视化图片。"""
    output_dir = Path(cfg_dict["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

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

    ann_file = Path(cfg_obj.yaml_cfg["val_dataloader"]["dataset"]["ann_file"])
    id_to_name, name_to_ids = _load_categories(ann_file)
    pred_selected_ids = _selected_class_ids(cfg_dict.get("selected_pred_classes"), name_to_ids)
    gt_selected_ids = pred_selected_ids if cfg_dict.get("filter_gt_to_selected_classes") else None

    selected_names = cfg_dict.get("selected_pred_classes")
    print(f"Selected prediction classes: {selected_names or 'ALL'}")
    print(f"Selected prediction category ids: {sorted(pred_selected_ids) if pred_selected_ids else 'ALL'}")

    saved = 0
    seen = 0
    max_images = cfg_dict.get("max_visualize_images")
    score_threshold = float(cfg_dict["score_threshold"])
    save_only_visible = bool(cfg_dict["save_only_images_with_visible_predictions"])

    for samples, targets in dataloader:
        samples = samples.to(device)
        targets_on_device = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}
            for target in targets
        ]

        outputs = model(samples)
        orig_target_sizes = torch.stack([target["orig_size"] for target in targets_on_device], dim=0)
        results = postprocessor(outputs, orig_target_sizes)

        resized_hw = [int(samples.shape[-2]), int(samples.shape[-1])]
        for result, target in zip(results, targets):
            seen += 1
            image_id = _tensor_to_int(target["image_id"])
            image_path = _image_path_from_target(target, dataloader)
            save_path = output_dir / f"{cfg_dict['file_prefix']}_{saved + 1:06d}_{image_id}_{image_path.stem}.jpg"

            visible_count = _visualize_one(
                image_path=image_path,
                save_path=save_path,
                result=result,
                target=target,
                id_to_name=id_to_name,
                pred_selected_ids=pred_selected_ids,
                gt_selected_ids=gt_selected_ids,
                score_threshold=score_threshold,
                max_predictions_per_image=cfg_dict.get("max_predictions_per_image"),
                resized_hw=resized_hw,
            )

            if save_only_visible and visible_count == 0:
                if save_path.exists():
                    save_path.unlink()
                continue

            saved += 1
            if max_images is not None and saved >= int(max_images):
                print(f"Saved {saved} visualized images to: {output_dir}")
                print(f"Scanned images: {seen}")
                return

    print(f"Saved {saved} visualized images to: {output_dir}")
    print(f"Scanned images: {seen}")


def _parse_class_list(value: Optional[str]) -> Optional[List[str]]:
    """解析命令行传入的类别列表，逗号分隔；空字符串表示不过滤。"""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    """命令行参数会覆盖 CONFIG 配置区。"""
    parser = argparse.ArgumentParser(description="DEIM 测试集预测可视化")
    parser.add_argument("--config-yml", default=None, help="DEIM 配置文件")
    parser.add_argument("--checkpoint", default=None, help="训练好的 checkpoint")
    parser.add_argument("--img-folder", default=None, help="COCO 图片目录")
    parser.add_argument("--ann-file", default=None, help="COCO 标注 JSON")
    parser.add_argument("--output-dir", default=None, help="可视化输出目录")
    parser.add_argument("--device", default=None, help="推理设备，例如 cuda:0 或 cpu")
    parser.add_argument("--batch-size", type=int, default=None, help="推理 batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="dataloader worker 数")
    parser.add_argument("--score-threshold", type=float, default=None, help="预测框置信度阈值")
    parser.add_argument("--max-visualize-images", type=int, default=None, help="最多保存多少张可视化图")
    parser.add_argument("--selected-classes", default=None, help="只显示这些预测类别，逗号分隔；空字符串表示全部")
    parser.add_argument("--filter-gt", action="store_true", help="左侧 GT 也按 selected classes 过滤")
    parser.add_argument("--save-only-visible", action="store_true", help="只保存至少有一个可见预测框的图片")
    parser.add_argument("--max-predictions-per-image", type=int, default=None, help="每张图最多显示多少个预测框")
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
        "max_predictions_per_image": args.max_predictions_per_image,
    }
    for key, value in cli_updates.items():
        if value is not None:
            cfg_dict[key] = value

    selected_classes = _parse_class_list(args.selected_classes)
    if selected_classes is not None:
        cfg_dict["selected_pred_classes"] = selected_classes
    if args.filter_gt:
        cfg_dict["filter_gt_to_selected_classes"] = True
    if args.save_only_visible:
        cfg_dict["save_only_images_with_visible_predictions"] = True

    cfg_obj = _build_cfg(cfg_dict)
    visualize_predictions(cfg_obj, cfg_dict)


if __name__ == "__main__":
    main()
