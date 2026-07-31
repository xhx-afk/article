"""将类别级 semantic maps 转换为逐实例 COCO RLE，供 SQ-Align 训练。"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from semantic_mask_data_utils import (
    EXPECTED_CLASSES,
    SemanticSpec,
    bbox_iou_xyxy,
    canonical_class_name,
    encode_rle,
    find_by_stem,
    load_class_map_json,
    parse_semantic_spec,
    semantic_binary_mask,
)


def _expanded_bbox(bbox: Sequence[float], width: int, height: int, ratio: float) -> Tuple[int, int, int, int]:
    x, y, w, h = (float(value) for value in bbox)
    x1 = max(0, math.floor(x - w * ratio))
    y1 = max(0, math.floor(y - h * ratio))
    x2 = min(width, math.ceil(x + w * (1.0 + ratio)))
    y2 = min(height, math.ceil(y + h * (1.0 + ratio)))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def _component_mask(
    class_mask: np.ndarray,
    bbox: Sequence[float],
    expand_ratio: float,
    min_area: int,
    score_weights: Tuple[float, float, float],
    min_score: float,
    merge_overlap: float,
) -> np.ndarray:
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("connected components 需要 scipy：pip install scipy") from exc

    height, width = class_mask.shape
    ex1, ey1, ex2, ey2 = _expanded_bbox(bbox, width, height, expand_ratio)
    local = class_mask[ey1:ey2, ex1:ex2]
    labels, count = ndimage.label(local, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(class_mask, dtype=bool)

    x, y, w, h = (float(value) for value in bbox)
    ann_box = (x, y, x + w, y + h)
    ann_cx, ann_cy = x + w / 2.0, y + h / 2.0
    diagonal = max(math.hypot(w, h), 1.0)
    candidates = []
    for component_id in range(1, count + 1):
        ys, xs = np.nonzero(labels == component_id)
        area = len(xs)
        if area < min_area:
            continue
        gx1, gy1 = ex1 + int(xs.min()), ey1 + int(ys.min())
        gx2, gy2 = ex1 + int(xs.max()) + 1, ey1 + int(ys.max()) + 1
        inside = (ex1 + xs >= x) & (ex1 + xs < x + w) & (ey1 + ys >= y) & (ey1 + ys < y + h)
        intersection_over_component = float(inside.sum()) / float(area)
        component_iou = bbox_iou_xyxy((gx1, gy1, gx2, gy2), ann_box)
        center_distance = math.hypot((gx1 + gx2) / 2.0 - ann_cx, (gy1 + gy2) / 2.0 - ann_cy)
        center_score = max(0.0, 1.0 - center_distance / diagonal)
        score = (
            score_weights[0] * intersection_over_component
            + score_weights[1] * component_iou
            + score_weights[2] * center_score
        )
        candidates.append((score, intersection_over_component, component_id))
    if not candidates:
        return np.zeros_like(class_mask, dtype=bool)
    candidates.sort(reverse=True)
    best_score = candidates[0][0]
    selected = [
        component_id for score, overlap, component_id in candidates
        if score >= min_score and (component_id == candidates[0][2] or overlap >= merge_overlap or score >= best_score * 0.8)
    ]
    if not selected:
        return np.zeros_like(class_mask, dtype=bool)
    output = np.zeros_like(class_mask, dtype=bool)
    output[ey1:ey2, ex1:ex2] = np.isin(labels, selected)
    return output


def _invalid(ann: Dict[str, object], reason: str, invalid_rows: List[Dict[str, object]]) -> None:
    ann["mask_valid"] = 0
    ann["segmentation"] = []
    invalid_rows.append({"annotation_id": ann.get("id"), "image_id": ann.get("image_id"), "reason": reason})


def _save_visualization(image_path: Path, anns: List[Dict[str, object]], masks: List[np.ndarray], output: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    colors = [(230, 57, 70, 100), (42, 157, 143, 100), (29, 53, 87, 100), (244, 162, 97, 100)]
    for index, (ann, mask) in enumerate(zip(anns, masks)):
        color = colors[index % len(colors)]
        alpha = Image.fromarray((mask.astype(np.uint8) * color[3]), mode="L")
        layer = Image.new("RGBA", image.size, color)
        overlay.alpha_composite(Image.composite(layer, Image.new("RGBA", image.size), alpha))
        x, y, w, h = ann["bbox"]
        ImageDraw.Draw(overlay).rectangle((x, y, x + w, y + h), outline=color[:3] + (255,), width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB").save(output)


def convert(args: argparse.Namespace) -> Dict[str, object]:
    input_data = json.loads(args.input_coco.read_text(encoding="utf-8"))
    output_data = copy.deepcopy(input_data)
    spec: SemanticSpec = load_class_map_json(args.class_map_json) if args.class_map_json else parse_semantic_spec(args.semantic_spec)

    category_by_id = {int(cat["id"]): cat for cat in input_data.get("categories", [])}
    canonical_by_id: Dict[int, str] = {}
    for cid, cat in category_by_id.items():
        canonical = canonical_class_name(str(cat["name"]))
        if canonical == "__overgrown__":
            raise ValueError("input COCO categories 中仍包含 overgrown，请先保持其删除状态。")
        if canonical is None or canonical.startswith("__"):
            raise ValueError(f"无法识别 COCO 类别：{cat['name']!r}")
        canonical_by_id[cid] = canonical
    expected = set(EXPECTED_CLASSES)
    if set(canonical_by_id.values()) != expected:
        raise ValueError(f"COCO 九类不符合预期。实际={sorted(canonical_by_id.values())}，预期={sorted(expected)}")
    missing_spec = sorted(expected - set(spec.class_values))
    if missing_spec:
        raise ValueError(f"semantic specification 缺少类别：{missing_spec}")

    images = {int(image["id"]): image for image in output_data.get("images", [])}
    anns_by_image: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for ann in output_data.get("annotations", []):
        anns_by_image[int(ann["image_id"])].append(ann)

    report_dir: Path = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    mismatch, missing_maps, empty_pixels, invalid_rows = [], [], [], []
    per_class = defaultdict(Counter)
    processed = 0
    vis_saved = 0

    for image_id, image_info in images.items():
        image_anns = anns_by_image.get(image_id, [])
        image_path = args.images_dir / str(image_info["file_name"])
        semantic_stem = Path(str(image_info["file_name"])).stem + args.semantic_suffix
        semantic_path = find_by_stem(args.semantic_maps_dir, semantic_stem)
        if semantic_path is None:
            missing_maps.append(str(image_info["file_name"]))
            for ann in image_anns:
                _invalid(ann, "missing_semantic_map", invalid_rows)
            continue
        semantic_image = Image.open(semantic_path)
        expected_size = (int(image_info["width"]), int(image_info["height"]))
        if semantic_image.size != expected_size:
            mismatch.append(f"{image_info['file_name']}\timage={expected_size}\tsemantic={semantic_image.size}")
            if args.resize_semantic_nearest:
                semantic_image = semantic_image.resize(expected_size, Image.Resampling.NEAREST)
            else:
                for ann in image_anns:
                    _invalid(ann, "size_mismatch", invalid_rows)
                continue
        semantic_array = np.asarray(semantic_image)
        class_masks: Dict[str, np.ndarray] = {}
        for canonical in expected:
            class_masks[canonical] = semantic_binary_mask(semantic_array, spec.class_values[canonical], spec.mode)
            if not class_masks[canonical].any():
                empty_pixels.append(f"{image_info['file_name']}\t{canonical}")

        valid_masks_for_vis: List[np.ndarray] = []
        valid_anns_for_vis: List[Dict[str, object]] = []
        for ann in image_anns:
            canonical = canonical_by_id[int(ann["category_id"])]
            mask = _component_mask(
                class_masks[canonical], ann["bbox"], args.expand_ratio, args.min_component_area,
                tuple(args.component_score_weights), args.min_component_score, args.merge_overlap,
            )
            per_class[canonical]["total"] += 1
            if not mask.any():
                per_class[canonical]["invalid"] += 1
                _invalid(ann, "empty_instance_mask", invalid_rows)
            else:
                ann["segmentation"] = encode_rle(mask)
                ann["mask_valid"] = 1
                per_class[canonical]["valid"] += 1
                valid_anns_for_vis.append(ann)
                valid_masks_for_vis.append(mask)
        processed += 1
        if vis_saved < args.num_visualizations and valid_masks_for_vis and image_path.exists():
            _save_visualization(image_path, valid_anns_for_vis, valid_masks_for_vis, report_dir / "sample_visualizations" / f"{image_id}.jpg")
            vis_saved += 1

    original_anns = {int(ann["id"]): ann for ann in input_data.get("annotations", [])}
    bbox_changed = category_changed = annotation_id_changed = 0
    for ann in output_data.get("annotations", []):
        original = original_anns.get(int(ann["id"]))
        if original is None:
            annotation_id_changed += 1
            continue
        bbox_changed += int(ann["bbox"] != original["bbox"])
        category_changed += int(ann["category_id"] != original["category_id"])
    annotation_id_changed += abs(len(original_anns) - len(output_data.get("annotations", [])))

    valid_count = sum(int(ann.get("mask_valid", 0)) for ann in output_data.get("annotations", []))
    total = len(output_data.get("annotations", []))
    summary = {
        "images_total": len(images),
        "images_processed": processed,
        "annotations_total": total,
        "valid_masks": valid_count,
        "invalid_masks": total - valid_count,
        "valid_ratio": valid_count / total if total else 0.0,
        "bbox_changed": bbox_changed,
        "category_changed": category_changed,
        "annotation_id_changed": annotation_id_changed,
    }
    args.output_coco.parent.mkdir(parents=True, exist_ok=True)
    args.output_coco.write_text(json.dumps(output_data, ensure_ascii=False), encoding="utf-8")
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for filename, rows in (
        ("size_mismatch.txt", mismatch),
        ("missing_semantic_map.txt", missing_maps),
        ("empty_class_pixels.txt", empty_pixels),
    ):
        (report_dir / filename).write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    with (report_dir / "invalid_masks.jsonl").open("w", encoding="utf-8") as file:
        for row in invalid_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (report_dir / "per_class_mask_valid.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["class_name", "total", "valid", "invalid", "valid_ratio"])
        writer.writeheader()
        for canonical in EXPECTED_CLASSES:
            counts = per_class[canonical]
            writer.writerow({
                "class_name": canonical,
                "total": counts["total"],
                "valid": counts["valid"],
                "invalid": counts["invalid"],
                "valid_ratio": counts["valid"] / counts["total"] if counts["total"] else 0.0,
            })
    if bbox_changed or category_changed or annotation_id_changed:
        raise RuntimeError(f"转换违反不变性要求：{summary}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment COCO annotations with semantic RLE masks")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--semantic-maps-dir", type=Path, required=True)
    parser.add_argument("--semantic-spec", type=Path)
    parser.add_argument("--class-map-json", type=Path)
    parser.add_argument("--input-coco", type=Path, required=True)
    parser.add_argument("--output-coco", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--semantic-suffix", default="_segm")
    parser.add_argument("--resize-semantic-nearest", action="store_true")
    parser.add_argument("--expand-ratio", type=float, default=0.03)
    parser.add_argument("--min-component-area", type=int, default=2)
    parser.add_argument("--component-score-weights", nargs=3, type=float, default=(0.50, 0.35, 0.15))
    parser.add_argument("--min-component-score", type=float, default=0.05)
    parser.add_argument("--merge-overlap", type=float, default=0.20)
    parser.add_argument("--num-visualizations", type=int, default=20)
    args = parser.parse_args()
    if bool(args.semantic_spec) == bool(args.class_map_json):
        parser.error("必须且只能指定 --semantic-spec 或 --class-map-json 其中一个。")
    if not math.isclose(sum(args.component_score_weights), 1.0, abs_tol=1e-6):
        parser.error("--component-score-weights 三项之和必须为 1。")
    return args


if __name__ == "__main__":
    result = convert(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
