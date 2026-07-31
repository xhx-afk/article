"""校验 semantic COCO 的 RLE、mask_valid 及 bbox/category/id 不变性。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from semantic_mask_data_utils import decode_segmentation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-coco", type=Path, required=True)
    parser.add_argument("--original-coco", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    data = json.loads(args.semantic_coco.read_text(encoding="utf-8"))
    images = {int(image["id"]): image for image in data.get("images", [])}
    errors = []
    valid = 0
    per_class = Counter()
    for ann in data.get("annotations", []):
        image = images.get(int(ann["image_id"]))
        if image is None:
            errors.append(f"ann={ann['id']}: missing image")
            continue
        mask_valid = bool(ann.get("mask_valid", bool(ann.get("segmentation"))))
        if mask_valid:
            try:
                mask = decode_segmentation(ann.get("segmentation"), int(image["height"]), int(image["width"]))
                if not mask.any():
                    errors.append(f"ann={ann['id']}: mask_valid=1 but empty mask")
                else:
                    valid += 1
                    per_class[int(ann["category_id"])] += 1
            except Exception as exc:
                errors.append(f"ann={ann['id']}: RLE decode failed: {exc}")
        elif ann.get("segmentation"):
            errors.append(f"ann={ann['id']}: mask_valid=0 but segmentation is not empty")

    changed = {"bbox_changed": 0, "category_changed": 0, "annotation_id_changed": 0}
    if args.original_coco:
        original = json.loads(args.original_coco.read_text(encoding="utf-8"))
        original_by_id = {int(ann["id"]): ann for ann in original.get("annotations", [])}
        for ann in data.get("annotations", []):
            old = original_by_id.get(int(ann["id"]))
            if old is None:
                changed["annotation_id_changed"] += 1
                continue
            changed["bbox_changed"] += int(ann["bbox"] != old["bbox"])
            changed["category_changed"] += int(ann["category_id"] != old["category_id"])
        changed["annotation_id_changed"] += abs(len(original_by_id) - len(data.get("annotations", [])))
    summary = {
        "images": len(images),
        "annotations": len(data.get("annotations", [])),
        "valid_masks": valid,
        "valid_ratio": valid / len(data.get("annotations", [])) if data.get("annotations") else 0.0,
        "per_category_valid_masks": dict(sorted(per_class.items())),
        **changed,
        "errors": errors,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    if errors or any(changed.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
