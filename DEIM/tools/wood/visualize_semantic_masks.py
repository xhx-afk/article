"""抽样可视化 COCO bbox 与 semantic RLE mask。"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from semantic_mask_data_utils import decode_segmentation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data = json.loads(args.coco.read_text(encoding="utf-8"))
    categories = {int(cat["id"]): str(cat["name"]) for cat in data.get("categories", [])}
    anns_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image[int(ann["image_id"])].append(ann)
    candidates = [image for image in data.get("images", []) if anns_by_image[int(image["id"])]]
    random.Random(args.seed).shuffle(candidates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    colors = [(230, 57, 70), (42, 157, 143), (29, 53, 87), (244, 162, 97), (131, 56, 236)]
    for image_info in candidates[: args.num_images]:
        image = Image.open(args.images_dir / str(image_info["file_name"])).convert("RGB")
        overlay = image.convert("RGBA")
        for index, ann in enumerate(anns_by_image[int(image_info["id"])]):
            color = colors[index % len(colors)]
            if ann.get("mask_valid"):
                mask = decode_segmentation(ann["segmentation"], image.height, image.width)
                alpha = Image.fromarray(mask.astype(np.uint8) * 100, mode="L")
                layer = Image.new("RGBA", image.size, color + (100,))
                overlay = Image.composite(layer, overlay, alpha)
            draw = ImageDraw.Draw(overlay)
            x, y, w, h = ann["bbox"]
            draw.rectangle((x, y, x + w, y + h), outline=color + (255,), width=2)
            draw.text((x, max(0, y - 12)), categories[int(ann["category_id"])], fill=color + (255,))
        overlay.convert("RGB").save(args.output_dir / f"{image_info['id']}_{Path(str(image_info['file_name'])).stem}.jpg")


if __name__ == "__main__":
    main()
