from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "wood"))
from augment_coco_with_semantic_masks import convert  # noqa: E402
from sqmal_data_utils import EXPECTED_CLASSES, decode_segmentation  # noqa: E402


class DatasetToolTest(unittest.TestCase):
    def test_gray_map_to_instance_rle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images_dir = root / "images"
            maps_dir = root / "maps"
            report_dir = root / "report"
            images_dir.mkdir()
            maps_dir.mkdir()
            width, height = 90, 20
            Image.new("RGB", (width, height), "white").save(images_dir / "sample.jpg")
            semantic = np.zeros((height, width), dtype=np.uint8)
            categories, annotations = [], []
            for index, name in enumerate(EXPECTED_CLASSES, start=1):
                semantic[:, (index - 1) * 10:index * 10] = index
                categories.append({"id": index, "name": name})
                annotations.append({
                    "id": index,
                    "image_id": 1,
                    "category_id": index,
                    "bbox": [(index - 1) * 10, 0, 10, height],
                    "area": 10 * height,
                    "iscrowd": 0,
                    "segmentation": [],
                })
            Image.fromarray(semantic).save(maps_dir / "sample_segm.png")
            coco = {
                "images": [{"id": 1, "file_name": "sample.jpg", "width": width, "height": height}],
                "categories": categories,
                "annotations": annotations,
            }
            input_coco = root / "input.json"
            output_coco = root / "output.json"
            spec = root / "spec.txt"
            input_coco.write_text(json.dumps(coco), encoding="utf-8")
            spec.write_text("\n".join(f"{name}: {i}" for i, name in enumerate(EXPECTED_CLASSES, 1)), encoding="utf-8")
            args = argparse.Namespace(
                images_dir=images_dir,
                semantic_maps_dir=maps_dir,
                semantic_spec=spec,
                class_map_json=None,
                input_coco=input_coco,
                output_coco=output_coco,
                report_dir=report_dir,
                semantic_suffix="_segm",
                resize_semantic_nearest=False,
                expand_ratio=0.03,
                min_component_area=2,
                component_score_weights=(0.5, 0.35, 0.15),
                min_component_score=0.05,
                merge_overlap=0.2,
                num_visualizations=1,
            )
            summary = convert(args)
            self.assertEqual(summary["valid_masks"], 9)
            self.assertEqual(summary["bbox_changed"], 0)
            self.assertEqual(summary["category_changed"], 0)
            self.assertEqual(summary["annotation_id_changed"], 0)
            output = json.loads(output_coco.read_text(encoding="utf-8"))
            for old, new in zip(annotations, output["annotations"]):
                self.assertEqual(old["bbox"], new["bbox"])
                self.assertEqual(old["category_id"], new["category_id"])
                self.assertEqual(old["id"], new["id"])
                self.assertEqual(new["mask_valid"], 1)
                self.assertTrue(decode_segmentation(new["segmentation"], height, width).any())


if __name__ == "__main__":
    unittest.main()

