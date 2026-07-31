from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.test_unified_evaluator import evaluator


class TideBBoxLoaderTest(unittest.TestCase):
    def test_empty_segmentation_does_not_break_bbox_tide(self):
        annotation = {
            "images": [
                {"id": 1, "file_name": "1.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "2.jpg", "width": 100, "height": 100},
                {"id": 3, "file_name": "3.jpg", "width": 100, "height": 100},
            ],
            "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0, "segmentation": []},
                {"id": 2, "image_id": 2, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0, "segmentation": []},
            ],
        }
        detections = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "score": 0.9},
            {"image_id": 2, "category_id": 2, "bbox": [10, 10, 30, 30], "score": 0.8},
            {"image_id": 2, "category_id": 1, "bbox": [45, 45, 30, 30], "score": 0.7},
            {"image_id": 3, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.6},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.write_text(json.dumps(annotation), encoding="utf-8")
            result = evaluator.tide_metrics(path, detections)
        self.assertTrue(result["available"], result.get("reason"))
        self.assertEqual(result["run_name"], "sqalign_v3_bbox")
        for name in (
            "Cls_dAP", "Loc_dAP", "Both_dAP", "Dupe_dAP",
            "Bkg_dAP", "Miss_dAP", "FalsePos_dAP", "FalseNeg_dAP",
        ):
            self.assertIn(name, result)


if __name__ == "__main__":
    unittest.main()
