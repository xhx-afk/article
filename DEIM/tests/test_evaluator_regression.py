from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.test_unified_evaluator import evaluator


class EvaluatorRegressionTest(unittest.TestCase):
    def test_fixed_ten_image_class_only_coco_path_is_bit_identical(self):
        dataset = {
            "images": [
                {"id": index, "width": 32, "height": 32, "file_name": f"{index}.jpg"}
                for index in range(10)
            ],
            "categories": [{"id": 1, "name": "defect"}],
            "annotations": [
                {"id": index + 1, "image_id": index, "category_id": 1,
                 "bbox": [8, 8, 16, 16], "area": 256, "iscrowd": 0}
                for index in range(10)
            ],
        }
        detections = [
            {"image_id": index, "category_id": 1, "bbox": [8, 8, 16, 16], "score": 0.9}
            for index in range(10)
        ]
        with tempfile.TemporaryDirectory() as directory:
            annotation = Path(directory) / "annotations.json"
            annotation.write_text(json.dumps(dataset), encoding="utf-8")
            before = evaluator.coco_metrics(annotation, detections, 300)
            after = evaluator.coco_metrics(annotation, list(detections), 300)
        for key in ("AP@[0.50:0.95]", "AP50", "AP75"):
            self.assertLess(abs(before[key] - after[key]), 1e-8)

    def test_impossible_full_v4_candidate_zero_is_error(self):
        report = evaluator.empty_report()
        report["meta"] = {"config_path": "v4_near_gt_candidate_rank.yml"}
        report["dataset"] = {"num_images": 10, "evaluated_images": 10, "per_class_GT": {"a": 30}}
        report["candidate_ranking"] = {
            "num_GT": 10, "GT_with_at_least_2_candidates": 0, "ranking_pair_count": 0
        }
        report["tide"] = {"available": True}
        diagnosis = evaluator.automatic_diagnosis(report)
        self.assertIn("candidate_diagnostics_impossible_zero", diagnosis["errors"])


if __name__ == "__main__":
    unittest.main()
