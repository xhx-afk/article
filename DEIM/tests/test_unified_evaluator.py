from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tests.sqalign_test_utils import ROOT, install_optional_dependency_stubs


install_optional_dependency_stubs()


def load_evaluator():
    path = ROOT / "tools" / "wood" / "evaluate_sqalign_v3.py"
    spec = importlib.util.spec_from_file_location("sqalign_v3_evaluator_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


evaluator = load_evaluator()


class UnifiedEvaluatorTest(unittest.TestCase):
    def test_ten_image_metrics_serialize_to_one_complete_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = {
                "info": {}, "licenses": [],
                "images": [{"id": index, "width": 32, "height": 32, "file_name": f"{index}.jpg"}
                           for index in range(10)],
                "categories": [{"id": 0, "name": "defect"}],
                "annotations": [{"id": index + 1, "image_id": index, "category_id": 0,
                                 "bbox": [8, 8, 16, 16], "area": 256, "iscrowd": 0}
                                for index in range(10)],
            }
            annotation = root / "annotations.json"
            annotation.write_text(json.dumps(dataset), encoding="utf-8")
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"sqalign-v3-test-checkpoint")
            predictions = [{"image_id": index, "category_id": 0, "bbox": [8, 8, 16, 16], "score": .9}
                           for index in range(10)]
            rows = {
                index: [{
                    "image_id": index, "label": 0,
                    "box": [8, 8, 24, 24], "score": .9,
                }]
                for index in range(10)
            }
            ground_truth = {
                index: [{"box": [8, 8, 24, 24], "label": 0}]
                for index in range(10)
            }
            result = evaluator.empty_report()
            result["meta"] = {"checkpoint_sha256": evaluator.sha256_file(checkpoint)}
            result["coco"] = evaluator.coco_metrics(annotation, predictions, 300)
            _, overall = evaluator.taxonomy_report(
                rows, ground_truth, .05
            )
            result["fixed_thresholds"] = {"0.05": overall}
            result["error_taxonomy"] = overall
            result["automatic_diagnosis"] = evaluator.automatic_diagnosis(result)
            output = root / "unified.json"
            output.write_text(
                json.dumps(evaluator.finite_json_value(result), allow_nan=False),
                encoding="utf-8",
            )
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(set(loaded), set(evaluator.TOP_LEVEL_KEYS))
            self.assertEqual(len(list(root.glob("unified.json"))), 1)
            self.assertEqual(len(loaded["meta"]["checkpoint_sha256"]), 64)
            self.assertIn("AP@[0.50:0.95]", loaded["coco"])
            self.assertIn("AR100", loaded["coco"])
            self.assertEqual(loaded["fixed_thresholds"]["0.05"]["TP"], 10)

    def test_annotation_free_image_background_fp_is_counted(self):
        rows = {
            1: [{"image_id": 1, "label": 0, "box": [1, 1, 4, 4], "score": .8}],
        }
        _, summary = evaluator.taxonomy_report(rows, {1: []}, .05)
        self.assertEqual(summary["far_background_FP"], 1)
        self.assertEqual(summary["FP"], 1)


if __name__ == "__main__":
    unittest.main()
