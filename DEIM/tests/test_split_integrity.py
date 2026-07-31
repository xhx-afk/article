from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.wood.check_sqalign_split_integrity import check_split_integrity
from tools.wood.compare_sqalign_val_test import compare_reports


def write_split(root: Path, name: str, image_id: int, file_name: str, content: bytes, category_count: int = 1):
    images = root / name
    images.mkdir()
    (images / file_name).write_bytes(content)
    annotation = {
        "images": [{"id": image_id, "file_name": file_name, "width": 20, "height": 20}],
        "categories": [{"id": 1, "name": "defect"}],
        "annotations": [
            {"id": index + 1, "image_id": image_id, "category_id": 1,
             "bbox": [1, 1, 5, 4], "iscrowd": 0, "segmentation": [[1, 1, 6, 1, 6, 5]]}
            for index in range(category_count)
        ],
    }
    path = root / f"{name}.json"
    path.write_text(json.dumps(annotation), encoding="utf-8")
    return images, path


class SplitIntegrityTest(unittest.TestCase):
    def test_cross_split_file_id_hash_stem_and_source_group_leaks_are_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = write_split(root, "train", 1, "board01_crop.jpg", b"same")
            val = write_split(root, "val", 1, "board01_crop.jpg", b"same")
            test = write_split(root, "test", 3, "board03_crop.jpg", b"different")
            report = check_split_integrity(
                {"train": train, "val": val, "test": test}, r"board(\d+)"
            )
        self.assertGreater(report["overlaps"]["exact_file_overlap_count"], 0)
        self.assertGreater(report["overlaps"]["sha256_overlap_count"], 0)
        self.assertGreater(report["overlaps"]["source_group_overlap_count"], 0)
        self.assertIn("split_exact_file_overlap", report["errors"])
        self.assertIn("split_sha256_overlap", report["errors"])
        self.assertIn("split_image_id_overlap", report["errors"])

    def test_clean_splits_have_no_overlap_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "train": write_split(root, "train", 1, "board01.jpg", b"train"),
                "val": write_split(root, "val", 2, "board02.jpg", b"val"),
                "test": write_split(root, "test", 3, "board03.jpg", b"test"),
            }
            report = check_split_integrity(paths, r"board(\d+)")
        self.assertEqual(report["errors"], [])

    def test_val_test_comparison_includes_tail_filtered_macro_and_bootstrap(self):
        def report(ap, counts):
            return {"coco": {
                "AP@[0.50:0.95]": ap, "AP50": ap + 0.1, "AP75": ap - 0.1,
                "per_class": {
                    name: {"AP": value, "GT_count": counts[name]}
                    for name, value in {"a": ap, "b": ap - 0.2, "c": ap + 0.1}.items()
                },
            }}
        result = compare_reports(
            report(0.5, {"a": 50, "b": 5, "c": 40}),
            report(0.4, {"a": 55, "b": 4, "c": 45}),
            bootstrap_samples=100,
            seed=7,
        )
        self.assertAlmostEqual(result["overall"]["AP"]["difference_test_minus_val"], -0.1)
        self.assertEqual(result["macro_AP"]["GT_at_least_20_diagnostic"]["class_count"], 2)
        self.assertTrue(result["bootstrap"]["available"])
        self.assertTrue(result["largest_gap_contributing_classes"])


if __name__ == "__main__":
    unittest.main()
