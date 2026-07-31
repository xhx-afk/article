"""Check SQ-Align train/val/test splits for leakage and distribution drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _mask_is_valid(annotation: Mapping[str, object]) -> bool:
    if "mask_valid" in annotation:
        return bool(annotation["mask_valid"])
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list):
        return any(bool(polygon) for polygon in segmentation)
    if isinstance(segmentation, Mapping):
        return bool(segmentation.get("counts"))
    return False


def inspect_split(
    name: str,
    images_dir: Path,
    annotation_path: Path,
    source_group_pattern: re.Pattern[str] | None = None,
) -> tuple[dict[str, object], dict[str, set[object]]]:
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = annotation.get("images", [])
    annotations = annotation.get("annotations", [])
    categories = {
        int(item["id"]): str(item.get("name", item["id"]))
        for item in annotation.get("categories", [])
    }
    image_ids = [int(item["id"]) for item in images]
    declared_ids = set(image_ids)
    files: dict[str, Path] = {}
    stems: set[str] = set()
    groups: set[str] = set()
    hashes: set[str] = set()
    missing_files: list[str] = []
    hash_failures: list[str] = []
    for item in images:
        file_name = str(item.get("file_name", ""))
        normalized = Path(file_name).as_posix().casefold()
        files[normalized] = images_dir / Path(file_name)
        stems.add(Path(file_name).stem.casefold())
        if source_group_pattern is not None:
            match = source_group_pattern.search(file_name)
            if match:
                groups.add((match.group(1) if match.groups() else match.group(0)).casefold())
        path = files[normalized]
        if not path.is_file():
            missing_files.append(file_name)
            continue
        try:
            hashes.add(sha256_file(path))
        except OSError:
            hash_failures.append(file_name)

    annotations_missing_image = sorted({
        int(item["image_id"]) for item in annotations
        if int(item["image_id"]) not in declared_ids
    })
    per_image = Counter(int(item["image_id"]) for item in annotations if not int(item.get("iscrowd", 0)))
    images_without_annotations = sorted(
        image_id for image_id in declared_ids if per_image.get(image_id, 0) == 0
    )
    per_class = Counter(
        int(item["category_id"]) for item in annotations if not int(item.get("iscrowd", 0))
    )
    bbox_areas: list[float] = []
    bbox_aspects: list[float] = []
    invalid_bbox_count = 0
    valid_masks = 0
    considered_annotations = 0
    for item in annotations:
        if int(item.get("iscrowd", 0)):
            continue
        considered_annotations += 1
        bbox = item.get("bbox", [])
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            invalid_bbox_count += 1
        else:
            width, height = float(bbox[2]), float(bbox[3])
            if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
                invalid_bbox_count += 1
            else:
                bbox_areas.append(width * height)
                bbox_aspects.append(width / height)
        valid_masks += int(_mask_is_valid(item))

    total_gt = sum(per_class.values())
    split_report = {
        "images_dir": str(images_dir.resolve()),
        "annotation_path": str(annotation_path.resolve()),
        "num_images": len(images),
        "num_unique_image_ids": len(declared_ids),
        "duplicate_image_id_count": len(image_ids) - len(declared_ids),
        "num_annotations": considered_annotations,
        "missing_image_files": missing_files,
        "annotation_image_ids_missing_from_json": annotations_missing_image,
        "hash_failures": hash_failures,
        "per_class_GT": {
            categories.get(class_id, str(class_id)): {
                "count": int(count),
                "ratio": count / total_gt if total_gt else None,
            }
            for class_id in sorted(set(categories) | set(per_class))
            for count in (per_class.get(class_id, 0),)
        },
        "images_without_annotations": images_without_annotations,
        "instances_per_image": distribution([per_image.get(image_id, 0) for image_id in image_ids]),
        "bbox_area": distribution(bbox_areas),
        "bbox_aspect_ratio": distribution(bbox_aspects),
        "invalid_bbox_count": invalid_bbox_count,
        "valid_mask_count": valid_masks,
        "invalid_mask_count": considered_annotations - valid_masks,
        "valid_mask_ratio": valid_masks / considered_annotations if considered_annotations else None,
        "source_group_match_count": len(groups),
    }
    identity = {
        "file_name": set(files),
        "image_id": set(declared_ids),
        "stem": stems,
        "sha256": hashes,
        "source_group": groups,
    }
    return split_report, identity


def check_split_integrity(
    split_paths: Mapping[str, tuple[Path, Path]],
    source_group_regex: str | None = None,
) -> dict[str, object]:
    pattern = re.compile(source_group_regex) if source_group_regex else None
    splits: dict[str, object] = {}
    identities: dict[str, dict[str, set[object]]] = {}
    errors: list[str] = []
    warnings: list[str] = []
    for name, (images_dir, annotation_path) in split_paths.items():
        report, identity = inspect_split(name, images_dir, annotation_path, pattern)
        splits[name], identities[name] = report, identity
        if report["missing_image_files"]:
            errors.append(f"{name}_json_file_name_missing_on_disk")
        if report["annotation_image_ids_missing_from_json"]:
            errors.append(f"{name}_annotation_image_missing")
        if report["duplicate_image_id_count"]:
            errors.append(f"{name}_duplicate_image_id")
        if report["images_without_annotations"]:
            warnings.append(f"{name}_images_without_annotations")
        if report["invalid_bbox_count"]:
            warnings.append(f"{name}_invalid_bbox")
        if report["invalid_mask_count"]:
            warnings.append(f"{name}_invalid_masks_present")

    pair_reports: dict[str, object] = {}
    totals = Counter()
    labels = {
        "file_name": "exact_file",
        "image_id": "image_id",
        "stem": "stem",
        "sha256": "sha256",
        "source_group": "source_group",
    }
    for left, right in combinations(split_paths, 2):
        pair_name = f"{left}__{right}"
        pair_report = {}
        for identity_name, output_name in labels.items():
            overlap = identities[left][identity_name] & identities[right][identity_name]
            values = sorted(str(value) for value in overlap)
            pair_report[f"{output_name}_overlap_count"] = len(values)
            pair_report[f"{output_name}_overlap_examples"] = values[:50]
            totals[output_name] += len(values)
        pair_reports[pair_name] = pair_report

    error_names = {
        "exact_file": "split_exact_file_overlap",
        "image_id": "split_image_id_overlap",
        "stem": "split_stem_overlap",
        "sha256": "split_sha256_overlap",
        "source_group": "split_source_group_overlap",
    }
    for name, error in error_names.items():
        if totals[name]:
            errors.append(error)
    overlap_report = {
        "exact_file_overlap_count": int(totals["exact_file"]),
        "image_id_overlap_count": int(totals["image_id"]),
        "stem_overlap_count": int(totals["stem"]),
        "sha256_overlap_count": int(totals["sha256"]),
        "source_group_overlap_count": int(totals["source_group"]),
        "pairs": pair_reports,
    }
    return {
        "meta": {"source_group_regex": source_group_regex},
        "splits": splits,
        "overlaps": overlap_report,
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check SQ-Align split integrity")
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-images", type=Path, required=True)
        parser.add_argument(f"--{split}-json", type=Path, required=True)
    parser.add_argument("--source-group-regex")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_paths = {
        name: (getattr(args, f"{name}_images"), getattr(args, f"{name}_json"))
        for name in ("train", "val", "test")
    }
    for images_dir, annotation_path in split_paths.values():
        if not images_dir.is_dir():
            raise FileNotFoundError(images_dir)
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
    report = check_split_integrity(split_paths, args.source_group_regex)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SQ-Align split integrity report: {args.output_json}")
    if report["errors"]:
        print("Errors: " + ", ".join(report["errors"]))


if __name__ == "__main__":
    main()
