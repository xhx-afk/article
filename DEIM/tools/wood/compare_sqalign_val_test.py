"""Compare unified SQ-Align validation and test reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np


def _number(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None


def _metric(coco: Mapping[str, object], name: str) -> float | None:
    return _number(coco.get(name))


def _difference(test: float | None, val: float | None) -> float | None:
    return test - val if test is not None and val is not None else None


def _macro(rows: list[dict[str, object]], minimum_gt: int = 0) -> dict[str, object]:
    selected = [
        row for row in rows
        if int(row["val_GT_count"]) >= minimum_gt
        and int(row["test_GT_count"]) >= minimum_gt
        and row["val_AP"] is not None
        and row["test_AP"] is not None
    ]
    if not selected:
        return {"class_count": 0, "val": None, "test": None, "difference_test_minus_val": None}
    val = float(np.mean([row["val_AP"] for row in selected]))
    test = float(np.mean([row["test_AP"] for row in selected]))
    return {
        "class_count": len(selected),
        "val": val,
        "test": test,
        "difference_test_minus_val": test - val,
    }


def compare_reports(
    val_report: Mapping[str, object],
    test_report: Mapping[str, object],
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    val_coco = val_report.get("coco", {})
    test_coco = test_report.get("coco", {})
    if not isinstance(val_coco, Mapping) or not isinstance(test_coco, Mapping):
        raise ValueError("both inputs must contain a coco object")
    overall = {}
    for output_name, input_name in (
        ("AP", "AP@[0.50:0.95]"), ("AP50", "AP50"), ("AP75", "AP75")
    ):
        val = _metric(val_coco, input_name)
        test = _metric(test_coco, input_name)
        overall[output_name] = {
            "val": val,
            "test": test,
            "difference_test_minus_val": _difference(test, val),
        }

    val_classes = val_coco.get("per_class", {})
    test_classes = test_coco.get("per_class", {})
    val_classes = val_classes if isinstance(val_classes, Mapping) else {}
    test_classes = test_classes if isinstance(test_classes, Mapping) else {}
    names = sorted(set(val_classes) | set(test_classes))
    rows: list[dict[str, object]] = []
    per_class = {}
    for name in names:
        val_values = val_classes.get(name, {})
        test_values = test_classes.get(name, {})
        val_values = val_values if isinstance(val_values, Mapping) else {}
        test_values = test_values if isinstance(test_values, Mapping) else {}
        row = {
            "class_name": name,
            "val_AP": _number(val_values.get("AP")),
            "test_AP": _number(test_values.get("AP")),
            "val_GT_count": int(val_values.get("GT_count", 0)),
            "test_GT_count": int(test_values.get("GT_count", 0)),
        }
        row["AP_difference_test_minus_val"] = _difference(row["test_AP"], row["val_AP"])
        rows.append(row)
    valid_class_count = sum(row["AP_difference_test_minus_val"] is not None for row in rows)
    for row in rows:
        difference = row["AP_difference_test_minus_val"]
        per_class[str(row["class_name"])] = {
            "val_AP": row["val_AP"],
            "test_AP": row["test_AP"],
            "AP_difference_test_minus_val": difference,
            "val_GT_count": row["val_GT_count"],
            "test_GT_count": row["test_GT_count"],
            "contribution_to_macro_AP_gap": (
                difference / valid_class_count if difference is not None and valid_class_count else None
            ),
        }

    macro = {
        "all_classes": _macro(rows, 0),
        "GT_at_least_20_diagnostic": _macro(rows, 20),
        "GT_at_least_30_diagnostic": _macro(rows, 30),
        "note": "GT-filtered macro AP is diagnostic only; official all-class COCO AP remains primary.",
    }
    differences = np.asarray(
        [row["AP_difference_test_minus_val"] for row in rows if row["AP_difference_test_minus_val"] is not None],
        dtype=np.float64,
    )
    bootstrap_samples = max(int(bootstrap_samples), 0)
    bootstrap_values = np.zeros((bootstrap_samples,), dtype=np.float64)
    if differences.size and bootstrap_samples:
        rng = np.random.default_rng(seed)
        for index in range(bootstrap_samples):
            bootstrap_values[index] = rng.choice(differences, size=differences.size, replace=True).mean()
        bootstrap = {
            "available": True,
            "samples": bootstrap_samples,
            "seed": seed,
            "metric": "paired class-bootstrap macro AP difference (test - val)",
            "mean": float(bootstrap_values.mean()),
            "CI95_low": float(np.percentile(bootstrap_values, 2.5)),
            "CI95_high": float(np.percentile(bootstrap_values, 97.5)),
        }
    else:
        bootstrap = {
            "available": False,
            "samples": bootstrap_samples,
            "seed": seed,
            "reason": "no paired classes or bootstrap_samples is zero",
        }

    largest = sorted(
        (
            {
                "class_name": name,
                **values,
            }
            for name, values in per_class.items()
            if values["contribution_to_macro_AP_gap"] is not None
        ),
        key=lambda item: abs(item["contribution_to_macro_AP_gap"]),
        reverse=True,
    )[:10]
    warnings = []
    low_support_contribution = sum(
        abs(values["contribution_to_macro_AP_gap"])
        for values in per_class.values()
        if values["contribution_to_macro_AP_gap"] is not None
        and min(values["val_GT_count"], values["test_GT_count"]) < 20
    )
    total_contribution = sum(
        abs(values["contribution_to_macro_AP_gap"])
        for values in per_class.values()
        if values["contribution_to_macro_AP_gap"] is not None
    )
    if total_contribution and low_support_contribution / total_contribution > 0.50:
        warnings.append("validation_test_gap_dominated_by_low_support_class")
    if any(min(row["val_GT_count"], row["test_GT_count"]) < 20 for row in rows):
        warnings.append("tail_class_macro_ap_instability")
    return {
        "meta": {"primary_metric": "official all-class COCO AP"},
        "overall": overall,
        "per_class": per_class,
        "macro_AP": macro,
        "bootstrap": bootstrap,
        "largest_gap_contributing_classes": largest,
        "automatic_diagnosis": {"errors": [], "warnings": warnings},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SQ-Align val/test unified JSON reports")
    parser.add_argument("--val-json", type=Path, required=True)
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val = json.loads(args.val_json.read_text(encoding="utf-8"))
    test = json.loads(args.test_json.read_text(encoding="utf-8"))
    report = compare_reports(val, test, args.bootstrap_samples, args.seed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SQ-Align val/test comparison: {args.output_json}")


if __name__ == "__main__":
    main()
