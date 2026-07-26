"""Create an objective R0-R4 CrossMamba V2 evaluation comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPERIMENTS = ("R0", "R1", "R2", "R3", "R4")
DEFAULT_SUBDIRECTORIES = {
    "R0": "r0_baseline",
    "R1": "r1_postfpn_local",
    "R2": "r2_cross_hv_only",
    "R3": "r3_cross_hv_vh_mean",
    "R4": "r4_cross_hv_vh_gate",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=None,
        help="Directory containing the evaluator's r0_baseline through r4_cross_hv_vh_gate subdirectories",
    )
    parser.add_argument(
        "--experiment", action="append", default=[], help="R0=/path/to/experiment_dir"
    )
    parser.add_argument(
        "--output-csv", default="artifacts/crossmamba_v2/results_comparison.csv"
    )
    parser.add_argument(
        "--output-md", default="artifacts/crossmamba_v2/results_comparison.md"
    )
    return parser.parse_args()


def _paths(args):
    result = {}
    if args.root:
        root = Path(args.root)
        result.update(
            {name: root / DEFAULT_SUBDIRECTORIES[name] for name in EXPERIMENTS}
        )
    for item in args.experiment:
        if "=" not in item:
            raise ValueError("--experiment must use R0=PATH")
        name, path = item.split("=", 1)
        result[name.upper()] = Path(path)
    return result


def _load(directory: Path):
    summary = directory / "all_metrics_summary.json"
    if not summary.exists():
        summary = directory / "class_and_all_metrics.json"
    if not summary.exists():
        raise FileNotFoundError("missing all_metrics_summary.json in {}".format(directory))
    payload = json.loads(summary.read_text(encoding="utf-8"))
    if "COCO_bbox" not in payload:
        # A class_and_all_metrics.json can still be compared for COCO values,
        # while unavailable diagnostics remain explicitly null.
        payload = {
            "COCO_bbox": {
                "overall": payload.get("overall", {}),
                "per_class": payload.get("per_class", []),
            },
            "counts": {},
            "TIDE_bbox": {},
            "calibration": {},
            "per_class_background_FP": [],
            "quality_diagnostics": {},
        }
    return payload


def _metric(payload, *keys):
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _row(name, payload):
    overall = _metric(payload, "COCO_bbox", "overall") or {}
    row = {
        "experiment": name,
        "AP": overall.get("AP"),
        "AP50": overall.get("AP50"),
        "AP75": overall.get("AP75"),
        "APM": overall.get("APM"),
        "APL": overall.get("APL"),
        "AR100": overall.get("AR100"),
        "predictions": _metric(payload, "counts", "predictions"),
        "TIDE_available": _metric(payload, "TIDE_bbox", "available"),
        "TIDE_Cls": _metric(payload, "TIDE_bbox", "metrics", "main_errors_dAP", "Cls"),
        "TIDE_Loc": _metric(payload, "TIDE_bbox", "metrics", "main_errors_dAP", "Loc"),
        "TIDE_Bkg": _metric(payload, "TIDE_bbox", "metrics", "main_errors_dAP", "Bkg"),
        "TIDE_Miss": _metric(payload, "TIDE_bbox", "metrics", "main_errors_dAP", "Miss"),
        "TIDE_FalsePos": _metric(payload, "TIDE_bbox", "metrics", "main_errors_dAP", "FalsePos"),
        "TIDE_FalseNeg": _metric(payload, "TIDE_bbox", "metrics", "main_errors_dAP", "FalseNeg"),
        "ECE": None,
        "LaECE": None,
        "background_FP": 0,
        "background_FP_score_threshold": _metric(
            payload, "evaluation_parameters", "background_fp_score_threshold"
        ),
        "background_FP_iou_threshold": _metric(
            payload, "evaluation_parameters", "background_fp_iou_threshold"
        ),
        "quality_vs_true_iou_spearman": _metric(
            payload, "quality_diagnostics", "quality_vs_true_iou_spearman", "rho"
        ),
        "quality_vs_joint_target_spearman": _metric(
            payload, "quality_diagnostics", "quality_vs_joint_target_spearman", "rho"
        ),
    }
    calibration_rows = _metric(payload, "calibration", "rows") or []
    overall_calibration = next(
        (item for item in calibration_rows if item.get("category_id") == "all"), None
    )
    if overall_calibration:
        row["ECE"] = overall_calibration.get("ECE")
        row["LaECE"] = overall_calibration.get("LaECE")
    background_rows = payload.get("per_class_background_FP") or []
    row["background_FP"] = sum(
        int(item.get("background_fp", 0) or 0) for item in background_rows
    )
    for item in background_rows:
        class_name = str(item.get("class_name", item.get("category_id", "unknown")))
        safe_name = "".join(char if char.isalnum() else "_" for char in class_name)
        row["background_FP_class_{}".format(safe_name)] = item.get("background_fp")
    for item in _metric(payload, "COCO_bbox", "per_class") or []:
        class_name = str(item.get("class_name", item.get("category_id", "unknown")))
        safe_name = "".join(char if char.isalnum() else "_" for char in class_name)
        row["AP_class_{}".format(safe_name)] = item.get("AP")
    return row


def _relative_r1(rows):
    baseline = next((row for row in rows if row["experiment"] == "R1"), None)
    if baseline is None:
        return
    numeric_metrics = sorted(
        key
        for key, value in baseline.items()
        if (
            key != "experiment"
            and "threshold" not in key
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
    )
    for row in rows:
        for metric in numeric_metrics:
            value = row.get(metric)
            reference = baseline.get(metric)
            if isinstance(value, (int, float)) and isinstance(reference, (int, float)):
                row[metric + "_delta_vs_R1"] = value - reference
            else:
                row[metric + "_delta_vs_R1"] = None


def _write_markdown(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "experiment", "AP", "AP50", "AP75", "APM", "APL", "AR100", "predictions",
        "TIDE_Cls", "TIDE_Loc", "TIDE_Bkg", "TIDE_Miss", "TIDE_FalsePos", "TIDE_FalseNeg",
        "ECE", "LaECE", "background_FP", "quality_vs_true_iou_spearman",
        "quality_vs_joint_target_spearman",
    ]
    lines = ["# CrossMamba V2 R0-R4 客观指标比较", "", "| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if row.get(column) is None else str(row.get(column)) for column in columns) + " |")
    lines.extend(["", "逐类 AP 作为 CSV 中的 `AP_class_*` 列输出；所有 `_delta_vs_R1` 列均以 R1 post-FPN local 为参照，未评价或不可用指标保留为空。"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    directories = _paths(args)
    rows = []
    for name in EXPERIMENTS:
        if name not in directories:
            continue
        directory = directories[name]
        if not directory.exists():
            print("skip {}: directory does not exist: {}".format(name, directory))
            continue
        rows.append(_row(name, _load(directory)))
    if not rows:
        raise RuntimeError("no R0-R4 evaluation directories were provided")
    _relative_r1(rows)
    columns = sorted({key for row in rows for key in row})
    # Keep the principal comparison fields first, then per-class and deltas.
    preferred = [
        "experiment", "AP", "AP50", "AP75", "APM", "APL", "AR100", "predictions",
        "ECE", "LaECE", "background_FP", "quality_vs_true_iou_spearman",
        "quality_vs_joint_target_spearman",
    ]
    columns = [column for column in preferred if column in columns] + [column for column in columns if column not in preferred]
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    _write_markdown(Path(args.output_md), rows)
    print("CrossMamba V2 comparison written to {} and {}".format(csv_path, args.output_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
