"""Create objective S0/S1/S2 LABS-Mamba metric, drift, and quality tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DEFAULT_STAGES = ("S0", "S1", "S2", "S3", "S4", "S5")
DEFAULT_DIRECTORIES = {
    "S0": "labs_mamba_v3_s0_local_anchor",
    "S1": "labs_mamba_v3_s1_hv_b64",
    "S2": "labs_mamba_v3_s2_vh_b64",
    "S3": "labs_mamba_v3_s3_hv_b128",
    "S4": "labs_mamba_v3_s4_vh_b128",
    "S5": "labs_mamba_v3_s5_best_no_center",
}
GROUPS = {
    "Head": ("Live_knot", "Dead_knot"),
    "Medium": ("resin", "knot_with_crack", "Crack", "Marrow", "Quartzity"),
    "Tail": ("Knot_missing", "Blue_stain"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", help="evaluation root containing LABS-Mamba experiment directories"
    )
    parser.add_argument(
        "--experiment", action="append", default=[], help="S0=/path/to/result_dir_or_json"
    )
    parser.add_argument("--baseline", help="compatibility alias for S0 result path")
    parser.add_argument(
        "--results", nargs="*", default=[], help="additional result paths, assigned S1, S2, ..."
    )
    parser.add_argument(
        "--output-dir", help="write results_comparison.csv/.md below this directory"
    )
    parser.add_argument(
        "--output-csv", default="artifacts/labs_mamba_v3/results_comparison.csv"
    )
    parser.add_argument(
        "--output-md", default="artifacts/labs_mamba_v3/results_comparison.md"
    )
    return parser.parse_args()


def _inputs(args):
    paths = {}
    if args.root:
        root = Path(args.root)
        for stage in DEFAULT_STAGES:
            candidate = root / DEFAULT_DIRECTORIES[stage]
            if candidate.exists():
                paths[stage] = candidate
    for item in args.experiment:
        if "=" not in item:
            raise ValueError("--experiment must use STAGE=PATH")
        stage, path = item.split("=", 1)
        paths[stage.upper()] = Path(path)
    if args.baseline:
        paths["S0"] = Path(args.baseline)
    for index, path in enumerate(args.results, start=1):
        paths["S{}".format(index)] = Path(path)
    return paths


def _load(path):
    directory = path if path.is_dir() else path.parent
    summary = path
    if path.is_dir():
        summary = path / "all_metrics_summary.json"
        if not summary.exists():
            summary = path / "class_and_all_metrics.json"
    if not summary.exists():
        raise FileNotFoundError("metric JSON does not exist: {}".format(summary))
    payload = json.loads(summary.read_text(encoding="utf-8"))
    predictions_path = directory / "predictions.json"
    predictions = None
    if predictions_path.exists():
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    argmax_path = directory / "raw_query_class_argmax_distribution.json"
    argmax = payload.get("raw_query_class_argmax_distribution")
    if argmax is None and argmax_path.exists():
        argmax = json.loads(argmax_path.read_text(encoding="utf-8"))
    return payload, predictions, argmax


def _nested(payload, *keys):
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _mean(values):
    clean = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(clean) / len(clean) if clean else None


def _class_map(payload):
    rows = _nested(payload, "COCO_bbox", "per_class") or payload.get("per_class", [])
    result = {}
    for item in rows:
        name = str(item.get("class_name", item.get("name", item.get("category_id"))))
        result[name.lower()] = item.get("AP")
    return result


def _distribution_map(argmax):
    if not argmax:
        return {}
    rows = argmax.get("rows", argmax) if isinstance(argmax, dict) else argmax
    result = {}
    if isinstance(rows, dict):
        for name, value in rows.items():
            result[str(name)] = float(value.get("count", value) if isinstance(value, dict) else value)
    elif isinstance(rows, list):
        for item in rows:
            name = item.get("class_name", item.get("category_id", item.get("model_label")))
            result[str(name)] = float(item.get("count", 0))
    total = sum(result.values())
    return {name: value / total for name, value in result.items()} if total else {}


def _js_divergence(left, right):
    if not left and not right:
        return None
    keys = set(left) | set(right)
    divergence = 0.0
    for key in keys:
        p = float(left.get(key, 0.0))
        q = float(right.get(key, 0.0))
        midpoint = 0.5 * (p + q)
        if p > 0.0:
            divergence += 0.5 * p * math.log(p / midpoint, 2)
        if q > 0.0:
            divergence += 0.5 * q * math.log(q / midpoint, 2)
    return divergence


def _calibration(payload):
    rows = _nested(payload, "calibration", "rows") or []
    return next(
        (row for row in rows if str(row.get("category_id")).lower() == "all"), {}
    )


def _all_bbox_row(payload):
    rows = payload.get("prediction_bbox_statistics", [])
    return next(
        (
            row
            for row in rows
            if str(row.get("category_id")).lower() == "all"
            or str(row.get("class_name")).upper() == "ALL_CLASSES"
        ),
        {},
    )


def _row(stage, payload, predictions, argmax):
    overall = _nested(payload, "COCO_bbox", "overall") or payload.get("overall", {})
    class_ap = _class_map(payload)
    background_rows = payload.get("per_class_background_FP", []) or []
    confusion_rows = _nested(payload, "confusion_and_FP_FN", "per_class") or []
    quality = payload.get("quality_diagnostics", {}) or {}
    tide = _nested(payload, "TIDE_bbox", "metrics") or {}
    tide_main = tide.get("main_errors_dAP", {}) or {}
    tide_special = tide.get("special_errors_dAP", {}) or {}
    bbox = _all_bbox_row(payload)
    calibration = _calibration(payload)
    score_threshold = float(
        _nested(payload, "evaluation_parameters", "background_fp_score_threshold") or 0.05
    )
    detections = (
        sum(1 for item in predictions if float(item.get("score", 0.0)) >= score_threshold)
        if isinstance(predictions, list)
        else _nested(payload, "counts", "predictions")
    )
    background_count = sum(int(row.get("background_fp", 0) or 0) for row in background_rows)
    background_detections = sum(int(row.get("detections", 0) or 0) for row in background_rows)
    row = {
        "experiment": stage,
        "AP": overall.get("AP"),
        "AP50": overall.get("AP50"),
        "AP75": overall.get("AP75"),
        "APM": overall.get("APM"),
        "APL": overall.get("APL"),
        "AR100": overall.get("AR100"),
        "ARM": overall.get("ARM"),
        "ARL": overall.get("ARL"),
        "detections_score_ge_0_05": detections,
        "background_FP_count": background_count,
        "background_FP_rate": (
            background_count / background_detections if background_detections else None
        ),
        "TIDE_FalsePos_dAP": tide_special.get("FalsePos"),
        "TIDE_Background_dAP": tide_main.get("Bkg", tide_main.get("Background")),
        "TP": sum(int(item.get("TP", 0) or 0) for item in confusion_rows),
        "FP": sum(int(item.get("FP", 0) or 0) for item in confusion_rows),
        "FN": sum(int(item.get("FN", 0) or 0) for item in confusion_rows),
        "width_median": bbox.get("width_median"),
        "height_median": bbox.get("height_median"),
        "area_median": bbox.get("area_median"),
        "aspect_median": bbox.get("aspect_median"),
        "ECE": calibration.get("ECE"),
        "LaECE": calibration.get("LaECE"),
        "positive_quality_IoU_spearman": _nested(
            quality, "quality_vs_true_iou_spearman", "rho"
        ),
        "positive_quality_joint_target_spearman": _nested(
            quality, "quality_vs_joint_target_spearman", "rho"
        ),
        "all_query_quality_best_IoU_spearman": _nested(
            quality, "all_query_quality_vs_best_iou_spearman", "rho"
        ),
        "positive_quality_mean": _nested(
            quality, "distributions", "positive_queries", "mean"
        ),
        "negative_quality_mean": _nested(
            quality, "distributions", "negative_queries", "mean"
        ),
        "TP_quality_mean": _nested(
            quality, "distributions", "true_positive_detections", "mean"
        ),
        "background_FP_quality_mean": _nested(
            quality, "distributions", "background_false_positives", "mean"
        ),
    }
    for group, names in GROUPS.items():
        row[group + "_macro_AP"] = _mean(class_ap.get(name.lower()) for name in names)
    row["all_9_macro_AP"] = _mean(class_ap.values())
    row["exclude_Blue_stain_macro_AP"] = _mean(
        value for name, value in class_ap.items() if name != "blue_stain"
    )
    for name, value in class_ap.items():
        row["AP_class_" + name] = value
    row["_argmax_distribution"] = _distribution_map(argmax)
    return row


def _add_s0_deltas(rows):
    baseline = next((row for row in rows if row["experiment"] == "S0"), None)
    if baseline is None:
        return
    drift_metrics = ("width_median", "height_median", "area_median", "aspect_median")
    for row in rows:
        for metric in drift_metrics:
            value, reference = row.get(metric), baseline.get(metric)
            row[metric + "_pct_vs_S0"] = (
                100.0 * (float(value) - float(reference)) / float(reference)
                if isinstance(value, (int, float))
                and isinstance(reference, (int, float))
                and float(reference) != 0.0
                else None
            )
        current_dist = row.pop("_argmax_distribution", {})
        baseline_dist = baseline.get("_baseline_argmax", baseline.get("_argmax_distribution", {}))
        if row is baseline:
            baseline["_baseline_argmax"] = current_dist
            baseline_dist = current_dist
        row["class_argmax_L1_vs_S0"] = (
            sum(abs(current_dist.get(key, 0.0) - baseline_dist.get(key, 0.0)) for key in set(current_dist) | set(baseline_dist))
            if current_dist or baseline_dist else None
        )
        row["class_argmax_JS_vs_S0"] = _js_divergence(current_dist, baseline_dist)
        for name in sorted(set(current_dist) | set(baseline_dist)):
            safe = "".join(char if char.isalnum() else "_" for char in name)
            row["class_argmax_share_" + safe] = current_dist.get(name, 0.0)
            row["class_argmax_share_delta_vs_S0_" + safe] = (
                current_dist.get(name, 0.0) - baseline_dist.get(name, 0.0)
            )
        macro_delta = None
        if isinstance(row.get("all_9_macro_AP"), (int, float)) and isinstance(
            baseline.get("all_9_macro_AP"), (int, float)
        ):
            macro_delta = row["all_9_macro_AP"] - baseline["all_9_macro_AP"]
        blue = row.get("AP_class_blue_stain")
        blue_base = baseline.get("AP_class_blue_stain")
        row["Blue_stain_macro_change_contribution_ratio"] = (
            ((blue - blue_base) / 9.0) / macro_delta
            if isinstance(blue, (int, float))
            and isinstance(blue_base, (int, float))
            and isinstance(macro_delta, (int, float))
            and macro_delta != 0.0
            else None
        )
    baseline.pop("_baseline_argmax", None)


def _write_markdown(path, rows, columns):
    shown = [
        column for column in (
            "experiment", "AP", "AP50", "AP75", "APM", "APL", "AR100", "ARM", "ARL",
            "Head_macro_AP", "Medium_macro_AP", "Tail_macro_AP", "all_9_macro_AP",
            "exclude_Blue_stain_macro_AP", "background_FP_count", "background_FP_rate",
            "area_median", "area_median_pct_vs_S0", "class_argmax_L1_vs_S0",
            "class_argmax_JS_vs_S0", "positive_quality_IoU_spearman",
            "positive_quality_joint_target_spearman",
            "Blue_stain_macro_change_contribution_ratio",
        ) if column in columns
    ]
    lines = [
        "# LABS-Mamba V3 客观结果汇总",
        "",
        "| " + " | ".join(shown) + " |",
        "|" + "|".join(["---"] * len(shown)) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(
                "" if row.get(column) is None else str(row.get(column)) for column in shown
            ) + " |"
        )
    lines.extend(
        [
            "",
            "逐类 AP、类别 argmax 占比/相对 S0 变化、TP/FP/FN、TIDE、框尺度、"
            "ECE/LaECE 与质量关系列在 CSV 中完整保留；缺失的上游指标保持为空，不做评价。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.output_dir:
        output_dir = Path(args.output_dir)
        args.output_csv = str(output_dir / "results_comparison.csv")
        args.output_md = str(output_dir / "results_comparison.md")
    paths = _inputs(args)
    if not paths:
        raise RuntimeError("provide --root or at least one --experiment STAGE=PATH")
    rows = []
    for stage in DEFAULT_STAGES:
        if stage not in paths:
            continue
        payload, predictions, argmax = _load(paths[stage])
        rows.append(_row(stage, payload, predictions, argmax))
    for stage in sorted(set(paths) - set(DEFAULT_STAGES)):
        payload, predictions, argmax = _load(paths[stage])
        rows.append(_row(stage, payload, predictions, argmax))
    _add_s0_deltas(rows)
    columns = sorted({key for row in rows for key in row if not key.startswith("_")})
    preferred = ["experiment", "AP", "AP50", "AP75", "APM", "APL", "AR100", "ARM", "ARL"]
    columns = [key for key in preferred if key in columns] + [key for key in columns if key not in preferred]
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)
    _write_markdown(Path(args.output_md), rows, columns)
    print("LABS-Mamba comparison written to {} and {}".format(output_csv, args.output_md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
