"""Validation-only sweep of the SQ-Align V3 suppression gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = Path(__file__).with_name("evaluate_sqalign_v3.py")


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep SQ-Align V3 semantic gate on validation")
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("-r", "--checkpoint", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--ann-file", type=Path, required=True)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0, 0.05, 0.10, 0.15, 0.20, 0.30])
    parser.add_argument("--gammas", nargs="+", type=float, default=[1, 2, 3])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-dets", type=int, default=300)
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    with tempfile.TemporaryDirectory(prefix="sqalign_v3_gate_") as temporary:
        temporary = Path(temporary)
        for gate_lambda in args.lambdas:
            for gamma in args.gammas:
                result_path = temporary / f"lambda_{gate_lambda}_gamma_{gamma}.json"
                command = [
                    sys.executable, str(EVALUATOR), "-c", str(args.config),
                    "-r", str(args.checkpoint), "--images-dir", str(args.images_dir),
                    "--ann-file", str(args.ann_file), "--output-json", str(result_path),
                    "--device", args.device, "--batch-size", str(args.batch_size),
                    "--num-workers", str(args.num_workers), "--max-dets", str(args.max_dets),
                    "--semantic-gate-lambda", str(gate_lambda),
                    "--semantic-gate-gamma", str(gamma),
                ]
                if args.max_images is not None:
                    command.extend(("--max-images", str(args.max_images)))
                subprocess.run(command, cwd=ROOT, check=True)
                report = json.loads(result_path.read_text(encoding="utf-8"))
                score = report["score_analysis"]["class_score_x_gate"]
                coco = report["coco"]
                calibration = report["calibration"]["overall"]
                results.append({
                    "lambda": gate_lambda, "gamma": gamma,
                    "AP": coco["AP@[0.50:0.95]"], "AP50": coco["AP50"],
                    "AP75": coco["AP75"], "AP80": coco["AP80"], "AP90": coco["AP90"],
                    "background_FP": score["background_FP"], "near_GT_FP": score["near_GT_FP"],
                    "LaECE": calibration["LaECE"],
                    "TP_vs_all_FP_AUC": score["TP_vs_all_FP_ROC_AUC"],
                })
    ranked = sorted(
        results,
        key=lambda row: (
            -float(row["AP"]), -float(row["AP75"]),
            -float(row["TP_vs_all_FP_AUC"] or 0.0), int(row["background_FP"]),
        ),
    )
    output = {
        "selection_dataset": "validation_only",
        "ranking_priority": ["AP", "AP75", "TP_vs_all_FP_AUC", "background_FP"],
        "best": ranked[0] if ranked else None,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Gate sweep report: {args.output_json}")


if __name__ == "__main__":
    main()

