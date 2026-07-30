"""Sweep SQ-Align localization/semantic score powers on validation only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOL_DIR))

from evaluate_sqalign import evaluate  # noqa: E402


def run(args: argparse.Namespace) -> None:
    if args.split != "val":
        raise ValueError("power selection is restricted to --split val")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for loc_power in args.loc_powers:
        for sem_power in args.sem_powers:
            name = f"loc_{loc_power:g}_sem_{sem_power:g}".replace(".", "p")
            metrics = evaluate(SimpleNamespace(
                config=args.config,
                checkpoint=args.checkpoint,
                output_dir=str(output_root / name),
                images_dir=args.images_dir,
                ann_file=args.ann_file,
                device=args.device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_dets=args.max_dets,
                score_threshold=args.score_threshold,
                recall_score_threshold=args.recall_score_threshold,
                tp_iou=0.50,
                background_iou=0.10,
                recall_ious=[0.50, 0.75],
                loc_power=loc_power,
                sem_power=sem_power,
            ))
            row = {
                "loc_power": loc_power,
                "sem_power": sem_power,
                **{key: metrics["coco"][key] for key in ("AP", "AP50", "AP75")},
                "background_FP": metrics["fixed_threshold"]["background_FP"],
                "ECE": metrics["calibration"]["ECE"],
                "LaECE": metrics["calibration"]["LaECE"],
                "semantic_AUC": metrics["query_alignment"]["semantic_TP_vs_background_ROC_AUC"],
                "final_AUC": metrics["query_alignment"]["final_score_TP_vs_background_ROC_AUC"],
            }
            rows.append(row)
    (output_root / "score_power_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output_root / "score_power_sweep.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-r", "--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--loc-powers", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5, 0.75])
    parser.add_argument("--sem-powers", type=float, nargs="+", default=[0, 0.1, 0.25, 0.5])
    parser.add_argument("--split", default="val")
    parser.add_argument("--images-dir")
    parser.add_argument("--ann-file")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-dets", type=int, default=300)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--recall-score-threshold", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
