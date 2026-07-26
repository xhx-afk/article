"""Audit CrossMamba V2 parameters, optimizer groups and checkpoint loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crossmamba_debug_utils import (
    CROSSMAMBA_PREFIX,
    audit_checkpoint_load,
    build_config,
    load_checkpoint,
    parameter_counts,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--output", default="artifacts/crossmamba_v2/checkpoint_load_report.txt"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args.config)
    model = cfg.model
    checkpoint_report = None
    if args.checkpoint:
        checkpoint_report = audit_checkpoint_load(model, load_checkpoint(args.checkpoint))

    cross_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith(CROSSMAMBA_PREFIX) and parameter.requires_grad
    }
    optimizer = cfg.optimizer
    parameter_groups = []
    occurrences = {}
    for index, group in enumerate(optimizer.param_groups):
        names = []
        for parameter in group["params"]:
            names_for_parameter = [
                name for name, candidate in cross_parameters.items() if candidate is parameter
            ]
            for name in names_for_parameter:
                names.append(name)
                occurrences[name] = occurrences.get(name, 0) + 1
        parameter_groups.append(
            {
                "index": index,
                "lr": group.get("lr"),
                "weight_decay": group.get("weight_decay"),
                "crossmamba_parameters": sorted(names),
                "crossmamba_count": len(names),
            }
        )

    missing_from_optimizer = sorted(
        name for name in cross_parameters if occurrences.get(name, 0) == 0
    )
    duplicate_optimizer_parameters = sorted(
        name for name, count in occurrences.items() if count > 1
    )
    report = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "parameter_counts": parameter_counts(model),
        "crossmamba_parameters": sorted(cross_parameters),
        "optimizer_groups": parameter_groups,
        "missing_from_optimizer": missing_from_optimizer,
        "duplicate_optimizer_parameters": duplicate_optimizer_parameters,
        "optimizer_audit_passed": not missing_from_optimizer and not duplicate_optimizer_parameters,
        "checkpoint_load": checkpoint_report,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "CrossMamba V2 model/optimizer audit",
        "config: {}".format(args.config),
        "checkpoint: {}".format(args.checkpoint or "<none>"),
        "parameter_counts: {}".format(json.dumps(report["parameter_counts"], ensure_ascii=False)),
        "",
        "[CrossMamba trainable parameters]",
    ]
    lines.extend("- {}: {}".format(name, tuple(cross_parameters[name].shape)) for name in report["crossmamba_parameters"])
    lines.append("")
    lines.append("[Optimizer groups]")
    for group in parameter_groups:
        lines.append(
            "group {index}: lr={lr}, weight_decay={weight_decay}, crossmamba_count={crossmamba_count}".format(
                **group
            )
        )
        lines.extend("  - {}".format(name) for name in group["crossmamba_parameters"])
    lines.extend(
        [
            "",
            "missing_from_optimizer: {}".format(missing_from_optimizer),
            "duplicate_optimizer_parameters: {}".format(duplicate_optimizer_parameters),
            "optimizer_audit_passed: {}".format(report["optimizer_audit_passed"]),
        ]
    )
    if checkpoint_report is not None:
        lines.extend(
            [
                "",
                "[Checkpoint load]",
                json.dumps(checkpoint_report, ensure_ascii=False, indent=2),
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not report["optimizer_audit_passed"]:
        raise RuntimeError("CrossMamba optimizer audit failed: {}".format(report))
    if checkpoint_report is not None and not checkpoint_report["compatible"]:
        raise RuntimeError("CrossMamba checkpoint audit failed: {}".format(checkpoint_report))
    print("CrossMamba V2 audit passed; report: {}".format(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
