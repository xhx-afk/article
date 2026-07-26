"""Report the optimizer assignment of every CrossMamba V2 parameter."""

from __future__ import annotations

import argparse

from crossmamba_debug_utils import CROSSMAMBA_PREFIX, build_config, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output", default="artifacts/crossmamba_v2/optimizer_audit.json"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args.config)
    model = cfg.model
    optimizer = cfg.optimizer
    cross_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith(CROSSMAMBA_PREFIX) and parameter.requires_grad
    }
    assignments = {name: [] for name in cross_parameters}
    groups = []
    for group_index, group in enumerate(optimizer.param_groups):
        member_ids = {id(parameter) for parameter in group["params"]}
        member_names = sorted(
            name
            for name, parameter in cross_parameters.items()
            if id(parameter) in member_ids
        )
        for name in member_names:
            assignments[name].append(group_index)
        groups.append(
            {
                "index": group_index,
                "lr": group.get("lr"),
                "weight_decay": group.get("weight_decay"),
                "crossmamba_parameters": member_names,
            }
        )
    missing = sorted(name for name, indices in assignments.items() if not indices)
    duplicate = sorted(name for name, indices in assignments.items() if len(indices) > 1)
    parameter_rows = [
        {
            "name": name,
            "shape": list(cross_parameters[name].shape),
            "numel": cross_parameters[name].numel(),
            "optimizer_groups": assignments[name],
            "lr": [groups[index]["lr"] for index in assignments[name]],
            "weight_decay": [
                groups[index]["weight_decay"] for index in assignments[name]
            ],
        }
        for name in sorted(cross_parameters)
    ]
    report = {
        "config": args.config,
        "parameters": parameter_rows,
        "groups": groups,
        "missing": missing,
        "duplicate": duplicate,
        "passed": bool(parameter_rows) and not missing and not duplicate,
    }
    write_json(args.output, report)
    if not report["passed"]:
        raise RuntimeError("CrossMamba optimizer audit failed: {}".format(report))
    print("CrossMamba optimizer audit passed; report: {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
