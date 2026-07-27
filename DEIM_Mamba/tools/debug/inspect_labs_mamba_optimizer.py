"""Audit optimizer assignment and LR for every LABS-Mamba V3 parameter."""

from __future__ import annotations

import argparse

from labs_mamba_debug_utils import (
    LOCAL_PREFIX,
    SIDECAR_PREFIX,
    build_config,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output", default="artifacts/labs_mamba_v3/optimizer_audit.json"
    )
    return parser.parse_args()


def _component(name):
    if name.startswith(LOCAL_PREFIX):
        return "local_anchor"
    if not name.startswith(SIDECAR_PREFIX):
        return None
    relative = name[len(SIDECAR_PREFIX):]
    if relative == "beta_raw":
        return "sidecar_beta"
    for component in ("reduce", "horizontal", "vertical", "expand", "pre_norm"):
        if relative.startswith(component + "."):
            return "sidecar_" + component
    return "sidecar_other"


def main():
    args = parse_args()
    cfg = build_config(args.config)
    model = cfg.model
    optimizer = cfg.optimizer
    trainable = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assignments = {name: [] for name in trainable}
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        member_ids = {id(parameter) for parameter in group["params"]}
        member_names = sorted(
            name for name, parameter in trainable.items() if id(parameter) in member_ids
        )
        for name in member_names:
            assignments[name].append(index)
        groups.append(
            {
                "index": index,
                "lr": group.get("lr"),
                "weight_decay": group.get("weight_decay"),
                "parameter_count": len(member_names),
                "numel": sum(trainable[name].numel() for name in member_names),
            }
        )

    missing = sorted(name for name, indices in assignments.items() if not indices)
    duplicate = sorted(name for name, indices in assignments.items() if len(indices) != 1)
    rows = []
    for name, parameter in sorted(trainable.items()):
        component = _component(name)
        if component is None:
            continue
        indices = assignments[name]
        rows.append(
            {
                "name": name,
                "component": component,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "optimizer_groups": indices,
                "lr": [groups[index]["lr"] for index in indices],
                "weight_decay": [groups[index]["weight_decay"] for index in indices],
            }
        )

    base_lr = float(cfg.yaml_cfg["optimizer"]["lr"])
    sidecar_rows = [row for row in rows if row["component"].startswith("sidecar_")]
    sidecar_bad_lr = [
        row["name"]
        for row in sidecar_rows
        if len(row["lr"]) != 1
        or row["lr"][0] is None
        or not base_lr <= float(row["lr"][0]) <= 2.0 * base_lr
    ]
    backbone_lr_errors = []
    norm_decay_errors = []
    for name, parameter in trainable.items():
        indices = assignments[name]
        if len(indices) != 1:
            continue
        group = groups[indices[0]]
        lowered = name.lower()
        if name.startswith("backbone.") and "norm" not in lowered and "bn" not in lowered:
            if abs(float(group["lr"]) - 0.0000125) > 1e-12:
                backbone_lr_errors.append(name)
        if (
            name.startswith(("encoder.", "decoder."))
            and ("norm" in lowered or "bn" in lowered)
            and not name.startswith(SIDECAR_PREFIX)
            and float(group["weight_decay"]) != 0.0
        ):
            norm_decay_errors.append(name)

    local_rows = [row for row in rows if row["component"] == "local_anchor"]
    expected_sidecar = bool(getattr(model.encoder, "use_mamba_sidecar", False))
    structural_errors = []
    if not local_rows:
        structural_errors.append("no local-anchor trainable parameters")
    if expected_sidecar and not sidecar_rows:
        structural_errors.append("sidecar is enabled but has no trainable parameters")
    if not expected_sidecar and sidecar_rows:
        structural_errors.append("sidecar is disabled but optimizer contains sidecar parameters")

    report = {
        "config": args.config,
        "base_encoder_lr": base_lr,
        "groups": groups,
        "labs_parameters": rows,
        "component_numel": {
            component: sum(row["numel"] for row in rows if row["component"] == component)
            for component in sorted({row["component"] for row in rows})
        },
        "missing_trainable_parameters": missing,
        "duplicate_trainable_parameters": duplicate,
        "sidecar_bad_lr": sidecar_bad_lr,
        "backbone_lr_errors": backbone_lr_errors,
        "encoder_decoder_norm_decay_errors": norm_decay_errors,
        "structural_errors": structural_errors,
    }
    report["passed"] = not any(
        (
            missing,
            duplicate,
            sidecar_bad_lr,
            backbone_lr_errors,
            norm_decay_errors,
            structural_errors,
        )
    )
    write_json(args.output, report)
    if not report["passed"]:
        raise RuntimeError("LABS-Mamba optimizer audit failed: {}".format(args.output))
    print("LABS-Mamba optimizer audit passed: {}".format(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
