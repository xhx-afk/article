"""Audit AxMamba parameters, optimizer groups, and checkpoint compatibility."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import torch

from axmamba_debug_utils import (
    build_model,
    count_parameters,
    load_checkpoint_state,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/deim_dfine/ablation_axmamba_v1/m3_axis_gate_p4p5.yml",
    )
    parser.add_argument(
        "--baseline", default="configs/deim_dfine/deim_hgnetv2_l_wood.yml"
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default="artifacts/axmamba_v1/checkpoint_load_report.txt",
    )
    return parser.parse_args()


def optimizer_audit(config, model):
    optimizer = config.optimizer
    name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    memberships = defaultdict(list)
    groups = []
    for group_index, group in enumerate(optimizer.param_groups):
        names = []
        for parameter in group["params"]:
            name = name_by_id.get(id(parameter), "<unknown>")
            names.append(name)
            memberships[id(parameter)].append(group_index)
        groups.append(
            {
                "index": group_index,
                "lr": group.get("lr"),
                "weight_decay": group.get("weight_decay"),
                "axmamba_parameters": sorted(
                    name for name in names if "axmamba_blocks" in name
                ),
                "parameter_count": len(names),
            }
        )

    trainable = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    missing = sorted(
        name for parameter_id, name in trainable.items() if parameter_id not in memberships
    )
    duplicates = sorted(
        trainable.get(parameter_id, "<unknown>")
        for parameter_id, indices in memberships.items()
        if len(indices) > 1
    )
    return groups, missing, duplicates


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    _, baseline = build_model(args.baseline, torch.device("cpu"))
    if args.checkpoint:
        source_state, state_source = load_checkpoint_state(args.checkpoint)
    else:
        source_state = dict(baseline.state_dict())
        state_source = "fresh baseline model state_dict"
    baseline_target_state = baseline.state_dict()
    baseline_keys = set(baseline_target_state)
    baseline_matched = {
        name: tensor
        for name, tensor in source_state.items()
        if name in baseline_target_state
        and tensor.shape == baseline_target_state[name].shape
    }
    baseline_missing = sorted(
        name for name in baseline_target_state if name not in baseline_matched
    )
    baseline_unexpected = sorted(
        name for name in source_state if name not in baseline_target_state
    )
    baseline_shape_mismatch = sorted(
        name
        for name, tensor in source_state.items()
        if name in baseline_target_state
        and tensor.shape != baseline_target_state[name].shape
    )
    del baseline

    config, model = build_model(args.config, device)
    target_state = model.state_dict()
    matched = {
        name: tensor
        for name, tensor in source_state.items()
        if name in target_state and tensor.shape == target_state[name].shape
    }
    missing = sorted(name for name in target_state if name not in matched)
    unexpected = sorted(name for name in source_state if name not in target_state)
    shape_mismatch = sorted(
        name
        for name, tensor in source_state.items()
        if name in target_state and tensor.shape != target_state[name].shape
    )
    incompatible = model.load_state_dict(matched, strict=False)
    loader_missing = sorted(incompatible.missing_keys)
    loader_unexpected = sorted(incompatible.unexpected_keys)
    additional_missing = sorted(set(loader_missing) - set(baseline_missing))
    allowed_axmamba_missing = [
        name
        for name in additional_missing
        if name.startswith("encoder.axmamba_blocks.")
    ]
    disallowed_missing = sorted(
        set(additional_missing) - set(allowed_axmamba_missing)
    )
    additional_unexpected = sorted(set(unexpected) - set(baseline_unexpected))
    additional_shape_mismatch = sorted(
        set(shape_mismatch) - set(baseline_shape_mismatch)
    )
    checkpoint_audit_passed = not (
        disallowed_missing
        or loader_unexpected
        or additional_unexpected
        or additional_shape_mismatch
    )

    groups, optimizer_missing, optimizer_duplicates = optimizer_audit(config, model)
    optimizer_audit_passed = not optimizer_missing and not optimizer_duplicates
    parameters = count_parameters(model)
    axmamba_names = sorted(
        name for name, _ in model.named_parameters() if "axmamba_blocks" in name
    )

    report = {
        "config": str(resolve_repo_path(args.config)),
        "baseline": str(resolve_repo_path(args.baseline)),
        "state_source": state_source,
        "parameters": parameters,
        "axmamba_parameter_names": axmamba_names,
        "optimizer_groups": groups,
        "optimizer_missing_trainable_parameters": optimizer_missing,
        "optimizer_duplicate_parameters": optimizer_duplicates,
        "optimizer_audit_passed": optimizer_audit_passed,
        "baseline_key_count": len(baseline_keys),
        "checkpoint_source_key_count": len(source_state),
        "checkpoint_matched_key_count": len(matched),
        "baseline_checkpoint_missing_keys": baseline_missing,
        "baseline_checkpoint_unexpected_keys": baseline_unexpected,
        "baseline_checkpoint_shape_mismatch_keys": baseline_shape_mismatch,
        "baseline_checkpoint_strictly_compatible": not (
            baseline_missing or baseline_unexpected or baseline_shape_mismatch
        ),
        "checkpoint_missing_keys": loader_missing,
        "checkpoint_additional_missing_keys_vs_baseline": additional_missing,
        "checkpoint_allowed_new_axmamba_missing_keys": allowed_axmamba_missing,
        "checkpoint_disallowed_additional_missing_keys": disallowed_missing,
        "checkpoint_unexpected_keys": sorted(set(unexpected + loader_unexpected)),
        "checkpoint_additional_unexpected_keys_vs_baseline": additional_unexpected,
        "checkpoint_shape_mismatch_keys": shape_mismatch,
        "checkpoint_additional_shape_mismatch_keys_vs_baseline": (
            additional_shape_mismatch
        ),
        "checkpoint_audit_passed": checkpoint_audit_passed,
    }

    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("report:", output_path)
    return 0 if checkpoint_audit_passed and optimizer_audit_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
