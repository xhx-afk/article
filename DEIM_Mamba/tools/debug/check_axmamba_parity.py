"""Compare M0 and the original baseline on identical weights and input."""

from __future__ import annotations

import argparse
import json

import torch

from axmamba_debug_utils import (
    build_model,
    load_checkpoint_state,
    resolve_repo_path,
    save_json,
    tensor_leaves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", default="configs/deim_dfine/deim_hgnetv2_l_wood.yml"
    )
    parser.add_argument(
        "--m0",
        default="configs/deim_dfine/ablation_axmamba_v1/m0_baseline.yml",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", default="artifacts/axmamba_v1/baseline_parity.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.height, args.width, args.batch_size) <= 0:
        raise ValueError("height, width, and batch-size must be positive")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    _, baseline = build_model(args.baseline, device)
    if args.checkpoint:
        state, state_source = load_checkpoint_state(args.checkpoint)
        baseline.load_state_dict(state, strict=True)
    else:
        state = baseline.state_dict()
        state_source = "fresh baseline model state_dict"

    _, m0 = build_model(args.m0, device)
    m0.load_state_dict(state, strict=True)
    baseline.eval()
    m0.eval()
    images = torch.randn(
        args.batch_size, 3, args.height, args.width, device=device
    )
    with torch.no_grad():
        baseline_output = baseline(images)
        m0_output = m0(images)

    baseline_leaves = tensor_leaves(baseline_output)
    m0_leaves = tensor_leaves(m0_output)
    if [name for name, _ in baseline_leaves] != [name for name, _ in m0_leaves]:
        raise RuntimeError("baseline and M0 output structures differ")

    max_abs_error = 0.0
    max_rel_error = 0.0
    all_close = True
    tensor_results = {}
    for (name, expected), (_, actual) in zip(baseline_leaves, m0_leaves):
        if expected.shape != actual.shape:
            raise RuntimeError("shape mismatch for {}".format(name))
        difference = (expected - actual).abs()
        relative = difference / expected.abs().clamp_min(1e-12)
        tensor_abs = float(difference.max().item()) if difference.numel() else 0.0
        tensor_rel = float(relative.max().item()) if relative.numel() else 0.0
        close = bool(torch.allclose(expected, actual, rtol=1e-5, atol=1e-6))
        max_abs_error = max(max_abs_error, tensor_abs)
        max_rel_error = max(max_rel_error, tensor_rel)
        all_close = all_close and close
        tensor_results[name] = {
            "max_abs_error": tensor_abs,
            "max_rel_error": tensor_rel,
            "passed": close,
        }

    report = {
        "baseline": str(resolve_repo_path(args.baseline)),
        "m0": str(resolve_repo_path(args.m0)),
        "state_source": state_source,
        "input_shape": list(images.shape),
        "rtol": 1e-5,
        "atol": 1e-6,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "passed": all_close,
        "tensors": tensor_results,
    }
    save_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if all_close else 1


if __name__ == "__main__":
    raise SystemExit(main())
