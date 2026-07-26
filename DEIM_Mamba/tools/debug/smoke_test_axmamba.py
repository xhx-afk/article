"""Full-model AxMamba forward/backward smoke test."""

from __future__ import annotations

import argparse
import json
import time

import torch

from axmamba_debug_utils import (
    assert_finite,
    build_model,
    count_parameters,
    debug_state_to_json,
    make_targets,
    resolve_repo_path,
    save_json,
    scalar_loss,
    structure_signature,
)


BASELINE_CONFIG = "configs/deim_dfine/deim_hgnetv2_l_wood.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--output", default="artifacts/axmamba_v1/smoke_test.json"
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, device: torch.device) -> None:
    if args.height <= 0 or args.width <= 0 or args.batch_size <= 0:
        raise ValueError("height, width, and batch-size must be positive")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp currently requires a CUDA device")


def run_baseline_signature(
    images: torch.Tensor, targets, training: bool, amp: bool
):
    baseline_config, baseline = build_model(BASELINE_CONFIG, images.device)
    baseline.train(training)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        output = baseline(images, targets if training else None)
    signature = structure_signature(output)
    del output, baseline, baseline_config
    if images.device.type == "cuda":
        torch.cuda.empty_cache()
    return signature


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    validate_args(args, device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats(device)

    images = torch.randn(
        args.batch_size, 3, args.height, args.width, device=device
    )
    targets = make_targets(args.batch_size, device)
    baseline_signature = run_baseline_signature(
        images, targets, args.backward, args.amp
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    _, model = build_model(args.config, device)
    if hasattr(model, "encoder") and hasattr(model.encoder, "axmamba_blocks"):
        for block in model.encoder.axmamba_blocks.values():
            block.collect_debug = True
    model.train(args.backward)
    # The synthetic scalar loss used by this smoke test is not the detector's
    # real training loss.  Using GradScaler's large default initial scale can
    # therefore overflow a few otherwise valid gradients on the first and only
    # backward pass.  Real training retries later steps with a reduced dynamic
    # scale; this one-shot connectivity/finite-gradient check starts at 1 so it
    # tests the model's AMP backward numerics instead of the scaler heuristic.
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp, init_scale=1.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.cuda.amp.autocast(enabled=args.amp):
        output = model(images, targets if args.backward else None)
        loss = scalar_loss(output) if args.backward else None
    if args.backward:
        scaler.scale(loss).backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert_finite(output)
    signature = structure_signature(output)
    if signature != baseline_signature:
        raise RuntimeError("AxMamba output structure differs from baseline")

    axmamba_gradients = {}
    if args.backward:
        for name, parameter in model.named_parameters():
            if "axmamba_blocks" not in name or not parameter.requires_grad:
                continue
            axmamba_gradients[name] = {
                "present": parameter.grad is not None,
                "finite": bool(
                    parameter.grad is not None
                    and torch.isfinite(parameter.grad).all().item()
                ),
            }
        missing_gradients = [
            name
            for name, state in axmamba_gradients.items()
            if not state["present"]
        ]
        non_finite_gradients = [
            name
            for name, state in axmamba_gradients.items()
            if state["present"] and not state["finite"]
        ]
        if missing_gradients or non_finite_gradients:
            raise RuntimeError(
                "AxMamba gradient check failed: missing={}, non_finite={}, "
                "amp_grad_scale={}".format(
                    missing_gradients,
                    non_finite_gradients,
                    scaler.get_scale(),
                )
            )

    debug_state = {}
    if hasattr(model, "encoder") and hasattr(
        model.encoder, "get_axmamba_debug_state"
    ):
        debug_state = model.encoder.get_axmamba_debug_state()

    report = {
        "config": str(resolve_repo_path(args.config)),
        "device": str(device),
        "amp": args.amp,
        "amp_grad_scale": scaler.get_scale() if args.backward else None,
        "backward": args.backward,
        "input_shape": list(images.shape),
        "output_structure_matches_baseline": True,
        "output_structure": signature,
        "loss": None if loss is None else float(loss.detach().float().item()),
        "elapsed_ms": elapsed_ms,
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "parameters": count_parameters(model),
        "axmamba_gradients": axmamba_gradients,
        "axmamba_debug_state": debug_state_to_json(debug_state),
        "status": "passed",
    }

    output_path = resolve_repo_path(args.output)
    existing = {}
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    runs = existing.get("runs", {})
    runs[resolve_repo_path(args.config).stem] = report
    save_json(args.output, {"runs": runs})
    print(json.dumps(report, indent=2))
    print("report:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
