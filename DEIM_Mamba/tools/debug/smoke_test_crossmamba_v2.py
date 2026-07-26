"""Full-model CrossMamba V2 forward/backward smoke test."""

from __future__ import annotations

import argparse
import gc
from contextlib import nullcontext

import torch

from crossmamba_debug_utils import (
    audit_checkpoint_load,
    build_config,
    debug_state_to_json,
    gradients_report,
    load_checkpoint,
    parameter_counts,
    tensor_tree_signature,
    output_surrogate_loss,
    synchronize,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--baseline-config",
        default="configs/deim_dfine/crossmamba_v2/r0_baseline.yml",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--output", default="artifacts/crossmamba_v2/smoke_test.json"
    )
    return parser.parse_args()


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def _structure_only(value):
    if isinstance(value, dict) and "shape" in value:
        return {"shape": value["shape"], "dtype": value["dtype"]}
    if isinstance(value, dict):
        return {key: _structure_only(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_structure_only(item) for item in value]
    return value


def _baseline_signature(config_path, images, device, amp):
    cfg = build_config(config_path)
    model = cfg.model.to(device).eval()
    with torch.no_grad(), _autocast(device, amp):
        outputs = model(images)
    signature = tensor_tree_signature(outputs)
    del outputs, model, cfg
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return signature


def main():
    args = parse_args()
    if args.height <= 0 or args.width <= 0 or args.batch_size <= 0:
        raise ValueError("height, width and batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")

    images = torch.randn(
        args.batch_size, 3, args.height, args.width, device=device
    )
    baseline_signature = _baseline_signature(
        args.baseline_config, images, device, args.amp
    )

    cfg = build_config(args.config)
    model = cfg.model
    checkpoint_report = None
    if args.checkpoint:
        checkpoint_report = audit_checkpoint_load(model, load_checkpoint(args.checkpoint))
        if not checkpoint_report["compatible"]:
            raise RuntimeError(
                "checkpoint has illegal missing/unexpected keys: {}".format(
                    checkpoint_report
                )
            )
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "crossmamba_p4"):
        raise RuntimeError("selected config does not create encoder.crossmamba_p4")
    model.encoder.crossmamba_p4.collect_debug = True
    model = model.to(device).eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)
    with _autocast(device, args.amp):
        outputs = model(images)
        loss = output_surrogate_loss(outputs) if args.backward else None
    output_signature = tensor_tree_signature(outputs)
    if args.backward:
        loss.backward()
    synchronize(device)

    gradient_state = gradients_report(model)
    if args.backward and not gradient_state["valid"]:
        raise RuntimeError(
            "CrossMamba gradient check failed: missing={}, non_finite={}".format(
                gradient_state["missing"], gradient_state["non_finite"]
            )
        )
    if not all(
        entry.get("finite", True)
        for entry in _flatten_tensor_signatures(output_signature)
    ):
        raise RuntimeError("model output contains NaN or Inf")

    baseline_structure = _structure_only(baseline_signature)
    output_structure = _structure_only(output_signature)
    same_structure = baseline_structure == output_structure
    if not same_structure:
        raise RuntimeError("CrossMamba output structure differs from R0 baseline")

    peak_memory_gb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda"
        else None
    )
    report = {
        "config": args.config,
        "baseline_config": args.baseline_config,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "amp": bool(args.amp),
        "backward": bool(args.backward),
        "input_shape": list(images.shape),
        "output_signature": output_signature,
        "baseline_structure_matches": same_structure,
        "loss": float(loss.detach().item()) if loss is not None else None,
        "peak_memory_gb": peak_memory_gb,
        "parameters": parameter_counts(model),
        "crossmamba_gradients": gradient_state,
        "crossmamba_debug_state": debug_state_to_json(
            model.encoder.get_crossmamba_debug_state()
        ),
        "checkpoint_load": checkpoint_report,
    }
    write_json(args.output, report)
    print("CrossMamba V2 smoke test passed; report: {}".format(args.output))
    return 0


def _flatten_tensor_signatures(value):
    if isinstance(value, dict) and "shape" in value:
        return [value]
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_flatten_tensor_signatures(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_flatten_tensor_signatures(item))
        return result
    return []


if __name__ == "__main__":
    raise SystemExit(main())
