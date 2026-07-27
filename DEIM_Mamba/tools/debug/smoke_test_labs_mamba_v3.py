"""Full-model LABS-Mamba V3 forward/backward, parity, and memory smoke test."""

from __future__ import annotations

import argparse
import gc
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from labs_mamba_debug_utils import (
    SIDECAR_PREFIX,
    audit_checkpoint_load,
    build_config,
    debug_state_to_json,
    gradients_report,
    load_checkpoint,
    parameter_counts,
    synchronize,
    tensor_tree_signature,
    output_surrogate_loss,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--collect-debug", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def _default_output(config_path):
    stem = Path(config_path).stem
    for stage in ("s0", "s1", "s2"):
        if stem.startswith(stage + "_"):
            return "artifacts/labs_mamba_v3/smoke_{}.json".format(stage)
    return "artifacts/labs_mamba_v3/smoke_{}.json".format(stem)


def _flatten_signatures(value):
    if isinstance(value, dict) and "shape" in value:
        return [value]
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_flatten_signatures(item))
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            result.extend(_flatten_signatures(item))
        return result
    return []


def main():
    args = parse_args()
    if args.height <= 0 or args.width <= 0 or args.batch_size <= 0:
        raise ValueError("height, width and batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    images = torch.randn(
        args.batch_size, 3, args.height, args.width, device=device
    )
    cfg = build_config(args.config)
    model = cfg.model
    checkpoint_report = None
    if args.checkpoint:
        checkpoint_report = audit_checkpoint_load(
            model, load_checkpoint(args.checkpoint), require_all_sidecar_missing=False
        )
        if not checkpoint_report["compatible"]:
            raise RuntimeError(
                "checkpoint audit failed: missing={}, unexpected={}, shape={}".format(
                    checkpoint_report["illegal_missing_keys"],
                    checkpoint_report["unexpected_keys"],
                    checkpoint_report["shape_mismatches"],
                )
            )
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "local_anchor_p4"):
        raise RuntimeError("selected config must create encoder.local_anchor_p4")
    sidecar_enabled = hasattr(model.encoder, "mamba_sidecar_p4")
    if sidecar_enabled:
        model.encoder.mamba_sidecar_p4.collect_debug = bool(args.collect_debug)
    model = model.to(device).eval()

    # Build a sidecar-off twin and copy every non-sidecar parameter.  This is
    # the direct P3/P5 invariant check with identical main-path weights.
    off_cfg = build_config(
        args.config,
        HybridEncoder={"use_mamba_sidecar": False},
    )
    off_model = off_cfg.model
    non_sidecar_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith(SIDECAR_PREFIX)
    }
    incompatible = off_model.load_state_dict(non_sidecar_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "sidecar-off parity model load failed: missing={}, unexpected={}".format(
                incompatible.missing_keys, incompatible.unexpected_keys
            )
        )
    off_model = off_model.to(device).eval()
    with torch.no_grad(), _autocast(device, args.amp):
        base_features = [
            feature.detach().clone()
            for feature in off_model.encoder(off_model.backbone(images))
        ]
    del off_model, off_cfg
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    captured = {}

    def capture_encoder_output(_module, _inputs, output):
        captured["features"] = output

    hook = model.encoder.register_forward_hook(capture_encoder_output)
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    with _autocast(device, args.amp):
        outputs = model(images)
        loss = output_surrogate_loss(outputs) if args.backward else None
    synchronize(device)
    forward_ms = 1000.0 * (time.perf_counter() - start)
    hook.remove()
    feature_outputs = captured.get("features")
    if not isinstance(feature_outputs, (list, tuple)) or len(feature_outputs) != 3:
        raise RuntimeError("HybridEncoder did not return [P3, P4, P5]")

    feature_diffs = [
        float((base - current.detach()).float().abs().max().item())
        for base, current in zip(base_features, feature_outputs)
    ]
    p3_equal = feature_diffs[0] == 0.0
    p4_equal = feature_diffs[1] == 0.0
    p5_equal = feature_diffs[2] == 0.0
    if not p3_equal or not p5_equal:
        raise RuntimeError("sidecar changed P3/P5; insertion is not isolated to final P4")
    if sidecar_enabled and p4_equal:
        raise RuntimeError("sidecar is enabled but final P4 did not change")

    if args.backward:
        loss.backward()
        synchronize(device)
    gradient_state = gradients_report(model)
    if args.backward and not gradient_state["valid"]:
        raise RuntimeError(
            "LABS-Mamba gradients failed: missing={}, non_finite={}".format(
                gradient_state["missing"], gradient_state["non_finite"]
            )
        )
    output_signature = tensor_tree_signature(outputs)
    if not all(
        entry.get("finite", True) for entry in _flatten_signatures(output_signature)
    ):
        raise RuntimeError("model output contains NaN or Inf")

    debug_state = model.encoder.get_labs_mamba_debug_state()
    beta = (
        float(model.encoder.mamba_sidecar_p4.effective_beta().detach().item())
        if sidecar_enabled
        else None
    )
    peak_memory_gb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda"
        else None
    )
    report = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "amp": bool(args.amp),
        "backward": bool(args.backward),
        "collect_debug": bool(args.collect_debug),
        "input_shape": list(images.shape),
        "full_model_output": output_signature,
        "encoder_feature_shapes": [list(feature.shape) for feature in feature_outputs],
        "encoder_features_finite": [
            bool(torch.isfinite(feature.detach()).all().item()) for feature in feature_outputs
        ],
        "sidecar_enabled": sidecar_enabled,
        "p3_exact_parity": p3_equal,
        "p4_exact_parity": p4_equal,
        "p5_exact_parity": p5_equal,
        "p3_max_abs_diff": feature_diffs[0],
        "p4_max_abs_diff": feature_diffs[1],
        "p5_max_abs_diff": feature_diffs[2],
        "forward_ms_single_measurement": forward_ms,
        "peak_memory_gb": peak_memory_gb,
        "parameters": parameter_counts(model),
        "beta_effective": beta,
        "labs_gradients": gradient_state,
        "labs_debug_state": debug_state_to_json(debug_state),
        "loss": float(loss.detach().item()) if loss is not None else None,
        "checkpoint_load": checkpoint_report,
    }
    output_path = args.output or _default_output(args.config)
    write_json(output_path, report)
    print("LABS-Mamba V3 smoke test passed: {}".format(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
