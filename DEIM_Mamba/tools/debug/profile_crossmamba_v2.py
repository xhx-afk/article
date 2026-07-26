"""Profile R0-R4 parameter counts, memory and latency.

FLOPs are intentionally reported as ``unsupported`` because generic FLOPs
tools do not account for the official ``mamba_ssm`` kernels reliably.
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from crossmamba_debug_utils import (
    audit_checkpoint_load,
    build_config,
    load_checkpoint,
    output_surrogate_loss,
    parameter_counts,
    synchronize,
)


DEFAULT_CONFIGS = {
    "R0": "configs/deim_dfine/crossmamba_v2/r0_baseline.yml",
    "R1": "configs/deim_dfine/crossmamba_v2/r1_postfpn_local.yml",
    "R2": "configs/deim_dfine/crossmamba_v2/r2_cross_hv_only.yml",
    "R3": "configs/deim_dfine/crossmamba_v2/r3_cross_hv_vh_mean.yml",
    "R4": "configs/deim_dfine/crossmamba_v2/r4_cross_hv_vh_gate.yml",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpoint", action="append", default=[], help="R4=/path/to/checkpoint.pth")
    parser.add_argument(
        "--output", default="artifacts/crossmamba_v2/model_complexity.csv"
    )
    return parser.parse_args()


def _autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast()
    return nullcontext()


def _parse_checkpoints(items):
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--checkpoint must use EXPERIMENT=PATH")
        experiment, path = item.split("=", 1)
        result[experiment.upper()] = path
    return result


def _measure_inference(model, images, device, amp, warmup, iterations):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            with _autocast(device, amp):
                model(images)
        synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            with _autocast(device, amp):
                model(images)
        synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iterations


def _measure_train_step(model, images, device, amp, warmup, iterations):
    # A real train-mode decoder call is used with one synthetic target.  This
    # exercises the same CrossMamba autograd path as training without reading
    # or changing the project's dataset protocol.
    targets = [
        {
            "labels": torch.zeros(1, dtype=torch.long, device=device),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device),
        }
        for _ in range(images.shape[0])
    ]
    model.train()
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            loss = output_surrogate_loss(model(images, targets))
        loss.backward()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        model.zero_grad(set_to_none=True)
        with _autocast(device, amp):
            loss = output_surrogate_loss(model(images, targets))
        loss.backward()
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iterations


def main():
    args = parse_args()
    if args.height <= 0 or args.width <= 0 or args.batch_size <= 0:
        raise ValueError("height, width and batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    checkpoints = _parse_checkpoints(args.checkpoint)
    images = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    rows = []
    for experiment, config_path in DEFAULT_CONFIGS.items():
        cfg = None
        model = None
        checkpoint_report = None
        counts = {"total": "not_available", "crossmamba": "not_available"}
        infer_memory = train_memory = None
        infer_latency = train_latency = "unsupported"
        try:
            cfg = build_config(config_path)
            model = cfg.model
            if experiment in checkpoints:
                checkpoint_report = audit_checkpoint_load(model, load_checkpoint(checkpoints[experiment]))
                if not checkpoint_report["compatible"]:
                    raise RuntimeError("checkpoint audit failed for {}".format(experiment))
            model.to(device)
            counts = parameter_counts(model)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            infer_latency = _measure_inference(
                model, images, device, args.amp, args.warmup, args.iterations
            )
            if device.type == "cuda":
                infer_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_latency = _measure_train_step(
                model, images, device, args.amp, args.warmup, args.iterations
            )
            if device.type == "cuda":
                train_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        except (RuntimeError, ImportError) as exc:
            print("{} profiling unsupported: {}".format(experiment, exc))
        rows.append(
            {
                "experiment": experiment,
                "total_params": counts["total"],
                "crossmamba_params": counts["crossmamba"],
                "peak_infer_mem_gb": infer_memory if infer_memory is not None else "unsupported",
                "peak_train_mem_gb": train_memory if train_memory is not None else "unsupported",
                "infer_latency_ms": infer_latency,
                "train_step_ms": train_latency,
                "flops_status": "unsupported",
                "config": config_path,
                "checkpoint_load": checkpoint_report or {},
            }
        )
        del model, cfg
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment",
        "total_params",
        "crossmamba_params",
        "peak_infer_mem_gb",
        "peak_train_mem_gb",
        "infer_latency_ms",
        "train_step_ms",
        "flops_status",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("CrossMamba V2 complexity report: {}".format(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
