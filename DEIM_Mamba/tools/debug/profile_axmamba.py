"""Profile parameters, MACs, latency, and memory for M0-M3."""

from __future__ import annotations

import argparse
import csv
import time

import torch

from axmamba_debug_utils import (
    build_model,
    count_parameters,
    make_targets,
    resolve_repo_path,
    scalar_loss,
)


DEFAULT_CONFIGS = [
    "configs/deim_dfine/ablation_axmamba_v1/m0_baseline.yml",
    "configs/deim_dfine/ablation_axmamba_v1/m1_local_p4p5.yml",
    "configs/deim_dfine/ablation_axmamba_v1/m2_axis_mean_p4p5.yml",
    "configs/deim_dfine/ablation_axmamba_v1/m3_axis_gate_p4p5.yml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--output", default="artifacts/axmamba_v1/model_complexity.csv"
    )
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_one(config_path: str, args: argparse.Namespace):
    device = torch.device(args.device)
    row = {
        "config": resolve_repo_path(config_path).stem,
        "config_path": str(resolve_repo_path(config_path)),
        "status": "failed",
        "error": "",
        "params": "",
        "trainable_params": "",
        "axmamba_params": "",
        "macs_g": "",
        "macs_status": "not_run",
        "amp_inference_latency_ms": "",
        "amp_inference_peak_memory_bytes": "",
        "amp_train_step_peak_memory_bytes": "",
    }
    try:
        model_config, model = build_model(config_path, device)
        counts = count_parameters(model)
        row.update(
            {
                "params": counts["total"],
                "trainable_params": counts["trainable"],
                "axmamba_params": counts["axmamba"],
            }
        )
        images = torch.randn(1, 3, args.height, args.width, device=device)

        try:
            from thop import profile

            model.eval()
            with torch.no_grad():
                macs, _ = profile(model, inputs=(images,), verbose=False)
            row["macs_g"] = macs / 1e9
            if "axis_" in resolve_repo_path(config_path).stem:
                row["macs_status"] = (
                    "partial_thop_selective_scan_not_counted_reliably"
                )
            else:
                row["macs_status"] = "thop_reported"
        except Exception as exc:
            row["macs_status"] = "unavailable: {}: {}".format(
                type(exc).__name__, exc
            )

        model.eval()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for _ in range(args.warmup):
                with torch.cuda.amp.autocast(enabled=args.amp):
                    model(images)
            synchronize(device)
            started = time.perf_counter()
            for _ in range(args.iterations):
                with torch.cuda.amp.autocast(enabled=args.amp):
                    model(images)
            synchronize(device)
        row["amp_inference_latency_ms"] = (
            time.perf_counter() - started
        ) * 1000.0 / args.iterations
        if device.type == "cuda":
            row["amp_inference_peak_memory_bytes"] = torch.cuda.max_memory_allocated(
                device
            )

        model.train()
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        targets = make_targets(1, device)
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(images, targets)
            loss = scalar_loss(output)
        scaler.scale(loss).backward()
        synchronize(device)
        if device.type == "cuda":
            row["amp_train_step_peak_memory_bytes"] = torch.cuda.max_memory_allocated(
                device
            )
        row["status"] = "passed"
        del output, loss, images, model, model_config
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:
        row["error"] = "{}: {}".format(type(exc).__name__, exc)
    return row


def main() -> int:
    args = parse_args()
    if min(args.height, args.width, args.iterations) <= 0 or args.warmup < 0:
        raise ValueError("invalid spatial size or iteration count")
    if args.amp and torch.device(args.device).type != "cuda":
        raise ValueError("--amp requires CUDA")

    rows = [profile_one(config, args) for config in args.configs]
    output_path = resolve_repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print("report:", output_path)
    return 0 if all(row["status"] == "passed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
