"""Run a bounded synthetic SQ-Align forward/backward optimizer smoke test."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from optional_dependency_stubs import install  # noqa: E402

install()

from engine.core import YAMLConfig  # noqa: E402


def _load_matching(model: torch.nn.Module, checkpoint_path: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pretrained = checkpoint.get("ema", {}).get("module", checkpoint.get("model", checkpoint))
    current = model.state_dict()
    compatible = {
        key: value for key, value in pretrained.items()
        if key in current and value.shape == current[key].shape
    }
    result = model.load_state_dict(compatible, strict=False)
    return {"loaded": len(compatible), "missing": list(result.missing_keys)}


def _targets(batch_size: int, image_size: int, num_classes: int, device: torch.device, step: int):
    targets = []
    for batch_index in range(batch_size):
        shift = ((step + batch_index) % 7 - 3) / 100.0
        mask = torch.zeros((1, image_size, image_size), device=device, dtype=torch.bool)
        lower = image_size // 4
        upper = 3 * image_size // 4
        mask[:, lower:upper, lower:upper] = True
        targets.append({
            "labels": torch.tensor([(step + batch_index) % num_classes], device=device),
            "boxes": torch.tensor([[0.5 + shift, 0.5, 0.5, 0.5]], device=device),
            "masks": mask,
            "mask_valid": torch.tensor([True], device=device),
            "orig_size": torch.tensor([image_size, image_size], device=device),
        })
    return targets


def run(args: argparse.Namespace) -> dict:
    if args.iterations <= 0 or args.iterations > 100:
        raise ValueError("iterations must be in [1, 100]; this tool must not start full training")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    config = YAMLConfig(
        args.config,
        eval_spatial_size=[args.image_size, args.image_size],
        num_classes=args.num_classes,
    )
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    model = config.model.to(device).train()
    criterion = config.criterion.to(device).train()
    criterion.set_epoch(args.epoch)
    load_result = None
    if args.checkpoint:
        load_result = _load_matching(model, Path(args.checkpoint))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    aggregates = defaultdict(float)
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(args.iterations):
        generator = torch.Generator(device=device).manual_seed(args.seed + step)
        images = torch.randn(
            (args.batch_size, 3, args.image_size, args.image_size), device=device, generator=generator
        )
        targets = _targets(args.batch_size, args.image_size, args.num_classes, device, step)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=args.amp,
        ):
            outputs = model(images, targets=targets)
            losses = criterion(outputs, targets)
            total_loss = sum(value for key, value in losses.items() if key.startswith("loss_"))
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite total loss at step {step}: {float(total_loss)}")
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        aggregates["total_loss"] += float(total_loss.detach())
        for key, value in losses.items():
            if key.startswith("loss_") or key.startswith("metric_"):
                aggregates[key] += float(value.detach())
        if step == 0 or (step + 1) % 10 == 0:
            loss_value = lambda key: float(losses.get(key, total_loss.detach() * 0.0))
            print(
                f"step={step + 1}/{args.iterations} total={float(total_loss):.5f} "
                f"loc={loss_value('loss_loc_quality'):.5f} "
                f"defect={loss_value('loss_defect'):.5f} "
                f"semantic={loss_value('loss_query_semantic'):.5f} "
                f"rank={loss_value('loss_final_bg_rank'):.5f}"
            )

    elapsed = time.perf_counter() - start
    nonfinite_parameters = [
        name for name, parameter in model.named_parameters()
        if not torch.isfinite(parameter.detach()).all()
    ]
    if nonfinite_parameters:
        raise RuntimeError(f"non-finite model parameters after smoke: {nonfinite_parameters[:20]}")
    result = {
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "amp": args.amp,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "iterations_per_second": args.iterations / elapsed,
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else None
        ),
        "mean": {key: value / args.iterations for key, value in sorted(aggregates.items())},
        "checkpoint_load": load_result,
        "nonfinite_parameters": nonfinite_parameters,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c", "--config", default=str(ROOT / "configs/deim_dfine/ablation_sqalign/r4_final_score_rank.yml")
    )
    parser.add_argument("-t", "--checkpoint")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output-json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
