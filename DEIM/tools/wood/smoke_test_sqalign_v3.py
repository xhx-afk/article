"""Bounded (<=100 iteration) SQ-Align V3 forward/backward smoke runner."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.wood.optional_dependency_stubs import install as install_optional_dependency_stubs  # noqa: E402


install_optional_dependency_stubs()

from engine.core import YAMLConfig  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="SQ-Align V3 bounded smoke test")
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("-t", "--tuning", type=Path)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    # Synthetic 128px batches can produce a large aggregate D-FINE auxiliary
    # loss.  A conservative initial scale keeps this bounded smoke focused on
    # graph correctness; production GradScaler may grow it dynamically.
    parser.add_argument("--amp-init-scale", type=float, default=1.0)
    parser.add_argument("--detect-anomaly", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def compatible_load(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("ema", {}).get("module", checkpoint.get("model", checkpoint))
    current = model.state_dict()
    compatible = {key: value for key, value in state.items() if key in current and current[key].shape == value.shape}
    result = model.load_state_dict(compatible, strict=False)
    return {"matched": len(compatible), "missing": len(result.missing_keys)}


def synthetic_batch(batch_size, image_size, num_classes, device, step):
    images = torch.randn((batch_size, 3, image_size, image_size), device=device)
    targets = []
    for batch_index in range(batch_size):
        offset = (step + batch_index) % max(image_size // 16, 1)
        mask = torch.zeros((1, image_size, image_size), device=device, dtype=torch.bool)
        start, end = image_size // 4 + offset, image_size * 3 // 4 + offset
        end = min(end, image_size)
        mask[:, start:end, start:end] = True
        center = (start + end) / (2 * image_size)
        size = (end - start) / image_size
        targets.append({
            "labels": torch.tensor([(step + batch_index) % num_classes], device=device),
            "boxes": torch.tensor([[center, center, size, size]], device=device),
            "masks": mask, "mask_valid": torch.tensor([True], device=device),
            "orig_size": torch.tensor([image_size, image_size], device=device),
        })
    return images, targets


def main() -> None:
    args = parse_args()
    if not 1 <= args.iterations <= 100:
        raise ValueError("iterations must be within [1,100]")
    torch.manual_seed(args.seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly, check_nan=True)
    config = YAMLConfig(
        str(args.config), eval_spatial_size=[args.image_size, args.image_size],
        num_classes=args.num_classes,
    )
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    device = torch.device(args.device)
    model = config.model.to(device).train()
    criterion = config.criterion.to(device).train()
    criterion.set_epoch(20, total_epochs=25)
    checkpoint_load = compatible_load(model, args.tuning) if args.tuning else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.0)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=args.amp and device.type == "cuda",
        init_scale=float(args.amp_init_scale),
    )
    history = []
    started = time.perf_counter()
    for step in range(args.iterations):
        images, targets = synthetic_batch(
            args.batch_size, args.image_size, args.num_classes, device, step
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            outputs = model(images, targets=targets)
            losses = criterion(outputs, targets)
            total = sum(value for key, value in losses.items() if key.startswith("loss_"))
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite total loss at iteration {step}: {total}")
        anomaly_context = (
            torch.autograd.detect_anomaly(check_nan=True)
            if args.detect_anomaly else contextlib.nullcontext()
        )
        with anomaly_context:
            scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        nonfinite_gradients = [
            name for name, parameter in model.named_parameters()
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        ]
        if nonfinite_gradients:
            raise FloatingPointError(
                f"non-finite gradient at iteration {step}: "
                + ", ".join(nonfinite_gradients[:20])
            )
        scaler.step(optimizer)
        scaler.update()
        history.append({
            "iteration": step + 1, "total_loss": float(total.detach()),
            **{
                key: float(value.detach()) for key, value in losses.items()
                if key in {
                    "loss_mal", "loss_bbox", "loss_giou", "loss_fgl", "loss_ddf",
                    "loss_defect", "loss_query_mask", "loss_semantic_quality",
                    "loss_semantic_rank", "loss_candidate_rank",
                }
            },
        })
    elapsed = time.perf_counter() - started
    required_shapes = {
        key: list(outputs[key].shape) for key in (
            "pred_logits", "pred_boxes", "pred_query_features", "pred_defect_logits",
            "pred_query_mask_logits", "pred_sem_quality",
        ) if key in outputs
    }
    report = {
        "config": str(args.config), "checkpoint_load": checkpoint_load,
        "iterations": args.iterations, "batch_size": args.batch_size,
        "image_size": args.image_size, "device": str(device), "amp": args.amp,
        "amp_init_scale": args.amp_init_scale,
        "elapsed_seconds": elapsed, "mean_seconds_per_iteration": elapsed / args.iterations,
        "output_shapes": required_shapes, "last_iteration": history[-1],
        "all_finite": True,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
