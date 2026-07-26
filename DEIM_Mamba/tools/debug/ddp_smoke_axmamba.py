"""Two-GPU full-model DDP/AMP/checkpoint/EMA smoke test with random targets."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from axmamba_debug_utils import (
    assert_finite,
    build_model,
    make_targets,
    resolve_repo_path,
    save_json,
    scalar_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/deim_dfine/ablation_axmamba_v1/m3_axis_gate_p4p5.yml",
    )
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--checkpoint",
        default="artifacts/axmamba_v1/ddp_smoke_checkpoint.pth",
    )
    parser.add_argument(
        "--output", default="artifacts/axmamba_v1/ddp_smoke.json"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.height, args.width, args.batch_size, args.iterations) <= 0:
        raise ValueError("spatial size, batch size, and iterations must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("DDP smoke test requires CUDA")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("AxMamba DDP smoke test requires exactly two ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    torch.manual_seed(1234 + rank)

    config, model = build_model(args.config, device)
    from engine.optim.ema import ModelEMA

    ema = ModelEMA(model, decay=0.9999, warmups=0)
    ema_keys = set(ema.module.state_dict())
    model_ax_keys = {
        name for name in model.state_dict() if "encoder.axmamba_blocks" in name
    }
    if not model_ax_keys.issubset(ema_keys):
        raise RuntimeError("EMA is missing AxMamba state")

    ddp = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    optimizer = config.optimizer
    # Keep the one-shot synthetic-loss gradient check independent of
    # GradScaler's large default initial scale.  Production training still uses
    # the normal dynamic scaler configured by the solver.
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp, init_scale=1.0)
    ax_gradient_checks = []

    for _ in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        images = torch.randn(
            args.batch_size, 3, args.height, args.width, device=device
        )
        targets = make_targets(args.batch_size, device)
        with torch.cuda.amp.autocast(enabled=args.amp):
            output = ddp(images, targets)
            loss = scalar_loss(output)
        assert_finite(output)
        scaler.scale(loss).backward()
        checks = {
            name: bool(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all().item()
            )
            for name, parameter in ddp.module.named_parameters()
            if "axmamba_blocks" in name and parameter.requires_grad
        }
        missing_gradients = [
            name
            for name, parameter in ddp.module.named_parameters()
            if "axmamba_blocks" in name
            and parameter.requires_grad
            and parameter.grad is None
        ]
        non_finite_gradients = [
            name
            for name, parameter in ddp.module.named_parameters()
            if "axmamba_blocks" in name
            and parameter.requires_grad
            and parameter.grad is not None
            and not torch.isfinite(parameter.grad).all().item()
        ]
        if not checks or missing_gradients or non_finite_gradients:
            raise RuntimeError(
                "AxMamba DDP gradient check failed: missing={}, "
                "non_finite={}, amp_grad_scale={}".format(
                    missing_gradients,
                    non_finite_gradients,
                    scaler.get_scale(),
                )
            )
        ax_gradient_checks.append(checks)
        scaler.step(optimizer)
        scaler.update()
        ema.update(ddp)

    checkpoint_path = resolve_repo_path(args.checkpoint)
    if rank == 0:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": ddp.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": ema.state_dict(),
            },
            str(checkpoint_path),
        )
    dist.barrier()
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    ddp.module.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    ema.load_state_dict(checkpoint["ema"], strict=True)
    dist.barrier()

    if rank == 0:
        report = {
            "config": str(resolve_repo_path(args.config)),
            "world_size": world_size,
            "input_shape_per_rank": [
                args.batch_size,
                3,
                args.height,
                args.width,
            ],
            "iterations": args.iterations,
            "amp": args.amp,
            "find_unused_parameters": False,
            "axmamba_gradients_finite": True,
            "ema_contains_axmamba": True,
            "checkpoint_save_restore": True,
            "checkpoint": str(checkpoint_path),
            "status": "passed",
        }
        save_json(args.output, report)
        print(json.dumps(report, indent=2))

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
