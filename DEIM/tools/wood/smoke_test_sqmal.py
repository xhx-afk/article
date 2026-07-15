"""SQ-MAL dataset/model forward/backward smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig  # noqa: E402


def _checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if "ema" in checkpoint:
        ema = checkpoint["ema"]
        state = ema.get("module", ema) if isinstance(ema, dict) else ema
    else:
        state = checkpoint.get("model", checkpoint)
    return {key.removeprefix("module."): value for key, value in state.items()}


def _grad_sum(model: torch.nn.Module, name_parts) -> float:
    return float(sum(
        parameter.grad.detach().abs().sum().item()
        for name, parameter in model.named_parameters()
        if any(part in name for part in name_parts) and parameter.grad is not None
    ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--coco-json", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("-t", "--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但 torch.cuda.is_available() 为 False。")
    update = {
        "train_dataloader": {
            "total_batch_size": args.batch_size,
            "num_workers": 0,
            "dataset": {
                "ann_file": str(args.coco_json),
                "img_folder": str(args.images_dir),
                "return_masks": True,
            },
        }
    }
    cfg = YAMLConfig(args.config, **update)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    device = torch.device(args.device)
    model = cfg.model.to(device).train()
    criterion = cfg.criterion.to(device).train()
    if hasattr(criterion, "set_epoch"):
        criterion.set_epoch(5)

    if args.checkpoint:
        missing, unexpected = model.load_state_dict(_checkpoint_state(args.checkpoint), strict=False)
        allowed_missing = ("dec_quality_head", "defectness_head")
        invalid_missing = [key for key in missing if not any(part in key for part in allowed_missing)]
        if invalid_missing or unexpected:
            raise RuntimeError(
                f"checkpoint 不兼容：invalid_missing={invalid_missing[:20]}, unexpected={unexpected[:20]}"
            )
        print(f"checkpoint_missing_new_keys: {len(missing)}")

    data_loader = cfg.train_dataloader
    data_loader.set_epoch(0)
    samples, targets = next(iter(data_loader))
    print(f"image tensor shape: {tuple(samples.shape)}")
    for index, target in enumerate(targets):
        print(
            f"target[{index}]: boxes={tuple(target['boxes'].shape)}, "
            f"labels={tuple(target['labels'].shape)}, masks={tuple(target['masks'].shape)}, "
            f"mask_valid={tuple(target['mask_valid'].shape)}"
        )
        count = target["boxes"].shape[0]
        assert count == target["labels"].shape[0] == target["masks"].shape[0] == target["mask_valid"].shape[0]

    samples = samples.to(device)
    targets = [{key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()} for target in targets]
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        outputs = model(samples, targets=targets)
    for key in ("pred_logits", "pred_boxes", "pred_quality", "pred_defect_logits"):
        value = outputs.get(key)
        print(f"{key}: {None if value is None else tuple(value.shape)}")
    assert outputs["pred_logits"].shape[:2] == outputs["pred_boxes"].shape[:2]
    if "pred_quality" in outputs:
        assert outputs["pred_quality"].shape == (*outputs["pred_logits"].shape[:2], 1)

    with torch.autocast(device_type=device.type, enabled=False):
        losses = criterion(outputs, targets)
        total_loss = sum(value for key, value in losses.items() if key.startswith("loss_"))
    for key, value in losses.items():
        if key.startswith("loss_"):
            print(f"{key}: {float(value.detach()):.6f}")
    print(f"total_loss: {float(total_loss.detach()):.6f}")
    if not torch.isfinite(total_loss):
        raise RuntimeError("total loss is NaN/Inf")

    if not args.forward_only:
        total_loss.backward()
        gradients = {
            "quality_head": _grad_sum(model, ("dec_quality_head",)),
            "defectness_head": _grad_sum(model, ("defectness_head",)),
            "classifier": _grad_sum(model, ("dec_score_head",)),
            "bbox_head": _grad_sum(model, ("dec_bbox_head",)),
        }
        print(f"gradients: {gradients}")
        for name, value in gradients.items():
            if value <= 0:
                raise RuntimeError(f"{name} gradient is zero")
    print("SQ-MAL smoke test: PASS")


if __name__ == "__main__":
    main()

