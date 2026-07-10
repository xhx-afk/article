"""
Build a complete DEIM model with YAMLConfig and run a random forward pass.

Run:
    python tools/debug/test_tsem_model_forward.py -c configs/deim_dfine/deim_hgnetv2_l_wood_tsem_full.yml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from engine.core import YAMLConfig  # noqa: E402


def _resolve_image_hw(args: argparse.Namespace, cfg: YAMLConfig) -> tuple[int, int]:
    """Use config eval_spatial_size by default to match cached eval positional embedding."""
    if args.image_height is not None or args.image_width is not None:
        if args.image_height is None or args.image_width is None:
            raise ValueError("--image-height and --image-width must be provided together")
        return int(args.image_height), int(args.image_width)

    if args.image_size is not None:
        return int(args.image_size), int(args.image_size)

    eval_size = cfg.yaml_cfg.get("eval_spatial_size")
    if isinstance(eval_size, (list, tuple)) and len(eval_size) == 2:
        return int(eval_size[0]), int(eval_size[1])

    encoder_cfg = cfg.yaml_cfg.get("HybridEncoder", {})
    eval_size = encoder_cfg.get("eval_spatial_size")
    if isinstance(eval_size, (list, tuple)) and len(eval_size) == 2:
        return int(eval_size[0]), int(eval_size[1])

    return 640, 640


def main() -> None:
    parser = argparse.ArgumentParser(description="TSEM full model forward test")
    parser.add_argument("-c", "--config", default="configs/deim_dfine/deim_hgnetv2_l_wood_tsem_full.yml")
    parser.add_argument("--image-size", type=int, default=None, help="Square input size. Defaults to config eval_spatial_size.")
    parser.add_argument("--image-height", type=int, default=None, help="Custom input height.")
    parser.add_argument("--image-width", type=int, default=None, help="Custom input width.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-path", default="output/tsem_debug_state.pth")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg = YAMLConfig(args.config, HGNetv2={"pretrained": False})
    image_h, image_w = _resolve_image_hw(args, cfg)
    model = cfg.model.to(device).eval()
    images = torch.randn(args.batch_size, 3, image_h, image_w, device=device)
    print(f"input_size: {(image_h, image_w)}")

    with torch.no_grad():
        if args.amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(images)
        else:
            outputs = model(images)

    assert isinstance(outputs, dict), type(outputs)
    for key in ("pred_logits", "pred_boxes"):
        assert key in outputs, outputs.keys()
        assert torch.isfinite(outputs[key]).all(), f"{key} contains NaN or Inf"
    print(f"forward PASS: logits={tuple(outputs['pred_logits'].shape)}, boxes={tuple(outputs['pred_boxes'].shape)}")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    reload_cfg = YAMLConfig(args.config, HGNetv2={"pretrained": False})
    reloaded = reload_cfg.model.to(device).eval()
    reloaded.load_state_dict(torch.load(save_path, map_location=device))
    print(f"state_dict save/load PASS: {save_path}")


if __name__ == "__main__":
    main()
