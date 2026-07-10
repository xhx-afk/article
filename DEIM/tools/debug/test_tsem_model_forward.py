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


def main() -> None:
    parser = argparse.ArgumentParser(description="TSEM full model forward test")
    parser.add_argument("-c", "--config", default="configs/deim_dfine/deim_hgnetv2_l_wood_tsem_full.yml")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-path", default="output/tsem_debug_state.pth")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg = YAMLConfig(args.config, HGNetv2={"pretrained": False})
    model = cfg.model.to(device).eval()
    images = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)

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
