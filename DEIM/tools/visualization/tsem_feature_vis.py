"""
Visualize cached TSEM debug maps for one image.

Run:
    python tools/visualization/tsem_feature_vis.py \
      -c configs/deim_dfine/deim_hgnetv2_l_wood_tsem_full.yml \
      --input /path/to/image.jpg \
      --output-dir output/tsem_vis \
      --level 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from engine.core import YAMLConfig  # noqa: E402


def _load_state(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    if "ema" in checkpoint:
        ema = checkpoint["ema"]
        return ema["module"] if isinstance(ema, dict) and "module" in ema else ema
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _prepare_image(path: Path, image_size: int, device: torch.device):
    image = Image.open(path).convert("RGB")
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(resized.tobytes()))
    data = data.reshape(image_size, image_size, 3).permute(2, 0, 1).float() / 255.0
    return image, data.unsqueeze(0).to(device)


def _normalize_map(tensor: torch.Tensor, output_size) -> Image.Image:
    tensor = F.interpolate(tensor.float(), size=output_size[::-1], mode="bilinear", align_corners=False)
    tensor = tensor[0, 0].detach().cpu()
    min_v = float(tensor.min())
    max_v = float(tensor.max())
    if max_v > min_v:
        tensor = (tensor - min_v) / (max_v - min_v)
    else:
        tensor = tensor * 0
    array = (tensor.clamp(0, 1).numpy() * 255).astype("uint8")
    return Image.fromarray(array, mode="L")


def _overlay(image: Image.Image, heat: Image.Image, alpha: float = 0.45) -> Image.Image:
    heat_rgb = Image.merge("RGB", (heat, Image.new("L", heat.size, 0), Image.eval(heat, lambda v: 255 - v)))
    return Image.blend(image.convert("RGB"), heat_rgb, alpha)


def _save_map(output_dir: Path, image: Image.Image, name: str, tensor: Optional[torch.Tensor]) -> None:
    if tensor is None:
        return
    gray = _normalize_map(tensor, image.size)
    gray.save(output_dir / f"{name}_gray.png")
    _overlay(image, gray).save(output_dir / f"{name}_overlay.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="TSEM feature visualization")
    parser.add_argument("-c", "--config", default="configs/deim_dfine/deim_hgnetv2_l_wood_tsem_full.yml")
    parser.add_argument("-r", "--resume", default=None, help="checkpoint path")
    parser.add_argument("--input", required=True, help="input image")
    parser.add_argument("--output-dir", default="output/tsem_vis")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg = YAMLConfig(
        args.config,
        HGNetv2={"pretrained": False},
        HybridEncoder={"tsem_debug": True},
    )
    model = cfg.model.to(device).eval()
    if args.resume:
        missing, unexpected = model.load_state_dict(_load_state(Path(args.resume)), strict=False)
        print(f"missing_keys: {len(missing)}, unexpected_keys: {len(unexpected)}")

    image, tensor = _prepare_image(Path(args.input), args.image_size, device)
    with torch.no_grad():
        _ = model(tensor)

    tsem = getattr(model.encoder, "tsem", None)
    if tsem is None:
        raise RuntimeError("This model has no active TSEM. Check use_tsem/tsem_mode in config.")

    block_debug = tsem.last_debug.get("blocks", {}).get(str(args.level), {})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_map(output_dir, image, f"level{args.level}_high", block_debug.get("high_heatmap"))
    _save_map(output_dir, image, f"level{args.level}_context", block_debug.get("context_heatmap"))
    _save_map(output_dir, image, f"level{args.level}_gate", block_debug.get("gate_heatmap"))

    stats = {
        "level": args.level,
        "gamma": [float(v) for v in tsem.gamma.detach().cpu()],
        "available_block_keys": sorted(block_debug.keys()),
    }
    if "scale_weight_channel_mean" in tsem.last_debug:
        stats["scale_weight_channel_mean"] = tsem.last_debug["scale_weight_channel_mean"].tolist()
    (output_dir / "tsem_debug_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved TSEM visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
