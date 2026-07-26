"""Shared helpers for AxMamba V1 validation scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def build_model(config_path: str, device: torch.device):
    """Build a full detector without downloading backbone weights."""
    from engine.core import YAMLConfig

    config = YAMLConfig(
        str(resolve_repo_path(config_path)),
        HGNetv2={"pretrained": False},
    )
    model = config.model.to(device)
    return config, model


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "axmamba": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if "axmamba_blocks" in name
        ),
    }


def make_targets(
    batch_size: int, device: torch.device, num_classes: int = 9
) -> List[Dict[str, torch.Tensor]]:
    targets = []
    for index in range(batch_size):
        label = index % max(num_classes, 1)
        targets.append(
            {
                "labels": torch.tensor([label], dtype=torch.long, device=device),
                "boxes": torch.tensor(
                    [[0.5, 0.5, 0.2, 0.2]],
                    dtype=torch.float32,
                    device=device,
                ),
            }
        )
    return targets


def tensor_leaves(value: Any, prefix: str = "output") -> List[Tuple[str, torch.Tensor]]:
    leaves = []
    if isinstance(value, torch.Tensor):
        leaves.append((prefix, value))
    elif isinstance(value, Mapping):
        for key in sorted(value.keys()):
            leaves.extend(tensor_leaves(value[key], "{}.{}".format(prefix, key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            leaves.extend(tensor_leaves(item, "{}[{}]".format(prefix, index)))
    return leaves


def structure_signature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        return {
            "type": "dict",
            "items": {
                str(key): structure_signature(value[key]) for key in sorted(value.keys())
            },
        }
    if isinstance(value, list):
        return {"type": "list", "items": [structure_signature(item) for item in value]}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [structure_signature(item) for item in value]}
    if value is None:
        return {"type": "none"}
    return {"type": type(value).__name__}


def assert_finite(value: Any) -> None:
    leaves = tensor_leaves(value)
    if not leaves:
        raise RuntimeError("model output contains no tensors")
    non_finite = [name for name, tensor in leaves if not torch.isfinite(tensor).all()]
    if non_finite:
        raise RuntimeError("non-finite model outputs: {}".format(non_finite))


def scalar_loss(value: Any) -> torch.Tensor:
    floating = [
        tensor.float().mean()
        for _, tensor in tensor_leaves(value)
        if tensor.is_floating_point()
    ]
    if not floating:
        raise RuntimeError("model output contains no floating-point tensors")
    return torch.stack(floating).mean()


def debug_state_to_json(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        detached = value.detach().float().cpu()
        return detached.item() if detached.numel() == 1 else detached.tolist()
    if isinstance(value, Mapping):
        return {str(key): debug_state_to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [debug_state_to_json(item) for item in value]
    return value


def save_json(path: str, payload: Any) -> Path:
    output_path = resolve_repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path


def extract_state_dict(checkpoint: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    """Extract a model state dict and report the selected checkpoint field."""
    selected = checkpoint
    source = "root"
    if isinstance(checkpoint, Mapping):
        if isinstance(checkpoint.get("ema"), Mapping):
            ema = checkpoint["ema"]
            if isinstance(ema.get("module"), Mapping):
                selected = ema["module"]
                source = "ema.module"
        if selected is checkpoint and isinstance(checkpoint.get("model"), Mapping):
            selected = checkpoint["model"]
            source = "model"
        if selected is checkpoint and isinstance(checkpoint.get("state_dict"), Mapping):
            selected = checkpoint["state_dict"]
            source = "state_dict"

    if not isinstance(selected, Mapping) or not all(
        isinstance(value, torch.Tensor) for value in selected.values()
    ):
        raise ValueError("could not find a tensor model state_dict in checkpoint")

    state = dict(selected)
    if state and all(name.startswith("module.") for name in state):
        state = {name[len("module.") :]: value for name, value in state.items()}
        source += " (uniform module. prefix removed)"
    return state, source


def load_checkpoint_state(path: str) -> Tuple[Dict[str, torch.Tensor], str]:
    checkpoint_path = resolve_repo_path(path)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    return extract_state_dict(checkpoint)
