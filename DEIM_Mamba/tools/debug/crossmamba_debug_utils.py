"""Shared helpers for CrossMamba V2 validation and profiling scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


CROSSMAMBA_PREFIX = "encoder.crossmamba_p4."


def build_config(config_path: str, **overrides):
    """Build a YAMLConfig without downloading backbone pretrained weights."""
    from engine.core import YAMLConfig

    cfg = YAMLConfig(config_path, **overrides)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    return cfg


def extract_checkpoint_state(checkpoint) -> Mapping[str, torch.Tensor]:
    """Extract model weights from DEIM model/EMA or a raw state dictionary."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    state = None
    if "ema" in checkpoint:
        ema = checkpoint["ema"]
        if isinstance(ema, Mapping) and "module" in ema:
            state = ema["module"]
    if "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
        state = checkpoint["model"]
    if state is None and all(isinstance(key, str) for key in checkpoint):
        state = checkpoint
    if state is None or not isinstance(state, Mapping):
        raise ValueError("cannot locate model/ema weights in checkpoint")
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def load_checkpoint(path: str) -> Mapping[str, torch.Tensor]:
    """Load a checkpoint on CPU and return its model state."""
    return extract_checkpoint_state(torch.load(path, map_location="cpu"))


def audit_checkpoint_load(model, state: Mapping[str, torch.Tensor]) -> Dict[str, object]:
    """Load non-strictly and verify that only CrossMamba V2 keys are missing."""
    incompatible = model.load_state_dict(state, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    illegal_missing = [key for key in missing if not key.startswith(CROSSMAMBA_PREFIX)]
    report = {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "allowed_crossmamba_missing_keys": [
            key for key in missing if key.startswith(CROSSMAMBA_PREFIX)
        ],
        "illegal_missing_keys": illegal_missing,
        "compatible": not illegal_missing and not unexpected,
    }
    return report


def parameter_counts(model) -> Dict[str, int]:
    """Return total/trainable/CrossMamba parameter counts."""
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "crossmamba": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith(CROSSMAMBA_PREFIX)
        ),
    }


def tensor_tree_signature(value) -> object:
    """Describe nested model output structure without serializing tensor data."""
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "finite": bool(torch.isfinite(value.detach()).all().item()),
        }
    if isinstance(value, Mapping):
        return {key: tensor_tree_signature(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [tensor_tree_signature(item) for item in value]
    return type(value).__name__


def floating_tensors(value) -> List[torch.Tensor]:
    """Collect floating-point tensors from a nested model output."""
    tensors: List[torch.Tensor] = []
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            tensors.append(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            tensors.extend(floating_tensors(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(floating_tensors(item))
    return tensors


def output_surrogate_loss(outputs) -> torch.Tensor:
    """Create a finite differentiable scalar from a nested model output."""
    tensors = [tensor for tensor in floating_tensors(outputs) if tensor.requires_grad]
    if not tensors:
        raise RuntimeError("model output contains no differentiable floating tensor")
    return sum(tensor.float().square().mean() for tensor in tensors)


def gradients_report(model, prefix: str = CROSSMAMBA_PREFIX) -> Dict[str, object]:
    """Report missing/non-finite gradients for trainable prefixed parameters."""
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith(prefix) and parameter.requires_grad
    }
    missing = sorted(name for name, parameter in parameters.items() if parameter.grad is None)
    non_finite = sorted(
        name
        for name, parameter in parameters.items()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    )
    norms = {
        name: float(parameter.grad.detach().float().norm().item())
        for name, parameter in parameters.items()
        if parameter.grad is not None and torch.isfinite(parameter.grad).all()
    }
    return {
        "parameter_count": len(parameters),
        "missing": missing,
        "non_finite": non_finite,
        "grad_norms": norms,
        "valid": bool(parameters) and not missing and not non_finite,
    }


def debug_state_to_json(state) -> object:
    """Convert detached scalar/tiny debug tensors into JSON-compatible data."""
    if isinstance(state, torch.Tensor):
        value = state.detach().cpu()
        return float(value.item()) if value.numel() == 1 else value.tolist()
    if isinstance(state, Mapping):
        return {key: debug_state_to_json(value) for key, value in state.items()}
    return state


def write_json(path: str, payload: object) -> None:
    """Write UTF-8 indented JSON and create the parent directory."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def synchronize(device: torch.device) -> None:
    """Synchronize only for CUDA devices."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
