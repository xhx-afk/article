"""Shared LABS-Mamba V3 checkpoint, model, and smoke-test helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LOCAL_PREFIX = "encoder.local_anchor_p4."
SIDECAR_PREFIX = "encoder.mamba_sidecar_p4."
LABS_PREFIXES = (LOCAL_PREFIX, SIDECAR_PREFIX)


def build_config(config_path: str, **overrides):
    """Build YAMLConfig without downloading backbone pretrained weights."""
    from engine.core import YAMLConfig

    cfg = YAMLConfig(config_path, **overrides)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    return cfg


def extract_checkpoint_state(checkpoint) -> Mapping[str, torch.Tensor]:
    """Match train.py -t semantics: prefer EMA, then model, then raw weights."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a mapping")
    state = None
    ema = checkpoint.get("ema")
    if isinstance(ema, Mapping) and isinstance(ema.get("module"), Mapping):
        state = ema["module"]
    elif isinstance(checkpoint.get("model"), Mapping):
        state = checkpoint["model"]
    elif checkpoint and all(isinstance(key, str) for key in checkpoint):
        state = checkpoint
    if state is None:
        raise ValueError("cannot locate EMA/model/raw model weights in checkpoint")
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state.items()
    }


def load_checkpoint(path: str) -> Mapping[str, torch.Tensor]:
    return extract_checkpoint_state(torch.load(path, map_location="cpu"))


def audit_checkpoint_load(
    model,
    state: Mapping[str, torch.Tensor],
    require_all_sidecar_missing: bool = False,
) -> Dict[str, object]:
    """Load shape-compatible keys and fully audit the non-strict result."""
    model_state = model.state_dict()
    unexpected = sorted(key for key in state if key not in model_state)
    shape_mismatches = sorted(
        [
        {
            "key": key,
            "checkpoint_shape": list(value.shape),
            "model_shape": list(model_state[key].shape),
        }
        for key, value in state.items()
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape)
        ],
        key=lambda row: row["key"],
    )
    mismatch_names = {row["key"] for row in shape_mismatches}
    compatible_state = {
        key: value
        for key, value in state.items()
        if key in model_state and key not in mismatch_names
    }
    incompatible = model.load_state_dict(compatible_state, strict=False)
    missing = sorted(set(incompatible.missing_keys) | mismatch_names)
    sidecar_model_keys = sorted(
        key for key in model_state if key.startswith(SIDECAR_PREFIX)
    )
    allowed_missing = [key for key in missing if key.startswith(SIDECAR_PREFIX)]
    illegal_missing = [key for key in missing if not key.startswith(SIDECAR_PREFIX)]
    all_sidecar_missing = bool(sidecar_model_keys) and allowed_missing == sidecar_model_keys
    compatible = not illegal_missing and not unexpected and not shape_mismatches
    if require_all_sidecar_missing:
        compatible = compatible and all_sidecar_missing
    return {
        "model_key_count": len(model_state),
        "checkpoint_key_count": len(state),
        "loaded_key_count": len(compatible_state),
        "missing_keys": missing,
        "allowed_sidecar_missing_keys": allowed_missing,
        "illegal_missing_keys": illegal_missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "sidecar_model_keys": sidecar_model_keys,
        "all_sidecar_keys_missing": all_sidecar_missing,
        "require_all_sidecar_missing": bool(require_all_sidecar_missing),
        "compatible": compatible,
    }


def parameter_counts(model) -> Dict[str, int]:
    named = list(model.named_parameters())
    return {
        "total": sum(parameter.numel() for _, parameter in named),
        "trainable": sum(
            parameter.numel() for _, parameter in named if parameter.requires_grad
        ),
        "local_anchor": sum(
            parameter.numel() for name, parameter in named if name.startswith(LOCAL_PREFIX)
        ),
        "mamba_sidecar": sum(
            parameter.numel() for name, parameter in named if name.startswith(SIDECAR_PREFIX)
        ),
    }


def tensor_tree_signature(value) -> object:
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
    tensors = []  # type: List[torch.Tensor]
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        tensors.append(value)
    elif isinstance(value, Mapping):
        for item in value.values():
            tensors.extend(floating_tensors(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(floating_tensors(item))
    return tensors


def output_surrogate_loss(outputs) -> torch.Tensor:
    tensors = [tensor for tensor in floating_tensors(outputs) if tensor.requires_grad]
    if not tensors:
        raise RuntimeError("model output contains no differentiable floating tensor")
    return sum(tensor.float().square().mean() for tensor in tensors)


def gradients_report(
    model, prefixes: Sequence[str] = LABS_PREFIXES
) -> Dict[str, object]:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith(tuple(prefixes)) and parameter.requires_grad
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
    if isinstance(state, torch.Tensor):
        value = state.detach().cpu()
        return float(value.item()) if value.numel() == 1 else value.tolist()
    if isinstance(state, Mapping):
        return {key: debug_state_to_json(value) for key, value in state.items()}
    return state


def write_json(path: str, payload: object) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
