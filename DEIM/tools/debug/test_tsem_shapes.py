"""
Lightweight TSEM shape, identity, AMP and backward checks.

Run:
    python tools/debug/test_tsem_shapes.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import Iterable, List

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

_TSEM_PATH = os.path.join(ROOT, "engine", "deim", "tsem.py")
_SPEC = importlib.util.spec_from_file_location("tsem_module", _TSEM_PATH)
_TSEM_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_TSEM_MODULE)
TextureScaleEnhancementModule = _TSEM_MODULE.TextureScaleEnhancementModule


MODES = ["high_only", "context_only", "dual_sum", "dual_gate", "full"]
LEVEL_SETS = [(0,), (0, 1), (0, 1, 2), (1, 2)]


def _make_feats(device: torch.device, dtype: torch.dtype, batch_size: int = 2) -> List[torch.Tensor]:
    shapes = [(batch_size, 256, 120, 120), (batch_size, 256, 60, 60), (batch_size, 256, 30, 30)]
    return [torch.randn(shape, device=device, dtype=dtype, requires_grad=True) for shape in shapes]


def _assert_same(outputs: Iterable[torch.Tensor], inputs: Iterable[torch.Tensor], identity_tol: float) -> None:
    for out, inp in zip(outputs, inputs):
        assert out.shape == inp.shape, (out.shape, inp.shape)
        assert out.dtype == inp.dtype, (out.dtype, inp.dtype)
        assert out.device == inp.device, (out.device, inp.device)
        assert torch.isfinite(out).all(), "output contains NaN or Inf"
        max_error = (out.detach().float() - inp.detach().float()).abs().max().item()
        assert max_error < identity_tol, f"gamma=0 identity check failed: {max_error}"


def _check_backward(module: torch.nn.Module, outputs: List[torch.Tensor]) -> None:
    loss = sum(out.float().mean() for out in outputs)
    loss.backward()
    missing = []
    bad = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing.append(name)
        elif not torch.isfinite(param.grad).all():
            bad.append(name)
    assert not missing, f"parameters without grad: {missing[:20]}"
    assert not bad, f"parameters with non-finite grad: {bad[:20]}"


def run_once(device: torch.device, dtype: torch.dtype, use_autocast: bool = False) -> None:
    identity_tol = 1e-6 if dtype == torch.float32 else 1e-3
    for mode in MODES:
        for levels in LEVEL_SETS:
            module = TextureScaleEnhancementModule(
                channels=256,
                num_levels=3,
                mode=mode,
                levels=levels,
                smooth_kernel=5,
                dilations=(1, 2, 3),
                reduction=16,
                init_scale=0.0,
                learnable_smoothing=True,
                debug=True,
            ).to(device=device)
            feats = _make_feats(device, dtype)
            if dtype != torch.float32:
                module = module.to(dtype=dtype)
            if use_autocast:
                with torch.cuda.amp.autocast():
                    outputs = module(feats)
            else:
                outputs = module(feats)
            _assert_same(outputs, feats, identity_tol)
            _check_backward(module, outputs)
            print(f"PASS mode={mode} levels={levels} device={device} dtype={dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TSEM shape/backward test")
    parser.add_argument("--skip-cuda", action="store_true")
    args = parser.parse_args()

    run_once(torch.device("cpu"), torch.float32)
    if torch.cuda.is_available() and not args.skip_cuda:
        run_once(torch.device("cuda"), torch.float16, use_autocast=True)


if __name__ == "__main__":
    main()
