"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
AxMamba V1 uses the public ``mamba_ssm.Mamba`` interface. It is an original
axial/local fusion implementation and does not copy third-party selective-scan
kernels or VMamba source code.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

try:
    from typing import Literal
except ImportError:  # Python 3.7 can still parse postponed annotations.
    Literal = str

import torch
import torch.nn as nn


try:
    from mamba_ssm import Mamba

    _MAMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the active environment.
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc


__all__ = [
    "DropPath",
    "LocalDetailBranch",
    "BidirectionalAxisMamba",
    "AxMambaBlock",
]


_MAMBA_UNAVAILABLE_MESSAGE = (
    "AxMamba is enabled but mamba_ssm is unavailable. Install a compatible "
    "mamba-ssm build without changing the existing PyTorch/CUDA stack."
)


def _validate_feature_map(x: torch.Tensor, dim: int, module_name: str) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError("{} expects a torch.Tensor input".format(module_name))
    if x.ndim != 4:
        raise ValueError(
            "{} expects [B, C, H, W], got {}".format(module_name, tuple(x.shape))
        )
    if x.shape[1] != dim:
        raise ValueError(
            "{} expects {} channels, got {}".format(module_name, dim, x.shape[1])
        )


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("{} must be a positive integer".format(name))


class DropPath(nn.Module):
    """Per-sample stochastic depth with shape-preserving output.

    Input/output: arbitrary matching tensor shapes, normally ``[B, C, H, W]``.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not isinstance(drop_prob, (float, int)):
            raise TypeError("drop_prob must be a number")
        if not 0.0 <= float(drop_prob) < 1.0:
            raise ValueError("drop_prob must be in [0, 1)")
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth without changing shape, dtype, or device."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("DropPath expects a torch.Tensor input")
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(mask_shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class LocalDetailBranch(nn.Module):
    """Parallel 3x3/5x5 depthwise local-detail extraction.

    Input/output: ``[B, C, H, W] -> [B, C, H, W]``.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        _validate_positive_int("dim", dim)
        self.dim = dim
        self.dw3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.dw5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.SiLU(inplace=True)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.out_norm = nn.BatchNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return local features with the same shape and layout as ``x``."""
        _validate_feature_map(x, self.dim, self.__class__.__name__)
        features = self.dw3(x) + self.dw5(x)
        features = self.act(self.norm(features))
        return self.out_norm(self.pw(features))


class BidirectionalAxisMamba(nn.Module):
    """Shared-parameter forward/reverse Mamba scan on one spatial axis.

    Input/output: ``[B, C, H, W] -> [B, C, H, W]``. Horizontal mode creates
    ``B*H`` sequences of length ``W``; vertical mode creates ``B*W`` sequences
    of length ``H``. The two-dimensional plane is never flattened into one
    ``H*W`` sequence.
    """

    def __init__(
        self,
        dim: int,
        axis: Literal["horizontal", "vertical"],
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        share_reverse: bool = True,
    ) -> None:
        super().__init__()
        _validate_positive_int("dim", dim)
        _validate_positive_int("d_state", d_state)
        _validate_positive_int("d_conv", d_conv)
        _validate_positive_int("expand", expand)
        if axis not in ("horizontal", "vertical"):
            raise ValueError("axis must be 'horizontal' or 'vertical'")
        if share_reverse is not True:
            raise ValueError("AxMamba V1 requires share_reverse=True")
        if Mamba is None:
            raise ImportError(_MAMBA_UNAVAILABLE_MESSAGE) from _MAMBA_IMPORT_ERROR

        self.dim = dim
        self.axis = axis
        self.share_reverse = share_reverse
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scan both directions along the configured spatial axis."""
        _validate_feature_map(x, self.dim, self.__class__.__name__)
        batch, channels, height, width = x.shape
        input_dtype = x.dtype

        if self.axis == "horizontal":
            seq = (
                x.permute(0, 2, 3, 1)
                .contiguous()
                .reshape(batch * height, width, channels)
            )
        else:
            seq = (
                x.permute(0, 3, 2, 1)
                .contiguous()
                .reshape(batch * width, height, channels)
            )

        seq = self.norm(seq)
        forward_features = self.mamba(seq)
        reverse_features = torch.flip(
            self.mamba(torch.flip(seq, dims=[1])), dims=[1]
        )
        features = 0.5 * (forward_features + reverse_features)
        if features.dtype != input_dtype:
            features = features.to(dtype=input_dtype)

        if self.axis == "horizontal":
            return (
                features.reshape(batch, height, width, channels)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
        return (
            features.reshape(batch, width, height, channels)
            .permute(0, 3, 2, 1)
            .contiguous()
        )


class AxMambaBlock(nn.Module):
    """Axial/local fusion followed by a small LayerScale residual update.

    Input/output: ``[B, C, H, W] -> [B, C, H, W]``.

    ``local`` does not instantiate Mamba. ``axis_mean`` uses a fixed mean and
    ``axis_gate`` uses a learned per-pixel three-way softmax gate.
    """

    VALID_MODES = ("local", "axis_mean", "axis_gate")

    def __init__(
        self,
        dim: int,
        mode: Literal["local", "axis_mean", "axis_gate"],
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        gate_reduction: int = 4,
        gamma_init: float = 1e-3,
        drop_path: float = 0.0,
        collect_debug: bool = False,
    ) -> None:
        super().__init__()
        _validate_positive_int("dim", dim)
        _validate_positive_int("d_state", d_state)
        _validate_positive_int("d_conv", d_conv)
        _validate_positive_int("expand", expand)
        _validate_positive_int("gate_reduction", gate_reduction)
        if mode not in self.VALID_MODES:
            raise ValueError(
                "mode must be one of {}, got {!r}".format(self.VALID_MODES, mode)
            )
        if not isinstance(gamma_init, (float, int)) or float(gamma_init) < 0.0:
            raise ValueError("gamma_init must be non-negative")
        if not isinstance(collect_debug, bool):
            raise TypeError("collect_debug must be a bool")

        self.dim = dim
        self.mode = mode
        self.collect_debug = collect_debug
        self.local_branch = LocalDetailBranch(dim)

        if mode in ("axis_mean", "axis_gate"):
            self.mamba_h = BidirectionalAxisMamba(
                dim, "horizontal", d_state, d_conv, expand, share_reverse=True
            )
            self.mamba_v = BidirectionalAxisMamba(
                dim, "vertical", d_state, d_conv, expand, share_reverse=True
            )

        if mode == "axis_gate":
            hidden = max(dim // gate_reduction, 16)
            self.gate = nn.Sequential(
                OrderedDict(
                    [
                        ("conv_in", nn.Conv2d(2 * dim, hidden, 1, bias=False)),
                        ("norm", nn.BatchNorm2d(hidden)),
                        ("act", nn.SiLU(inplace=True)),
                        ("conv_out", nn.Conv2d(hidden, 3, 1, bias=True)),
                    ]
                )
            )

        self.gamma = nn.Parameter(
            torch.full((1, dim, 1, 1), float(gamma_init))
        )
        self.out_proj = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(dim, dim, 1, bias=False)),
                    ("norm", nn.BatchNorm2d(dim)),
                ]
            )
        )
        self.drop_path = DropPath(float(drop_path))
        self.debug_state: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _rms_norm(x: torch.Tensor) -> torch.Tensor:
        return x.detach().float().norm() / (x.numel() ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured fusion and preserve the input tensor shape."""
        _validate_feature_map(x, self.dim, self.__class__.__name__)
        local_features = self.local_branch(x)
        horizontal_features = None
        vertical_features = None

        if self.mode == "local":
            fused = local_features
            gate_mean = x.new_tensor([0.0, 0.0, 1.0])
        else:
            horizontal_features = self.mamba_h(x)
            vertical_features = self.mamba_v(x)
            if self.mode == "axis_mean":
                fused = (
                    horizontal_features + vertical_features + local_features
                ) / 3.0
                gate_mean = x.new_full((3,), 1.0 / 3.0)
            else:
                logits = self.gate(torch.cat([x, local_features], dim=1))
                weights = torch.softmax(logits, dim=1)
                fused = (
                    weights[:, 0:1] * horizontal_features
                    + weights[:, 1:2] * vertical_features
                    + weights[:, 2:3] * local_features
                )
                gate_mean = weights.detach().mean(dim=(0, 2, 3))

        projected = self.out_proj(fused)
        gamma = self.gamma
        if gamma.dtype != projected.dtype:
            gamma = gamma.to(dtype=projected.dtype)
        out = x + self.drop_path(gamma * projected)
        if out.dtype != x.dtype:
            out = out.to(dtype=x.dtype)

        if self.collect_debug:
            zero = x.detach().new_zeros(())
            self.debug_state = {
                "gate_mean": gate_mean.detach(),
                "gamma_mean": self.gamma.detach().mean(),
                "horizontal_norm": zero
                if horizontal_features is None
                else self._rms_norm(horizontal_features),
                "vertical_norm": zero
                if vertical_features is None
                else self._rms_norm(vertical_features),
                "local_norm": self._rms_norm(local_features),
            }

        return out

    def get_debug_state(self) -> Dict[str, torch.Tensor]:
        """Return detached tensors collected by the latest debug forward."""
        return dict(self.debug_state)
