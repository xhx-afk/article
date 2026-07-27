"""LABS-Mamba V3 local anchor and bounded final-P4 sidecar.

The local anchor remains on the first top-down fused P4.  The Mamba adapter is
a separate, single-path sidecar intended exclusively for the final PAN P4.
This module deliberately contains no semantic input, content gate, or dual-path
residual fusion.
"""

from __future__ import annotations

import math
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from mamba_ssm import Mamba

    _MAMBA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the server environment.
    Mamba = None
    _MAMBA_IMPORT_ERROR = exc


__all__ = [
    "DropPath",
    "LayerNorm2d",
    "LocalAnchorBlock",
    "BidirectionalAxisMamba",
    "BoundedMambaSidecar",
]


_MAMBA_UNAVAILABLE_MESSAGE = (
    "LABS-Mamba sidecar is enabled, but mamba_ssm is unavailable. Install a "
    "mamba_ssm build compatible with the existing PyTorch/CUDA environment. "
    "The S0 local-only configuration does not require mamba_ssm."
)


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("{} must be a positive integer".format(name))


def _validate_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError("{} must be a bool".format(name))


def _validate_feature_map(x: torch.Tensor, dim: int, module_name: str) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError("{} expects a torch.Tensor".format(module_name))
    if x.ndim != 4:
        raise ValueError(
            "{} expects [B, C, H, W], got {}".format(module_name, tuple(x.shape))
        )
    if x.shape[1] != dim:
        raise ValueError(
            "{} expects {} channels, got {}".format(module_name, dim, x.shape[1])
        )


def _group_count(channels: int, maximum: int = 32) -> int:
    """Return the largest conventional GroupNorm group count that divides C."""
    _validate_positive_int("channels", channels)
    for groups in (32, 16, 8, 4, 2, 1):
        if groups <= maximum and groups <= channels and channels % groups == 0:
            return groups
    return 1


class _SafeGroupNorm(nn.GroupNorm):
    """GroupNorm with a defined fallback for ``[1, C, 1, 1]`` inputs."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        values_per_group = input.numel() // self.num_groups
        if values_per_group > 1:
            return super().forward(input)
        if input.shape[1] > 1:
            return F.group_norm(input, 1, self.weight, self.bias, self.eps)
        weight = self.weight.reshape(1, 1, 1, 1)
        bias = self.bias.reshape(1, 1, 1, 1)
        return input * weight + bias


class DropPath(nn.Module):
    """Per-sample stochastic depth that preserves shape, dtype, and device."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not isinstance(drop_prob, (float, int)) or isinstance(drop_prob, bool):
            raise TypeError("drop_prob must be a number")
        if not 0.0 <= float(drop_prob) < 1.0:
            raise ValueError("drop_prob must be in [0, 1)")
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("DropPath expects a torch.Tensor")
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(mask_shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for ``[B, C, H, W]`` feature maps."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        _validate_positive_int("dim", dim)
        if not isinstance(eps, (float, int)) or float(eps) <= 0.0:
            raise ValueError("eps must be positive")
        self.dim = dim
        self.norm = nn.LayerNorm(dim, eps=float(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_feature_map(x, self.dim, self.__class__.__name__)
        return (
            self.norm(x.permute(0, 2, 3, 1))
            .permute(0, 3, 1, 2)
            .contiguous()
        )


class _LocalAnchorBranch(nn.Module):
    """Pre-normalized parallel 3x3/5x5 depthwise local features."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.pre_norm = LayerNorm2d(dim)
        self.dw3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.dw5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_feature_map(x, self.dim, self.__class__.__name__)
        normalized = self.pre_norm(x)
        return self.act(self.dw3(normalized) + self.dw5(normalized))


class LocalAnchorBlock(nn.Module):
    """Local multi-kernel residual anchor used on the first fused top-down P4."""

    def __init__(
        self,
        dim: int = 256,
        gamma_init: float = 0.001,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_positive_int("dim", dim)
        if not isinstance(gamma_init, (float, int)) or isinstance(gamma_init, bool):
            raise TypeError("gamma_init must be a number")
        if float(gamma_init) < 0.0:
            raise ValueError("gamma_init must be non-negative")
        self.dim = dim
        self.local_branch = _LocalAnchorBranch(dim)
        self.projector = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            _SafeGroupNorm(_group_count(dim), dim),
        )
        self.gamma_local = nn.Parameter(
            torch.full((1, dim, 1, 1), float(gamma_init))
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_feature_map(x, self.dim, self.__class__.__name__)
        local = self.projector(self.local_branch(x))
        update = self.gamma_local.to(dtype=local.dtype) * local
        output = x + self.drop_path(update)
        return output if output.dtype == x.dtype else output.to(dtype=x.dtype)


class BidirectionalAxisMamba(nn.Module):
    """Vectorized forward/reverse Mamba scan along one spatial axis."""

    def __init__(
        self,
        dim: int,
        axis: str,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        _validate_positive_int("dim", dim)
        _validate_positive_int("d_state", d_state)
        _validate_positive_int("d_conv", d_conv)
        _validate_positive_int("expand", expand)
        if axis not in ("horizontal", "vertical"):
            raise ValueError("axis must be 'horizontal' or 'vertical'")
        if Mamba is None:
            raise ImportError(_MAMBA_UNAVAILABLE_MESSAGE) from _MAMBA_IMPORT_ERROR
        self.dim = dim
        self.axis = axis
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_feature_map(x, self.dim, self.__class__.__name__)
        batch, channels, height, width = x.shape
        input_dtype = x.dtype
        if self.axis == "horizontal":
            sequence = (
                x.permute(0, 2, 3, 1)
                .contiguous()
                .reshape(batch * height, width, channels)
            )
        else:
            sequence = (
                x.permute(0, 3, 2, 1)
                .contiguous()
                .reshape(batch * width, height, channels)
            )

        sequence = self.norm(sequence)
        forward_features = self.mamba(sequence)
        reverse_features = torch.flip(
            self.mamba(torch.flip(sequence, dims=[1])), dims=[1]
        )
        merged = 0.5 * (forward_features + reverse_features)

        if self.axis == "horizontal":
            output = merged.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        else:
            output = merged.reshape(batch, width, height, channels).permute(0, 3, 2, 1)
        output = output.contiguous()
        return output if output.dtype == input_dtype else output.to(dtype=input_dtype)


class BoundedMambaSidecar(nn.Module):
    """Low-dimensional, single-path, bounded Mamba residual for final PAN P4."""

    def __init__(
        self,
        dim: int = 256,
        bottleneck_dim: int = 64,
        path: str = "hv",
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        beta_init: float = 0.005,
        beta_max: float = 0.05,
        center_residual: bool = True,
        rms_align: bool = True,
        rms_eps: float = 1e-6,
        max_align_scale: float = 10.0,
        drop_path: float = 0.0,
        collect_debug: bool = False,
    ) -> None:
        super().__init__()
        _validate_positive_int("dim", dim)
        _validate_positive_int("bottleneck_dim", bottleneck_dim)
        _validate_positive_int("d_state", d_state)
        _validate_positive_int("d_conv", d_conv)
        _validate_positive_int("expand", expand)
        if path not in ("hv", "vh"):
            raise ValueError("path must be 'hv' or 'vh'")
        for name, value in (
            ("center_residual", center_residual),
            ("rms_align", rms_align),
            ("collect_debug", collect_debug),
        ):
            _validate_bool(name, value)
        for name, value in (
            ("beta_init", beta_init),
            ("beta_max", beta_max),
            ("rms_eps", rms_eps),
            ("max_align_scale", max_align_scale),
        ):
            if not isinstance(value, (float, int)) or isinstance(value, bool):
                raise TypeError("{} must be a number".format(name))
        if not 0.0 < float(beta_init) < float(beta_max) <= 0.2:
            raise ValueError("beta must satisfy 0 < beta_init < beta_max <= 0.2")
        if float(rms_eps) <= 0.0:
            raise ValueError("rms_eps must be positive")
        if float(max_align_scale) < 1.0:
            raise ValueError("max_align_scale must be at least 1")

        self.dim = dim
        self.bottleneck_dim = bottleneck_dim
        self.path = path
        self.center_residual = center_residual
        self.rms_align = rms_align
        self.rms_eps = float(rms_eps)
        self.max_align_scale = float(max_align_scale)
        self.beta_max = float(beta_max)
        self.collect_debug = collect_debug

        groups = _group_count(bottleneck_dim)
        self.pre_norm = LayerNorm2d(dim)
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, bottleneck_dim, 1, bias=False),
            _SafeGroupNorm(groups, bottleneck_dim),
            nn.SiLU(inplace=False),
        )
        self.horizontal = BidirectionalAxisMamba(
            bottleneck_dim,
            "horizontal",
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.vertical = BidirectionalAxisMamba(
            bottleneck_dim,
            "vertical",
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.expand = nn.Sequential(
            nn.Conv2d(bottleneck_dim, dim, 1, bias=False),
            _SafeGroupNorm(_group_count(dim), dim),
        )

        raw_init = math.atanh(float(beta_init) / self.beta_max)
        self.beta_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        self.drop_path = DropPath(drop_path)
        self.debug_state = {}  # type: Dict[str, Union[str, torch.Tensor]]

    @staticmethod
    def _sample_rms(value: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
        squared_mean = value.float().pow(2).mean(dim=(1, 2, 3), keepdim=True)
        return torch.sqrt(squared_mean + eps)

    def effective_beta(self) -> torch.Tensor:
        """Return the differentiable, globally scalar bounded beta."""
        return self.beta_max * torch.tanh(self.beta_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _validate_feature_map(x, self.dim, self.__class__.__name__)
        reduced = self.reduce(self.pre_norm(x))
        if self.path == "hv":
            candidate = self.vertical(self.horizontal(reduced))
        else:
            candidate = self.horizontal(self.vertical(reduced))
        candidate = self.expand(candidate)

        candidate_mean_before = candidate.float().mean(dim=(2, 3), keepdim=True)
        if self.center_residual:
            candidate = candidate - candidate.mean(dim=(2, 3), keepdim=True)
        candidate_mean_after = candidate.float().mean(dim=(2, 3), keepdim=True)

        input_rms = self._sample_rms(x, self.rms_eps)
        candidate_raw_rms = self._sample_rms(candidate, self.rms_eps)
        if self.rms_align:
            align_scale = (
                input_rms.detach() / candidate_raw_rms.detach()
            ).clamp(max=self.max_align_scale)
            candidate = candidate * align_scale.to(dtype=candidate.dtype)
        candidate_aligned_rms = self._sample_rms(candidate, self.rms_eps)

        beta = self.effective_beta()
        update = beta.to(dtype=candidate.dtype) * candidate
        output = x + self.drop_path(update)
        if output.dtype != x.dtype:
            output = output.to(dtype=x.dtype)

        if self.collect_debug:
            update_rms = self._sample_rms(update)
            self.debug_state = {
                "path": self.path,
                "beta_effective": beta.detach(),
                "input_rms": input_rms.detach().reshape(-1),
                "candidate_raw_rms": candidate_raw_rms.detach().reshape(-1),
                "candidate_aligned_rms": candidate_aligned_rms.detach().reshape(-1),
                "update_rms": update_rms.detach().reshape(-1),
                "update_input_ratio": (
                    update_rms / (input_rms + self.rms_eps)
                ).detach().reshape(-1),
                "input_spatial_mean_abs": x.detach().float().mean(
                    dim=(2, 3)
                ).abs().mean(),
                "input_spatial_mean": x.detach().float().mean(),
                "candidate_spatial_mean_abs_before_center": candidate_mean_before.detach().abs().mean(),
                "candidate_spatial_mean_abs_after_center": candidate_mean_after.detach().abs().mean(),
                "input_std": x.detach().float().std(unbiased=False),
                "output_std": output.detach().float().std(unbiased=False),
                "output_spatial_mean": output.detach().float().mean(),
                "feature_delta_mean_abs": (output.detach().float() - x.detach().float()).abs().mean(),
            }
        return output

    def get_debug_state(self) -> Dict[str, Union[str, torch.Tensor]]:
        """Return detached tensors from the latest debug-enabled forward."""
        state = {}  # type: Dict[str, Union[str, torch.Tensor]]
        for name, value in self.debug_state.items():
            state[name] = value.detach() if isinstance(value, torch.Tensor) else value
        return state
