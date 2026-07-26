"""Post-FPN local-anchored CrossMamba V2 building blocks.

CrossMamba V2 applies shared-parameter H->V and V->H bidirectional axial
Mamba paths to a fused P4 feature.  It has no semantic-map input and keeps the
local multi-kernel branch as an independent residual update.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict

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
    "LocalAnchorBranch",
    "BidirectionalAxisMamba",
    "BranchProjector",
    "ContentAwareResidualGate",
    "CrossMambaBlock",
]


_MAMBA_UNAVAILABLE_MESSAGE = (
    "CrossMamba is enabled with an axial path, but mamba_ssm is unavailable. "
    "Install a build compatible with the existing PyTorch/CUDA environment."
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
    _validate_positive_int("maximum", maximum)
    for groups in (32, 16, 8, 4, 2, 1):
        if groups <= maximum and groups <= channels and channels % groups == 0:
            return groups
    return 1


class SafeGroupNorm(nn.GroupNorm):
    """GroupNorm that remains defined for a single-sample singleton map.

    PyTorch rejects a training-time GroupNorm call when each group contains
    exactly one value (for example ``[1, C, 1, 1]`` with ``C`` groups).
    CrossMamba must support such feature maps, so the fallback normalizes over
    one group; for the degenerate one-channel case it applies the affine
    parameters without attempting a zero-variance normalization.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        values_per_group = input.numel() // self.num_groups
        if values_per_group > 1:
            return super().forward(input)
        if input.shape[1] > 1:
            return F.group_norm(
                input,
                1,
                self.weight,
                self.bias,
                self.eps,
            )
        weight = self.weight.reshape(1, 1, 1, 1)
        bias = self.bias.reshape(1, 1, 1, 1)
        return input * weight + bias


class DropPath(nn.Module):
    """Per-sample stochastic depth.

    Input/output have identical arbitrary shapes, normally ``[B, C, H, W]``.
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
            raise TypeError("DropPath expects a torch.Tensor")
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
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
        """Normalize channels and preserve ``[B, C, H, W]``."""
        _validate_feature_map(x, self.dim, self.__class__.__name__)
        return (
            self.norm(x.permute(0, 2, 3, 1))
            .permute(0, 3, 1, 2)
            .contiguous()
        )


class LocalAnchorBranch(nn.Module):
    """Pre-normalized parallel 3x3/5x5 depthwise local anchor.

    Shape: ``[B, C, H, W] -> [B, C, H, W]``.  Projection and GroupNorm are
    intentionally performed by the branch-specific :class:`BranchProjector`.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        _validate_positive_int("dim", dim)
        self.dim = dim
        self.pre_norm = LayerNorm2d(dim)
        self.dw3 = nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False)
        self.dw5 = nn.Conv2d(dim, dim, 5, stride=1, padding=2, groups=dim, bias=False)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return local multi-kernel features with the input shape."""
        _validate_feature_map(x, self.dim, self.__class__.__name__)
        normalized = self.pre_norm(x)
        return self.act(self.dw3(normalized) + self.dw5(normalized))


class BidirectionalAxisMamba(nn.Module):
    """Shared-parameter forward/reverse Mamba scan on one spatial axis.

    Shape: ``[B,C,H,W] -> [B,C,H,W]``.  Horizontal scans use
    ``[B*H,W,C]`` and vertical scans use ``[B*W,H,C]``.  The same Mamba
    parameters are shared by forward and reverse directions of one axis.
    """

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
        """Normalize and scan the configured axis in both directions."""
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


class BranchProjector(nn.Sequential):
    """Independent 1x1 projection and GroupNorm for one ``[B,C,H,W]`` branch."""

    def __init__(self, dim: int) -> None:
        _validate_positive_int("dim", dim)
        super().__init__(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(dim, dim, 1, bias=False)),
                    ("norm", SafeGroupNorm(_group_count(dim), dim)),
                ]
            )
        )


class ContentAwareResidualGate(nn.Module):
    """Independent candidate-aware sigmoid residual gate.

    Inputs are reduced feature maps ``x``, local anchor and one CrossMamba
    candidate, each ``[B, gate_dim, H, W]``.  Output is ``[B,1,H,W]``.
    """

    def __init__(self, gate_dim: int, bias_init: float = -2.0) -> None:
        super().__init__()
        _validate_positive_int("gate_dim", gate_dim)
        if not isinstance(bias_init, (float, int)):
            raise TypeError("bias_init must be a number")
        self.gate_dim = gate_dim
        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("conv_in", nn.Conv2d(5 * gate_dim, gate_dim, 1, bias=False)),
                    ("norm", SafeGroupNorm(_group_count(gate_dim), gate_dim)),
                    ("act", nn.SiLU(inplace=False)),
                    ("conv_out", nn.Conv2d(gate_dim, 1, 1, bias=True)),
                ]
            )
        )
        nn.init.constant_(self.net.conv_out.bias, float(bias_init))

    def forward(
        self,
        x_reduced: torch.Tensor,
        local_reduced: torch.Tensor,
        candidate_reduced: torch.Tensor,
    ) -> torch.Tensor:
        """Return an unconstrained-by-other-branches sigmoid gate."""
        for name, tensor in (
            ("x_reduced", x_reduced),
            ("local_reduced", local_reduced),
            ("candidate_reduced", candidate_reduced),
        ):
            _validate_feature_map(tensor, self.gate_dim, name)
        if x_reduced.shape != local_reduced.shape or x_reduced.shape != candidate_reduced.shape:
            raise ValueError("all reduced gate inputs must have identical shapes")
        features = torch.cat(
            [
                x_reduced,
                local_reduced,
                candidate_reduced,
                torch.abs(candidate_reduced - local_reduced),
                candidate_reduced * local_reduced,
            ],
            dim=1,
        )
        return torch.sigmoid(self.net(features))


class CrossMambaBlock(nn.Module):
    """Local-anchored H->V / V->H CrossMamba residual block.

    Input/output: ``[B,C,H,W] -> [B,C,H,W]``.  The local update never passes
    through a gate.  Enabled CrossMamba candidates use independent sigmoid
    gates, or fixed gates equal to one when ``use_gate=False``.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        gate_reduction: int = 8,
        gamma_local_init: float = 1e-3,
        gamma_mamba_init: float = 1e-3,
        gate_bias_init: float = -2.0,
        drop_path: float = 0.0,
        use_gate: bool = True,
        use_hv: bool = True,
        use_vh: bool = True,
        collect_debug: bool = False,
    ) -> None:
        super().__init__()
        _validate_positive_int("dim", dim)
        _validate_positive_int("d_state", d_state)
        _validate_positive_int("d_conv", d_conv)
        _validate_positive_int("expand", expand)
        _validate_positive_int("gate_reduction", gate_reduction)
        for name, value in (
            ("use_gate", use_gate),
            ("use_hv", use_hv),
            ("use_vh", use_vh),
            ("collect_debug", collect_debug),
        ):
            _validate_bool(name, value)
        for name, value in (
            ("gamma_local_init", gamma_local_init),
            ("gamma_mamba_init", gamma_mamba_init),
        ):
            if not isinstance(value, (float, int)) or float(value) < 0.0:
                raise ValueError("{} must be non-negative".format(name))
        if not isinstance(gate_bias_init, (float, int)):
            raise TypeError("gate_bias_init must be a number")

        self.dim = dim
        self.use_gate = use_gate
        self.use_hv = use_hv
        self.use_vh = use_vh
        self.collect_debug = collect_debug

        self.local_branch = LocalAnchorBranch(dim)
        self.local_projector = BranchProjector(dim)
        self.gamma_local = nn.Parameter(
            torch.full((1, dim, 1, 1), float(gamma_local_init))
        )

        if use_hv or use_vh:
            self.pre_norm = LayerNorm2d(dim)
            self.horizontal = BidirectionalAxisMamba(
                dim, "horizontal", d_state=d_state, d_conv=d_conv, expand=expand
            )
            self.vertical = BidirectionalAxisMamba(
                dim, "vertical", d_state=d_state, d_conv=d_conv, expand=expand
            )

        if use_hv:
            self.hv_projector = BranchProjector(dim)
            self.gamma_hv = nn.Parameter(
                torch.full((1, dim, 1, 1), float(gamma_mamba_init))
            )
        if use_vh:
            self.vh_projector = BranchProjector(dim)
            self.gamma_vh = nn.Parameter(
                torch.full((1, dim, 1, 1), float(gamma_mamba_init))
            )

        if use_gate and (use_hv or use_vh):
            gate_dim = max(dim // gate_reduction, 32)
            self.reduce_x = nn.Conv2d(dim, gate_dim, 1, bias=False)
            self.reduce_local = nn.Conv2d(dim, gate_dim, 1, bias=False)
            if use_hv:
                self.reduce_hv = nn.Conv2d(dim, gate_dim, 1, bias=False)
                self.gate_hv = ContentAwareResidualGate(gate_dim, gate_bias_init)
            if use_vh:
                self.reduce_vh = nn.Conv2d(dim, gate_dim, 1, bias=False)
                self.gate_vh = ContentAwareResidualGate(gate_dim, gate_bias_init)

        self.drop_path = DropPath(float(drop_path))
        self.debug_state: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _rms(x: torch.Tensor) -> torch.Tensor:
        detached = x.detach().float()
        return torch.sqrt(torch.mean(detached * detached))

    @staticmethod
    def _quantiles(x: torch.Tensor) -> torch.Tensor:
        flattened = x.detach().float().reshape(-1)
        levels = flattened.new_tensor([0.1, 0.5, 0.9])
        return torch.quantile(flattened, levels)

    @staticmethod
    def _scaled(parameter: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return parameter.to(dtype=value.dtype) * value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local anchor plus enabled sequential cross-axis increments."""
        _validate_feature_map(x, self.dim, self.__class__.__name__)
        local = self.local_projector(self.local_branch(x))
        local_update = self._scaled(self.gamma_local, local)
        residual_update = local_update

        zero = x.detach().float().new_zeros(())
        one_gate = x.new_ones((x.shape[0], 1, x.shape[2], x.shape[3]))
        hv = None
        vh = None
        hv_update = None
        vh_update = None
        gate_hv = None
        gate_vh = None

        if self.use_hv or self.use_vh:
            normalized = self.pre_norm(x)
            if self.use_hv:
                hv = self.hv_projector(self.vertical(self.horizontal(normalized)))
            if self.use_vh:
                vh = self.vh_projector(self.horizontal(self.vertical(normalized)))

            if self.use_gate:
                x_reduced = self.reduce_x(x)
                local_reduced = self.reduce_local(local)
                if self.use_hv:
                    gate_hv = self.gate_hv(
                        x_reduced, local_reduced, self.reduce_hv(hv)
                    )
                if self.use_vh:
                    gate_vh = self.gate_vh(
                        x_reduced, local_reduced, self.reduce_vh(vh)
                    )
            else:
                gate_hv = one_gate if self.use_hv else None
                gate_vh = one_gate if self.use_vh else None

            if self.use_hv:
                hv_update = self._scaled(self.gamma_hv, gate_hv * hv)
                residual_update = residual_update + hv_update
            if self.use_vh:
                vh_update = self._scaled(self.gamma_vh, gate_vh * vh)
                residual_update = residual_update + vh_update

        output = x + self.drop_path(residual_update)
        if output.dtype != x.dtype:
            output = output.to(dtype=x.dtype)

        if self.collect_debug:
            hv_quantiles = (
                self._quantiles(gate_hv) if gate_hv is not None else zero.repeat(3)
            )
            vh_quantiles = (
                self._quantiles(gate_vh) if gate_vh is not None else zero.repeat(3)
            )
            self.debug_state = {
                "gamma_local_mean": self.gamma_local.detach().mean(),
                "gamma_hv_mean": self.gamma_hv.detach().mean() if self.use_hv else zero,
                "gamma_vh_mean": self.gamma_vh.detach().mean() if self.use_vh else zero,
                "gate_hv_mean": gate_hv.detach().mean() if gate_hv is not None else zero,
                "gate_vh_mean": gate_vh.detach().mean() if gate_vh is not None else zero,
                "gate_hv_q10": hv_quantiles[0],
                "gate_hv_q50": hv_quantiles[1],
                "gate_hv_q90": hv_quantiles[2],
                "gate_vh_q10": vh_quantiles[0],
                "gate_vh_q50": vh_quantiles[1],
                "gate_vh_q90": vh_quantiles[2],
                "local_rms": self._rms(local),
                "hv_rms": self._rms(hv) if hv is not None else zero,
                "vh_rms": self._rms(vh) if vh is not None else zero,
                "local_update_rms": self._rms(local_update),
                "hv_update_rms": self._rms(hv_update) if hv_update is not None else zero,
                "vh_update_rms": self._rms(vh_update) if vh_update is not None else zero,
            }
        return output

    def get_debug_state(self) -> Dict[str, torch.Tensor]:
        """Return detached tensors recorded by the latest debug forward."""
        return {name: value.detach() for name, value in self.debug_state.items()}
