"""
Texture-Scale Enhancement Module (TSEM).

TSEM is inserted after HybridEncoder.input_proj and before the original
Transformer/FPN/PAN path. It keeps feature shapes unchanged and uses a
zero-initialized residual scale so the initial output is an identity mapping.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "AdaptiveSmoothing",
    "TextureScaleBlock",
    "CrossScaleGate",
    "TextureScaleEnhancementModule",
]

_VALID_MODES = {"off", "high_only", "context_only", "dual_sum", "dual_gate", "full"}


def _as_tuple(values: Sequence[int], name: str) -> Tuple[int, ...]:
    if values is None:
        raise ValueError(f"{name} must not be None")
    result = tuple(int(v) for v in values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _validate_mode(mode: str) -> str:
    mode = str(mode)
    if mode not in _VALID_MODES:
        raise ValueError(f"Unsupported tsem mode {mode!r}; expected one of {sorted(_VALID_MODES)}")
    return mode


class ConvBNAct(nn.Module):
    """Small Conv2d + BatchNorm2d + optional SiLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class AdaptiveSmoothing(nn.Module):
    """Depthwise learnable smoothing initialized as average filtering."""

    def __init__(self, channels: int, kernel_size: int = 5, learnable: bool = True) -> None:
        super().__init__()
        channels = int(channels)
        kernel_size = int(kernel_size)
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        self.channels = channels
        self.kernel_size = kernel_size
        self.smooth = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self._reset_parameters()
        self.smooth.weight.requires_grad_(bool(learnable))

    def _reset_parameters(self) -> None:
        with torch.no_grad():
            self.smooth.weight.fill_(1.0 / float(self.kernel_size * self.kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.smooth(x)


class TextureScaleBlock(nn.Module):
    """Single-level TSEM block returning enhancement residual only."""

    def __init__(
        self,
        channels: int = 256,
        mode: str = "full",
        smooth_kernel: int = 5,
        dilations: Sequence[int] = (1, 2, 3),
        gate_reduction: int = 16,
        learnable_smoothing: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.mode = _validate_mode(mode)
        self.dilations = _as_tuple(dilations, "dilations")
        self.debug = bool(debug)
        self.last_debug: Dict[str, torch.Tensor] = {}

        if self.mode == "off":
            raise ValueError("TextureScaleBlock should not be instantiated for mode='off'")
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if int(gate_reduction) <= 0:
            raise ValueError(f"gate_reduction must be positive, got {gate_reduction}")
        for dilation in self.dilations:
            if dilation <= 0:
                raise ValueError(f"all dilations must be positive, got {self.dilations}")

        hidden = max(self.channels // int(gate_reduction), 8)
        self.smooth = AdaptiveSmoothing(self.channels, smooth_kernel, learnable_smoothing)

        self.use_high = self.mode in {"high_only", "dual_sum", "dual_gate", "full"}
        self.use_context = self.mode in {"context_only", "dual_sum", "dual_gate", "full"}
        self.use_spatial_gate = self.mode in {"dual_gate", "full"}

        if self.use_high:
            self.high_branch = nn.Sequential(
                ConvBNAct(self.channels, self.channels, 3, padding=1, groups=self.channels, act=True),
                ConvBNAct(self.channels, self.channels, 1, act=False),
            )

        if self.use_context:
            self.context_branches = nn.ModuleList(
                [
                    nn.Sequential(
                        ConvBNAct(
                            self.channels,
                            self.channels,
                            3,
                            padding=dilation,
                            dilation=dilation,
                            groups=self.channels,
                            act=True,
                        ),
                        ConvBNAct(self.channels, self.channels, 1, act=False),
                    )
                    for dilation in self.dilations
                ]
            )
            self.context_gate = nn.Sequential(
                ConvBNAct(self.channels * 2, hidden, 1, act=True),
                nn.Conv2d(hidden, len(self.dilations), kernel_size=1, bias=True),
            )

        if self.use_spatial_gate:
            self.spatial_gate = nn.Sequential(
                ConvBNAct(self.channels * 2, hidden, 1, act=True),
                nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            )

    def _make_context(self, low: torch.Tensor, x: torch.Tensor, high_raw: torch.Tensor) -> torch.Tensor:
        branch_feats = [branch(low) for branch in self.context_branches]
        descriptor = torch.cat([x, high_raw.abs()], dim=1)
        weights = torch.softmax(self.context_gate(descriptor), dim=1)
        context = branch_feats[0] * weights[:, 0:1]
        for idx in range(1, len(branch_feats)):
            context = context + branch_feats[idx] * weights[:, idx:idx + 1]
        return context

    def _cache_debug(
        self,
        high: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        gate: Optional[torch.Tensor],
    ) -> None:
        if not self.debug:
            return
        data: Dict[str, torch.Tensor] = {}
        if high is not None:
            data["high_abs_mean"] = high.detach().abs().mean()
            data["high_heatmap"] = high.detach().abs().mean(dim=1, keepdim=True)
        if context is not None:
            data["context_abs_mean"] = context.detach().abs().mean()
            data["context_heatmap"] = context.detach().abs().mean(dim=1, keepdim=True)
        if gate is not None:
            detached_gate = gate.detach()
            data["gate_mean"] = detached_gate.mean()
            data["gate_min"] = detached_gate.amin()
            data["gate_max"] = detached_gate.amax()
            data["gate_heatmap"] = detached_gate
        self.last_debug = data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return enhancement residual with the same shape as input."""
        low = self.smooth(x)
        high_raw = x - low

        high = self.high_branch(high_raw) if self.use_high else None
        context = self._make_context(low, x, high_raw) if self.use_context else None
        gate = None

        if self.mode == "high_only":
            residual = high
        elif self.mode == "context_only":
            residual = context
        elif self.mode == "dual_sum":
            residual = 0.5 * (high + context)
        else:
            gate = torch.sigmoid(self.spatial_gate(torch.cat([high, context], dim=1)))
            residual = gate * high + (1.0 - gate) * context

        self._cache_debug(high, context, gate)
        return residual


class CrossScaleGate(nn.Module):
    """Cross-level channel gate returning weights shaped [B, L, C, 1, 1]."""

    def __init__(self, channels: int, num_levels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_levels = int(num_levels)
        reduction = int(reduction)
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if self.num_levels <= 0:
            raise ValueError(f"num_levels must be positive, got {self.num_levels}")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}")

        hidden = max(self.channels // reduction, 8)
        self.fc1 = nn.Linear(self.num_levels * self.channels, hidden)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Linear(hidden, self.num_levels * self.channels)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """Return scale weights normalized over level dimension."""
        if len(feats) != self.num_levels:
            raise ValueError(f"expected {self.num_levels} levels, got {len(feats)}")
        pooled = [F.adaptive_avg_pool2d(feat, 1).flatten(1) for feat in feats]
        logits = self.fc2(self.act(self.fc1(torch.cat(pooled, dim=1))))
        logits = logits.reshape(feats[0].shape[0], self.num_levels, self.channels, 1, 1)
        return torch.softmax(logits, dim=1) * float(self.num_levels)


class TextureScaleEnhancementModule(nn.Module):
    """Apply TSEM blocks to selected multi-scale feature levels."""

    def __init__(
        self,
        channels: int,
        num_levels: int,
        mode: str = "full",
        levels: Sequence[int] = (0, 1, 2),
        smooth_kernel: int = 5,
        dilations: Sequence[int] = (1, 2, 3),
        reduction: int = 16,
        init_scale: float = 0.0,
        learnable_smoothing: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_levels = int(num_levels)
        self.mode = _validate_mode(mode)
        self.levels = _as_tuple(levels, "levels")
        self.debug = bool(debug)
        self.last_debug: Dict[str, object] = {}

        if self.mode == "off":
            raise ValueError("TextureScaleEnhancementModule should not be instantiated for mode='off'")
        if self.num_levels <= 0:
            raise ValueError(f"num_levels must be positive, got {self.num_levels}")
        for level in self.levels:
            if level < 0 or level >= self.num_levels:
                raise ValueError(f"tsem level {level} out of range for num_levels={self.num_levels}")

        blocks = {}
        for level in self.levels:
            blocks[str(level)] = TextureScaleBlock(
                channels=self.channels,
                mode=self.mode,
                smooth_kernel=smooth_kernel,
                dilations=dilations,
                gate_reduction=reduction,
                learnable_smoothing=learnable_smoothing,
                debug=debug,
            )
        self.blocks = nn.ModuleDict(blocks)
        self.gamma = nn.Parameter(torch.full((self.num_levels,), float(init_scale)))
        self.cross_scale_gate = CrossScaleGate(self.channels, self.num_levels, reduction) if self.mode == "full" else None

    def _cache_debug(self, scale_weights: Optional[torch.Tensor]) -> None:
        if not self.debug:
            return
        data: Dict[str, object] = {
            "gamma": self.gamma.detach().cpu(),
            "levels": list(self.levels),
            "blocks": {level: block.last_debug for level, block in self.blocks.items()},
        }
        if scale_weights is not None:
            data["scale_weight_channel_mean"] = scale_weights.detach().mean(dim=2).cpu()
        self.last_debug = data

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Return a new feature list; input tensors and list are not modified."""
        if len(feats) != self.num_levels:
            raise ValueError(f"expected {self.num_levels} feature levels, got {len(feats)}")

        scale_weights = self.cross_scale_gate(feats) if self.cross_scale_gate is not None else None
        outs: List[torch.Tensor] = []
        for idx, feat in enumerate(feats):
            level_key = str(idx)
            if level_key not in self.blocks:
                outs.append(feat)
                continue

            block = self.blocks[level_key]
            residual = block(feat)
            gamma = self.gamma[idx].reshape(1, 1, 1, 1).to(dtype=feat.dtype, device=feat.device)
            if scale_weights is None:
                scale = 1.0
            else:
                scale = scale_weights[:, idx].to(dtype=feat.dtype, device=feat.device)
            outs.append(feat + gamma * scale * residual)

        self._cache_debug(scale_weights)
        return outs
