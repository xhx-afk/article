"""Unit tests for AxMamba V1."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from engine.deim import axmamba as axmamba_module
from engine.deim.axmamba import (
    AxMambaBlock,
    BidirectionalAxisMamba,
    LocalDetailBranch,
)
from engine.deim.hybrid_encoder import HybridEncoder


class FakeMamba(nn.Module):
    """CPU-test double with the official Mamba constructor/shape contract."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.proj(x)


@pytest.fixture
def fake_mamba(monkeypatch):
    monkeypatch.setattr(axmamba_module, "Mamba", FakeMamba)
    return FakeMamba


def assert_group_has_finite_gradient(module, prefix):
    gradients = [
        parameter.grad
        for name, parameter in module.named_parameters()
        if name == prefix or name.startswith(prefix + ".")
    ]
    assert gradients, "no parameters found for {}".format(prefix)
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_local_detail_branch_shape():
    branch = LocalDetailBranch(32)
    x = torch.randn(2, 32, 13, 19)
    assert branch(x).shape == x.shape


@pytest.mark.parametrize("axis", ["horizontal", "vertical"])
def test_bidirectional_axis_mamba_shape(fake_mamba, axis):
    branch = BidirectionalAxisMamba(32, axis)
    x = torch.randn(2, 32, 9, 17)
    output = branch(x)
    assert output.shape == x.shape
    assert output.dtype == x.dtype


def test_axis_mean_shape(fake_mamba):
    block = AxMambaBlock(32, "axis_mean")
    x = torch.randn(2, 32, 9, 17)
    assert block(x).shape == x.shape


def test_axis_gate_shape_and_debug(fake_mamba):
    block = AxMambaBlock(32, "axis_gate", collect_debug=True)
    x = torch.randn(2, 32, 9, 17)
    assert block(x).shape == x.shape
    state = block.get_debug_state()
    assert set(state) == {
        "gate_mean",
        "gamma_mean",
        "horizontal_norm",
        "vertical_norm",
        "local_norm",
    }
    assert state["gate_mean"].shape == (3,)


def test_axis_gate_pixelwise_weights_sum_to_one(fake_mamba):
    block = AxMambaBlock(32, "axis_gate").eval()
    x = torch.randn(2, 32, 7, 11)
    with torch.no_grad():
        local = block.local_branch(x)
        weights = torch.softmax(block.gate(torch.cat([x, local], dim=1)), dim=1)
    expected = torch.ones_like(weights[:, 0])
    assert torch.allclose(weights.sum(dim=1), expected, rtol=1e-6, atol=1e-6)


def test_axis_gate_backward_all_branches(fake_mamba):
    block = AxMambaBlock(32, "axis_gate", gamma_init=1e-3)
    x = torch.randn(2, 32, 7, 11, requires_grad=True)
    loss = block(x).float().square().mean()
    loss.backward()
    for prefix in ("mamba_h", "mamba_v", "local_branch", "gate", "gamma"):
        assert_group_has_finite_gradient(block, prefix)


def test_non_contiguous_input(fake_mamba):
    block = AxMambaBlock(32, "axis_gate")
    x = torch.randn(2, 32, 11, 7).transpose(2, 3)
    assert not x.is_contiguous()
    assert block(x).shape == x.shape


def test_non_square_input(fake_mamba):
    block = AxMambaBlock(32, "axis_mean")
    x = torch.randn(2, 32, 36, 80)
    assert block(x).shape == x.shape


@pytest.mark.parametrize("shape", [(2, 32, 1, 11), (2, 32, 11, 1)])
def test_singleton_spatial_axis(fake_mamba, shape):
    block = AxMambaBlock(32, "axis_gate")
    x = torch.randn(*shape)
    assert block(x).shape == x.shape


def test_hybrid_encoder_disabled_has_no_axmamba_parameters():
    encoder = HybridEncoder(
        in_channels=[16, 32, 64],
        feat_strides=[8, 16, 32],
        hidden_dim=32,
        nhead=4,
        dim_feedforward=64,
        num_encoder_layers=0,
        eval_spatial_size=None,
        use_axmamba=False,
    )
    assert not hasattr(encoder, "axmamba_blocks")
    assert not any("axmamba" in name for name, _ in encoder.named_parameters())
    assert encoder.get_axmamba_debug_state() == {}


def test_local_mode_does_not_require_mamba(monkeypatch):
    monkeypatch.setattr(axmamba_module, "Mamba", None)
    block = AxMambaBlock(32, "local")
    assert not hasattr(block, "mamba_h")
    assert not hasattr(block, "mamba_v")
    x = torch.randn(2, 32, 7, 11)
    assert block(x).shape == x.shape


def test_hybrid_encoder_local_without_mamba(monkeypatch):
    monkeypatch.setattr(axmamba_module, "Mamba", None)
    encoder = HybridEncoder(
        in_channels=[16, 32, 64],
        feat_strides=[8, 16, 32],
        hidden_dim=32,
        nhead=4,
        dim_feedforward=64,
        num_encoder_layers=0,
        eval_spatial_size=None,
        use_axmamba=True,
        axmamba_levels=[1, 2],
        axmamba_mode="local",
    )
    features = [
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 64, 4, 4),
    ]
    outputs = encoder(features)
    assert [tuple(output.shape) for output in outputs] == [
        (2, 32, 16, 16),
        (2, 32, 8, 8),
        (2, 32, 4, 4),
    ]


def test_axis_mode_has_clear_missing_dependency_error(monkeypatch):
    monkeypatch.setattr(axmamba_module, "Mamba", None)
    monkeypatch.setattr(
        axmamba_module,
        "_MAMBA_IMPORT_ERROR",
        ModuleNotFoundError("mamba_ssm"),
    )
    with pytest.raises(ImportError, match="mamba_ssm is unavailable"):
        AxMambaBlock(32, "axis_mean")


@pytest.mark.parametrize("mode", ["bad", "horizontal", ""])
def test_invalid_mode(mode):
    with pytest.raises(ValueError, match="mode must be one of"):
        AxMambaBlock(32, mode)


@pytest.mark.parametrize("levels", [[1, 1], [-1, 2], [1, 3]])
def test_invalid_hybrid_encoder_levels(levels):
    with pytest.raises((ValueError, TypeError), match="axmamba_levels"):
        HybridEncoder(
            in_channels=[16, 32, 64],
            hidden_dim=32,
            use_axmamba=True,
            axmamba_levels=levels,
            axmamba_mode="local",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or axmamba_module.Mamba is None,
    reason="official mamba_ssm CUDA environment is required",
)
def test_official_mamba_amp_forward_backward():
    device = torch.device("cuda")
    block = AxMambaBlock(32, "axis_gate").to(device)
    x = torch.randn(2, 32, 7, 11, device=device, requires_grad=True)
    scaler = torch.cuda.amp.GradScaler()
    with torch.cuda.amp.autocast():
        output = block(x)
        loss = output.float().square().mean()
    scaler.scale(loss).backward()
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    for prefix in ("mamba_h", "mamba_v", "local_branch", "gate", "gamma"):
        assert_group_has_finite_gradient(block, prefix)
