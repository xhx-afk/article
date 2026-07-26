"""CrossMamba V2 structural and gradient tests.

The tests use a small Mamba-compatible CPU double.  The server smoke test
still exercises the official ``mamba_ssm`` implementation.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from engine.deim import axmamba as axmamba_module
from engine.deim.axmamba import (
    BidirectionalAxisMamba,
    CrossMambaBlock,
    LocalAnchorBranch,
)
from engine.deim.hybrid_encoder import HybridEncoder


class FakeMamba(nn.Module):
    """CPU test double with the official Mamba constructor/shape contract."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        return x * self.scale + self.bias


@pytest.fixture
def fake_mamba(monkeypatch):
    monkeypatch.setattr(axmamba_module, "Mamba", FakeMamba)


def _assert_finite_gradients(module):
    missing = []
    non_finite = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            non_finite.append(name)
    assert not missing, "missing gradients: {}".format(missing)
    assert not non_finite, "non-finite gradients: {}".format(non_finite)


def test_local_shape(fake_mamba):
    x = torch.randn(2, 32, 13, 19)
    assert LocalAnchorBranch(32)(x).shape == x.shape
    assert CrossMambaBlock(32, use_hv=False, use_vh=False)(x).shape == x.shape


@pytest.mark.parametrize("enabled", [(True, False), (False, True), (True, True)])
def test_cross_axis_shapes(fake_mamba, enabled):
    block = CrossMambaBlock(32, use_hv=enabled[0], use_vh=enabled[1])
    x = torch.randn(2, 32, 9, 17)
    y = block(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device


def test_hv_vh_call_order(fake_mamba):
    block = CrossMambaBlock(8, use_gate=False, collect_debug=False)
    calls = []

    class Spy(nn.Module):
        def __init__(self, label):
            super().__init__()
            self.label = label

        def forward(self, x):
            calls.append(self.label)
            return x

    block.horizontal = Spy("H")
    block.vertical = Spy("V")
    block.hv_projector = nn.Identity()
    block.vh_projector = nn.Identity()
    block.local_projector = nn.Identity()
    block.local_branch = nn.Identity()
    block.gamma_local.data.zero_()
    block.gamma_hv.data.fill_(1.0)
    block.gamma_vh.data.fill_(1.0)
    block(torch.randn(1, 8, 4, 5))
    assert calls == ["H", "V", "V", "H"]


def test_non_square_large_shape_and_non_contiguous(fake_mamba):
    x = torch.randn(2, 256, 80, 36).transpose(2, 3)
    assert x.shape == (2, 256, 36, 80)
    assert not x.is_contiguous()
    horizontal = BidirectionalAxisMamba(256, "horizontal")
    vertical = BidirectionalAxisMamba(256, "vertical")
    assert horizontal(x).shape == x.shape
    assert vertical(x).shape == x.shape


@pytest.mark.parametrize("shape", [(1, 32, 1, 1), (1, 32, 1, 11), (1, 32, 11, 1)])
def test_singleton_spatial_axis(fake_mamba, shape):
    block = CrossMambaBlock(32, use_gate=True)
    x = torch.randn(*shape)
    assert block(x).shape == x.shape


def test_independent_sigmoid_gates_and_not_softmax(fake_mamba):
    block = CrossMambaBlock(32, use_gate=True)
    reduced = torch.zeros(2, block.reduce_x.out_channels, 7, 11)
    gate_hv = block.gate_hv(reduced, reduced, reduced)
    gate_vh = block.gate_vh(reduced, reduced, reduced)
    assert torch.all((gate_hv >= 0) & (gate_hv <= 1))
    assert torch.all((gate_vh >= 0) & (gate_vh <= 1))
    # Independent sigmoid gates have no sum-to-one constraint.
    assert not torch.allclose(gate_hv + gate_vh, torch.ones_like(gate_hv))


def test_local_update_is_not_gated(fake_mamba):
    block = CrossMambaBlock(8, use_gate=True, use_hv=True, use_vh=False)
    block.local_branch = nn.Identity()
    block.local_projector = nn.Identity()
    block.hv_projector = nn.Identity()
    block.horizontal = nn.Identity()
    block.vertical = nn.Identity()
    block.pre_norm = nn.Identity()
    block.gamma_local.data.fill_(1.0)
    block.gamma_hv.data.zero_()
    x = torch.randn(1, 8, 4, 5)
    expected = x + x
    assert torch.allclose(block(x), expected)


def test_backward_has_local_cross_gate_and_gamma_gradients(fake_mamba):
    block = CrossMambaBlock(32, use_gate=True, gamma_local_init=1e-3, gamma_mamba_init=1e-3)
    x = torch.randn(2, 32, 7, 11, requires_grad=True)
    block(x).float().square().mean().backward()
    _assert_finite_gradients(block)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP test")
def test_amp_forward_backward(fake_mamba):
    device = torch.device("cuda")
    block = CrossMambaBlock(32, use_gate=True).to(device)
    x = torch.randn(2, 32, 7, 11, device=device, requires_grad=True)
    scaler = torch.cuda.amp.GradScaler()
    with torch.cuda.amp.autocast():
        output = block(x)
        loss = output.float().square().mean()
    scaler.scale(loss).backward()
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    _assert_finite_gradients(block)


def test_disabled_encoder_has_no_crossmamba_parameters():
    encoder = HybridEncoder(
        in_channels=[16, 32, 64],
        feat_strides=[8, 16, 32],
        hidden_dim=32,
        nhead=4,
        dim_feedforward=64,
        num_encoder_layers=0,
        use_crossmamba=False,
    )
    assert not hasattr(encoder, "crossmamba_p4")
    assert not any("crossmamba" in name for name, _ in encoder.named_parameters())
    assert encoder.get_crossmamba_debug_state() == {}


def test_hybrid_encoder_three_scale_shapes_and_fused_p4_only(fake_mamba):
    encoder = HybridEncoder(
        in_channels=[16, 32, 64],
        feat_strides=[8, 16, 32],
        hidden_dim=32,
        nhead=4,
        dim_feedforward=64,
        num_encoder_layers=0,
        use_crossmamba=True,
        crossmamba_use_hv=False,
        crossmamba_use_vh=False,
    )
    calls = []

    class Spy(nn.Module):
        def forward(self, x):
            calls.append(tuple(x.shape))
            return x

    encoder.crossmamba_p4 = Spy()
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
    assert calls == [(2, 32, 8, 8)]


def test_no_semantic_map_interface():
    signature = inspect.signature(CrossMambaBlock.__init__)
    assert not any("semantic" in name.lower() for name in signature.parameters)
    assert not any("semantic" in name.lower() for name in inspect.signature(HybridEncoder.__init__).parameters)
    assert not hasattr(CrossMambaBlock, "semantic_map")
