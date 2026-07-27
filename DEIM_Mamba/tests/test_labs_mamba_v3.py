"""LABS-Mamba V3 structural, numerical, gradient, and placement tests."""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from engine.deim import axmamba as axmamba_module
from engine.deim.axmamba import (
    BidirectionalAxisMamba,
    BoundedMambaSidecar,
    LocalAnchorBlock,
)
from engine.deim.hybrid_encoder import HybridEncoder


class FakeMamba(nn.Module):
    """Small CPU/CUDA test double with the mamba_ssm shape contract."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        return x * self.scale + self.bias


@pytest.fixture
def fake_mamba(monkeypatch):
    monkeypatch.setattr(axmamba_module, "Mamba", FakeMamba)


def _encoder(use_sidecar, path="hv"):
    return HybridEncoder(
        in_channels=[16, 32, 64],
        feat_strides=[8, 16, 32],
        hidden_dim=32,
        nhead=4,
        dim_feedforward=64,
        num_encoder_layers=0,
        use_local_anchor=True,
        use_mamba_sidecar=use_sidecar,
        mamba_sidecar_path=path,
        mamba_sidecar_bottleneck_dim=16,
        mamba_sidecar_collect_debug=True,
    )


def _features(batch=2):
    return [
        torch.randn(batch, 16, 16, 16),
        torch.randn(batch, 32, 8, 8),
        torch.randn(batch, 64, 4, 4),
    ]


def _assert_all_finite_gradients(module):
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


def test_local_anchor_shape():
    x = torch.randn(2, 256, 13, 19)
    output = LocalAnchorBlock(256)(x)
    assert output.shape == x.shape
    assert output.dtype == x.dtype


@pytest.mark.parametrize("path", ["hv", "vh"])
def test_hv_and_vh_sidecar_shapes_and_single_path(fake_mamba, path):
    block = BoundedMambaSidecar(32, 16, path=path)
    calls = []

    class Spy(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, value):
            calls.append(self.name)
            return value

    block.horizontal = Spy("H")
    block.vertical = Spy("V")
    x = torch.randn(2, 32, 9, 17)
    output = block(x)
    assert output.shape == x.shape
    assert calls == (["H", "V"] if path == "hv" else ["V", "H"])


def test_non_square_256_shape_and_non_contiguous(fake_mamba):
    contiguous = torch.randn(2, 256, 80, 36)
    x = contiguous.transpose(2, 3)
    assert x.shape == (2, 256, 36, 80)
    assert not x.is_contiguous()
    block = BoundedMambaSidecar(256, 64, path="hv")
    assert block(x).shape == x.shape


@pytest.mark.parametrize("shape", [(1, 32, 1, 1), (1, 32, 1, 11), (1, 32, 11, 1)])
def test_singleton_spatial_axes(fake_mamba, shape):
    block = BoundedMambaSidecar(32, 16, path="vh")
    x = torch.randn(*shape)
    assert block(x).shape == x.shape


def test_reduce_axes_expand_and_beta_have_finite_gradients(fake_mamba):
    block = BoundedMambaSidecar(32, 16, path="hv", collect_debug=True)
    x = torch.randn(2, 32, 7, 11, requires_grad=True)
    block(x).float().square().mean().backward()
    _assert_all_finite_gradients(block)
    required = ("reduce.", "horizontal.", "vertical.", "expand.", "beta_raw")
    names = [name for name, _ in block.named_parameters()]
    for prefix in required:
        assert any(name.startswith(prefix) for name in names), prefix


def test_beta_bound_centering_rms_alignment_and_update_ratio(fake_mamba):
    block = BoundedMambaSidecar(
        32,
        16,
        beta_init=0.005,
        beta_max=0.05,
        center_residual=True,
        rms_align=True,
        collect_debug=True,
    ).eval()
    with torch.no_grad():
        block.beta_raw.fill_(100.0)
        block(torch.randn(2, 32, 13, 19))
    state = block.get_debug_state()
    assert abs(float(state["beta_effective"])) <= block.beta_max + 1e-7
    assert float(state["candidate_spatial_mean_abs_after_center"]) < 1e-5
    torch.testing.assert_close(
        state["candidate_aligned_rms"], state["input_rms"], rtol=2e-4, atol=2e-4
    )
    assert torch.all(state["update_input_ratio"] <= block.beta_max + 2e-4)


def test_disabled_sidecar_has_no_mamba_parameters_or_module():
    encoder = _encoder(use_sidecar=False)
    assert not hasattr(encoder, "mamba_sidecar_p4")
    assert not any("mamba_sidecar" in name for name, _ in encoder.named_parameters())
    assert encoder.get_labs_mamba_debug_state() == {}


def test_local_only_builds_when_mamba_ssm_is_unavailable(monkeypatch):
    monkeypatch.setattr(axmamba_module, "Mamba", None)
    encoder = _encoder(use_sidecar=False)
    assert hasattr(encoder, "local_anchor_p4")
    with pytest.raises(ImportError, match="mamba_ssm"):
        _encoder(use_sidecar=True)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"path": "both"}, "path"),
        ({"bottleneck_dim": 0}, "bottleneck_dim"),
        ({"beta_init": 0.0}, "beta"),
        ({"beta_init": 0.05, "beta_max": 0.05}, "beta"),
        ({"beta_init": 0.005, "beta_max": 0.21}, "beta"),
    ],
)
def test_invalid_sidecar_parameters_are_clear(fake_mamba, kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        BoundedMambaSidecar(32, **kwargs)


def test_hybrid_encoder_three_scale_shapes(fake_mamba):
    encoder = _encoder(use_sidecar=True).eval()
    outputs = encoder(_features())
    assert [tuple(output.shape) for output in outputs] == [
        (2, 32, 16, 16),
        (2, 32, 8, 8),
        (2, 32, 4, 4),
    ]


def test_sidecar_changes_only_final_p4(fake_mamba):
    torch.manual_seed(7)
    off = _encoder(use_sidecar=False).eval()
    on = _encoder(use_sidecar=True).eval()
    incompatible = on.load_state_dict(off.state_dict(), strict=False)
    assert all(key.startswith("mamba_sidecar_p4.") for key in incompatible.missing_keys)
    assert not incompatible.unexpected_keys
    features = _features()
    with torch.no_grad():
        outputs_off = off(features)
        outputs_on = on(features)
    torch.testing.assert_close(outputs_off[0], outputs_on[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(outputs_off[2], outputs_on[2], rtol=0.0, atol=0.0)
    assert not torch.allclose(outputs_off[1], outputs_on[1])


def test_sidecar_called_once_after_all_pan_blocks(fake_mamba):
    encoder = _encoder(use_sidecar=True).eval()
    events = []
    handles = []
    for index, block in enumerate(encoder.pan_blocks):
        handles.append(
            block.register_forward_hook(
                lambda _module, _inputs, _output, index=index: events.append("PAN{}".format(index))
            )
        )

    class SidecarSpy(nn.Module):
        def forward(self, value):
            events.append("SIDECAR")
            return value + 0.01

        def get_debug_state(self):
            return {}

    encoder.mamba_sidecar_p4 = SidecarSpy()
    with torch.no_grad():
        encoder(_features())
    for handle in handles:
        handle.remove()
    assert events == ["PAN0", "PAN1", "SIDECAR"]
    assert events.count("SIDECAR") == 1


def test_no_v2_gate_or_semantic_interface():
    sidecar_signature = inspect.signature(BoundedMambaSidecar.__init__)
    encoder_signature = inspect.signature(HybridEncoder.__init__)
    forbidden = ("semantic", "gate", "crossmamba", "use_hv", "use_vh")
    for signature in (sidecar_signature, encoder_signature):
        assert not any(
            token in name.lower() for name in signature.parameters for token in forbidden
        )
    assert not hasattr(axmamba_module, "CrossMambaBlock")
    assert not hasattr(axmamba_module, "ContentAwareResidualGate")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP")
def test_amp_forward_backward(fake_mamba):
    device = torch.device("cuda")
    block = BoundedMambaSidecar(32, 16, path="hv").to(device)
    x = torch.randn(2, 32, 7, 11, device=device, requires_grad=True)
    scaler = torch.cuda.amp.GradScaler()
    with torch.cuda.amp.autocast():
        output = block(x)
        loss = output.float().square().mean()
    scaler.scale(loss).backward()
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    _assert_all_finite_gradients(block)


def test_axis_module_uses_vectorized_shared_forward_reverse(fake_mamba):
    axis = BidirectionalAxisMamba(16, "horizontal")
    assert sum(1 for _ in axis.modules() if isinstance(_, FakeMamba)) == 1
    x = torch.randn(2, 16, 5, 9)
    assert axis(x).shape == x.shape
