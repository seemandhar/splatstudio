from dataclasses import replace

import pytest
import torch

from splatstudio import FieldSpec
from splatstudio.losses import check_loss, ssim
from splatstudio.methods import vanilla
from splatstudio.probes import state_transitions
from splatstudio.state import check_state, sample_scene


def test_real_topology_operations_preserve_custom_fields_and_adam_state():
    method = replace(vanilla(), fields=(FieldSpec("latent", (8,), initial_value=0.25),))
    result = state_transitions(method, "cpu")
    assert result["final_gaussians"] == 13
    assert "latent" in result["checked_fields"]


def test_stale_optimizer_parameter_is_detected():
    method = vanilla()
    params, opts = sample_scene(method)
    params["means"] = torch.nn.Parameter(params["means"].detach().clone())
    with pytest.raises(ValueError, match="stale or wrong"):
        check_state(method, params, opts)


def test_wrong_moment_shape_is_detected():
    method = vanilla()
    params, opts = sample_scene(method)
    p = params["means"]
    opts["means"].state[p]["exp_avg"] = torch.zeros(1, 3)
    with pytest.raises(ValueError, match="optimizer moment"):
        check_state(method, params, opts)


def test_detached_loss_is_detected():
    x = torch.tensor(2.0, requires_grad=True)
    with pytest.raises(ValueError, match="detached"):
        check_loss(x.square().detach(), "mutant")


def test_nonfinite_loss_is_detected():
    with pytest.raises(ValueError, match="nonfinite"):
        check_loss(torch.tensor(float("nan"), requires_grad=True), "mutant")


def test_ssim_identity_and_constant_image_oracle():
    a = torch.full((16, 16, 3), 0.2)
    b = torch.full_like(a, 0.8)
    torch.testing.assert_close(ssim(a, a), torch.tensor(1.0))
    expected = (2 * 0.2 * 0.8 + 0.01**2) / (0.2**2 + 0.8**2 + 0.01**2)
    assert ssim(a, b).item() == pytest.approx(expected, abs=1e-4)
