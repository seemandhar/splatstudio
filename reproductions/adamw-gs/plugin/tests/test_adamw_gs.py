"""CPU unit tests of the AdamW-GS pure functions and probes."""

import math

import pytest
import torch

import ss_adamw_gs
from ss_adamw_gs import probes
from ss_adamw_gs.density import DEFAULTS, rsr_active, rsr_ratio
from ss_adamw_gs.optim import img_scale


def test_img_scale_official_examples():
    assert img_scale(1297 * 840) == 100000
    assert img_scale(1558 * 1038) == 160000
    with pytest.raises(ValueError):
        img_scale(50)


def test_stss_outdoor_and_indoor():
    c = DEFAULTS
    assert [rsr_ratio(c, s) for s in (999, 1000, 6999, 7000, 9999, 10000, 15000)] == [
        0.0, 0.1, 0.1, 0.3, 0.3, 0.5, 0.5]
    indoor = next(x for x in ss_adamw_gs.indoor().components if x.kind == "density").config
    assert rsr_ratio(indoor, 12000) == 0.3 and indoor["noise"] is False
    assert [rsr_active(c, s) for s in (900, 1000, 1050, 15000, 15100)] == [
        False, True, False, True, False]


def test_alpha2_is_code_effective_value():
    assert DEFAULTS["rsr_alpha1"] == 0.2 and DEFAULTS["rsr_alpha2"] == 0.01


@pytest.mark.parametrize("name", ["install", "sparse_step", "dar", "image_scale", "rsr", "topology"])
@pytest.mark.parametrize("make", [ss_adamw_gs.method, ss_adamw_gs.indoor, ss_adamw_gs.gs8])
def test_probes_cpu(name, make):
    getattr(probes, name)(make(), "cpu")


def test_dense_adam_would_fail_sparse_probe(monkeypatch):
    """Removing sparsity (update every row) must make the sparse-step probe fail."""
    from ss_adamw_gs import optim

    original = optim.sparse_adamwgs_update

    def dense(param, grad, m, v, t, visible, *a, **k):
        return original(param, grad, m, v, t.clamp_min(1), torch.ones_like(visible), *a, **k)

    monkeypatch.setattr(optim, "sparse_adamwgs_update", dense)
    with pytest.raises(AssertionError):
        probes.sparse_step(ss_adamw_gs.method(), "cpu")


def test_coupled_regularisation_would_fail_dar_probe(monkeypatch):
    """Switching DAR off must make the DAR probe fail."""
    from ss_adamw_gs import optim

    monkeypatch.setattr(optim.AdamWGSContext, "regularizer", lambda self, f: (0, 0.0, 0.0))
    with pytest.raises(AssertionError):
        probes.dar(ss_adamw_gs.method(), "cpu")


def test_opacity_correction_formula():
    o = torch.tensor([0.2, 0.5, 0.99])
    assert torch.allclose(1 - torch.sqrt(1 - o), torch.tensor([1 - math.sqrt(0.8),
                                                              1 - math.sqrt(0.5), 0.9]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat covariance kernel needs CUDA")
def test_noise_probe_cuda():
    probes.noise(ss_adamw_gs.method(), "cuda")


def test_install_probe_fails_without_swap(monkeypatch):
    from ss_adamw_gs import density as dmod

    def plain_init(self, params, optimizers, scene_scale=1.0):
        self.state = {"asystep": torch.zeros(len(params["means"]), dtype=torch.int32),
                      "grad2d": None, "count": None, "max_radii": None, "scene_scale": 1.0}
        self.context = dmod.AdamWGSContext(self.config, self.state)

    monkeypatch.setattr(dmod.AdamWGSDensity, "initialize", plain_init)
    with pytest.raises(AssertionError):
        probes.install(ss_adamw_gs.method(), "cpu")


def test_gs8_config():
    c = next(x for x in ss_adamw_gs.gs8().components if x.kind == "density").config
    assert c["opacity_reset_every"] == 3000 and c["noise"] and c["rsr_ratios"] == [0.1, 0.3, 0.5]
    assert c["big_prune_world"] == 0.1 and "big_prune_screen" not in c

