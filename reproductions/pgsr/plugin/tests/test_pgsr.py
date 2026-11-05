"""CPU unit tests of the PGSR port's pure functions and probes."""

import pytest
import torch

import ss_pgsr
from ss_pgsr import geometry as G
from ss_pgsr import probes


@pytest.fixture(scope="module")
def method():
    return ss_pgsr.method()


@pytest.mark.parametrize("name", ["flatten_loss", "single_view", "mv_geometry", "mv_ncc",
                                  "exposure", "neighbour_graph", "density_recipe"])
def test_cpu_probes(method, name):
    out = getattr(probes, name)(method, "cpu")
    assert isinstance(out, dict)


def test_lncc_identical_patches_zero():
    x = torch.rand(10, 49)
    ncc, mask = G.lncc(x, x * 2 + 0.3)  # affine-invariant
    assert ncc.max() < 1e-4 and mask.all()


def test_lncc_constant_patch_masked():
    ncc, mask = G.lncc(torch.rand(4, 49), torch.full((4, 49), 0.5))
    assert (ncc > 0.99).all() and not mask.any()


def test_image_grad_weight_range():
    w = G.image_grad_weight(torch.rand(20, 30, 3))
    assert w.shape == (20, 30) and w.min() >= 0 and w.max() <= 1
    assert (w[0] == 1).all() and (w[:, -1] == 1).all()


def test_variants_configs():
    tnt = ss_pgsr.method_tnt()
    cfg = {c.name: c.config for c in tnt.components}
    assert cfg["pgsr-rgb"]["exposure_compensation"] is True
    assert cfg["pgsr-density"]["densify_abs_grad_threshold"] == 0.00015
    base = {c.name: c.config for c in ss_pgsr.method().components}
    assert base["pgsr-rgb"]["exposure_compensation"] is False
    assert base["pgsr-density"]["percent_dense"] == 0.001
    assert base["pgsr-flatten"]["scale_loss_weight"] == 100.0
    assert base["pgsr-single-view"]["single_view_weight"] == 0.015
    assert base["pgsr-mv-geo"]["multi_view_geo_weight"] == 0.03
    assert base["pgsr-mv-ncc"]["multi_view_ncc_weight"] == 0.15
