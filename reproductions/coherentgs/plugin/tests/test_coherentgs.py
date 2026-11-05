"""CPU unit tests: run the CPU mechanism probes and check pure helpers."""

import pytest
import torch

import ss_coherentgs
from ss_coherentgs import probes
from ss_coherentgs.renderer import pool, scaled_camera

METHOD = ss_coherentgs.method()


@pytest.mark.parametrize("name", ["ray_structure", "decoder_architecture", "masked_residual",
                                  "alignment", "radius", "tv_loss", "flow_loss",
                                  "photometric_terms"])
def test_cpu_probe(name):
    out = getattr(probes, name)(METHOD, "cpu")
    assert isinstance(out, dict)


def test_scaled_camera_keeps_fov():
    cam = {"K": torch.tensor([[10.0, 0, 8], [0, 10.0, 6], [0, 0, 1]]), "width": 16, "height": 12,
           "viewmat": torch.eye(4)}
    c3 = scaled_camera(cam, 3)
    assert (c3["width"], c3["height"]) == (48, 36)
    assert torch.allclose(c3["K"][:2], cam["K"][:2] * 3)
    assert abs(c3["K"][0, 2] / c3["width"] - 0.5) < 1e-6


def test_pool_is_average():
    x = torch.arange(36.0).reshape(6, 6, 1)
    p = pool(x, 3)
    assert p.shape == (2, 2, 1)
    assert float(p[0, 0, 0]) == float(x[:3, :3].mean())


def test_alignment_probe_fails_without_alignment(monkeypatch):
    from ss_coherentgs.bank import Bank

    monkeypatch.setattr(Bank, "align", lambda self: [0.0] * 1000)
    with pytest.raises(AssertionError):
        probes.alignment(METHOD, "cpu")


def test_flow_probe_fails_with_wrong_warp(monkeypatch):
    from ss_coherentgs import bank

    monkeypatch.setattr(bank, "backwarp", lambda img, flow: img)
    with pytest.raises(AssertionError):
        probes.flow_loss(METHOD, "cpu")
