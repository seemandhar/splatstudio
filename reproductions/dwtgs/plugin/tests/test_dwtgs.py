import torch

from ss_dwtgs import method, method_lf, probes
from ss_dwtgs.losses import hf_active, hf_loss, lf_loss, sh_degree_at
from ss_dwtgs.wavelet import haar_dwt, ll_pyramid


def test_haar_shapes_and_constant():
    x = torch.ones(3, 378, 504)
    ll, _lh, _hl, hh = haar_dwt(x)
    assert ll.shape == (3, 189, 252)
    assert torch.allclose(ll, torch.full_like(ll, 1.0))  # averaging Haar: LL = block mean
    assert float(hh.abs().max()) == 0.0
    lv = ll_pyramid(x, 2)
    assert lv[1].shape == (3, 95, 126)
    assert torch.allclose(lv[1][:, :94], torch.full_like(lv[1][:, :94], 1.0))
    assert torch.allclose(haar_dwt(x, "orthonormal")[0], torch.full_like(ll, 2.0))


def test_losses_zero_on_identical():
    x = torch.rand(3, 16, 16)
    assert float(lf_loss(x, x, 0.5, 2)) == 0.0
    assert float(hf_loss(torch.ones(3, 16, 16))) == 0.0


def test_schedule():
    assert hf_active(10, 10, 1, 5000) and hf_active(5000, 10, 1, 5000)
    assert not hf_active(5010, 10, 1, 5000) and not hf_active(11, 10, 1, 5000)
    assert sh_degree_at(1) == 0 and sh_degree_at(1001) == 1 and sh_degree_at(9999) == 3


def test_cpu_probes():
    m = method()
    for fn in (probes.haar_dwt_probe, probes.lf_loss_probe, probes.hf_schedule_probe,
               probes.pseudo_poses_probe):
        fn(m, "cpu")


def test_variants():
    names = {c.name for c in method().components}
    assert {"dwtgs-lf", "dwtgs-hf", "photometric", "dropgaussian-renderer"} <= names
    assert "dwtgs-hf" not in {c.name for c in method_lf().components}


def test_probes_reject_orthonormal():
    import dataclasses

    import pytest

    m = method()
    comps = tuple(dataclasses.replace(c, config={**c.config, "haar_norm": "orthonormal"})
                  if c.name in ("dwtgs-lf", "dwtgs-hf") else c for c in m.components)
    bad = dataclasses.replace(m, components=comps)
    for fn in (probes.haar_dwt_probe, probes.lf_loss_probe):
        with pytest.raises(AssertionError):
            fn(bad, "cpu")
    comps = tuple(dataclasses.replace(c, config={**c.config, "novel_views": "fsgs_llff"})
                  if c.name == "dwtgs-hf" else c for c in m.components)
    with pytest.raises(AssertionError):
        probes.pseudo_poses_probe(dataclasses.replace(m, components=comps), "cpu")
