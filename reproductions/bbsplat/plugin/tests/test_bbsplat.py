import math
from pathlib import Path

import torch

from ss_bbsplat import method, method_300k, method_tnt, probes
from ss_bbsplat import texture as tex
from ss_bbsplat.density import expon_lr, noise_coefficient


def test_gaussian_table_matches_png():
    import numpy as np
    from PIL import Image

    png = Path(tex.__file__).with_name("alpha_init_gaussian_small.png")
    im = np.array(Image.open(png))
    if im.ndim == 3:
        im = im[..., 0]
    assert im.tolist() == [list(r) for r in tex.GAUSSIAN_PNG]


def test_safe_logit_caps_infinities():
    a = torch.tensor([0.0, 0.5, 1.0])
    out = tex.safe_logit(a)
    assert torch.isfinite(out).all() and out[1] == 0 and out[0] == -tex.LOGIT_CAP
    assert torch.equal(out.sigmoid(), torch.tensor([0.0, 0.5, 1.0]))


def test_quantize_round_trip():
    g = torch.Generator().manual_seed(0)
    a, c = torch.rand(3, 16, 16, generator=g), torch.rand(3, 3, 16, 16, generator=g)
    qa, qc = tex.quantize(a, c)
    assert float((qa - a).abs().max()) <= 1 / 255 + 1e-6
    assert float((qc - c).abs().max()) <= 0.5 / 255 + 1e-6


def test_expon_lr_and_noise():
    assert math.isclose(expon_lr(0, 1.6e-4, 1.6e-6, 30000), 1.6e-4)
    assert math.isclose(expon_lr(30000, 1.6e-4, 1.6e-6, 30000), 1.6e-6)
    assert noise_coefficient() == 0.0


def test_cpu_probes():
    m = method()
    assert probes.texture_regularizer(m)["value"] > 0
    assert probes.mcmc_relocation(m)["relocated"] == 7
    assert probes.init_sky_fps(m)["population"] == 250
    assert all(probes.schedule(m).values())


def test_variants_budgets():
    caps = {}
    for m in (method(), method_tnt(), method_300k()):
        d = next(c for c in m.components if c.kind == "density")
        caps[m.name] = (d.config["cap_max"], d.config["max_read_points"])
    assert caps == {"bbsplat": (60000, 50000), "bbsplat-tnt": (30000, 20000),
                    "bbsplat-300k": (300000, 290000)}


def test_sky2k_variant_and_official_xyz_lr():
    from splatstudio import ComponentSpec

    from ss_bbsplat import method_tnt_sky2k
    from ss_bbsplat.density import Density

    d = next(c for c in method_tnt_sky2k().components if c.kind == "density").config
    assert (d["cap_max"], d["max_read_points"], d["sky_points"]) == (30000, 28000, 2000)
    off = Density(ComponentSpec("d", "density", "x", (), (), {"xyz_lr_after_freeze": "official"}))
    zero = Density(ComponentSpec("d", "density", "x", (), (), {}))
    assert off.learning_rates(30001)["means"] == 0 and zero.learning_rates(30001)["means"] == 0
    assert math.isclose(off.learning_rates(30002)["means"], 1.6e-6)
    assert zero.learning_rates(30002)["means"] == 0
    assert off.learning_rates(30002)["sh0"] == 0 and off.learning_rates(32000)["means"] == 0
