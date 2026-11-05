import math

import torch

from ss_atomgs import method
from ss_atomgs.density import expon_lr, quantile_linear
from ss_atomgs.losses import ms_ssim, ssim_same
from ss_atomgs.probes import (
    atom_proliferation,
    atomize,
    edge_aware_normal,
    init_schedules,
    ms_ssim_probe,
    normal_from_depth,
)


def test_cpu_probes():
    m = method()
    assert atomize(m, "cpu")["decay"] == 0.99
    out = atom_proliferation(m, "cpu")
    assert out["step_200"]["split"] == 6 and out["step_7000"]["split"] == 4
    init_schedules(m, "cpu")
    edge_aware_normal(m, "cpu")
    ms_ssim_probe(m, "cpu")
    normal_from_depth(m, "cpu")


def test_ssim_identity():
    a = torch.rand(3, 24, 20)
    assert math.isclose(ssim_same(a, a).item(), 1.0, rel_tol=1e-5)
    assert math.isclose(ms_ssim(a, a).item(), 1.0, rel_tol=1e-5)


def test_schedules_and_quantile():
    assert expon_lr(0, 1.6e-4, 1.6e-6, max_steps=7000) == 1.6e-4
    assert math.isclose(expon_lr(7000, 1.6e-4, 1.6e-6, max_steps=7000), 1.6e-6, rel_tol=1e-9)
    x = torch.rand(1000)
    assert abs(quantile_linear(x, 0.01) - torch.quantile(x, 0.01).item()) < 1e-7


def test_spec_defaults():
    cfg = {c.name: c.config for c in method().components}
    d = cfg["atomgs-density"]
    assert d["opacity_reset_interval"] == 300 and d["opacity_reset_value"] == 0.1
    assert d["clone_threshold"] == 0.0002 and d["prune_threshold"] == 0.005
    assert d["position_lr_max_steps"] == 7000 and d["atom_init_quantile"] == 0.01
    assert cfg["edge-aware-normal"] == {"lambda_normal": 0.1, "until": 7000, "edge_power": 500}
    assert cfg["atomgs-photometric"]["lambda_ms_ssim"] == 0.1
    assert cfg["atomgs-raster"]["background"] == [0.0, 0.0, 0.0]
