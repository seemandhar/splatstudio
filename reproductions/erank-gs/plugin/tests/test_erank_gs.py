import numpy as np
import torch

from ss_erank_gs import method, method_3dgs, method_adc, probes
from ss_erank_gs.losses import DEFAULTS, effective_rank, erank_terms


def test_effective_rank_limits():
    s = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1e-6], [1.0, 1e-6, 1e-6]], dtype=torch.float64)
    er = effective_rank(s)
    assert abs(er[0] - 3) < 1e-9 and abs(er[1] - 2) < 1e-6 and abs(er[2] - 1) < 1e-6


def test_terms_match_numpy():
    rng = np.random.default_rng(0)
    s = rng.uniform(0.01, 1, (50, 3))
    b, t = erank_terms(torch.tensor(s), DEFAULTS)
    ref = probes._np_loss(s, DEFAULTS["erank_lambda"], DEFAULTS["thin_lambda"], DEFAULTS["eps"])
    assert abs((b + t).item() - ref) < 1e-10


def test_cpu_probes():
    m = method()
    probes.erank_formula(m, "cpu")
    probes.adc_abs_metric(m, "cpu")
    probes.density_recipe(m, "cpu")
    probes.density_recipe(method_3dgs(), "cpu")
    probes.density_recipe(method_adc(), "cpu")


def test_variants():
    names = {c.name for c in method_adc().components}
    assert "erank" not in names
    cfg = next(c for c in method_3dgs().components if c.name == "erank-density").config
    assert cfg["abs_gradient"] is False and cfg["min_opacity"] == 0.005
