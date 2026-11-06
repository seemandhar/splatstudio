import pytest
import torch

import ss_robustsplat
from ss_robustsplat import probes
from ss_robustsplat.density import DEFAULTS, refine_steps, reset_steps
from ss_robustsplat.losses import MLP_DEFAULTS, mlp_active

METHODS = [ss_robustsplat.method(), ss_robustsplat.method_nodg(), ss_robustsplat.method_nomb()]


@pytest.mark.parametrize("method", METHODS, ids=lambda m: m.name)
@pytest.mark.parametrize("probe", ["delayed_growth_probe", "mask_supervision_probe",
                                   "masked_photometric_probe", "mlp_schedule_probe"])
def test_cpu_probes(method, probe):
    getattr(probes, probe)(method, "cpu")


def test_paper_schedule():
    grow = refine_steps(DEFAULTS, 30000)
    assert grow[0] == 10100 and grow[-1] == 19900 and len(grow) == 99
    assert reset_steps(DEFAULTS, 30000) == [15000, 18000]
    assert MLP_DEFAULTS["bootstrap_iter"] == 20000
    assert not mlp_active(15100, MLP_DEFAULTS) and mlp_active(15300, MLP_DEFAULTS)


def test_config_rejects_unknown():
    from splatstudio import ComponentSpec

    from ss_robustsplat.density import Density
    with pytest.raises(ValueError):
        Density(ComponentSpec("d", "density", "x:y", (), (), {"bogus": 1}))


def test_cosine_target_range():
    from ss_robustsplat.masks import cosine_target
    f = torch.randn(384, 5, 5)
    t = cosine_target(f, -f, 10, 10)
    assert float(t.max()) == 0.0
    assert float(cosine_target(f, f, 10, 10).min()) > 0.999
