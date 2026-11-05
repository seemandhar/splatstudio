import pytest
import torch

from ss_dropgaussian import method, method_nodrop, probes
from ss_dropgaussian.renderer import compensation, drop_rate


def test_drop_rate_official():
    assert drop_rate(10000, 0.2, 10000) == pytest.approx(0.2)
    assert drop_rate(1, 0.2, 10000) == pytest.approx(2e-5)


def test_compensation_values():
    torch.manual_seed(0)
    m = compensation(1000, 0.25, "cpu")
    vals = torch.unique(m)
    assert torch.all((vals == 0) | torch.isclose(vals, torch.tensor(4 / 3)))
    with pytest.raises(ValueError):
        compensation(4, 1.0, "cpu")


@pytest.mark.parametrize(
    "probe",
    [probes.drop_compensation, probes.drop_schedule, probes.official_densification,
     probes.loss_weights],
)
def test_cpu_probes(probe):
    probe(method(), "cpu")


def test_nodrop_variant():
    m = method_nodrop()
    r = next(c for c in m.components if c.name == "dropgaussian-renderer")
    assert r.config["gamma"] == 0.0
    assert "drop-compensation" not in {p.name for p in m.probes}
    probes.official_densification(m, "cpu")
