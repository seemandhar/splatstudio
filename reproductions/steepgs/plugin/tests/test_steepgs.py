import torch

from ss_steepgs import core, method
from ss_steepgs.probes import saddle_criterion, schedule, split_direction, splitting_matrix


def test_inverse_covariance_matches_inverse():
    torch.manual_seed(0)
    s = torch.rand(5, 3) * 0.2 + 0.05
    q = torch.randn(5, 4)
    R = core.quat_to_rotmat(q)
    cov = R @ torch.diag_embed(s ** 2) @ R.transpose(1, 2)
    inv = core.inverse_covariance(s.log(), q)
    assert torch.allclose(inv @ cov, torch.eye(3).expand(5, 3, 3), atol=1e-3)


def test_split_offsets_sign_pattern_follows_repeat_rows():
    v = torch.tensor([[0.0, 0, 1], [0.0, 1, 0]])
    off = core.split_offsets(v, torch.full((2, 3), 0.1))
    # rows: [p0, p1, p0, p1]; signs +, -, +, - -> p0 both +, p1 both -
    assert off[0, 2] > 0 and off[2, 2] > 0 and off[1, 1] < 0 and off[3, 1] < 0


def test_select_requires_negative_threshold():
    import pytest

    with pytest.raises(ValueError):
        core.select((), torch.ones(1), torch.ones(1), 2e-4, 0.0)


def test_averaged_zero_denominator():
    out = core.averaged(torch.zeros(2, 3, 3), torch.tensor([0.0, 2.0]))
    assert torch.isfinite(out).all() and out.abs().sum() == 0


def test_cpu_probes():
    m = method()
    assert splitting_matrix(m, "cpu")["rel_err"] < 1e-4
    assert saddle_criterion(m, "cpu")["default"]["selected"] == 1
    assert "clone_shift" in split_direction(m, "cpu")
    assert schedule(m, "cpu")["first"] == 600
