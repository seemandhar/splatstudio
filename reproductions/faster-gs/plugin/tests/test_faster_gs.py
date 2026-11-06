import math

import torch
from torch import nn

from ss_faster_gs import method
from ss_faster_gs.density import (
    DEFAULTS,
    adaptive_density_control,
    adc_masks,
    permute_gaussians,
    quat_to_rotmat,
    schedule,
)
from ss_faster_gs.morton import morton_codes, morton_order
from ss_faster_gs.probes import _morton_py


def test_method_contract():
    from splatstudio.contracts import validate

    m = method()
    assert validate(m) == []
    assert {c.kind for c in m.components} == {"renderer", "density", "loss"}
    density = next(c for c in m.components if c.kind == "density")
    assert density.config["lr_opacities"] == 0.025 and density.config["fused_adam"] is True


def test_schedule_matches_official_callbacks():
    adc, morton, reset = schedule(DEFAULTS)
    assert adc[0] == 600 and adc[-1] == 14900 and len(adc) == 144
    assert morton == [0, 5000, 10000, 15000]
    assert reset == [3000, 6000, 9000, 12000]


def test_morton_codes_match_bit_interleave():
    g = torch.Generator().manual_seed(0)
    p = torch.randn(200, 3, generator=g) * torch.tensor([3.0, 1.0, 0.5])
    codes = morton_codes(p)
    lo, size = p.amin(0), float((p.amax(0) - p.amin(0)).max())
    ref = torch.tensor([_morton_py(x, lo, size) for x in p])
    assert torch.equal(codes, ref)
    order = morton_order(p)
    assert torch.all(codes[order][1:] >= codes[order][:-1])


def test_quat_to_rotmat_orthonormal_and_scale_invariant():
    q = torch.randn(10, 4)
    R = quat_to_rotmat(q)
    torch.testing.assert_close(R @ R.transpose(1, 2), torch.eye(3).expand(10, 3, 3), atol=1e-5,
                               rtol=0)
    torch.testing.assert_close(quat_to_rotmat(3 * q), R)


def _params(n=20):
    g = torch.Generator().manual_seed(1)
    vals = {"means": torch.randn(n, 3, generator=g),
            "scales": (torch.rand(n, 3, generator=g) * 0.2 + 0.001).log(),
            "quats": torch.randn(n, 4, generator=g),
            "opacities": torch.randn(n, generator=g),
            "sh0": torch.randn(n, 1, 3, generator=g), "shN": torch.randn(n, 15, 3, generator=g)}
    params = nn.ParameterDict({k: nn.Parameter(v) for k, v in vals.items()})
    opts = {k: torch.optim.Adam([p], lr=1e-3, eps=1e-15) for k, p in params.items()}
    for p in params.values():
        p.grad = torch.randn_like(p)
    for o in opts.values():
        o.step()
    return params, opts


def test_adc_masks_rule():
    params, _ = _params()
    info = torch.stack([torch.arange(20.0) % 4, torch.full((20,), 5e-4)])
    dup, split = adc_masks(params, info, DEFAULTS, extent=5.0)
    avg = info[1] / info[0].clamp_min(1)
    small = params["scales"].max(1).values <= math.log(0.05)
    assert torch.equal(dup, (avg >= 2e-4) & small)
    assert torch.equal(split, (avg >= 2e-4) & ~small)


def test_adc_keeps_optimizer_aligned_on_cpu():
    from splatstudio.state import check_state

    params, opts = _params()
    info = torch.stack([torch.ones(20), torch.linspace(0, 6e-4, 20)])
    ev = adaptive_density_control(params, opts, info, DEFAULTS, 5.0, True)
    check_state(method(), params, opts)
    assert ev["gaussians"] == len(params["means"])


def test_permute_moves_moments():
    params, opts = _params()
    m = opts["sh0"].state[params["sh0"]]["exp_avg"].clone()
    order = torch.randperm(20)
    permute_gaussians(params, opts, order)
    torch.testing.assert_close(opts["sh0"].state[params["sh0"]]["exp_avg"], m[order])
