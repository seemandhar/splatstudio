import math

import pytest
import torch

import ss_gaussianspa as G
from ss_gaussianspa import core, spa
from ss_gaussianspa.losses import create_penalty


def _comp(m, name):
    return next(c for c in m.components if c.name == name)


def test_methods_and_presets():
    m = G.method()
    assert m.name == "gaussianspa"
    assert [c.kind for c in m.components] == ["renderer", "density", "loss", "loss"]
    assert {f.name for f in m.fields} == {"spa_z", "spa_u"}
    assert all(0 < f.learning_rate <= 1e-12 for f in m.fields)
    d = _comp(m, "spa-density").config
    assert (d["criterion"], d["prune_ratio1"], d["prune_ratio2"]) == ("opacity", 0.5, 0.8)
    assert d["imp_metric"] == "indoor" and d["spa_start"] == 15200 and d["spa_stop"] == 25200
    assert _comp(m, "spa-penalty").config["stop_inclusive"] is True
    imp = G.importance()
    assert _comp(imp, "spa-density").config["criterion"] == "importance"
    assert _comp(imp, "spa-penalty").config["stop_inclusive"] is False
    b = _comp(G.preset_bicycle(), "spa-density").config
    assert (b["criterion"], b["prune_ratio1"], b["prune_ratio2"]) == ("importance", 0.5, 0.72)
    c = _comp(G.preset_counter(), "spa-density").config
    assert (c["criterion"], c["prune_ratio1"], c["prune_ratio2"]) == ("opacity", 0.6, 0.5)
    assert G.preset_garden().name == "gaussianspa-garden"
    for k in ("spa_start", "spa_stop", "spa_interval"):
        assert _comp(m, "spa-penalty").config[k] == d[k]


def test_project_opacity_matches_official_rule():
    v = torch.tensor([0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.0])
    z = spa.project_opacity(v, 0.8)  # index 8, thr = sorted[7] = 0.7 -> keep > 0.7
    assert torch.equal(z != 0, v > 0.7)
    assert torch.allclose(z[v > 0.7], v[v > 0.7])


def test_project_importance_matches_official_rule():
    v = torch.arange(10.0) + 1
    imp = torch.tensor([5.0, 1, 9, 3, 7, 2, 8, 4, 6, 0])
    z = spa.project_importance(v, imp, 0.5)  # thr = sorted[4] = 4 -> zero imp < 4
    assert torch.equal(z == 0, imp < 4)
    assert torch.equal(z[imp >= 4], v[imp >= 4])


def test_admm_step_multiplier():
    a = torch.tensor([0.9, 0.05, 0.6, 0.02])
    u = torch.tensor([0.0, 0.1, -0.5, 0.0])
    z, u2 = spa.admm_step(a, u, 0.5)  # v = [0.9, 0.15, 0.1, 0.02]; keep > 0.1
    assert torch.allclose(z, torch.tensor([0.9, 0.15, 0.0, 0.0]))
    assert torch.allclose(u2, u + a - z)


def test_penalty_value_and_schedule():
    spec = _comp(G.method(), "spa-penalty")
    loss = create_penalty(spec)
    logit = torch.nn.Parameter(torch.zeros(4))
    params = {"opacities": logit, "spa_z": torch.zeros(4), "spa_u": torch.full((4,), 0.5)}
    v = loss(params, {"step": 15250}, None)  # a = 0.5, r = 1.0
    assert math.isclose(float(v), 0.5 * 5e-4 * 4.0, rel_tol=1e-6)
    params["spa_z"].requires_grad_(True)
    params["spa_u"].requires_grad_(True)
    v = loss(params, {"step": 15250}, None)
    v.backward()
    assert params["spa_z"].grad is None and params["spa_u"].grad is None  # z, u detached
    assert torch.allclose(logit.grad, torch.full((4,), 5e-4 * 1.0 * 0.25))
    for s in (15200, 15251, 25250, 100):
        assert float(loss(params, {"step": s}, None)) == 0.0
    assert float(loss(params, {"step": 25200}, None)) > 0
    imp_loss = create_penalty(_comp(G.importance(), "spa-penalty"))
    assert float(imp_loss(params, {"step": 25200}, None)) == 0.0


def test_windows():
    assert not spa.update_active(15200, 15200, 25200, 50)
    assert spa.update_active(15250, 15200, 25200, 50)
    assert spa.update_active(25200, 15200, 25200, 50)
    assert not spa.update_active(25250, 15200, 25200, 50)
    assert spa.penalty_active(25200, 15200, 25200, 50, True)
    assert not spa.penalty_active(25200, 15200, 25200, 50, False)
    assert sum(spa.update_active(s, 15200, 25200, 50) for s in range(1, 40001)) == 200


def test_hard_prune_masks():
    a = torch.tensor([0.3, 0.9, 0.1, 0.7, 0.5])
    rm = spa.hard_prune_mask(0.6, opacity=a)  # keep top int(0.4*5)=2
    assert torch.equal(~rm, torch.tensor([False, True, False, True, False]))
    imp = torch.tensor([3.0, 1.0, 4.0, 0.0, 2.0])
    rm = spa.hard_prune_mask(0.4, importance=imp)  # thr = sorted[1] = 1 -> remove <= 1
    assert torch.equal(rm, imp <= 1.0)
    with pytest.raises(ValueError):
        spa.hard_prune_mask(0.5)


def test_sampling_rule_differs_from_mini_splatting():
    imp = torch.tensor([0.0, 0.0, 0.0, 5.0, 3.0, 0.0, 0.0, 1.0])
    assert spa.sampling_count(8, 0.5) == 4
    assert core.sampling_count(imp, 0.5) == 1  # Mini-Splatting int(N*0.5*nnz/N)
    w = spa.sampling_weights(imp)
    assert (w > 0).all()
    idx = core.weighted_sample(w, 6, replace=False, generator=torch.Generator().manual_seed(0))
    assert len(idx) == 6 and len(set(idx.tolist())) == 6


def test_density_rejects_bad_config():
    from dataclasses import replace

    from ss_gaussianspa.density import create_density

    spec = _comp(G.method(), "spa-density")
    with pytest.raises(ValueError):
        create_density(replace(spec, config={**spec.config, "criterion": "l1"}))
    with pytest.raises(ValueError):
        create_density(replace(spec, config={**spec.config, "spa_start": 14000}))
    with pytest.raises(ValueError):
        create_density(replace(spec, config={**spec.config, "simp_iteration2": 20000}))
