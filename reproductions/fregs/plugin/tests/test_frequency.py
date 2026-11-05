import math
from dataclasses import replace

import pytest
import torch

import ss_fregs
from ss_fregs import probes
from ss_fregs.defaults import DEFAULTS, THRESHOLD_GRID
from ss_fregs.frequency import band_radius, create_frequency_loss, parse_config, radius_grid

VARIANTS = [ss_fregs.method, ss_fregs.fregs_fr, ss_fregs.fregs_thr0002, ss_fregs.calib_g0003,
            ss_fregs.calib_g0004, ss_fregs.calib_g0005, ss_fregs.calib_g0006]


def _with(method, **cfg):
    comps = tuple(
        replace(c, config={**c.config, **cfg}) if c.name == "fregs-frequency" else c
        for c in method.components
    )
    return replace(method, components=comps)


@pytest.mark.parametrize("factory", VARIANTS)
def test_specs_valid(factory):
    from splatstudio.contracts import validate

    m = factory()
    assert validate(m) == []
    assert m.citations == ("https://arxiv.org/abs/2403.06908",)


def test_variant_configs():
    density = {m.name: next(c for c in m.components if c.kind == "density").config
               for m in (f() for f in VARIANTS)}
    assert density["fregs"]["grow_grad2d"] == DEFAULTS["density"]["grow_grad2d"]
    assert density["fregs-thr0002"]["grow_grad2d"] == 0.0002
    for g in THRESHOLD_GRID:
        assert density[f"fregs-calib-g{round(g * 1e4):04d}"]["grow_grad2d"] == g
    fr = next(c for c in ss_fregs.fregs_fr().components if c.name == "fregs-frequency")
    assert fr.config["anneal"] is False


@pytest.mark.parametrize("factory", [ss_fregs.method, ss_fregs.fregs_fr])
def test_oracle(factory):
    probes.freq_oracle(factory(), "cpu")


@pytest.mark.parametrize("factory", [ss_fregs.method, ss_fregs.fregs_fr])
def test_schedule(factory):
    probes.freq_schedule(factory(), "cpu")


def test_threshold():
    ev = probes.threshold(ss_fregs.method(), "cpu")
    assert ev["fregs_growth"] == 0 and ev["base_3dgs_growth"] > 0


def test_oracle_normalized_units():
    probes.freq_oracle(_with(ss_fregs.method(), radius_units="normalized"), "cpu")
    probes.freq_schedule(_with(ss_fregs.method(), radius_units="normalized"), "cpu")


def test_oracle_detects_wrong_loss():
    # Removing the phase term (a wrong mechanism) must make the oracle probe fail.
    import ss_fregs.frequency as fq

    original = fq.discrepancy
    fq.discrepancy = lambda a, b, m, c: (original(a, b, m, c)[0], a.real.sum() * 0)
    try:
        with pytest.raises(AssertionError):
            probes.freq_oracle(ss_fregs.method(), "cpu")
    finally:
        fq.discrepancy = original


def test_schedule_detects_no_annealing():
    # An annealed spec whose loss ignores the band (sees r* at t=1) must fail.
    with pytest.raises(AssertionError):
        probes.freq_schedule(_with(ss_fregs.method(), t0=0, t_end=1), "cpu")


@pytest.mark.parametrize("hw", [(23, 37), (24, 40), (31, 32)])
def test_odd_sizes_and_grid(hw):
    h, w = hw
    r, big_d = radius_grid(h, w, "cpu")
    assert r[h // 2, w // 2] == 0
    assert math.isclose(big_d, math.hypot(h // 2, w // 2))
    cfg = parse_config(DEFAULTS["frequency"])
    assert band_radius(cfg["t0"], cfg, big_d) == cfg["d0_frac"] * big_d
    assert band_radius(cfg["t_end"], cfg, big_d) == big_d
    loss = create_frequency_loss(replace(next(
        c for c in ss_fregs.method().components if c.name == "fregs-frequency")))
    x = torch.rand(h, w, 3, requires_grad=True)
    value = loss({}, {"rgb": x, "step": 5000}, torch.rand(h, w, 3))
    assert value.ndim == 0 and torch.isfinite(value)
    (g,) = torch.autograd.grad(value, x)
    assert torch.isfinite(g).all() and torch.count_nonzero(g)


def test_config_rejects_unknown_and_missing():
    with pytest.raises(ValueError):
        parse_config({**DEFAULTS["frequency"], "bogus": 1})
    cfg = dict(DEFAULTS["frequency"])
    cfg.pop("w_l")
    with pytest.raises(ValueError):
        parse_config(cfg)


def test_fr_per_band_weights_match_oracle():
    # fregs-fr applies w_l to r <= D0 and w_h to r > D0 from step 1 (deconfounded ablation).
    m = _with(ss_fregs.fregs_fr(), w_l=3e-6, w_h=5e-7)
    probes.freq_oracle(m, "cpu")


def test_threshold_probe_fails_when_not_raised():
    m = ss_fregs.method()
    comps = tuple(
        replace(c, config={**c.config, "grow_grad2d": 0.0002}) if c.name == "fregs-density" else c
        for c in m.components
    )
    with pytest.raises(AssertionError):
        probes.threshold(replace(m, components=comps), "cpu")
    assert "fregs-threshold" in [p.name for p in m.probes]
    assert "fregs-threshold" in [p.name for p in ss_fregs.fregs_fr().probes]
    assert "fregs-threshold" not in [p.name for p in ss_fregs.fregs_thr0002().probes]


def test_blinding_refuses_metrics(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "calibrate_threshold.py"
    spec = importlib.util.spec_from_file_location("calibrate_threshold", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    (tmp_path / "test.json").write_text("{}")
    with pytest.raises(PermissionError):
        mod.read_log(tmp_path / "test.json")
    hashes = mod.blind(tmp_path)
    assert (tmp_path / "test.json").exists() and len(hashes) == 1
    assert (tmp_path / "BLINDED.json").exists()


def test_solve_raises_when_unidentifiable():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "calibrate_weights.py"
    spec = importlib.util.spec_from_file_location("calibrate_weights", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(ValueError):
        mod.solve(lambda w: 1.5 + w, 1.2)  # already above target at w = 0
    with pytest.raises(ValueError):
        mod.solve(lambda w: 1.0, 1.2, w_max=1.0)  # never reaches target
    # Non-monotone (dip) ratio: the first upward crossing is bracketed.
    w = mod.solve(lambda w: 1.0 + (w - 1e-3) ** 2 * 1e4 - 1e-2, 1.5)
    assert abs((1.0 + (w - 1e-3) ** 2 * 1e4 - 1e-2) - 1.5) < 1e-3


def test_equalized_w_h():
    from ss_fregs.gradstats import equalized_w_h

    assert equalized_w_h(2e-6, {"low": 30.0, "high": 120.0}) == pytest.approx(5e-7)
    with pytest.raises(ValueError):
        equalized_w_h(2e-6, {"low": 30.0, "high": 0.0})


def test_default_phase_is_wrapped():
    assert DEFAULTS["frequency"]["phase"] == "wrapped"
    probes.freq_oracle(_with(ss_fregs.method(), phase="raw"), "cpu")  # raw still available


@pytest.mark.parametrize("spec_phase,impl_phase", [("wrapped", "raw"), ("raw", "wrapped")])
def test_oracle_detects_phase_mixup(spec_phase, impl_phase):
    import ss_fregs.frequency as fq

    original = fq.discrepancy
    fq.discrepancy = lambda a, b, m, c: original(a, b, m, {**c, "phase": impl_phase})
    try:
        with pytest.raises(AssertionError):
            probes.freq_oracle(_with(ss_fregs.method(), phase=spec_phase), "cpu")
    finally:
        fq.discrepancy = original


def test_wrapped_phase_bounded_and_differs_from_raw():
    from ss_fregs.frequency import discrepancy, spectrum

    g = torch.Generator().manual_seed(1)
    x, y = (torch.rand(24, 40, 3, generator=g, dtype=torch.float64) for _ in range(2))
    a, b = spectrum(x), spectrum(y)
    mask = torch.ones(24, 40, dtype=torch.bool)
    base = parse_config(DEFAULTS["frequency"])
    _, p_wrapped = discrepancy(a, b, mask, base)
    _, p_raw = discrepancy(a, b, mask, {**base, "phase": "raw"})
    n_bins = 3 * 24 * 40
    assert p_wrapped * math.sqrt(24 * 40) <= math.pi * n_bins + 1e-9
    assert p_raw > p_wrapped * 1.01
