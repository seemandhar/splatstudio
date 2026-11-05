import math

import torch

from ss_dofgs import method, probes
from ss_dofgs.bank import aperture_from_raw, focal_from_raw, get_2d_emb
from ss_dofgs.density import expon_lr


def test_camera_init_and_schedule():
    probes.camera_init(method(), "cpu")


def test_iln_and_mask_losses():
    probes.iln_mask_losses(method(), "cpu")


def test_objective_schedule():
    probes.objective_schedule(method(), "cpu")


def test_density_recipe_cpu():
    probes.density_recipe(method(), "cpu")


def test_reparameterisation_clamps():
    assert abs(float(focal_from_raw(torch.tensor(-1.0))) - 0.1) < 1e-6
    assert float(aperture_from_raw(torch.tensor(-5.0))) == 101.0
    assert float(aperture_from_raw(torch.tensor(1.0))) == 200.0


def test_positional_encoding_shape():
    pe = get_2d_emb(1, 8, 12, 16, "cpu")
    assert pe.shape == (1, 8, 12, 16) and torch.isfinite(pe).all()


def test_xyz_lr_schedule():
    assert math.isclose(expon_lr(0, 5e-4, 5e-6, max_steps=80000), 5e-4)
    assert math.isclose(expon_lr(80000, 5e-4, 5e-6, max_steps=80000), 5e-6)


def test_coc_constant():
    assert math.isclose(probes.C_COC, 0.36067376022224085, rel_tol=1e-12)


# -- per-scene official configuration (review-1 major + minor 5) ------------------------------
import pathlib  # noqa: E402

import pytest  # noqa: E402

from ss_dofgs import method_bl_m001, method_nobl  # noqa: E402
from ss_dofgs.variants import (  # noqa: E402
    NOT_PORTED,
    OFFICIAL_REAL,
    SCENE_VARIANT,
    SHARED_KEYS,
    SHARED_VALUES,
    VARIANTS,
)

CONFIGS = pathlib.Path(__file__).resolve().parents[2] / "source" / "code" / "config_files"
FACTORIES = {"dofgs": method, "dofgs-bl-m001": method_bl_m001, "dofgs-nobl": method_nobl}


def _yaml(scene):
    yaml = pytest.importorskip("yaml")
    path = CONFIGS / f"config_real_{scene}.yaml"
    if not path.exists():
        pytest.skip(f"official config missing: {path}")
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("scene", sorted(OFFICIAL_REAL))
def test_transcription_matches_official_yaml(scene):
    cfg = _yaml(scene)
    assert (cfg["band_limit"], cfg["lambda_mask_loss"]) == OFFICIAL_REAL[scene]
    for key, val in SHARED_VALUES.items():
        theirs = cfg[key]
        if isinstance(val, list):
            assert list(theirs) == val, key
        else:
            assert float(theirs) == float(val), key  # yaml 1.1 reads '5e-4' as str
    assert set(cfg) == set(SHARED_KEYS) | set(NOT_PORTED), set(cfg) ^ (set(SHARED_KEYS)
                                                                    | set(NOT_PORTED))


def test_every_scene_has_a_variant_and_entry_point():
    assert set(SCENE_VARIANT) == set(OFFICIAL_REAL)
    assert set(SCENE_VARIANT.values()) == set(VARIANTS) == set(FACTORIES)
    from importlib.metadata import entry_points
    eps = {e.name: e.value for e in entry_points(group="splatstudio.methods")}
    for name in VARIANTS:
        assert name in eps, f"entry point {name} not installed"


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_variant_pinned_to_official(name):
    m = FACTORIES[name]()
    assert m.name == name
    out = probes.official_config(m, "cpu")
    assert out["band_limit"] is VARIANTS[name][0]
    assert out["lambda_mask_loss"] == VARIANTS[name][1]
    assert math.isclose(out["lambda_reg"], 0.1 * VARIANTS[name][1])
    probes.iln_mask_losses(m, "cpu")


def test_plain_reset_cpu():
    from ss_dofgs.density import Density, plain_reset_logits
    o = torch.linspace(-8, 5, 50)
    ref = torch.sigmoid(o).clamp(max=0.01)
    assert torch.allclose(plain_reset_logits(o), torch.log(ref / (1 - ref)), atol=1e-4)
    d = Density(probes._Spec({"band_limit": False}))
    p = torch.nn.ParameterDict({"scales": torch.nn.Parameter(torch.full((50, 3), -3.0)),
                                "opacities": torch.nn.Parameter(o.clone()),
                                "filter_3d": torch.nn.Parameter(torch.full((50,), 0.5))})
    d.reset_opacity(p, {k: torch.optim.Adam([v]) for k, v in p.items()})
    assert torch.allclose(p["opacities"], plain_reset_logits(o))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel")
def test_band_limit_switch_cuda():
    probes.band_limit_switch(method_nobl(), "cuda")


def test_overflowed_adam_moment_is_frozen_like_official():
    """|g| > 1.8e19 overflows exp_avg_sq to +inf. Official (unchecked torch Adam): the element
    is frozen for good (m / inf = 0). The port pins it to a finite sentinel that keeps the same
    freeze, leaves other elements bit-identical and passes the runtime's moment check."""
    from ss_dofgs.density import clamp_overflowed_moments

    g = torch.Generator().manual_seed(0)
    init = torch.randn(64, 4, generator=g)
    grads = [torch.randn(64, 4, generator=g) * 1e-3 for _ in range(300)]
    grads[100] = grads[100].clone()
    grads[100][7, 2] = 1e21  # (1 - beta2) g**2 = inf in float32
    for t in range(101, 300):  # needle keeps receiving huge-but-finite grads
        grads[t] = grads[t].clone()
        grads[t][7, 2] = 1e17 * (1 if t % 3 else -1)

    def run(sanitize):
        p = torch.nn.Parameter(init.clone())
        opt = torch.optim.Adam([p], lr=1e-3, eps=1e-15)
        for t, gr in enumerate(grads):
            p.grad = gr.clone()
            opt.step()
            if sanitize:
                clamp_overflowed_moments({"quats": opt})
                st = opt.state[p]
                assert torch.isfinite(st["exp_avg_sq"]).all() and torch.isfinite(st["exp_avg"]).all()
            if t == 100:
                at_overflow = p.detach().clone()
        return p.detach(), at_overflow, opt.state[p]["exp_avg_sq"]

    official, off_at, v_off = run(False)
    ported, port_at, v_port = run(True)
    assert torch.isinf(v_off[7, 2]) and torch.isfinite(v_port).all()
    assert official[7, 2] == off_at[7, 2]  # official: frozen exactly
    # port: frozen to float noise (~1e-4 lr per step at |g| = 1e17; 199 steps here)
    assert abs(float(ported[7, 2] - port_at[7, 2])) < 199 * 1e-4 * 1e-3
    mask = torch.ones_like(official, dtype=torch.bool)
    mask[7, 2] = False
    assert torch.equal(official[mask], ported[mask])  # everything else bit-identical


def test_overflow_freeze_leaves_nan_moments():
    from ss_dofgs.density import clamp_overflowed_moments

    p = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.Adam([p], lr=1e-3)
    p.grad = torch.tensor([float("nan"), 1.0, 1.0])
    opt.step()
    assert clamp_overflowed_moments({"x": opt}) == 0
    assert torch.isnan(opt.state[p]["exp_avg_sq"][0])  # genuine NaN still fails check_state


# Gaussian 60079 of the cake run at step 27874 (instrumented reproduction of the benchmark
# failure "quats: invalid optimizer moment exp_avg_sq"), with its training camera (view 17).
NEEDLE = {
    "means": [-54.11765670776367, -109.65916442871094, 276.64739990234375],
    "scales": [8.447319984436035, 0.13965633511543274, 0.34074166417121887],
    "quats": [0.6730867624282837, 0.29312944412231445, -0.06852071732282639, -0.28250086307525635],
    "opacities": -3.703866720199585, "filter_3d": 0.27469080686569214,
    "sh0": [2.5075466632843018, 1.7953451871871948, 1.6030408143997192],
    "viewmat": [[0.9988856911659241, 9.767385199666023e-05, -0.04719509184360504, -0.5387605428695679],
                [-0.001721088425256312, 0.999407947063446, -0.034359272569417953, 1.8944599628448486],
                [0.04716380313038826, 0.03440207242965698, 0.9982945322990417, 1.4828492403030396],
                [0.0, 0.0, 0.0, 1.0]],
    "fx": 417.066650390625, "F": 63.4488639831543, "K": 252.88949584960938,
}


def test_needle_gaussian_overflows_adam_and_freeze_keeps_state_valid_cuda():
    """Official kernel, float32: for a needle (4663 x 1.2 x 1.4 units at depth 277) the 2D
    covariance entries are ~2.6e7, a*c - b^2 cancels to rounding noise (ulp ~6.7e7 vs true det
    2.3e8) and the backward divides by a clamped ~0 determinant: |dL/dq| jumps from ~1e-3 to
    >1e19 under 1e-4 changes of one log-scale. Adam's exp_avg_sq then overflows to +inf."""
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from ss_dofgs.density import clamp_overflowed_moments
    from ss_dofgs.renderer import render_dof

    dev = "cuda"
    cam = {"K": torch.tensor([[NEEDLE["fx"], 0, 300.0], [0, NEEDLE["fx"], 200.0], [0, 0, 1]],
                             device=dev),
           "viewmat": torch.tensor(NEEDLE["viewmat"], device=dev), "width": 600, "height": 400}
    F = torch.tensor([[NEEDLE["F"]]], device=dev)
    K = torch.tensor([[NEEDLE["K"]]], device=dev)
    worst = None
    for k in range(-10, 11):
        p = {"means": torch.tensor([NEEDLE["means"]], device=dev),
             "scales": torch.tensor([NEEDLE["scales"]], device=dev),
             "quats": torch.tensor([NEEDLE["quats"]], device=dev),
             "opacities": torch.tensor([NEEDLE["opacities"]], device=dev),
             "filter_3d": torch.tensor([NEEDLE["filter_3d"]], device=dev),
             "sh0": torch.tensor([[NEEDLE["sh0"]]], device=dev),
             "shN": torch.zeros(1, 15, 3, device=dev)}
        p["scales"][0, 0] += k * 2e-5
        p = {n: torch.nn.Parameter(v) for n, v in p.items()}
        out = render_dof(p, cam, 0, torch.zeros(3, device=dev), F, K, 0)
        w = torch.rand(out["rgb"].shape, generator=torch.Generator(device=dev).manual_seed(0),
                       device=dev) - 0.5
        (out["rgb"] * w * 1e-5).sum().backward()
        g = float(p["quats"].grad.abs().max())
        if worst is None or g > worst[0]:
            worst = (g, p)
    g, p = worst
    assert g > 1.9e19, g  # overflow class reached by the unmodified official kernel
    opt = torch.optim.Adam([p["quats"]], lr=0.001, eps=1e-15)
    opt.step()
    v = opt.state[p["quats"]]["exp_avg_sq"]
    assert torch.isinf(v).any()  # what the runtime rejected (state.py check_state)
    before = p["quats"].detach().clone()
    assert torch.isfinite(before).all()  # official: m / inf = 0 -> parameter unchanged
    assert clamp_overflowed_moments({"quats": opt}) > 0
    assert torch.isfinite(v).all() and torch.isfinite(opt.state[p["quats"]]["exp_avg"]).all()
