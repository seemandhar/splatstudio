import inspect
import re
from pathlib import Path

import numpy as np
import pytest
import torch

import ss_bags
from ss_bags import evaluate as ev
from ss_bags import probes
from ss_bags.evaluate import quantize, ssim_official, ssim_skimage
from ss_bags.variants import OFFICIAL, scene_config

SCRIPTS = Path(__file__).resolve().parents[2] / "source" / "code" / "scripts"
METHODS = ["bags", "bags-smoke", "bags-ball", "bags-caps", "bags-cake", "bags-tools",
           "bags-ball-nobpn", "bags-caps-nobpn"]


def _m(name):
    return ss_bags.method() if name == "bags" else (
        ss_bags.method_smoke() if name == "bags-smoke"
        else getattr(ss_bags, f"method_{name.removeprefix('bags-').replace('-', '_')}")())


def test_spatial_conv():
    probes.spatial_conv_reference(_m("bags"), "cpu")


@pytest.mark.parametrize("name", ["bags", "bags-cake"])
def test_bpn_and_objective(name):
    probes.bpn_architecture(_m(name), "cpu")
    probes.objective_terms(_m(name), "cpu")


@pytest.mark.parametrize("name", METHODS)
def test_schedule_and_config(name):
    probes.phase_schedule(_m(name), "cpu")
    probes.official_config(_m(name), "cpu")


def test_density_recipe_cpu():
    probes.density_recipe(_m("bags-ball"), "cpu")


@pytest.mark.skipif(not SCRIPTS.exists(), reason="official code not downloaded")
def test_variants_match_official_scripts():
    for scene in OFFICIAL:
        prefix = "blur" if scene_config(scene)["dataset"] == "motion" else "defocus"
        text = (SCRIPTS / f"{prefix}{scene}.sh").read_text()
        v = scene_config(scene)

        def arg(name, cast=float, text=text):
            m = re.search(rf"--{name} ([0-9.e-]+)", text)
            return cast(m.group(1)) if m else None

        assert int(re.search(r"llffhold=(\d+)", text).group(1)) == v["llffhold"]
        assert arg("init_dgt") == v["init_dgt"]
        assert arg("iterations", int) == v["iterations"]
        assert arg("ms_steps", int) == v["ms_steps"]
        assert arg("min_opacity") == 0.1
        assert ("--use_depth_loss" in text) == v["use_depth_loss"]
        assert (arg("depth_loss_alpha") or 0.001) == v["depth_loss_alpha"]
        assert ("--use_rgbtv_loss" in text) == v["use_rgbtv_loss"]
        assert ("--use_another_mlp" in text) == v["use_another_mlp"]
        kl = re.search(r"KERLR=([0-9.e-]+)", text)
        assert (float(kl.group(1)) if kl else 5e-4) == v["bpn_lr"]


def test_skimage_ssim_identity_and_quantize():
    x = torch.rand(20, 24, 3)
    assert abs(ssim_skimage(x, x) - 1) < 1e-9
    assert ssim_skimage(x, x.flip(0)) < 0.5
    q = quantize(torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(q * 255, torch.tensor([0.0, 128.0, 255.0]))


def _ssim_numpy(a, b, R, win=7):
    """Independent skimage-0.19-style SSIM: explicit window loop, sample covariance."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    H, W, C = a.shape
    c1, c2 = (0.01 * R) ** 2, (0.03 * R) ** 2
    vals = []
    for c in range(C):
        for y in range(H - win + 1):
            for x in range(W - win + 1):
                p, q = a[y:y + win, x:x + win, c].ravel(), b[y:y + win, x:x + win, c].ravel()
                ux, uy = p.mean(), q.mean()
                vx, vy = p.var(ddof=1), q.var(ddof=1)
                vxy = ((p - ux) * (q - uy)).sum() / (p.size - 1)
                vals.append(((2 * ux * uy + c1) * (2 * vxy + c2))
                            / ((ux**2 + uy**2 + c1) * (vx + vy + c2)))
    return float(np.mean(vals))


def test_official_ssim_convention():
    # constant images: SSIM = luminance term on the [-1, 1] values with data_range 2
    a, b = torch.full((9, 10, 3), 0.2), torch.full((9, 10, 3), 0.6)
    x, y, c1 = -0.6, 0.2, (0.01 * 2) ** 2
    expected = (2 * x * y + c1) / (x * x + y * y + c1)
    assert abs(ssim_official(a, b) - expected) < 1e-6
    # the old [0, 1] / data_range 1 convention gives a different value (review-1 major)
    assert abs(ssim_skimage(a, b) - expected) > 0.5
    # non-zero-mean random pair vs an independent window loop on [-1, 1], R = 2
    g = torch.Generator().manual_seed(0)
    p = 0.3 + 0.5 * torch.rand(12, 13, 3, generator=g)
    q = (p + 0.1 * torch.randn(12, 13, 3, generator=g)).clamp(0, 1)
    ref = _ssim_numpy((p * 2 - 1).numpy(), (q * 2 - 1).numpy(), 2.0)
    assert abs(ssim_official(p, q) - ref) < 1e-9
    assert abs(ssim_skimage(p, q) - ref) > 1e-3
    # the evaluator's headline SSIM uses the official convention
    assert '"ssim": ssim_official(q, gt)' in inspect.getsource(ev.metrics)


def test_nobpn_control():
    from splatstudio.runtime import specification_id

    from ss_bags.losses import active

    ball, ctl = _m("bags-ball"), _m("bags-ball-nobpn")
    cfg = probes._cfg(ctl, "bags-density")
    assert cfg["bpn_start"] == cfg["total_steps"] == 49000
    bank = probes._bank(ctl, "cpu")
    for step in (1000, 9000, 9300, 49000):
        out = probes._fake_outputs(bank, step)
        assert not active(out)
    # identical to bags-ball except the BPN start
    b = probes._cfg(ball, "bags-density")
    assert {k: v for k, v in b.items() if k != "bpn_start"} == \
        {k: v for k, v in cfg.items() if k != "bpn_start"}
    assert specification_id(ball) != specification_id(ctl)
