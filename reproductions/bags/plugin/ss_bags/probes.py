"""Mechanism probes for the BAGS port. Each raises AssertionError when the mechanism is removed
or wrong and returns its measurements."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from . import bank as bank_mod
from .blur import photometric, resize_target, spatial_conv, ssim_same
from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import Density, densify_and_prune
from .filters import filter_coefficient, filtered_scales_opacities
from .losses import (
    bpn_forward,
    create_depth_tv,
    create_mask_loss,
    create_objective,
    create_rgb_tv,
)
from .renderer import Renderer, render_sharp, scaled_camera
from .schedule import expon_lr, resolution_divisor, step_actions
from .variants import GENERIC, OFFICIAL, SMOKE, scene_config

SH_C0 = 0.28209479177387814


class _Spec:
    def __init__(self, config):
        self.config = config


def _cfg(method, name):
    return dict(next(c for c in method.components if c.name == name).config)


def _camera(device, W=64, H=48, f=60.0, tx=0.0):
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=device)
    V = torch.eye(4, device=device)
    V[0, 3] = tx
    return {"K": K, "viewmat": V, "width": W, "height": H}


def _bank(method, device, cameras=None):
    cams = cameras or [_camera(device, tx=0.1 * i) for i in range(3)]
    return bank_mod.BlurBank(cams, _cfg(method, "bags-density"), device)


# --- C1: spatially varying convolution ------------------------------------------------------
def shift_add_reference(image, weights, K):
    """Independent float64 implementation of Eq. 1 (no Unfold): explicit zero padding and one
    shifted copy per kernel tap; tap k = ky*K + kx reads (y + ky - r, x + kx - r)."""
    C, H, W = image.shape
    r = K // 2
    pad = torch.zeros(C, H + 2 * r, W + 2 * r, dtype=image.dtype, device=image.device)
    pad[:, r:r + H, r:r + W] = image
    out = torch.zeros_like(image)
    for ky in range(K):
        for kx in range(K):
            out = out + weights[ky * K + kx][None] * pad[:, ky:ky + H, kx:kx + W]
    return out


def spatial_conv_reference(method, device="cpu"):
    ev = {}
    g = torch.Generator().manual_seed(0)
    for K in (5, 9, 17):
        H, W = 13, 19
        img = torch.rand(3, H, W, generator=g, dtype=torch.float64).to(device).requires_grad_()
        logits = torch.randn(K * K, H, W, generator=g, dtype=torch.float64).to(device)
        wts = torch.softmax(logits, 0).requires_grad_()
        up = torch.randn(3, H, W, generator=g, dtype=torch.float64).to(device)
        out = spatial_conv(img, wts, K)
        gi, gw = torch.autograd.grad((out * up).sum(), (img, wts))
        img2, wts2 = img.detach().clone().requires_grad_(), wts.detach().clone().requires_grad_()
        ref = shift_add_reference(img2, wts2, K)
        ri, rw = torch.autograd.grad((ref * up).sum(), (img2, wts2))
        e = [float((out - ref).abs().max()), float((gi - ri).abs().max()),
             float((gw - rw).abs().max())]
        assert max(e) < 1e-10, (K, e)
        assert float((out - img).abs().max()) > 1e-3, "convolution is an identity"
        ev[f"K{K}_max_err_value_gimg_gkernel"] = e
    # impulse / orientation: one-hot tap (ky0, kx0) everywhere
    K, H, W, py, px, ky0, kx0 = 5, 9, 11, 4, 5, 0, 3
    img = torch.zeros(1, H, W, dtype=torch.float64)
    img[0, py, px] = 1.0
    wts = torch.zeros(K * K, H, W, dtype=torch.float64)
    wts[ky0 * K + kx0] = 1.0
    out = spatial_conv(img, wts, K)
    y, x = (out[0] == 1).nonzero()[0].tolist()
    assert (y + ky0 - K // 2, x + kx0 - K // 2) == (py, px), (y, x)
    assert float(out.sum()) == 1.0
    delta = torch.zeros(K * K, H, W, dtype=torch.float64)
    delta[(K // 2) * K + K // 2] = 1.0
    r = torch.rand(3, H, W, dtype=torch.float64)
    assert torch.equal(spatial_conv(r, delta, K), r), "centre-tap kernel must be identity"
    # gradcheck (float64, small)
    a = torch.rand(2, 5, 6, dtype=torch.float64, requires_grad=True)
    b = torch.softmax(torch.randn(9, 5, 6, dtype=torch.float64), 0).requires_grad_()
    assert torch.autograd.gradcheck(lambda u, v: spatial_conv(u, v, 3), (a, b))
    ev.update(impulse_output_at=[y, x], gradcheck=True)
    return ev


# --- C1: BPN architecture -------------------------------------------------------------------
def _pe_reference(H, W, ch=16):
    n = ch // 4  # frequencies per axis (out_ch = 8 -> 4)
    f = 1.0 / (10000 ** (np.arange(0, 2 * n, 2) / (2 * n)))
    pe = np.zeros((H, W, ch))
    for y in range(H):
        for x in range(W):
            for j in range(n):
                ay, ax = 2 * np.pi * y / H * f[j], 2 * np.pi * x / W * f[j]
                pe[y, x, 2 * j], pe[y, x, 2 * j + 1] = np.sin(ay), np.cos(ay)
                pe[y, x, 2 * n + 2 * j], pe[y, x, 2 * n + 2 * j + 1] = np.sin(ax), np.cos(ax)
    return pe


def _fake_outputs(bank, step, H=12, W=16, view=1, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    rgb = torch.rand(H, W, 3, generator=g).to(device).requires_grad_()
    depth = (1.0 + 3 * torch.rand(H, W, 1, generator=g)).to(device).requires_grad_()
    return {"rgb": rgb, "depth": depth, "step": step,
            "bags": {"view": view, "bank": bank, "divisor": 1}}


def bpn_architecture(method, device="cpu"):
    bank = _bank(method, device)
    ms = bank.ms
    ks = bank.ms.ks
    assert ks == (5, 9, 17), ks
    shapes = {n: tuple(p.shape) for n, p in ms.named_parameters()}
    assert shapes["mlp_base1.0.weight"] == (64, 80, 1, 1)
    assert shapes["mlp_head1.weight"] == (25, 64, 1, 1)
    assert shapes["mlp_head2.weight"] == (81, 64, 1, 1)
    assert shapes["mlp_head3.weight"] == (289, 64, 1, 1)
    assert shapes["embedding_camera.weight"] == (3, 32)
    assert shapes["conv_rgbd.6.weight"] == (32, 64, 3, 3)
    assert not any(n.startswith("mlp_") and n.endswith("bias") for n in shapes), "MLP has bias"
    H, W = 12, 16
    pe = bank.pos_enc(H, W)[0].cpu().numpy()
    assert np.abs(pe - _pe_reference(H, W)).max() < 1e-5
    ev = {}
    for level, K in ((1, 5), (2, 9), (3, 17)):
        w, m = ms(0, bank.pos_enc(H, W), torch.rand(1, 4, H, W, device=device), level)
        assert w.shape == (1, K * K, H, W) and m.shape == (1, 1, H, W)
        assert float((w.sum(1) - 1).abs().max()) < 1e-5 and 0 < float(m.min()) < float(m.max()) < 1
        ev[f"level{level}_K"] = K
    step = bank.cfg["upsample_steps"][1] + 300  # full resolution, K1 head, BPN active
    out = _fake_outputs(bank, step, H, W, device=device)
    res = bpn_forward(out)
    assert res["K"] == 17
    rgb = out["rgb"].permute(2, 0, 1)
    blend = res["mask"] * spatial_conv(rgb, res["kernel"], 17) + (1 - res["mask"]) * rgb
    assert float((blend - res["blur"]).abs().max()) < 1e-6
    gk = torch.autograd.grad(res["kernel"].sum() + res["mask"].sum(), [out["rgb"], out["depth"]],
                             allow_unused=True)
    assert all(t is None or float(t.abs().max()) == 0 for t in gk), "BPN input not detached"
    gb = torch.autograd.grad(res["blur"].sum(), out["rgb"])[0]
    assert float(gb.abs().max()) > 0
    ev["blend_err"] = float((blend - res["blur"]).abs().max())
    return ev


# --- C1: objective terms --------------------------------------------------------------------
def _tv_numpy(a):
    a = np.asarray(a, dtype=np.float64)  # [C,H,W]
    C, H, W = a.shape
    h = ((a[:, 1:, :] - a[:, :-1, :]) ** 2).sum() / (C * (H - 1) * W)
    w = ((a[:, :, 1:] - a[:, :, :-1]) ** 2).sum() / (C * H * (W - 1))
    return 2 * (h + w)


def objective_terms(method, device="cpu"):
    bank = _bank(method, device)
    c = bank.cfg
    ms, start = c["ms_steps"], c["bpn_start"]
    obj = create_objective(_Spec(_cfg(method, "bags-objective")))
    mask_l = create_mask_loss(_Spec({"use_mask_loss": True, "mask_loss_alpha": 0.001}))
    dtv = create_depth_tv(_Spec({"use_depth_loss": True, "depth_loss_alpha": 0.01}))
    rtv = create_rgb_tv(_Spec({"use_rgbtv_loss": True, "rgbtv_loss_alpha": 0.001}))
    H, W = 12, 16
    target = torch.rand(2 * H, 2 * W, 3, device=device)
    gt = resize_target(target, H, W)
    assert gt.shape == (3, H, W)
    ev = {}
    cases = ((start, False), (start + 1, True), (ms, True), (ms + start, False),
             (ms + start + 1, True))
    if start >= c["total_steps"]:  # -nobpn control: the BPN never shapes the loss
        cases = tuple((s, False) for s in (1, 251, ms, ms + 251, c["total_steps"]))
        ev["bpn_disabled"] = True
    for step, expect in cases:
        out = _fake_outputs(bank, step, H, W, device=device)
        loss = float(obj(None, out, target))
        sharp = float(photometric(out["rgb"].permute(2, 0, 1), gt))
        if not expect:
            assert abs(loss - sharp) < 1e-6, (step, loss, sharp)
            assert float(mask_l(None, out, target)) == 0.0
            assert "bpn" not in out["bags"], "BPN evaluated while inactive"
            continue
        res = out["bags"]["bpn"]
        assert abs(loss - float(photometric(res["blur"], gt))) < 1e-6
        assert abs(loss - sharp) > 1e-6, "loss ignores the BPN output"
        m = res["mask"].detach().cpu().numpy()
        assert abs(float(mask_l(None, out, target)) - 0.001 * m.mean()) < 1e-8
        d = out["depth"].detach().cpu().numpy()[..., 0]
        dn = (d - d.min()) / (d - d.min()).max()
        assert abs(float(dtv(None, out, target)) - 0.01 * _tv_numpy(dn[None])) < 1e-7
        r = out["rgb"].detach().cpu().numpy().transpose(2, 0, 1)
        assert abs(float(rtv(None, out, target)) - 0.001 * _tv_numpy(r)) < 1e-7
        ev[f"step{step}"] = {"loss": loss, "sharp_loss": sharp}
    # SSIM sanity: identical images -> 1
    x = torch.rand(3, H, W, device=device)
    assert abs(float(ssim_same(x, x)) - 1) < 1e-6
    return ev


# --- C2: multi-scale rendering --------------------------------------------------------------
def _gaussians(device, n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    means = torch.empty(n, 3)
    means[:, 0] = (torch.rand(n, generator=g) - 0.5) * 1.6
    means[:, 1] = (torch.rand(n, generator=g) - 0.5) * 1.2
    means[:, 2] = 2.0 + torch.rand(n, generator=g) * 3.0
    colors = 0.1 + torch.rand(n, 3, generator=g) * 0.8
    p = {"means": means, "scales": torch.log(0.03 + 0.08 * torch.rand(n, 3, generator=g)),
         "quats": F.normalize(torch.randn(n, 4, generator=g), dim=-1),
         "opacities": torch.logit(0.1 + 0.6 * torch.rand(n, generator=g)),
         "sh0": ((colors - 0.5) / SH_C0)[:, None], "shN": torch.zeros(n, 15, 3),
         "filter_3d": torch.full((n,), 0.002)}
    return {k: v.to(device) for k, v in p.items()}


def multiscale_schedule(method, device="cuda"):
    rcfg = _cfg(method, "bags-raster")
    up = rcfg["upsample_steps"]
    renderer = Renderer(_Spec(rcfg))
    cams = [_camera(device, tx=0.05 * i) for i in range(3)]
    bank = _bank(method, device, cams)
    params = _gaussians(device)
    prev = bank_mod.ACTIVE
    bank_mod.ACTIVE = bank
    ev = {}
    try:
        for step, (Hs, Ws, K) in ((1, (12, 16, 5)), (up[0] - 1, (12, 16, 5)),
                                  (up[0], (24, 32, 9)), (up[1] - 1, (24, 32, 9)),
                                  (up[1], (48, 64, 17))):
            bank.step = step - 1
            out = renderer.render(params, cams[1], 1)
            assert out["rgb"].shape == (Hs, Ws, 3), (step, out["rgb"].shape)
            assert out["bags"]["view"] == 1
            if not bank.cfg["use_another_mlp"] or step <= bank.cfg["ms_steps"]:
                assert bank.net_for(step)[2] == K, (step, bank.net_for(step))
            f = 60.0 * Ws / 64
            Kx = torch.tensor([[f, 0, Ws / 2], [0, f, Hs / 2], [0, 0, 1]], device=device)
            ref = render_sharp(params, {"K": Kx, "viewmat": cams[1]["viewmat"], "width": Ws,
                                        "height": Hs}, 1, torch.zeros(3, device=device))
            assert float((ref["rgb"] - out["rgb"]).abs().max()) < 1e-5
            ev[f"step{step}"] = [Hs, Ws, K]
        bank.step = 0
        test_cam = _camera(device, tx=0.5)
        out = renderer.render(params, test_cam, 1)
        assert out["rgb"].shape == (48, 64, 3) and out["bags"]["view"] is None
        with torch.no_grad():
            out = renderer.render(params, cams[1], 1)
        assert out["rgb"].shape == (48, 64, 3) and out["bags"]["view"] is None
    finally:
        bank_mod.ACTIVE = prev
    # depth = alpha-blended Euclidean distance: single off-axis Gaussian -> D / alpha == |mu - c|
    one = {k: v[:1].clone() for k, v in _gaussians(device, 1).items()}
    one["means"] = torch.tensor([[0.6, -0.4, 2.0]], device=device)
    o = render_sharp(one, _camera(device), 0, torch.zeros(3, device=device))
    a, d = o["alpha"][..., 0], o["depth"][..., 0]
    sel = a > 1e-3
    ratio = d[sel] / a[sel]
    dist, z = float(one["means"].norm()), 2.0
    assert float((ratio - dist).abs().max()) < 1e-3 * dist, (float(ratio.mean()), dist)
    assert abs(dist - z) > 0.1
    ev["depth_over_alpha"] = float(ratio.mean())
    ev["euclidean_distance"] = dist
    return ev


# --- C2: official loop clocks ---------------------------------------------------------------
def official_loop(cfg):
    """Independent transcription of the official train.py control flow (no plugin code)."""
    out = {"densify": set(), "reset": set(), "filter": set(), "bpn": set(), "lr": {},
           "dgt": {}}
    ms, total = cfg["ms_steps"], cfg["total_steps"]
    for ori in range(1, total + 1):
        it = ori - ms if ori > ms else ori
        out["lr"][ori] = (it, )
        if it > cfg["bpn_start"]:  # official: 250; control variants disable the BPN
            out["bpn"].add(ori)
        if ori == total:  # saved before the densification block of the last iteration
            continue
        if it < cfg["densify_until_iter"]:
            if it > cfg["densify_from_iter"] and it % 100 == 0:
                out["densify"].add(ori)
                out["filter"].add(ori)
                out["dgt"][ori] = cfg["init_dgt"] if ori <= ms else 0.0002
            if it % 3000 == 0:
                out["reset"].add(ori)
        if it % 100 == 0 and it > cfg["densify_until_iter"]:
            if it < (total - ms) - 100:
                out["filter"].add(ori)
    return out


def phase_schedule(method, device="cpu"):
    from .schedule import bpn_active, densify_thresholds

    cfg = _cfg(method, "bags-density")
    ref = official_loop(cfg)
    got = {"densify": set(), "reset": set(), "filter": set(), "bpn": set()}
    for s in range(1, cfg["total_steps"] + 1):
        a = step_actions(s, cfg)
        for k in ("densify", "reset", "filter"):
            if a[k]:
                got[k].add(s)
        if bpn_active(s, cfg["ms_steps"], cfg["bpn_start"]):
            got["bpn"].add(s)
        if a["densify"]:
            assert abs(densify_thresholds(s, cfg)[0] - ref["dgt"][s]) < 1e-12, s
    for k in got:
        assert got[k] == ref[k], (k, sorted(got[k] ^ ref[k])[:10])
    d = Density(_Spec(cfg))
    d.extent = 2.0
    ms = cfg["ms_steps"]
    lr_a, lr_b = d.xyz_lr(ms), d.xyz_lr(ms + 1)
    assert abs(lr_b - expon_lr(1, 1.6e-4 * 2, 1.6e-6 * 2, 0, 0.01, 60000)) < 1e-12
    assert lr_b > lr_a, "xyz LR does not restart after ms_steps"
    assert resolution_divisor(cfg["upsample_steps"][0] - 1, cfg["upsample_steps"]) == 4
    assert resolution_divisor(cfg["upsample_steps"][1], cfg["upsample_steps"]) == 1
    return {"densify_steps": len(got["densify"]), "resets": sorted(got["reset"])[:8],
            "filter_steps": len(got["filter"]), "bpn_off_steps": cfg["total_steps"] - len(got["bpn"]),
            "lr_before_after_restart": [lr_a, lr_b]}


# --- base: density recipe -------------------------------------------------------------------
def density_recipe(method, device="cuda"):
    from splatstudio.state import check_state

    cfg = _cfg(method, "bags-density")
    n = 10
    p = _gaussians(device, n, seed=3)
    p["scales"] = torch.log(torch.tensor([0.005, 0.03] * 5, device=device))[:, None].expand(n, 3) \
        .contiguous()
    p["opacities"] = torch.logit(torch.tensor([0.5, 0.5, 0.5, 0.05, 0.5, 0.05, 0.05, 0.5, 0.5, 0.5],
                                                  device=device))
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in p.items()})
    opts = {k: torch.optim.Adam([v], lr=1e-3, eps=1e-15) for k, v in params.items()}
    grads = torch.tensor([3e-4, 3e-4, 1e-5, 1e-5, 3e-4, 3e-4, 0, 0, 0, 0], device=device)
    extent, thr, opa = 2.0, 2e-4, 0.1  # percent_dense 0.01 * 2 = 0.02 separates 0.005 / 0.03
    before = {k: v.detach().clone() for k, v in params.items()}
    big = (before["scales"].exp().max(1).values > 0.01 * extent)
    sel = grads >= thr
    exp_clone, exp_split = int((sel & ~big).sum()), int((sel & big).sum())
    # survivors: originals not split + clones (2) + 2 per split; prune on sigmoid(o) < 0.1
    keep_o = before["opacities"].sigmoid() >= opa
    expected = int((keep_o & ~(sel & big)).sum()) + int((keep_o & sel & ~big).sum()) \
        + 2 * int((keep_o & sel & big).sum())
    counts = densify_and_prune(params, opts, grads, thr, opa, extent, 0.01)
    assert counts["clone"] == exp_clone and counts["split"] == exp_split, counts
    assert len(params["means"]) == expected, (len(params["means"]), expected)
    assert float(params["opacities"].sigmoid().min()) >= opa
    s_split = before["scales"][sel & big & keep_o][0, 0]
    assert torch.isclose(params["scales"][:, 0], s_split - math.log(1.6)).any(), "split scale"

    class _M:
        fields = (type("F", (), {"name": "filter_3d", "shape": ()})(),)
    check_state(_M, params, opts)
    # NDC gradient accumulation
    d = Density(_Spec(cfg))
    d._reset_stats(4, device)
    means2d = torch.zeros(1, 4, 2, device=device, requires_grad=True)
    means2d.grad = torch.tensor([[[1.0, 2.0], [0.5, 0.0], [3.0, 3.0], [1.0, 1.0]]], device=device)
    radii = torch.tensor([[[1, 1], [2, 2], [0, 0], [3, 3]]], device=device)
    d.accumulate({"means2d": means2d, "radii": radii, "width": 100, "height": 50})
    exp_acc = [math.hypot(50, 50), 25.0, 0.0, math.hypot(50, 25)]
    assert np.allclose(d.stats["grad_accum"].cpu().numpy(), exp_acc)
    assert d.stats["denom"].tolist() == [1, 1, 0, 1]
    # filter-aware reset and current-scale 3D filter
    d.params = params
    d.cameras = [_camera(device, tx=0.05 * i) for i in range(3)]
    d.update_filter(params, cfg["upsample_steps"][1])
    f_full = params["filter_3d"].detach().clone()
    d.update_filter(params, 1)
    f_coarse = params["filter_3d"].detach().clone()
    assert torch.allclose(f_coarse, 4 * f_full, rtol=1e-5), "filter ignores the training scale"
    _, op_before = filtered_scales_opacities(params["scales"], params["opacities"],
                                             params["filter_3d"])
    d.reset_opacity(params, opts)
    _, op_after = filtered_scales_opacities(params["scales"], params["opacities"],
                                            params["filter_3d"])
    assert torch.allclose(op_after, torch.minimum(op_before, torch.tensor(0.01, device=device)),
                          atol=1e-6)
    assert float(filter_coefficient(params["scales"], params["filter_3d"]).min()) < 1
    return {"counts": counts, "population": len(params["means"]),
            "filter_ratio_coarse_full": float((f_coarse / f_full).mean())}


# --- official configuration -----------------------------------------------------------------
ARGUMENT_DEFAULTS = {  # source/code/arguments/__init__.py OptimizationParams / ModelParams
    "densify_grad_threshold": 0.0002, "densify_from_iter": 500, "densify_until_iter": 35000,
    "densification_interval": 100, "opacity_reset_interval": 3000, "percent_dense": 0.01,
    "position_lr_init": 0.00016, "position_lr_final": 0.0000016, "position_lr_delay_mult": 0.01,
    "position_lr_max_steps": 60000, "init_opacity": -1.0, "kernel_sizes": [5, 9, 17],
    "kernel_size_ss": 17,
}


def official_config(method, device="cpu"):
    dens = _cfg(method, "bags-density")
    for k, v in ARGUMENT_DEFAULTS.items():
        assert dens[k] == v, (k, dens[k], v)
    name = method.name
    if name == "bags":
        v = dict(GENERIC)
    elif name == "bags-smoke":
        v = dict(SMOKE)
    else:
        scene = name.removeprefix("bags-").removesuffix("-nobpn")
        assert scene in OFFICIAL
        v = scene_config(scene)
    nobpn = name.endswith("-nobpn")
    assert dens["bpn_start"] == (dens["total_steps"] if nobpn else 250), dens["bpn_start"]
    v.setdefault("min_opacity", 0.1)
    for key, comp, opt in (("init_dgt", "bags-density", "init_dgt"),
                           ("iterations", "bags-density", "total_steps"),
                           ("ms_steps", "bags-density", "ms_steps"),
                           ("use_another_mlp", "bags-density", "use_another_mlp"),
                           ("bpn_lr", "bags-density", "bpn_lr"),
                           ("min_opacity", "bags-density", "min_opacity"),
                           ("use_depth_loss", "bags-depth-tv", "use_depth_loss"),
                           ("depth_loss_alpha", "bags-depth-tv", "depth_loss_alpha"),
                           ("use_rgbtv_loss", "bags-rgb-tv", "use_rgbtv_loss")):
        assert _cfg(method, comp)[opt] == v[key], (key, _cfg(method, comp)[opt], v[key])
    assert _cfg(method, "bags-mask") == {"use_mask_loss": True, "mask_loss_alpha": 0.001}
    assert _cfg(method, "bags-objective") == {"lambda_dssim": 0.2}
    r = _cfg(method, "bags-raster")
    assert r["kernel_size"] == 0.1 and r["background"] == [0.0, 0.0, 0.0]
    assert r["upsample_steps"] == dens["upsample_steps"]
    assert set(DENSITY_DEFAULTS) >= set(dens)
    return {"method": name, "iterations": dens["total_steps"], "ms_steps": dens["ms_steps"],
            "init_dgt": dens["init_dgt"]}


__all__ = ["scaled_camera"]
