"""Executable mechanism checks for c3dgs (each fails if its mechanism is removed or wrong).

Scene / camera helpers follow the pup-3dgs plugin probes (reproductions/pup-3dgs/plugin/
ss_pup3dgs/probes.py).
"""

import io
import math
import os
import tempfile
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F


def _require_cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ProbeUnavailable("c3dgs probes need CUDA (gsplat, official VQ kernel)")


def _scene(n, device, seed, sh_rest=0.3):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.empty(n, 3).uniform_(-0.6, 0.6, generator=g)
    means[:, 2] = torch.empty(n).uniform_(-0.3, 0.3, generator=g)
    params = {
        "means": means,
        "scales": torch.empty(n, 3).uniform_(math.log(0.06), math.log(0.2), generator=g),
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.logit(torch.empty(n).uniform_(0.2, 0.6, generator=g)),
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.8,
        "shN": torch.randn(n, 15, 3, generator=g) * sh_rest,
    }
    return {k: v.to(device) for k, v in params.items()}


def _camera(device, angle=0.15, width=96, height=64, cx=47.3, cy=33.1, f=80.0, dist=3.0):
    """OpenCV world-to-camera looking at the origin from `dist`, rotated about y by `angle`."""
    c, s = math.cos(angle), math.sin(angle)
    R = torch.tensor([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    viewmat = torch.eye(4)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = torch.tensor([0.05, -0.03, dist])
    K = torch.tensor([[f, 0.0, cx], [0.0, f * 1.02, cy], [0.0, 0.0, 1.0]])
    return {"K": K.to(device), "viewmat": viewmat.to(device), "width": width, "height": height}


def _close(a, b, name, rtol=1e-4, atol=1e-7):
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        err = float((a - b).abs().max())
        raise AssertionError(f"{name}: max abs error {err:.3e}")
    return float((a - b).abs().max())


# ---------------------------------------------------------------------------------------------
def sensitivity_oracle(method, device):
    """|dE/dsh| per view = W_g Y_k(d_g) (E linear in colours; W_g = total blend weight of g from a
    one-hot multi-channel render), summed as absolute values over views / total pixels; shape
    sensitivity = eta^2 |dE/dSigma| of an independent full-covariance leaf."""
    _require_cuda(device)
    from gsplat import rasterization, spherical_harmonics

    from . import sensitivity, vq
    from .quant import fake_quantizers, half_ste

    n = 14
    p = _scene(n, device, 3)
    p["sh0"][:4] = -3.0  # negative colours: a clamped evaluation would give zero gradient
    p["means"][-1] = torch.tensor([0.0, 40.0, 0.0], device=device)  # outside every view
    cams = [_camera(device, a) for a in (0.0, 2.6)]  # opposite sides: Y_1 changes sign
    q = fake_quantizers(device)
    color, shape, pixels = sensitivity.compute(p, q, cams)

    # independent oracle
    q2 = fake_quantizers(device)
    s_hat, log_eta = sensitivity.reparameterize(p)
    with torch.no_grad():
        cov_hat = vq.covariance(q2["scaling"](F.normalize(F.relu(s_hat))),
                                F.normalize(q2["rotation"](p["quats"])))
        eta2 = torch.exp(q2["scaling_factor"](log_eta)).square()
        opac = q2["opacity"](torch.sigmoid(p["opacities"])[:, None])[:, 0]
    m = half_ste(p["means"]).detach()
    eye = torch.eye(16, device=device)
    oracle_c = torch.zeros(n, 16, 3, device=device)
    oracle_c_signed = torch.zeros(n, 16, 3, device=device)
    oracle_s = torch.zeros(n, 6, device=device)
    for cam in cams:
        full = (cov_hat * eta2).clone().requires_grad_(True)
        with torch.no_grad():
            r, _, _ = rasterization(m, None, None, opac, torch.eye(n, device=device),
                                    cam["viewmat"][None], cam["K"][None], cam["width"],
                                    cam["height"], sh_degree=None, packed=False,
                                    covars=vq.to_full_cov(full))
            W = r[0].sum(dim=(0, 1))  # [n] total blending weight per Gaussian
            dirs = m - sensitivity.camera_center(cam)[None]
            Y = torch.stack([spherical_harmonics(3, dirs, eye[k][None, :, None].expand(n, 16, 3)
                                                 .contiguous())[:, 0] for k in range(16)], 1)
        g = (W[:, None] * Y)[:, :, None].expand(n, 16, 3)
        oracle_c += g.abs()
        oracle_c_signed += g
        sh = torch.cat((q2["features_dc"](p["sh0"]), q2["features_rest"](p["shN"])), 1)
        colors = spherical_harmonics(3, dirs, sh) + 0.5
        r, _, _ = rasterization(m, None, None, opac, colors, cam["viewmat"][None], cam["K"][None],
                                cam["width"], cam["height"], sh_degree=None, packed=False,
                                covars=vq.to_full_cov(full))
        (gf,) = torch.autograd.grad(r.sum(), full)
        oracle_s += (gf * eta2).abs()
    oracle_c = oracle_c.flatten(-2) / pixels
    oracle_s = oracle_s / pixels
    err_c = _close(color, oracle_c, "colour sensitivity", rtol=2e-3, atol=1e-9)
    err_s = _close(shape, oracle_s, "shape sensitivity", rtol=2e-3, atol=1e-9)
    # the probe can tell abs-per-view from abs-of-sum, and sees the negative-colour Gaussians
    signed = (oracle_c_signed.abs().flatten(-2) / pixels)
    gap = float((signed - oracle_c).abs().max() / oracle_c.max())
    if gap < 1e-2:
        raise AssertionError("probe scene cannot distinguish |sum| from sum |.|")
    if not bool((color[:4].amax(-1) > 0).all()):
        raise AssertionError("negative-colour Gaussians got zero sensitivity (clamped colours?)")
    if float(color[-1].abs().max()) != 0.0:
        raise AssertionError("an invisible Gaussian has nonzero sensitivity")
    return {"colour_max_err": err_c, "shape_max_err": err_s, "abs_vs_sum_gap": gap,
            "pixels": pixels, "invisible_color_importance": float(color[-1].amax())}


# ---------------------------------------------------------------------------------------------
def vq_official(method, device):
    """Official nearest kernel, EMA weighted update, trace re-normalisation, join, eigh split."""
    _require_cuda(device)
    from . import vq

    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(5000, 48, device=device, generator=g)
    cb = torch.randn(300, 48, device=device, generator=g)
    d, i = vq.nearest(x, cb)
    d_ref, i_ref = vq.nearest_reference(x, cb)
    if not torch.equal(i.cpu(), i_ref.cpu()):
        raise AssertionError("official nearest-centroid kernel != brute force argmin")
    err_d = _close(d, d_ref.to(d.device), "nearest distances", rtol=1e-4, atol=1e-4)

    # one EMA update against the paper's centroid formula
    w = torch.rand(5000, device=device, generator=g)
    cb0 = cb.clone()
    vq.update(cb, x, w, 0.8)
    want = cb0.clone()
    for k in range(len(cb0)):
        m = i_ref.to(device) == k
        new = (x[m] * w[m, None]).sum(0) / (w[m].sum() + vq.EPS)
        want[k] = 0.8 * cb0[k] + 0.2 * new
    err_u = _close(cb, want, "EMA centroid update", rtol=1e-4, atol=1e-5)

    # sensitivity-aware: weighted, not plain, mean (decay 0 -> centroid = weighted mean)
    pts = torch.tensor([[0.0], [1.0]], device=device)
    c1 = torch.tensor([[0.3]], device=device)
    vq.update(c1, pts, torch.tensor([1.0, 3.0], device=device), 0.0)
    if abs(float(c1) - 0.75) > 1e-4:
        raise AssertionError(f"weighted centroid {float(c1)} != 0.75 (plain mean 0.5)")

    # normalised covariance clustering keeps unit trace; eigh split reconstructs
    q = F.normalize(torch.randn(4000, 4, device=device, generator=g))
    s = F.normalize(torch.rand(4000, 3, device=device, generator=g) + 0.05)
    cov = vq.covariance(s, q)
    imp = torch.rand(4000, device=device, generator=g)
    cbk, idx = vq.vq_features(cov, imp, 64, 512, 20, 0.8, scale_normalize=True, generator=g)
    tr = cbk[:, [0, 3, 5]].sum(-1)
    err_tr = _close(tr, torch.ones_like(tr), "codebook trace", rtol=0, atol=1e-5)
    keep = imp > 0.97
    joined, jidx = vq.join_features(cov, keep, cbk, idx[~keep])
    if not torch.equal(joined[jidx[keep]], cov[keep]):
        raise AssertionError("high-sensitivity vectors are not stored exactly")
    rot, sc = vq.extract_rot_scale(vq.to_full_cov(cov))
    err_e = _close(vq.covariance(sc, rot), cov, "eigendecomposition", rtol=1e-3, atol=1e-5)
    return {"nearest_dist_err": err_d, "update_err": err_u, "trace_err": err_tr,
            "eigh_err": err_e, "weighted_centroid": float(c1)}


# ---------------------------------------------------------------------------------------------
def _tiny_bank(device, n=40, kc=5, kg=4, seed=0):
    from .bank import CompressedBank
    from .quant import fake_quantizers

    g = torch.Generator(device="cpu").manual_seed(seed)
    p = _scene(n, device, seed)
    color_idx = torch.randint(0, kc, (n,), generator=g).to(device)
    gauss_idx = torch.randint(0, kg, (n,), generator=g).to(device)
    bank = CompressedBank(
        torch.randn(kc, 1, 3, generator=g).to(device), torch.randn(kc, 15, 3, generator=g)
        .to(device) * 0.2, color_idx, F.normalize(torch.rand(kg, 3, generator=g) + 0.1)
        .to(device), F.normalize(torch.randn(kg, 4, generator=g)).to(device), gauss_idx,
        torch.empty(n, 1).uniform_(math.log(0.08), math.log(0.3), generator=g).to(device),
        fake_quantizers(device))
    return bank, p


def qa_codebook(method, device):
    """8-bit grids / float16 positions, STE gradients, codebook gradient = sum of per-Gaussian
    gradients, official fine-tune learning rates."""
    _require_cuda(device)
    from .bank import DEFAULT_LRS

    bank, p = _tiny_bank(device)
    p = {k: v.clone().requires_grad_(k in ("means", "opacities")) for k, v in p.items()}
    a = bank.attributes(p)
    uniq = {k: int(torch.unique(v.detach()).numel()) for k, v in
            (("dc", bank.q["features_dc"](bank.tensors["cb_dc"])),
             ("opacity", a["opacities"]),
             ("log_eta", bank.q["scaling_factor"](bank.tensors["log_eta"])))}
    if max(uniq.values()) > 256:
        raise AssertionError(f"more than 256 levels after 8-bit fake quantisation: {uniq}")
    for name in ("features_dc", "features_rest", "opacity", "scaling", "scaling_factor",
                 "rotation"):
        fq = bank.q[name]
        if int(fq.quant_min) != -128 or int(fq.quant_max) != 127:
            raise AssertionError(f"{name}: not an 8-bit quantiser")
    # values lie on the quantiser grid
    dc = bank.q["features_dc"](bank.tensors["cb_dc"]).detach()
    s, z = bank.q["features_dc"].scale, bank.q["features_dc"].zero_point
    grid = dc / s + z
    if float((grid - grid.round()).abs().max()) > 1e-3:
        raise AssertionError("fake-quantised colours are not on the 8-bit grid")
    if not torch.equal(a["means"], a["means"].half().float()):
        raise AssertionError("positions are not float16")
    # codebook gradient accumulation: L = sum_g w_g . colour_g
    w = torch.randn_like(a["colors"])
    (a["colors"] * w).sum().backward()
    gdc = bank.tensors["cb_dc"].grad
    want = torch.zeros_like(gdc).index_add_(0, bank.color_idx, w[:, :1])
    # STE: entries inside the quantiser range pass the gradient unchanged
    err = _close(gdc, want, "codebook gradient", rtol=1e-5, atol=1e-6)
    lrs = {gr["name"]: gr["lr"] for gr in bank.optimizer.param_groups}
    if lrs != DEFAULT_LRS or DEFAULT_LRS["cb_rest"] != 0.0025 / 20:
        raise AssertionError(f"fine-tune LRs {lrs} differ from the official training_setup")
    before = bank.tensors["cb_dc"].detach().clone()
    bank.step()
    if torch.equal(before, bank.tensors["cb_dc"].detach()):
        raise AssertionError("bank optimizer did not update the colour codebook")
    return {"unique_values": uniq, "codebook_grad_err": err, "lrs": lrs}


# ---------------------------------------------------------------------------------------------
def codec_roundtrip(method, device):
    """npz (official layout, savez_compressed) -> decode equals the fake-quantised model; size is
    the on-disk size; Morton order is applied and shrinks a coherent scene."""
    _require_cuda(device)
    from splatstudio.backends.gsplat import Renderer
    from splatstudio.specs import ComponentSpec

    from . import codec
    from .quant import morton_order
    from .renderer import rasterize

    n = 6000
    bank, p = _tiny_bank(device, n=n, kc=64, kg=32, seed=1)
    # spatially coherent scene in a random storage order: colour/shape index follow position
    g = torch.Generator(device="cpu").manual_seed(5)
    t = torch.rand(n, generator=g).to(device)
    p["means"] = torch.stack((t - 0.5, torch.sin(6 * t) * 0.3, 0.1 * t), 1)
    bank.color_idx = (t * 63.999).long()
    bank.gauss_idx = (t * 31.999).long()
    with torch.no_grad():
        bank.tensors["log_eta"].copy_((math.log(0.02) + 0.5 * t)[:, None])
        p["opacities"] = torch.logit(0.3 + 0.4 * t)
    cam = _camera(device, 0.1, width=120, height=80)
    with torch.no_grad():
        a = bank.attributes(p)  # sets the quantisers' qparams (last forward)
        ref = rasterize(a["means"], a["quats"], a["scales"], a["opacities"], a["colors"], cam, 3)
        data, order = codec.encode_npz(bank, p, sort_morton=True)
        dec = codec.decode_npz(data, device)
    _close(dec["sh0"], a["colors"][order][:, :1], "decoded dc", rtol=0, atol=1e-6)
    _close(dec["shN"], a["colors"][order][:, 1:], "decoded rest", rtol=0, atol=1e-6)
    _close(torch.sigmoid(dec["opacities"]), a["opacities"][order], "decoded opacity",
           rtol=0, atol=1e-5)
    _close(dec["scales"].exp(), a["scales"][order], "decoded scales", rtol=1e-5, atol=1e-7)
    qerr = float((F.normalize(dec["quats"]) - a["quats"][order]).abs().max())
    if qerr > 0.02:  # stored Q(normalize(r)) vs rendered normalize(FQ(r)) (official save_npz)
        raise AssertionError(f"decoded rotations differ by {qerr}")
    base = Renderer(ComponentSpec("gsplat", "renderer", "", (), (), {}))
    with torch.no_grad():
        out = base.render({**dec}, cam, 3)
    mse = float((out["rgb"] - ref["rgb"]).square().mean())
    psnr = -10 * math.log10(max(mse, 1e-12))
    if psnr < 45:
        raise AssertionError(f"decoded render differs from the fake-quantised render ({psnr:.1f})")
    # size is exactly np.savez_compressed's file size
    z = np.load(io.BytesIO(data))
    keys = list(z.keys())
    expected = ["quantization", "xyz", "features_dc", "features_dc_scale",
                "features_dc_zero_point", "features_rest", "features_rest_scale",
                "features_rest_zero_point", "opacity", "opacity_scale", "opacity_zero_point",
                "feature_indices", "gaussian_indices", "scaling", "scaling_scale",
                "scaling_zero_point", "scaling_factor", "scaling_factor_scale",
                "scaling_factor_zero_point", "rotation", "rotation_scale", "rotation_zero_point"]
    if keys != expected:
        raise AssertionError(f"npz keys {keys} differ from the official save_npz")
    dtypes = {k: str(z[k].dtype) for k in ("xyz", "features_dc", "opacity", "feature_indices",
                                            "scaling_factor")}
    if dtypes != {"xyz": "float16", "features_dc": "int8", "opacity": "int8",
                  "feature_indices": "int32", "scaling_factor": "int8"}:
        raise AssertionError(f"npz dtypes {dtypes}")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "point_cloud.npz")
        np.savez_compressed(path, **{k: z[k] for k in keys})
        disk = os.path.getsize(path)
    if disk != len(data):
        raise AssertionError(f"reported size {len(data)} != on-disk savez_compressed {disk}")
    if not torch.equal(order, morton_order(p["means"])) or not torch.equal(
            dec["means"], p["means"][order].half().float()):
        raise AssertionError("decoded Gaussians are not the Morton-sorted float16 positions")
    unsorted, _ = codec.encode_npz(bank, p, sort_morton=False)
    if len(data) >= len(unsorted):
        raise AssertionError("Morton order did not reduce the compressed size")
    return {"psnr_decoded_vs_bank": psnr, "npz_bytes": len(data),
            "npz_bytes_unsorted": len(unsorted), "rotation_max_err": qerr}


# ---------------------------------------------------------------------------------------------
def pipeline(method, device):
    """End-to-end tiny run through the density/renderer: prune zero-sensitivity Gaussians,
    cluster, QA fine-tune (codebooks move, means frozen), encode -> decode into the params."""
    _require_cuda(device)
    from splatstudio.components import build
    from splatstudio.state import check_state, initialize

    from . import bank as bank_mod
    from .density import C3DGSDensity

    small = {"base_steps": 2, "total_steps": 6, "color_codebook_size": 16,
             "gaussian_codebook_size": 8, "color_batch_size": 128, "gaussian_batch_size": 128,
             "color_cluster_iterations": 5, "gaussian_cluster_iterations": 5,
             "color_importance_include": 1.0, "gaussian_importance_include": 1.0}
    comps = tuple(replace(c, config={**c.config, **small}) if c.kind == "density" else c
                  for c in method.components)
    m = replace(method, components=comps)
    g = torch.Generator(device="cpu").manual_seed(2)
    n = 400
    pts = (torch.rand(n, 3, generator=g) - 0.5) * torch.tensor([1.2, 1.2, 0.6])
    pts[-3:] = torch.tensor([0.0, 40.0, 0.0])  # outside all views -> zero sensitivity
    cols = torch.rand(n, 3, generator=g) * 0.6 + 0.2
    params, optimizers = initialize(m, pts.to(device), cols.to(device))
    renderer, density, losses = build(m)
    if not isinstance(density, C3DGSDensity):
        raise AssertionError("method density is not the c3dgs density")
    cams = [_camera(device, a) for a in (-0.4, 0.0, 0.4, 0.8)]
    target = torch.rand(64, 96, 3, generator=g).to(device)
    density.initialize(params, optimizers, 1.0)
    density.set_training_cameras(cams)
    trace = {}
    try:
        for step in range(1, 7):
            out = renderer.render(params, cams[step % 4], sh_degree=3)
            out["step"] = step
            density.before_backward(params, optimizers, step, out)
            total = sum(fn(params, out, target) for _, fn in losses)
            total.backward()
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            means_before = params["means"].detach().clone()
            cb_before = (None if density.bank is None
                         else density.bank.tensors["cb_dc"].detach().clone())
            density.after_backward(params, optimizers, step, out)
            check_state(m, params, optimizers)
            if step == 2:
                if bank_mod.ACTIVE["bank"] is not density.bank or density.bank is None:
                    raise AssertionError("compressed bank not active after base_steps")
                rec = density.history[0]
                if rec["pruned"] < 3 or bool((params["means"][:, 1] > 20).any()):
                    raise AssertionError(f"zero-sensitivity Gaussians not pruned: {rec}")
                trace["compress"] = rec
            if step in (3, 4, 5):
                if torch.equal(cb_before, density.bank.tensors["cb_dc"].detach()):
                    raise AssertionError("QA fine-tune did not update the colour codebook")
                if not torch.equal(means_before, params["means"].detach()):
                    raise AssertionError("means moved during fine-tuning (official LR 0)")
            if step == 5:
                with torch.no_grad():
                    trace["bank_rgb"] = renderer.render(params, cams[0], 3)["rgb"]
        if density.encoded is None or not density.bank.decoded or bank_mod.ACTIVE["bank"]:
            raise AssertionError("final encode -> decode did not run")
        if (len(density.bank.tensors["cb_dc"]), len(density.bank.tensors["cb_rot"])) != (16, 8):
            raise AssertionError("tiny-run codebooks are not the clustered 16 / 8 entries")
        if density.bank.steps != 3:
            raise AssertionError(f"bank stepped {density.bank.steps} times, official 3 (skip last)")
        with torch.no_grad():
            dec_rgb = renderer.render(params, cams[0], 3)["rgb"]
        # the step-6 forward moved the quantiser ranges slightly after the step-5 render
        mse = float((dec_rgb - trace["bank_rgb"]).square().mean())
        psnr = -10 * math.log10(max(mse, 1e-12))
        if psnr < 30:
            raise AssertionError(f"decoded model renders differently ({psnr:.1f} dB)")
        enc = density.encoded
        if abs(enc["size_MB"] - len(density.npz) / 2**20) > 1e-12:
            raise AssertionError("size_MB is not the npz byte length / 2**20")
        state = density.state  # must survive torch.save / weights_only load
        buf = io.BytesIO()
        torch.save(state, buf)
        buf.seek(0)
        torch.load(buf, weights_only=True)
        return {"pruned": trace["compress"]["pruned"], "psnr_decoded_vs_bank": psnr,
                "size_MB": enc["size_MB"], "color_codebook": enc["color_codebook"],
                "shape_codebook": enc["shape_codebook"]}
    finally:
        bank_mod.ACTIVE["bank"] = None


# ---------------------------------------------------------------------------------------------
def schedule(method, device):
    """Compression after base_steps, bank steps up to total-1, encode at total, run-length guard."""
    from . import schedule as sch
    from .density import C3DGSDensity

    density = next(c for c in method.components if c.kind == "density")
    cfg = density.config
    b, t = cfg["base_steps"], cfg["total_steps"]
    if (b, t) != (30000, 35000) and method.name == "c3dgs":
        raise AssertionError(f"c3dgs schedule {b}/{t} != official 30000 + 5000")
    got = {s: sch.events(s, b, t) for s in (1, b - 1, b, b + 1, t - 1, t)}
    want = {1: ["base"], b - 1: ["base"], b: ["base", "compress"], b + 1: ["bank_step"],
            t - 1: ["bank_step"], t: ["encode"]}
    if got != want:
        raise AssertionError(f"events {got} != {want}")
    steps = sch.inferred_run_length(1.0, 0.01 ** (1 / t))
    if abs(steps - t) > 1e-6:
        raise AssertionError("run-length inference wrong")
    d = C3DGSDensity(density)
    d.lr0 = 1.0
    try:
        d.check_run_length(1, 0.01 ** (1 / 30000))
    except ValueError:
        pass
    else:
        raise AssertionError("a 30000-step run was not rejected")
    d.check_run_length(1, 0.01 ** (1 / t))
    lrs = {}
    for s in (b - 1, b, t - 1):
        opt = {"means": torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1.0)}
        d.set_learning_rates(opt, s)
        lrs[s] = opt["means"].param_groups[0]["lr"]
    if not math.isclose(lrs[b - 1], 0.01 ** ((b - 1) / b)) or lrs[b] != 0.0 or lrs[t - 1] != 0.0:
        raise AssertionError(f"means LRs {lrs}: 30k-run schedule, then frozen")
    return {"events": {str(k): v for k, v in got.items()}, "means_lr": lrs}
