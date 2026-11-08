"""Executable mechanism checks for LightGaussian (each fails if its mechanism is removed/wrong)."""

import math
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F


def _require_cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ProbeUnavailable("The LightGaussian significance kernel / renderer need CUDA")


def _component(method, kind, name=None):
    return next(c for c in method.components if c.kind == kind and (name is None or c.name == name))


def _scene(n, device, seed, opacity=(0.05, 0.6), stack=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.empty(n, 3).uniform_(-0.5, 0.5, generator=g)
    means[:, 2] = torch.empty(n).uniform_(-0.4, 0.4, generator=g)
    params = {
        "means": means,
        "scales": torch.empty(n, 3).uniform_(math.log(0.04), math.log(0.2), generator=g),
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.logit(torch.empty(n).uniform_(*opacity, generator=g)),
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.8,
        "shN": torch.randn(n, 15, 3, generator=g) * 0.1,
    }
    if stack:  # opaque Gaussians stacked along the optical axis (termination / clamp rules)
        params["means"][:stack] = torch.tensor([[0.0, 0.0, -0.3 + 0.15 * i] for i in range(stack)])
        params["scales"][:stack] = math.log(0.15)
        params["opacities"][:stack] = torch.logit(torch.tensor(0.995))
    return {k: v.to(device) for k, v in params.items()}


def _camera(device, angle=0.15, width=72, height=56, cx=35.3, cy=27.1, f=60.0, shift=0.0):
    """OpenCV world-to-camera looking at the origin from distance 3."""
    c, s = math.cos(angle), math.sin(angle)
    R = torch.tensor([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    viewmat = torch.eye(4)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = torch.tensor([0.05 + shift, -0.03, 3.0])
    K = torch.tensor([[f, 0.0, cx], [0.0, f * 1.02, cy], [0.0, 0.0, 1.0]])
    return {"K": K.to(device), "viewmat": viewmat.to(device), "width": width, "height": height}


def _brute_significance(params, camera, clamp=0.99, use_alpha=False):
    """Independent per-pixel composite of the official renderCUDA_count rules."""
    from . import significance

    means2d, radii, depths, conics, opac = significance.project(params, camera)
    W, H = int(camera["width"]), int(camera["height"])
    ys, xs = torch.meshgrid(torch.arange(H, device=opac.device), torch.arange(W, device=opac.device),
                            indexing="ij")
    px, py = xs.reshape(-1).float() + 0.5, ys.reshape(-1).float() + 0.5
    T = torch.ones_like(px)
    done = torch.zeros_like(px, dtype=torch.bool)
    n = len(opac)
    score = torch.zeros(n, device=opac.device, dtype=torch.float64)
    count = torch.zeros(n, device=opac.device, dtype=torch.long)
    visible = (radii > 0).all(dim=-1)
    for j in torch.argsort(depths).tolist():
        if not visible[j]:
            continue
        dx, dy = means2d[j, 0] - px, means2d[j, 1] - py
        a, b, c = conics[j]
        power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
        alpha = torch.clamp(opac[j] * torch.exp(power), max=clamp)
        valid = (~done) & (power <= 0) & (alpha >= 1.0 / 255.0)
        test_T = T * (1 - alpha)
        term = valid & (test_T < 1e-4)
        done = done | term
        hit = valid & ~term
        w = alpha * T if use_alpha else opac[j] * T
        score[j] += w[hit].double().sum()
        count[j] += int(hit.sum())
        T = torch.where(hit, test_T, T)
    return count, score.float()


@torch.no_grad()
def significance_oracle(method, device):
    _require_cuda(device)
    from . import significance

    params = _scene(40, device, seed=1, stack=4)
    cams = [_camera(device, angle=a, shift=0.02 * i) for i, a in enumerate((0.1, -0.2, 0.3))]
    count, score = significance.hit_scores(params, cams)
    ref_c = torch.zeros_like(count, dtype=torch.long)
    ref_s = torch.zeros_like(score)
    alt_c = torch.zeros_like(ref_c)
    alt_s = torch.zeros_like(ref_s)
    for cam in cams:
        c, s = _brute_significance(params, cam)
        ref_c += c
        ref_s += s
        c2, s2 = _brute_significance(params, cam, clamp=0.999, use_alpha=True)
        alt_c += c2
        alt_s += s2
    count_err = int((count.long() - ref_c).abs().sum())
    rel = float(((score - ref_s).abs().sum() / ref_s.abs().sum()))
    if count_err > max(3, int(0.002 * int(ref_c.sum()))) or rel > 1e-3:
        raise AssertionError(f"kernel vs brute force: count err {count_err}, score rel err {rel}")
    alt_rel = float(((score - alt_s).abs().sum() / alt_s.abs().sum()))
    alt_count = int((count.long() - alt_c).abs().sum())
    if alt_rel < 0.05 or alt_count == 0:
        raise AssertionError("the probe scene does not separate sigma*T/0.99 from alpha*T/0.999")
    return {"hits": int(ref_c.sum()), "count_abs_err": count_err, "score_rel_err": rel,
            "rel_diff_vs_alpha_T_0.999": alt_rel, "count_diff_vs_0.999": alt_count}


def prune_rule(method, device):
    from . import significance

    g = torch.Generator().manual_seed(0)
    n = 1001
    log_scales = torch.randn(n, 3, generator=g) * 0.7 - 3
    score = torch.rand(n, generator=g) * 10
    score[:50] = 0.0
    cfg = _component(method, "density").config
    v = significance.v_importance(score, log_scales, cfg["v_pow"])
    vol = np.prod(np.exp(log_scales.double().numpy()), axis=1)
    kth = np.sort(vol)[::-1][int(0.9 * n)]
    ref = (vol / kth) ** 0.1 * score.double().numpy()
    if not np.allclose(v.double().numpy(), ref, rtol=1e-5, atol=1e-7):
        raise AssertionError("v-score differs from score * (V / V_sorted_desc[int(0.9N)])^0.1")
    clipped = np.minimum(vol / kth, 1.0) ** 0.1 * score.double().numpy()
    if np.allclose(v.double().numpy(), clipped, rtol=1e-4):
        raise AssertionError("v-score indistinguishable from the clipped paper variant")
    if cfg["v_pow"] != 0.1 or cfg["prune_percent"] != 0.66:
        raise AssertionError(f"paper/script values changed: {cfg['v_pow']}, {cfg['prune_percent']}")
    mask = significance.prune_mask(v, cfg["prune_percent"])
    thr = np.sort(v.numpy())[int(0.66 * (n - 1))]
    if not torch.equal(mask, torch.from_numpy(v.numpy() <= thr)):
        raise AssertionError("prune mask differs from v <= sort(v)[int(p (N-1))]")
    return {"pruned": int(mask.sum()), "of": n, "expected": int(0.66 * (n - 1)) + 1}


def _activate(cfg, step, seed=0):
    from .shared import STATE

    STATE.reset()
    STATE.active = True
    STATE.cfg = cfg
    STATE.next_step = step
    STATE.generator = torch.Generator().manual_seed(seed)
    return STATE


def distill_views(method, device):
    _require_cuda(device)
    from splatstudio.losses import ssim

    from .losses import create_distill, create_photometric
    from .renderer import create_renderer

    cfg = dict(_component(method, "density").config)
    renderer = create_renderer(_component(method, "renderer"))
    distill = create_distill(_component(method, "loss", "lightgaussian-distill"))
    photometric = create_photometric(_component(method, "loss", "photometric"))
    params = _scene(60, device, seed=2)
    cam = _camera(device)
    center = torch.linalg.inv(cam["viewmat"])[:3, 3]
    try:
        d0 = cfg["distill_step"]
        state = _activate(cfg, d0 + 1)
        shifts, zero_steps = [], 0
        for k in range(d0 + 1, d0 + 601):
            state.next_step = k
            with torch.no_grad():
                out = renderer.render(params, cam, 3)
            moved = torch.linalg.inv(out["lg_camera"]["viewmat"])[:3, 3] - center
            if k % 3 == 0:
                if float(moved.abs().max()) > 1e-6:
                    raise AssertionError(f"step {k} (k%3==0) must use the training view")
                zero_steps += 1
            else:
                shifts.append(moved.cpu())
                rot = out["lg_camera"]["viewmat"][:3, :3] - cam["viewmat"][:3, :3]
                if float(rot.abs().max()) > 1e-6:
                    raise AssertionError("pseudo views must not rotate the camera (std 0)")
        shifts = torch.stack(shifts)
        std = float(shifts.std())
        if not 0.045 < std < 0.055 or float(shifts.mean().abs()) > 0.006:
            raise AssertionError(f"pseudo-view centre shift std {std} != 0.05")
        state.next_step = d0  # co-adaptation: no pseudo view
        with torch.no_grad():
            out = renderer.render(params, cam, 3)
        if not torch.equal(out["lg_camera"]["viewmat"], cam["viewmat"]):
            raise AssertionError("pseudo view outside the distillation stage")

        # distillation loss against the teacher at the camera used by the student
        state.next_step = d0 + 1
        state.teacher = {k: v.clone() for k, v in params.items()}
        student = {k: v.clone().requires_grad_(True) for k, v in params.items()}
        out = renderer.render(student, cam, 3)
        same = distill(student, out, torch.zeros_like(out["rgb"]))
        if float(same) > 1e-5:
            raise AssertionError(f"distill loss {float(same)} for identical teacher/student")
        with torch.no_grad():
            student["sh0"].add_(0.3)
        out = renderer.render(student, cam, 3)
        loss = distill(student, out, torch.zeros_like(out["rgb"]))
        teacher_rgb = _plain_render(params, out["lg_camera"], 3)
        ref = 0.8 * F.l1_loss(out["rgb"], teacher_rgb) + 0.2 * (1 - ssim(out["rgb"], teacher_rgb))
        if abs(float(loss) - float(ref)) > 1e-5 or float(loss) < 1e-3:
            raise AssertionError(f"distill loss {float(loss)} != L1+0.2DSSIM vs teacher {float(ref)}")
        gt = torch.rand_like(out["rgb"])
        p_off = photometric(student, out, gt)
        if float(p_off) != 0.0 or not p_off.requires_grad:
            raise AssertionError("photometric loss must be an attached zero during distillation")
        state.next_step = d0
        out = renderer.render(student, cam, 3)
        p_on = photometric(student, out, gt)
        ref_p = 0.8 * F.l1_loss(out["rgb"], gt) + 0.2 * (1 - ssim(out["rgb"], gt))
        if abs(float(p_on) - float(ref_p)) > 1e-6 or float(distill(student, out, gt)) != 0.0:
            raise AssertionError("outside distillation: photometric on, distill off")

        # SH degree 2 by zeroed degree-3 coefficients
        state.active = False
        trunc = {k: v.clone() for k, v in params.items()}
        trunc["shN"][:, 8:] = 0
        d3, d2 = _plain_render(trunc, cam, 3), _plain_render(trunc, cam, 2)
        full = _plain_render(params, cam, 3)
        sh_err = float((d3 - d2).abs().max())
        if sh_err > 1e-5 or float((full - d2).abs().max()) < 1e-3:
            raise AssertionError(f"zeroed degree-3 SH render differs from degree 2 by {sh_err}")
    finally:
        from .shared import STATE

        STATE.reset()
    return {"pseudo_shift_std": std, "training_view_steps": zero_steps,
            "pseudo_steps": len(shifts), "distill_loss": float(loss), "sh2_max_diff": sh_err}


def _plain_render(params, camera, sh_degree):
    from gsplat import rasterization

    with torch.no_grad():
        colors = torch.cat((params["sh0"], params["shN"]), dim=1)
        render, alpha, _ = rasterization(
            means=params["means"], quats=params["quats"], scales=params["scales"].exp(),
            opacities=params["opacities"].sigmoid(), colors=colors,
            viewmats=camera["viewmat"][None], Ks=camera["K"][None], width=camera["width"],
            height=camera["height"], sh_degree=sh_degree, packed=False,
        )
        return render[0] + (1 - alpha[0])


@torch.no_grad()
def vq_official(method, device):
    _require_cuda(device)
    from . import vq

    g = torch.Generator(device=device).manual_seed(0)
    k, d, n = 64, 27, 3000
    x = torch.randn(n, d, generator=g, device=device)
    w = torch.rand(n, generator=g, device=device) + 0.1
    embed = torch.randn(k, d, generator=g, device=device)
    cs = torch.rand(k, generator=g, device=device)
    # official EuclideanCodebook.forward (training, weighted), one-hot einsums
    ref_e, ref_cs = embed.clone()[None], cs.clone()[None]
    flat, wt = x[None], (w * w.numel() / w.sum()).reshape(1, -1, 1)
    ind = (-torch.cdist(flat, ref_e, p=2)).argmax(dim=-1)
    onehot = F.one_hot(ind, k).float()
    ref_cs.mul_(0.8).add_((onehot * wt).sum(dim=1), alpha=0.2)
    embed_sum = torch.einsum("hnd,hnc->hcd", flat * wt, onehot)
    sm = (ref_cs + 1e-5) / (ref_cs.sum() + k * 1e-5) * ref_cs.sum()
    ref_e.mul_(0.8).add_(embed_sum / sm[..., None], alpha=0.2)
    got_e, got_cs = embed.clone(), cs.clone()
    vq.ema_step(got_e, got_cs, x, w, 0.8)
    e_err = float((got_e - ref_e[0]).abs().max())
    c_err = float((got_cs - ref_cs[0]).abs().max())
    if e_err > 1e-4 or c_err > 1e-4:
        raise AssertionError(f"EMA step differs from official formula: {e_err}, {c_err}")
    init = vq.init_codebook(8192, 27, g, device)
    bound = math.sqrt(6.0 / (8192 * 27))
    if float(init.abs().max()) > bound or float(init.abs().max()) < 0.99 * bound:
        raise AssertionError("codebook init is not kaiming-uniform(1, K, D)")
    centers = torch.randn(24, d, generator=g, device=device) * 2
    data = centers[torch.randint(0, 24, (6000,), generator=g, device=device)]
    data = data + 0.05 * torch.randn(data.shape, generator=g, device=device)
    imp = torch.rand(len(data), generator=g, device=device)

    def err(cb):
        return float((data - cb[vq.nearest(data, cb)]).square().sum(1).mean())

    g1 = torch.Generator(device=device).manual_seed(1)
    cb10, _ = vq.learn_codebook(data, imp, 32, 60, 1000, 0.8, 10, g1)
    g1 = torch.Generator(device=device).manual_seed(1)
    cb0, _ = vq.learn_codebook(data, imp, 32, 60, 1000, 0.8, 0, g1)
    g1 = torch.Generator(device=device).manual_seed(1)
    e_init = err(vq.init_codebook(32, d, g1, device))
    e10, e0 = err(cb10), err(cb0)
    if not e10 < 0.2 * e_init or not e10 < e0:
        raise AssertionError(f"codebook learning ineffective: init {e_init}, k10 {e10}, k0 {e0}")
    imp2 = torch.rand(1000, generator=g, device=device)
    non_vq = vq.split_masks(imp2, 0.6)
    if int(non_vq.sum()) != 400 or float(imp2[non_vq].min()) < float(imp2[~non_vq].max()):
        raise AssertionError("non-VQ set must be the top int(0.4 N) by importance")
    return {"ema_max_err": e_err, "err_init": e_init, "err_dead_code_10": e10,
            "err_no_replacement": e0}


def codec_roundtrip(method, device):
    from . import codec, vq

    g = torch.Generator().manual_seed(0)
    n, k = 3000, 8192
    xyz = torch.randn(n, 3, generator=g) * 3
    feats = torch.randn(n, 27, generator=g)
    other = torch.randn(n, 8, generator=g)
    codebook = torch.randn(k, 27, generator=g)
    non_vq = vq.split_masks(torch.rand(n, generator=g), 0.6)
    idx = torch.randint(0, k, (int((~non_vq).sum()),), generator=g)
    data, files = codec.encode(xyz, feats, other, codebook, idx, non_vq)
    dec = codec.decode(data, "cpu")
    exp = feats.half().float()
    exp[~non_vq] = codebook.half().float()[idx]
    checks = {
        "xyz_fp32": torch.equal(dec["xyz"], xyz),
        "other_fp16": torch.equal(dec["other"], other.half().float()),
        "sh": torch.equal(dec["sh_feats"], exp),
        "indices": torch.equal(dec["indices"], idx),
        "mask": torch.equal(dec["non_vq_mask"], non_vq),
    }
    if not all(checks.values()):
        raise AssertionError(f"decode mismatch: {checks}")
    packed = np.load(__import__("io").BytesIO(files["vq_indexs.npz"]))["arr_0"]
    if len(packed) != math.ceil(len(idx) * 13 / 8):
        raise AssertionError("VQ indices are not 13-bit packed")
    rep = codec.size_report(data, files, n, len(idx))
    if rep["size_MiB"] != len(data) / 2**20 or len(files) != 7:
        raise AssertionError("size must be the zip bytes / 1024^2 of the 7 official files")
    raw_fp32 = n * (3 + 27 + 8) * 4
    return {"zip_bytes": len(data), "raw_fp32_bytes": raw_fp32, **checks}


def _tiny_run(method, device, steps_cfg):
    """Mimics runtime.train on a synthetic scene; returns the per-step observations."""
    from splatstudio.losses import ssim

    from . import codec, significance, vq
    from .density import create_density
    from .losses import create_distill, create_photometric
    from .renderer import create_renderer

    dspec = _component(method, "density")
    dspec = replace(dspec, config={**dspec.config, **steps_cfg})
    density = create_density(dspec)
    renderer = create_renderer(_component(method, "renderer"))
    losses = [create_photometric(_component(method, "loss", "photometric")),
              create_distill(_component(method, "loss", "lightgaussian-distill"))]
    base = _scene(600, device, seed=4)
    params = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in base.items()})
    rates = {"means": 1.6e-4, "scales": 0.005, "quats": 0.001, "opacities": 0.05,
             "sh0": 0.0025, "shN": 0.0025 / 20}
    opts = {k: torch.optim.Adam([p], lr=rates[k], eps=1e-15) for k, p in params.items()}
    cams = [_camera(device, angle=a, shift=0.03 * i) for i, a in enumerate((0.0, 0.25, -0.25, 0.5))]
    density.initialize(params, opts, 1.0)
    density.set_training_cameras(cams)
    targets = [_plain_render(_scene(600, device, seed=5), c, 3) for c in cams]
    c = density.cfg
    total = c["total_steps"]
    lr0 = opts["means"].param_groups[0]["lr"]
    obs = {}
    for step in range(1, total + 1):
        i = step % len(cams)
        out = renderer.render(params, cams[i], 3)
        out["step"] = step
        density.before_backward(params, opts, step, out)
        loss = sum(fn(params, out, targets[i]) for fn in losses)
        loss.backward()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        opts["means"].param_groups[0]["lr"] = lr0 * 0.01 ** (step / total)
        if step == c["prune_step"]:
            v, _, _ = significance.global_significance(params, cams, c["v_pow"])
            obs["expected_after_prune"] = int((~significance.prune_mask(v, c["prune_percent"])).sum())
        density.after_backward(params, opts, step, out)
        if step == c["prune_step"]:
            obs["after_prune"] = len(params["means"])
        if step == c["distill_step"]:
            obs["teacher_sh_nonzero"] = float(_teacher_shN_abs())
            obs["opacity_at_distill"] = params["opacities"].detach().clone()
        if c["distill_step"] < step <= c["vq_step"] and step != c["vq_step"]:
            obs.setdefault("opacity_drift", 0.0)
            obs["opacity_drift"] = max(obs["opacity_drift"], float(
                (params["opacities"] - obs["opacity_at_distill"]).abs().max()))
            obs["photometric_in_distill"] = float(losses[0](params, out, targets[i]))
        if step > c["distill_step"]:
            obs["shN_deg3_max"] = max(obs.get("shN_deg3_max", 0.0),
                                      float(params["shN"][:, 8:].abs().max()))
        if step == c["vq_step"]:
            d = density.vq
            obs["cb0"] = vq.sh_features(d["cb_dc"].detach(), d["cb_rest"].detach(), 8).clone()
            obs["sh0_nonvq_0"] = params["sh0"].detach()[d["non_vq_mask"]].clone()
        if c["vq_step"] < step < total:
            d = density.vq
            feats = vq.sh_features(params["sh0"].detach(), params["shN"].detach(), 8)
            cb = vq.sh_features(d["cb_dc"].detach(), d["cb_rest"].detach(), 8)
            obs["vq_row_err"] = max(obs.get("vq_row_err", 0.0),
                                    float((feats[d["rows"]] - cb[d["idx"]]).abs().max()))
            obs["cb_moved"] = float((cb - obs["cb0"]).abs().max())
            obs["nonvq_moved"] = float((params["sh0"].detach()[d["non_vq_mask"]]
                                        - obs["sh0_nonvq_0"]).abs().max())
    zip_bytes = density.zip.numpy().tobytes()
    dec = codec.decode(zip_bytes, device)
    sh0, rest = vq.from_sh_features(dec["sh_feats"], 8)
    obs["decode_err"] = max(float((params["sh0"] - sh0).abs().max()),
                            float((params["shN"][:, :8] - rest).abs().max()),
                            float((params["means"] - dec["xyz"]).abs().max()),
                            float((params["opacities"] - dec["other"][:, 0]).abs().max()))
    obs["leaves"] = all(p.is_leaf and p.requires_grad for p in params.values())
    obs["encoded"] = density.encoded
    obs["ssim_ok"] = float(ssim(targets[0], targets[0])) > 0.999
    from .shared import STATE

    obs["state_active_after"] = STATE.active
    return obs, density


def _teacher_shN_abs():
    from .shared import STATE

    return STATE.teacher["shN"][:, 8:].abs().max()


def pipeline(method, device):
    _require_cuda(device)
    from .shared import STATE

    steps_cfg = {"prune_step": 4, "distill_step": 8, "vq_step": 12, "total_steps": 16,
                 "codebook_size": 32, "vq_iterations": 5, "vq_batch": 200}
    try:
        obs, density = _tiny_run(method, device, steps_cfg)
    finally:
        STATE.reset()
    checks = {
        "prune_count": obs["after_prune"] == obs["expected_after_prune"] < 600,
        "teacher_full_sh": obs["teacher_sh_nonzero"] > 0,
        "opacity_frozen": obs["opacity_drift"] == 0.0,
        "photometric_off_in_distill": obs["photometric_in_distill"] == 0.0,
        "sh_degree_2": obs["shN_deg3_max"] == 0.0,
        "vq_rows_codebook": obs["vq_row_err"] == 0.0,
        "codebook_trained": obs["cb_moved"] > 0.0,
        "non_vq_trained": obs["nonvq_moved"] > 0.0,
        "decoded_model": obs["decode_err"] == 0.0,
        "trainable_leaves": obs["leaves"],
        "encoded": obs["encoded"] is not None and obs["encoded"]["zip_bytes"] > 0,
        "inactive_after": not obs["state_active_after"],
    }
    if not all(checks.values()):
        raise AssertionError(f"pipeline checks failed: {checks}")
    events = [e["lightgaussian"] for e in density.history]
    if events != ["prune", "start_distill", "vq", "start_vqft", "encode"]:
        raise AssertionError(f"stage events {events}")
    return {"checks": checks, "events": events, "after_prune": obs["after_prune"],
            "zip_bytes": obs["encoded"]["zip_bytes"]}


def schedule(method, device):
    from . import schedule as sch
    from .density import LightGaussianDensity

    spec = _component(method, "density")
    c = spec.config
    want = (30000, 35000, 40000, 45000) if method.name == "lightgaussian" else None
    got = (c["prune_step"], c["distill_step"], c["vq_step"], c["total_steps"])
    if want and got != want:
        raise AssertionError(f"lightgaussian schedule {got} != official {want}")
    p, d, v, t = got
    stages = {k: sch.stage(k, c) for k in (p, p + 1, d, d + 1, v, v + 1, t)}
    exp = {p: "3dgs", p + 1: "coadapt", d: "coadapt", d + 1: "distill", v: "distill",
           v + 1: "vqft" if t > v else "done", t: "vqft" if t > v else "distill"}
    if stages != exp:
        raise AssertionError(f"stages {stages} != {exp}")
    if method.name == "lightgaussian":
        f = {k: sch.other_lr_factor(k, c) for k in (30001, 30399, 30400, 35000, 35499, 35500,
                                                    40000, 40400, 45000)}
        ref = {30001: 1.0, 30399: 1.0, 30400: 0.95, 35000: 0.95 ** 12, 35499: 1.0,
               35500: 0.9, 40000: 0.9 ** 10, 40400: 0.95, 45000: 0.95 ** 12}
        for k_, val in ref.items():
            if not math.isclose(f[k_], val, rel_tol=1e-12):
                raise AssertionError(f"LR factor at {k_}: {f[k_]} != {val}")
        wd = [sch.weight_decay(k, c) for k in (30000, 30001, 35001, 40001)]
        if wd != [0.0, 0.0, 0.01, 0.01]:
            raise AssertionError(f"weight decay per stage {wd}")
        lr = {k: sch.means_lr(1.0, k, c) for k in (30000, 30001, 35000, 35001, 40000, 45000)}
        if not (math.isclose(lr[30000], 0.01 ** (29999 / 30000))
                and math.isclose(lr[30001], 0.01 ** (30001 / 35000))
                and math.isclose(lr[35000], 0.01) and math.isclose(lr[35001], 0.01 ** (35001 / 40000))
                and math.isclose(lr[40000], 0.01) and math.isclose(lr[45000], 0.01)):
            raise AssertionError(f"means LR schedule {lr}")
        pv = [sch.pseudo_view(k, c) for k in (35000, 35001, 35002, 35003, 40000, 40001)]
        if pv != [False, False, True, True, True, False]:
            raise AssertionError(f"pseudo-view cadence {pv}")
    dens = LightGaussianDensity(spec)
    dens.lr0 = 1.0
    try:
        dens.check_run_length(1, 0.01 ** (1 / 30000))
    except ValueError:
        pass
    else:
        raise AssertionError("a 30000-step run was not rejected")
    dens.check_run_length(1, 0.01 ** (1 / t))
    return {"stages": {str(k): s for k, s in stages.items()}}
