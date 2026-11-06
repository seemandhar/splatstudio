"""Mechanism probes for GS-IR. Each one can fail when its mechanism is removed or wrong."""

import math
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import torch
from splatstudio.components import build
from splatstudio.state import check_state, initialize

from . import baking, cubemap, geometry, light, volumes
from . import bank as bank_mod
from .losses import create_normal, normal_terms


def _cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")
    return torch.device(device)


@contextmanager
def _no_live_bank():
    saved = bank_mod.ACTIVE["bank"]
    bank_mod.ACTIVE["bank"] = None
    try:
        yield
    finally:
        bank_mod.ACTIVE["bank"] = saved


def _camera(device, size=16, f=20.0, c=None):
    c = size / 2.0 if c is None else c
    return {"viewmat": torch.eye(4, device=device),
            "K": torch.tensor([[f, 0, c], [0, f, c], [0, 0, 1.0]], device=device),
            "width": size, "height": size}


def _params(method, means, scale, opacity, device):
    n = len(means)
    params, opts = initialize(method, means.to(device), torch.full((n, 3), 0.5, device=device))
    with torch.no_grad():
        params["scales"].fill_(math.log(scale))
        params["opacities"].fill_(float(torch.logit(torch.tensor(opacity))))
    return params, opts


def _plane(device, n=24, z=3.0, extent=1.5):
    t = torch.linspace(-extent, extent, n, device=device)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten(), torch.full_like(xx.flatten(), z)), -1)


# ---------------------------------------------------------------- C1
def depth_linear_interp(method, device):
    dev = _cuda(device)
    renderer, _, _ = build(method)
    means = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]], device=dev)
    params, _ = _params(method, means, 0.5, 0.5, dev)
    cam = _camera(dev, 17, 20.0, 8.5)  # pixel (8, 8) centre projects exactly onto both means
    with _no_live_bank(), torch.no_grad():
        renderer.bank = None
        out = renderer.render(params, cam, sh_degree=0)
    d = float(out["depth"][8, 8, 0])
    a1, a2 = 0.5, 0.5
    w1, w2 = a1, a2 * (1 - a1)
    expected = (w1 * 2.0 + w2 * 3.0) / (w1 + w2)
    vol = w1 * 2.0 + w2 * 3.0
    if abs(d - expected) > 1e-3:
        raise AssertionError(f"depth {d} != linear interpolation {expected} (vol. accum {vol})")
    if not 2.0 <= d <= 3.0:
        raise AssertionError("depth outside [d_min, d_max]")
    return {"depth": d, "linear_interp": expected, "volumetric_accumulation": vol,
            "peak_selection": 2.0}


# ---------------------------------------------------------------- C2
def normal_from_depth_probe(method, device):
    dev = _cuda(device)
    # (a) analytic plane n.X = d in camera space, depth z = d / (n . ray)
    size, f = 48, 40.0
    K = torch.tensor([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]], device=dev)
    n = torch.nn.functional.normalize(torch.tensor([0.3, -0.2, -1.0], device=dev), dim=0)
    rays = geometry.camera_rays(K, size, size, dev)
    depth = (-3.0) / (rays @ n)  # n . X = -3 with n facing the camera
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(3)))
    rot = (q * torch.sign(torch.det(q))).to(dev)
    nd = geometry.normal_from_depth(depth, K, rot)
    valid = (nd != 0).all(-1)
    target = rot @ n
    cos = (nd[valid] @ target).clamp(-1, 1)
    err = float(torch.rad2deg(torch.acos(cos)).max())
    if valid.sum() < (size - 4) ** 2 * 0.9 or err > 0.5:
        raise AssertionError(f"plane normal error {err} deg on {int(valid.sum())} pixels")
    # (b) gradient isolation of L_n-p + TV
    renderer, _, _ = build(method)
    params, _ = _params(method, _plane(dev), 0.08, 0.95, dev)
    with torch.no_grad():
        params["gsir_normal"].copy_(torch.randn_like(params["gsir_normal"]))
    with _no_live_bank():
        renderer.bank = None
        out = renderer.render(params, _camera(dev, 32, 30.0), sh_degree=0)
    out["step"] = 1
    spec = next(c for c in method.components if c.name == "gsir-normal")
    value = create_normal(spec)(params, out, torch.rand(32, 32, 3, device=dev))
    lnp, tv = normal_terms(out, torch.rand(32, 32, 3, device=dev))
    value.backward()
    g = params["gsir_normal"].grad
    if g is None or not torch.count_nonzero(g):
        raise AssertionError("normal loss does not reach the Gaussian normals")
    for k in ("means", "scales", "quats", "opacities", "sh0"):
        if params[k].grad is not None and torch.count_nonzero(params[k].grad):
            raise AssertionError(f"normal loss leaks gradient into {k}")
    return {"plane_max_error_deg": err, "valid_pixels": int(valid.sum()),
            "normal_loss": float(lnp), "tv": float(tv), "normal_grad_norm": float(g.norm())}


# ---------------------------------------------------------------- C3
def _loop_masked_trilinear(coeffs, ids, aabb, p, nrm):
    res = ids.shape[0]
    lo, hi = aabb[:3], aabb[3:]
    grid = (hi - lo) / (res - 1)
    out = []
    for i in range(len(p)):
        nx = ((p[i] - lo) / grid).clamp(0, res - 1)
        fl = nx.floor()
        o = nx - fl
        acc, ws = torch.zeros(coeffs.shape[1], dtype=torch.float64), []
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    off = torch.tensor([dx, dy, dz], dtype=fl.dtype)
                    corner = (fl + off) * grid + lo
                    w = 1.0
                    for a, dd in enumerate((dx, dy, dz)):
                        w *= float(o[a]) if dd else 1 - float(o[a])
                    if float(((corner - p[i]) * nrm[i]).sum()) <= 0:
                        w = 0.0
                    idx = [min(int(fl[a]) + dd, res - 1) for a, dd in enumerate((dx, dy, dz))]
                    ws.append((w, int(ids[idx[0], idx[1], idx[2]])))
        s = sum(w for w, _ in ws) or 1e-4
        for w, cid in ws:
            acc += (w / s) * coeffs[cid].double()
        out.append(acc)
    return torch.stack(out).float()


def occlusion_baking(method, device):
    dev = _cuda(device)
    # wall of opaque Gaussians on x = 0.25 in front of a probe at the origin
    t = torch.linspace(-1.2, 1.2, 61, device=dev)
    zz, yy = torch.meshgrid(t, t, indexing="ij")
    wall = torch.stack((torch.full_like(yy.flatten(), 0.25), yy.flatten(), zz.flatten()), -1)
    n = len(wall)
    quats = torch.zeros(n, 4, device=dev)
    quats[:, 0] = 1
    scales = torch.full((n, 3), 0.04, device=dev)
    opac = torch.full((n,), 0.99, device=dev)
    pos = torch.zeros(1, 3, device=dev)
    depth, alpha = baking.render_probe_depths(wall, quats, scales, opac, pos, 1.5, 64, 0.2)
    ll = baking.LatLong(64, 4, dev)
    f = ll.coefficients(depth, alpha, 0.4)[0]
    f_off = ll.coefficients(depth, alpha, 0.1)[0]  # threshold below the wall distance
    dirs = torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]], device=dev)
    v = volumes.sh_components(4, dirs) @ f
    if not (float(v[0]) < 0.2 and float(v[1]) > 0.8):
        raise AssertionError(f"SH visibility toward / away from wall = {v.tolist()}")
    v_off = volumes.sh_components(4, dirs) @ f_off
    if float(v_off[0]) < 0.8:
        raise AssertionError("threshold has no effect on the baked visibility")
    # cosine-weighted hemisphere reconstruction (CUDA) through a 2^3 id grid
    ids = torch.zeros(2, 2, 2, dtype=torch.int32, device=dev)
    coeffs = f[None, :, None].contiguous()
    pts = torch.zeros(2, 3, device=dev)
    rec = volumes.recon_occlusion(coeffs, ids, 1.0, pts, dirs, 4, 256)[:, 0]
    if not (float(rec[0]) < 0.3 and float(rec[1]) > 0.8):
        raise AssertionError(f"reconstructed hemisphere visibility {rec.tolist()}")
    # SH projection of constant visibility
    const = torch.ones(1, ll.weights.shape[0], device=dev) @ ll.weights
    dc = 4 * math.pi * 0.28209479177387814
    # (official corner-sampled lat-long grid: small odd-band leakage is part of the method)
    if abs(float(const[0, 0]) - dc) > 1e-3 or float(const[0, 1:].abs().max()) > 0.05:
        raise AssertionError(f"SH projection of 1 = {const.tolist()}")
    # masked trilinear vs loops
    g = torch.Generator().manual_seed(0)
    res = 5
    ids5 = torch.randperm(res**3, generator=g).reshape(res, res, res).int()
    c5 = torch.randn(res**3, 16, generator=g)
    aabb = torch.tensor([-1.0, -1, -1, 1, 1, 1])
    p = torch.rand(40, 3, generator=g) * 2.4 - 1.2
    nrm = torch.nn.functional.normalize(torch.randn(40, 3, generator=g), dim=-1)
    ref = _loop_masked_trilinear(c5, ids5, aabb, p, nrm)
    got = volumes.masked_trilinear(c5.to(dev), ids5.to(dev), aabb.to(dev), p.to(dev),
                                   nrm.to(dev)).cpu()
    if float((ref - got).abs().max()) > 1e-4:
        raise AssertionError("masked trilinear interpolation differs from the reference loops")
    # dilation
    sparse = torch.full((6, 6, 6), -1, dtype=torch.int32, device=dev)
    sparse[1, 2, 3] = 7
    sparse[4, 4, 0] = 3
    dil = volumes.dilate_ids(sparse)
    if bool((dil < 0).any()) or set(dil.unique().tolist()) != {3, 7}:
        raise AssertionError("dilation did not fill the grid from baked cells")
    # probe placement: 8 corners around each centre
    m = baking.probe_mask(torch.tensor([[0.01, 0.01, 0.01]], device=dev), 9, 1.0)
    if int(m.sum()) != 8 or not bool(m[4:6, 4:6, 4:6].all()):
        raise AssertionError("probe cells are not the 8 corners of the containing cell")
    return {"visibility_toward_wall": float(v[0]), "visibility_away": float(v[1]),
            "hemisphere_visibility": rec.tolist(), "sh_dc_of_one": float(const[0, 0])}


# ---------------------------------------------------------------- C5 (cube ops)
def _ref_spec_weights(n, roughness, cos_cut, out_texel):
    """Independent numpy GGX pre-filter weights of one output texel over all input texels."""
    def dirs(face, x, y):
        fx, fy = 2 * (x + 0.5) / n - 1, 2 * (y + 0.5) / n - 1
        v = {0: (1, -fy, -fx), 1: (-1, -fy, fx), 2: (fx, 1, fy), 3: (fx, -1, -fy),
             4: (fx, -fy, 1), 5: (-fx, -fy, -1)}[face]
        v = np.array(v, dtype=np.float64)
        return v / np.linalg.norm(v)

    def area(x, y):
        h = n // 2
        ax, ay = abs(x - h), abs(y - h)
        return ((math.atan((ax + 1) / h) - math.atan(ax / h))
                * (math.atan((ay + 1) / h) - math.atan(ay / h)))

    f, oy, ox = out_texel
    r = dirs(f, ox, oy)
    a2 = (roughness**2) ** 2
    w = np.zeros((6, n, n))
    for s in range(6):
        for y in range(n):
            for x in range(n):
                L = dirs(s, x, y)
                if L @ r < cos_cut:
                    continue
                h = L + r
                h /= np.linalg.norm(h)
                c = min(max(r @ h, 0.0), 1.0)
                d = (c * a2 - c) * c + 1.0
                w[s, y, x] = max(L @ r, 0.0) * a2 / (d * d * math.pi) * area(x, y) / 4.0
    return w


def cube_ops(method, device):
    dev = _cuda(device)
    n = 16
    g = torch.Generator(device=dev).manual_seed(1)
    tex = torch.rand(6, n, n, 3, device=dev, generator=g)
    d = cubemap.texel_dirs(n, dev)
    rt = float((cubemap.sample(tex, d) - tex).abs().max())
    rot = baking.face_rotations(dev)
    ys, xs = torch.meshgrid(torch.arange(n, device=dev).float(), torch.arange(n, device=dev).float(),
                            indexing="ij")
    ray = torch.stack(((xs - n / 2 + 0.5) / (n / 2), (ys - n / 2 + 0.5) / (n / 2),
                       torch.ones_like(xs)), -1)
    face_err = max(float((torch.nn.functional.normalize(ray @ rot[f].T, dim=-1) - d[f]).abs().max())
                   for f in range(6))
    if rt > 1e-5 or face_err > 1e-5:
        raise AssertionError(f"cube lookup round trip {rt}, face rotation mismatch {face_err}")
    # seamless filtering: a linear function of the direction is reproduced across edges
    a = torch.tensor([0.3, -0.5, 0.8], device=dev)
    lin = (d @ a)[..., None] + 1.0
    q = torch.nn.functional.normalize(torch.randn(20000, 3, device=dev, generator=g), dim=-1)
    edge_err = float((cubemap.sample(lin, q)[:, 0] - (q @ a + 1.0)).abs().max())
    if edge_err > 0.05:
        raise AssertionError(f"cube filtering is not continuous across faces ({edge_err})")
    # diffuse operator vs explicit formula on one output texel
    op = cubemap.diffuse_operator(n, dev)
    area = cubemap.texel_area(n, dev)
    ref = (d.reshape(-1, 3) @ d[2, 5, 7]).clamp(0, 0.999) * area[None].expand(6, n, n).reshape(-1)
    diff_err = float((op[(2 * n + 5) * n + 7] - ref / 3.141592).abs().max())
    # GGX pre-filter: constant in -> constant out; weights vs independent numpy reference
    # (GS-IR level 2: 64^2 with roughness 0.36; nvdiffrec's tile culling assumes small tiles)
    ns, rs, texel = 64, 0.36, (2, 21, 40)
    const = torch.full((6, ns, ns, 3), 0.6, device=dev)
    sp = cubemap.specular_filter(const, rs)
    if float((sp - 0.6).abs().max()) > 1e-4:
        raise AssertionError("GGX pre-filter of a constant map is not constant")
    cos_cut = cubemap.ndf_cutoff(rs, 0.99)
    wref = torch.tensor(_ref_spec_weights(ns, rs, cos_cut, texel), device=dev).float()
    x = torch.rand(6, ns, ns, 3, device=dev, generator=g, requires_grad=True)
    out = cubemap.specular_filter(x, rs)
    exp_val = (wref[..., None] * x.detach()).sum((0, 1, 2)) / wref.sum()
    spec_err = float((out[texel] - exp_val).abs().max())
    out[texel].sum().backward()
    grad_err = float((x.grad - (wref / wref.sum())[..., None].expand(6, ns, ns, 3)).abs().max())
    if diff_err > 1e-5 or spec_err > 1e-4 or grad_err > 1e-4:
        raise AssertionError(f"filters differ from references: diffuse {diff_err}, "
                             f"specular {spec_err}, gradient {grad_err}")
    return {"roundtrip_err": rt, "face_rotation_err": face_err, "edge_err": edge_err,
            "diffuse_err": diff_err, "specular_err": spec_err, "specular_grad_err": grad_err,
            "cos_cutoff_r036": cos_cut}


# ---------------------------------------------------------------- C5 (shading)
def _lut_ref(lut, u, v):
    arr = lut[0].cpu().numpy().astype(np.float64)  # [256(v), 256(u), 2]
    x = min(max(u * 256 - 0.5, 0.0), 255.0)
    y = min(max(v * 256 - 0.5, 0.0), 255.0)
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, 255), min(y0 + 1, 255)
    fx, fy = x - x0, y - y0
    return ((1 - fx) * (1 - fy) * arr[y0, x0] + fx * (1 - fy) * arr[y0, x1]
            + (1 - fx) * fy * arr[y1, x0] + fx * fy * arr[y1, x1])


def pbr_shading_probe(method, device):
    dev = _cuda(device)
    e = 0.6
    base = torch.full((6, 64, 64, 3), e, device=dev, requires_grad=True)
    spec, diffuse = light.build_mips(base)
    lut = light.brdf_lut(dev)
    h = w = 4
    nrm = torch.nn.functional.normalize(torch.tensor([0.2, 0.4, -1.0], device=dev), dim=0)
    normals = nrm.expand(h, w, 3).contiguous()
    view = normals.clone()
    albedo = torch.tensor([0.2, 0.5, 0.8], device=dev).expand(h, w, 3).clone().requires_grad_(True)
    rough = torch.full((h, w, 1), 0.3, device=dev, requires_grad=True)
    metal = torch.full((h, w, 1), 0.25, device=dev, requires_grad=True)
    vis = torch.full((h, w, 1), 0.7, device=dev)
    irr = torch.full((h, w, 1), 0.1, device=dev)
    mask = torch.ones(h, w, 1, dtype=torch.bool, device=dev)
    out = light.pbr_shading(spec, diffuse, lut, normals, view, albedo, rough, mask,
                            metallic=metal, occlusion=vis, irradiance=irr)
    dval = float(cubemap.sample(diffuse.detach(), nrm[None])[0, 0])
    A, B = _lut_ref(lut, 1.0, 0.3)
    a = np.array([0.2, 0.5, 0.8])
    f0 = (1 - 0.25) * 0.04 + a * 0.25
    expected = np.clip(a * (dval * 0.7 + 0.3 * 0.1) + e * (f0 * A + B), 0, 1)
    err = float(np.abs(out["rgb"][0, 0].detach().cpu().numpy() - expected).max())
    if err > 2e-4:
        raise AssertionError(f"PBR shading differs from the closed form by {err}")
    out["rgb"].sum().backward()
    grads = {k: float(t.grad.abs().sum()) for k, t in (("albedo", albedo), ("roughness", rough),
                                                     ("metallic", metal), ("cube", base))}
    if not all(v > 0 for v in grads.values()):
        raise AssertionError(f"missing PBR gradients {grads}")
    # occlusion removed (V = 1) must change the diffuse term
    out1 = light.pbr_shading(spec, diffuse, lut, normals, view, albedo, rough, mask,
                             metallic=metal, occlusion=torch.ones_like(vis), irradiance=irr)
    if float((out1["rgb"] - out["rgb"]).abs().max()) < 1e-3:
        raise AssertionError("occlusion has no effect on shading")
    return {"closed_form_err": err, "diffuse_of_constant": dval, "lut_AB": [A, B], **grads}


# ---------------------------------------------------------------- C4
def indirect_volumes(method, device):
    dev = _cuda(device)
    g = torch.Generator().manual_seed(2)
    res, d2 = 4, 9
    coeffs = torch.randn(res, res, res, d2, 1, generator=g)
    aabb = torch.tensor([-1.0, -1, -1, 1, 1, 1])
    p = torch.rand(30, 3, generator=g) * 2.6 - 1.3
    nrm = torch.nn.functional.normalize(torch.randn(30, 3, generator=g), dim=-1)
    comps = volumes.sh_components(3, nrm.double())
    ref = []
    grid = 2.0 / (res - 1)
    for i in range(30):
        nx = ((p[i] + 1) / grid).clamp(0, res - 1)
        fl = nx.floor()
        o = nx - fl
        acc = torch.zeros(d2, dtype=torch.float64)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = 1.0
                    for a, dd in enumerate((dx, dy, dz)):
                        w *= float(o[a]) if dd else 1 - float(o[a])
                    ix = [min(int(fl[a]) + dd, res - 1) for a, dd in enumerate((dx, dy, dz))]
                    acc += w * coeffs[ix[0], ix[1], ix[2], :, 0].double()
        ref.append(max(float(acc @ comps[i]), 0.0))
    c = coeffs.to(dev).requires_grad_(True)
    got = volumes.query_irradiance(c, aabb.to(dev), p.to(dev), nrm.to(dev))[:, 0]
    err = float((got.cpu().double() - torch.tensor(ref)).abs().max())
    if err > 1e-4:
        raise AssertionError(f"irradiance query differs from loops by {err}")
    got.sum().backward()
    if c.grad is None or not torch.count_nonzero(c.grad):
        raise AssertionError("no gradient reaches the irradiance coefficients")
    # V = 0 -> the diffuse light is the irradiance
    base = torch.full((6, 64, 64, 3), 0.6, device=dev)
    spec, diffuse = light.build_mips(base)
    h = 2
    nn_ = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(h, h, 3).contiguous()
    irr = torch.full((h, h, 1), 0.37, device=dev)
    out = light.pbr_shading(spec, diffuse, light.brdf_lut(dev), nn_, nn_,
                            torch.ones(h, h, 3, device=dev), torch.ones(h, h, 1, device=dev),
                            torch.ones(h, h, 1, dtype=torch.bool, device=dev),
                            occlusion=torch.zeros(h, h, 1, device=dev), irradiance=irr)
    if float((out["diffuse_light"] - 0.37).abs().max()) > 1e-6:
        raise AssertionError("fully occluded pixels do not use the irradiance volume")
    return {"irradiance_err": err, "coeff_grad_norm": float(c.grad.norm())}


# ---------------------------------------------------------------- staged pipeline
def _small_method(method):
    comps = []
    for c in method.components:
        if c.name == "gsir-density":
            cfg = dict(c.config)
            cfg.update({"pbr_start": 2, "stage1_steps": 2, "occ_res": 12, "occ_bound": 4.0,
                        "occ_cube_res": 32, "cube_res": 64, "irr_res": 8, "occ_batch": 8})
            c = replace(c, config=cfg)
        comps.append(c)
    return replace(method, components=tuple(comps))


def stage_switch(method, device):
    dev = _cuda(device)
    small = _small_method(method)
    renderer, density, losses = build(small)
    renderer.bank = None
    params, opts = _params(small, _plane(dev, 20, 3.0, 1.2), 0.08, 0.9, dev)
    cam = _camera(dev, 24, 20.0)
    target = torch.rand(24, 24, 3, device=dev, generator=torch.Generator(device=dev).manual_seed(0))
    saved = bank_mod.ACTIVE["bank"]
    log = []
    try:
        density.initialize(params, opts, 1.0)
        if float((torch.nn.functional.normalize(params["gsir_normal"], dim=-1)
                  - torch.tensor([0.0, 0, 1], device=dev)).abs().max()) > 0:
            raise AssertionError("normals not initialised to (0, 0, 1)")
        for step in range(1, 5):
            out = renderer.render(params, cam, sh_degree=0)
            out["step"] = step
            density.before_backward(params, opts, step, out)
            vals = {n: f(params, out, target) for n, f in losses}
            sum(vals.values()).backward()
            grads = {k: params[k].grad is not None and bool(torch.count_nonzero(params[k].grad))
                     for k in ("means", "opacities", "sh0", "gsir_normal", "gsir_albedo")}
            for o in opts.values():
                o.step()
                o.zero_grad(set_to_none=True)
            density.after_backward(params, opts, step, out)
            check_state(small, params, opts)
            log.append({"step": step, "stage": out["gsir_stage"], "grads": grads,
                        "losses": {k: float(v) for k, v in vals.items()}})
        bank = density.bank
        if log[1]["stage"] != 1 or log[2]["stage"] != 2 or bank.stage != 2:
            raise AssertionError(f"stage sequence wrong: {[r['stage'] for r in log]}")
        if bank.occ_ids is None or bool((bank.occ_ids < 0).any()):
            raise AssertionError("occlusion volumes were not baked at pbr_start")
        g1, g3 = log[0]["grads"], log[2]["grads"]
        if not (g1["means"] and g1["gsir_normal"]) or g1["gsir_albedo"]:
            raise AssertionError(f"stage-1 gradients wrong: {g1}")
        if g3["means"] or g3["opacities"] or g3["sh0"] or not g3["gsir_albedo"]:
            raise AssertionError(f"stage-2 gradients wrong: {g3}")
        if bank.light_steps != 2:
            raise AssertionError(f"light optimizer stepped {bank.light_steps} times, expected 2")
        state = density.state["gsir"]
        if state["stage"] != 2 or "base" not in state or "occ_ids" not in state:
            raise AssertionError("bank is not exported in density.state")
        # a bank restored from the export renders the same evaluation image
        restored = bank_mod.GSIRBank.from_export(state, {}, dev)
        with torch.no_grad():
            a = renderer.render(params, cam)["rgb"]
            renderer.bank = restored
            b = renderer.render(params, cam)["rgb"]
            renderer.bank = None
        if float((a - b).abs().max()) > 1e-5:
            raise AssertionError("restored bank renders differently")
    finally:
        bank_mod.ACTIVE["bank"] = saved
    return {"stages": [r["stage"] for r in log], "light_steps": bank.light_steps,
            "probes_baked": bank.bake_stats["probes"], "losses": log[-1]["losses"]}
