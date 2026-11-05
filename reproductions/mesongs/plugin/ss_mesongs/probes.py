"""Executable mechanism checks for MesonGS (each fails if its mechanism is removed or wrong).

Scene / camera helpers follow the c3dgs plugin probes (reproductions/c3dgs/plugin/ss_c3dgs/
probes.py, after pup-3dgs).
"""

import math
from dataclasses import replace

import numpy as np
import torch


def _require_cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ProbeUnavailable("mesongs probes need CUDA (gsplat, official VQ kernel)")


def _scene(n, device, seed, spread=0.6):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.empty(n, 3).uniform_(-spread, spread, generator=g)
    means[:, 2] = torch.empty(n).uniform_(-0.3, 0.3, generator=g)
    params = {
        "means": means,
        "scales": torch.empty(n, 3).uniform_(math.log(0.04), math.log(0.15), generator=g),
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.logit(torch.empty(n).uniform_(0.2, 0.7, generator=g)),
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.8,
        "shN": torch.randn(n, 15, 3, generator=g) * 0.2,
    }
    return {k: v.to(device) for k, v in params.items()}


def _camera(device, angle=0.15, width=64, height=48, f=60.0, dist=3.0):
    c, s = math.cos(angle), math.sin(angle)
    R = torch.tensor([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    viewmat = torch.eye(4)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = torch.tensor([0.05, -0.03, dist])
    K = torch.tensor([[f, 0.0, width / 2 - 0.3], [0.0, f * 1.02, height / 2 + 0.2],
                      [0.0, 0.0, 1.0]])
    return {"K": K.to(device), "viewmat": viewmat.to(device), "width": width, "height": height}


def _close(a, b, name, rtol=1e-4, atol=1e-6):
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    if a.shape != b.shape or not torch.allclose(a, b, rtol=rtol, atol=atol):
        err = float((a - b).abs().max()) if a.shape == b.shape else float("inf")
        raise AssertionError(f"{name}: max abs error {err:.3e}")
    return float((a - b).abs().max()) if a.numel() else 0.0


# ---------------------------------------------------------------------------------------------
def importance(method, device):
    """I_d from the gsplat gradient trick vs an independent compositor of gsplat's per-Gaussian
    alphas (one-hot colour renders give alpha_i T_i per pixel); volume term and prune quantile."""
    _require_cuda(device)
    from gsplat import rasterization

    from . import importance as imp

    n = 9
    p = _scene(n, device, 1)
    p["means"][-1] = torch.tensor([0.0, 50.0, 0.0], device=device)  # invisible -> I_d = 0
    cams = [_camera(device, a) for a in (0.0, 0.4)]
    got = imp.view_dependent(p, cams)
    # oracle: blend weight of Gaussian g at each pixel = render of colour e_g (one-hot channels)
    ref = torch.zeros(n, device=device, dtype=torch.float64)
    eye = torch.eye(n, device=device)
    for cam in cams:
        render, _, _ = rasterization(
            means=p["means"], quats=p["quats"], scales=torch.exp(p["scales"]),
            opacities=torch.sigmoid(p["opacities"]), colors=eye, viewmats=cam["viewmat"][None],
            Ks=cam["K"][None], width=cam["width"], height=cam["height"], sh_degree=None,
            packed=False)
        ref += render[0].double().sum((0, 1))
    err = _close(got, ref, "I_d", rtol=1e-4, atol=1e-5)
    if float(got[-1]) != 0.0 or float(got[:-1].min()) <= 0:
        raise AssertionError("I_d must be 0 exactly for invisible and > 0 for visible Gaussians")
    v = torch.prod(torch.exp(p["scales"]), 1)
    vref = torch.sort(v, descending=True).values[int(0.9 * n)]
    _close(imp.volume_term(p["scales"], 0.1), (v / vref) ** 0.1, "volume term")
    x = torch.tensor([5.0, 1.0, 3.0, 2.0, 4.0, 0.5, 6.0, 7.0, 8.0, 9.0])
    m = imp.prune_mask(0.28, x)  # int(0.28 * 9) = 2 -> value 2.0 -> prune {0.5, 1, 2}
    if m.tolist() != [False, True, False, True, False, True, False, False, False, False]:
        raise AssertionError(f"prune mask {m.tolist()} differs from the official quantile rule")
    return {"I_d_max_error": err, "pruned": int(m.sum())}


def octree(method, device):
    from . import octree as oc

    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, (500, 3)).astype(np.float32)
    pts[:50] = pts[50:100] + 1e-5  # forced duplicates
    feats = rng.normal(size=(500, 6)).astype(np.float32)
    keys_ref, merged_ref, par_ref = oc.octree_codes_reference(pts, feats, 6)
    xyz, merged, V, keys, par, inverse = oc.voxelize(pts, feats, 6)
    if not np.array_equal(keys, keys_ref) or not np.array_equal(par, par_ref):
        raise AssertionError("voxel keys / bounds differ from the official octreecodes")
    err = float(np.abs(merged - merged_ref).max())
    if err > 1e-5:
        raise AssertionError(f"mean merge differs from the official np.mean: {err}")
    if len(keys) >= len(pts):
        raise AssertionError("duplicates were not merged")
    lo = par[[0, 2, 4]]
    step = (par[[1, 3, 5]] - lo) / 2**6
    if not np.allclose(xyz, lo + V * step, atol=1e-6):
        raise AssertionError("decoded positions are not the lower voxel corners")
    if not np.allclose(merged[inverse[:3]].sum(), merged[inverse][:3].sum()):
        raise AssertionError("inverse map")
    return {"voxels": int(len(keys)), "merge_error": err}


def raht(method, device):
    from . import raht as R

    dev = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    depth = 6
    V = np.unique(rng.integers(0, 2**depth, (1500, 3)), axis=0)
    w, val, _ = R.copy_and_sort(V)
    a, b = R.haar3d_param(depth, w, val), R.haar3d_param_reference(depth, w, val)
    for k in ("iW1", "iW2", "iLeft_idx", "iRight_idx"):
        if not all(np.array_equal(x, y) for x, y in zip(a[k], b[k])):
            raise AssertionError(f"fast haar3D_param {k} differs from the verbatim port")
    a, b = R.inv_haar3d_param(V, depth), R.inv_haar3d_param_reference(V, depth)
    if not np.array_equal(a["pos"], b["pos"]) or not all(
            np.array_equal(np.asarray(x), np.asarray(y)) for k in a if k != "pos"
            for x, y in zip(a[k], b[k])):
        raise AssertionError("fast inv_haar3D_param differs from the verbatim port")
    T = R.RAHTTables(V, depth, dev)
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(len(V), 7, generator=g).to(dev)
    C = R.forward_transform(x, T)
    rt = _close(R.inverse_transform(C, T), x, "inverse(forward(x))", atol=2e-5)
    _close(C.norm(), x.norm(), "orthonormality", rtol=1e-5)
    _close(C[0], x.sum(0) / math.sqrt(len(V)), "DC = sum / sqrt(N)", atol=2e-5)
    smooth = torch.from_numpy(V.astype(np.float32)).to(dev) @ torch.ones(3, 7, device=dev)
    Cs = R.forward_transform(smooth, T)
    if float(Cs[1:].abs().mean()) > 0.2 * float(smooth.abs().mean()):
        raise AssertionError("RAHT does not compact a smooth signal into few coefficients")
    # custom adjoint backward vs autograd through the functional loops
    xa = x.clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True)
    wgt = torch.randn(len(V), 7, generator=g).to(dev)
    (R.iraht(R.raht(xa, T) * wgt, T) * wgt).sum().backward()
    (R.inverse_transform_functional(R.forward_transform_functional(xb, T) * wgt, T)
     * wgt).sum().backward()
    gerr = _close(xa.grad, xb.grad, "RAHT adjoint gradient", atol=5e-5)
    return {"roundtrip_error": rt, "gradient_error": gerr, "points": int(len(V))}


def block_quant(method, device):
    from .quant import BlockQuantizer, split_length

    if split_length(10, 3) != [4, 3, 3] or split_length(9, 3) != [3, 3, 3]:
        raise AssertionError("split_length")
    g = torch.Generator().manual_seed(0)
    x = torch.randn(10, 2, generator=g)
    q = BlockQuantizer(10, 2, 3, 8, "cpu")
    y = q(x)
    for c in range(2):
        start = 0
        for b, length in enumerate(split_length(10, 3)):
            blk = x[start:start + length, c]
            mx, mn = max(float(blk.max()), 0.0), min(float(blk.min()), 0.0)
            s = (mx - mn) / 255.0
            z = round(min(max(255.0 - mx / s, 0.0), 255.0))
            ref = s * (torch.clamp(z + blk / s, 0, 255).round() - z)
            _close(y[start:start + length, c], ref, f"block ({c},{b})", atol=1e-6)
            start += length
    # running range only widens
    q(x * 0.1)
    _close(q.max_val, BlockQuantizer(10, 2, 3, 8, "cpu")._block_minmax(x)[1].clamp(min=0),
           "running max kept")
    xr = x.clone().requires_grad_(True)
    q(xr).sum().backward()
    if not torch.all((xr.grad == 0) | (xr.grad == 1)) or float(xr.grad.sum()) < 8:
        raise AssertionError("straight-through gradient missing")
    err = float((y - x).abs().max())
    if err == 0 or err > 0.05:
        raise AssertionError(f"quantisation error {err} not 8-bit-like")
    return {"max_error": err}


def euler(method, device):
    from . import euler as E

    g = torch.Generator().manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(500, 4, generator=g, dtype=torch.float64), dim=1)
    e = E.to_euler(q)
    err = _close(E.rotation_of(e), E.quat_to_matrix(q), "R(euler(q)) = R(q)", atol=1e-6)
    q2 = E.matrix_to_quaternion(E.rotation_of(e))
    _close(E.quat_to_matrix(q2), E.quat_to_matrix(q), "quaternion recovery", atol=1e-6)
    swapped = E.rotation_from_euler(e[:, 0], e[:, 1], e[:, 2])
    if torch.allclose(swapped, E.quat_to_matrix(q), atol=1e-3):
        raise AssertionError("argument order of build_rotation_from_euler is not exercised")
    s = torch.rand(500, 3, generator=g, dtype=torch.float64)
    cov = E.covariance_from_euler(s, e)
    L = E.quat_to_matrix(q) * s[:, None, :]
    _close(cov, L @ L.transpose(1, 2), "covariance from Euler", atol=1e-6)
    return {"max_error": err}


def _bank_setup(device, n=400, depth=8, n_block=4):
    from . import bank as B
    from . import octree as oc
    from . import raht as R
    from . import vq

    p = _scene(n, device, 5, spread=0.5)
    feats = torch.cat([p["opacities"][:, None], p["sh0"].flatten(1), p["shN"].flatten(1),
                       p["scales"], p["quats"]], -1).cpu().numpy()
    xyz, merged, V, keys, par, _ = oc.voxelize(p["means"].cpu().numpy(), feats, depth)
    m = torch.from_numpy(merged).to(device)
    k = len(xyz)
    q = {"means": torch.from_numpy(xyz).float().to(device), "opacities": m[:, 0],
         "sh0": m[:, 1:4].reshape(k, 1, 3), "shN": m[:, 4:49].reshape(k, 15, 3),
         "scales": m[:, 49:52], "quats": m[:, 52:56]}
    tables = R.RAHTTables(V, depth, device)
    gen = torch.Generator(device=device).manual_seed(0)
    cb, idx = vq.vq_features(q["shN"].flatten(1), torch.rand(k, device=device), 32, 256, 20,
                             0.8, generator=gen)
    bank = B.MesonBank(tables, n_block, 8, cb, idx, 1e-3, device)
    return q, bank, {"keys": keys, "params": par, "depth": depth}, tables


def codec_roundtrip(method, device):
    _require_cuda(device)
    import io
    import zipfile

    from splatstudio.backends.gsplat import Renderer
    from splatstudio.specs import ComponentSpec

    from . import codec
    from .renderer import rasterize

    q, bank, geo, tables = _bank_setup(device)
    cam = _camera(device)
    with torch.no_grad():
        ref = rasterize(bank.attributes(q), cam, 3)["rgb"]
    data, members = codec.encode(bank, q, geo["keys"], geo["params"], geo["depth"])
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = sorted(z.namelist())
    if names != sorted(codec.MEMBERS):
        raise AssertionError(f"bitstream members {names}")
    dec = codec.decode(data, device)
    dec2 = codec.decode(data, device, tables=tables)
    for k in dec:
        _close(dec[k], dec2[k], f"decode with rebuilt tables ({k})", atol=1e-6)
    _close(dec["means"], q["means"], "positions", atol=0)
    base = Renderer(ComponentSpec("gsplat", "renderer", "", (), (), {"absgrad": False,
                                                                     "antialiased": False}))
    with torch.no_grad():
        out = base.render({k: v.contiguous() for k, v in dec.items()}, cam, 3)["rgb"]
    err = _close(out, ref, "decoded render = bank render", atol=2e-4)
    # the decoded attributes are quantised (not the raw ones)
    if torch.allclose(dec["scales"], q["scales"]):
        raise AssertionError("scales are not quantised in the bitstream")
    return {"zip_bytes": len(data), "members": members, "render_error": err}


def pipeline(method, device):
    """Tiny end-to-end run of the density: compress (prune, merge, VQ), fine-tune steps through
    the renderer, frozen last step, encode -> decode into params."""
    _require_cuda(device)
    from splatstudio.components import build

    from . import bank as B

    n = 600
    small = replace(method, components=tuple(
        replace(c, config={**c.config, "base_steps": 3, "total_steps": 8, "vq_steps": 10,
                           "vq_batch_size": 256, "codebook_size": 16, "depth": 8, "n_block": 3,
                           "prune_percent": 0.25})
        if c.kind == "density" else c for c in method.components))
    renderer, density, losses = build(small)
    p = {k: torch.nn.Parameter(v) for k, v in _scene(n, device, 7, spread=0.5).items()}
    lrs = {"means": 1.6e-4, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3,
           "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([v], lr=lrs[k], eps=1e-15) for k, v in p.items()}
    density.initialize(p, opts, 1.0)
    cams = [_camera(device, a) for a in (0.0, 0.3, -0.3)]
    density.set_training_cameras(cams)
    target = torch.full((48, 64, 3), 0.4, device=device)
    record = {}
    for step in range(1, 9):
        cam = cams[step % 3]
        out = renderer.render(p, cam, 3)
        out["step"] = step
        density.before_backward(p, opts, step, out)
        loss = sum(fn(p, out, target) for _, fn in losses)
        loss.backward()
        before = {k: v.detach().clone() for k, v in p.items()}
        cb_before = density.bank.codebook.detach().clone() if density.bank else None
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        opts["means"].param_groups[0]["lr"] = 1.6e-4 * 0.01 ** (step / 8)
        if step == 8:
            for k in ("opacities", "sh0", "scales", "quats"):
                if not torch.equal(before[k], p[k]):
                    raise AssertionError("last runtime step was not frozen")
        density.after_backward(p, opts, step, out)
        if step == 3:
            ev = density.history[-1]
            record["pruned"] = ev["pruned"]
            record["voxels"] = ev["gaussians"]
            if ev["pruned"] != int(0.25 * (n - 1)) + 1:
                raise AssertionError(f"pruned {ev['pruned']} != official quantile count")
            if B.ACTIVE["bank"] is None or len(p["means"]) != ev["gaussians"]:
                raise AssertionError("bank not active / params not voxelised")
        if 3 < step < 8:
            if not torch.equal(before["means"], p["means"]):
                raise AssertionError("positions moved during the fine-tune")
            if step < 7 and torch.equal(before["opacities"], p["opacities"]):
                raise AssertionError("opacities did not change during the fine-tune")
            if step < 7 and torch.equal(cb_before, density.bank.codebook):
                raise AssertionError("SH codebook did not change during the fine-tune")
    if density.encoded is None or B.ACTIVE["bank"] is not None:
        raise AssertionError("encode did not run / bank still active")
    if density.bank.steps != 4:
        raise AssertionError(f"codebook steps {density.bank.steps} != 4 (5 FT iterations - 1)")
    record.update(zip_bytes=density.encoded["zip_bytes"], bank_steps=density.bank.steps)
    return record


def schedule(method, device):
    from . import schedule as S

    d = next(c for c in method.components if c.kind == "density").config
    b, t = d["base_steps"], d["total_steps"]
    if S.events(b, b, t)[:2] != ["base", "compress"] or "encode" not in S.events(t, b, t):
        raise AssertionError("compress / encode events")
    if t > b:
        steps = sum("bank_step" in S.events(s, b, t) for s in range(1, t + 1))
        if steps != t - b - 1 or "freeze" not in S.events(t - 1, b, t):
            raise AssertionError("codebook steps / freeze")
    if S.events(30000, 30000, 30000) != ["base", "compress", "encode"]:
        raise AssertionError("no-finetune schedule")
    if abs(S.inferred_run_length(1.0, 0.01 ** (1 / t)) - t) > 1e-6:
        raise AssertionError("run-length inference")
    return {"base_steps": b, "total_steps": t}
