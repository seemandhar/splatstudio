"""Mechanism probes for the SuGaR port. Each raises when its mechanism is removed or wrong."""

import math
from dataclasses import replace

import torch
import torch.nn.functional as F

from .context import SugarContext
from .geometry import (
    bound_gaussians,
    edge_face_pairs,
    expon_lr,
    inverse_scaled_rotation,
    level_set_points,
    neighbour_density,
    normal_consistency,
    quat_to_rotmat,
    sdf_values,
)


def _unavailable(msg):
    from splatstudio.verification import ProbeUnavailable

    raise ProbeUnavailable(msg)


def _loss_spec(name, factory, config):
    from splatstudio import ComponentSpec

    return ComponentSpec(name, "loss", f"ss_sugar.losses:{factory}", (), (), config)


def _plane_scene(device, n_side=6, z0=4.0, tilt=0.0, seed=0):
    """Flat opaque Gaussians tiling the plane z = z0 (normal = local z axis)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    xs = torch.linspace(-1, 1, n_side)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    means = torch.stack((gx.reshape(-1), gy.reshape(-1), torch.full((n_side**2,), z0)), -1)
    n = len(means)
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1
    if tilt:
        # tilt every other Gaussian about x by `tilt` radians
        quats[::2] = torch.tensor([math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0])
    scales = torch.tensor([0.25, 0.25, 0.01]).expand(n, 3).clone()
    scales = scales * (1 + 0.05 * torch.rand(n, 3, generator=g))
    opac = torch.full((n,), 0.95)
    return {k: v.to(device) for k, v in
            dict(means=means, quats=quats, scales=scales, opac=opac).items()}


def _outputs(scene, depth_offset, device, size=32, step=10000):
    means, quats, scales, opac = (scene[k] for k in ("means", "quats", "scales", "opac"))
    K = torch.tensor([[24.0, 0, size / 2], [0, 24.0, size / 2], [0, 0, 1]], device=device)
    cam = {"viewmat": torch.eye(4, device=device), "K": K, "width": size, "height": size}
    z0 = float(means[:, 2].mean())
    alpha = torch.ones(size, size, 1, device=device)
    depth = torch.full((size, size, 1), z0 + depth_offset, device=device)
    n = len(means)
    return {
        "rgb": torch.zeros(size, size, 3, device=device, requires_grad=True),
        "alpha": alpha, "depth_accumulated": depth, "opacities": opac,
        "gaussians": (means, quats, scales), "camera": cam, "step": step,
        "info": {"radii": torch.ones(1, n, 2, device=device)},
    }


def _ctx_for(scene, k=16):
    from .geometry import knn

    ctx = SugarContext()
    ctx.knn = knn(scene["means"], min(k, len(scene["means"])))
    ctx.extent = 4.0
    return ctx


def _build_loss(factory, config, ctx):
    from . import context as C

    C._CURRENT[:] = [ctx]
    from .losses import create_normal, create_sdf

    fn = {"create_sdf": create_sdf, "create_normal": create_normal}[factory]
    return fn(_loss_spec(factory, factory, config))


def sdf_field(method, device):
    device = "cpu"  # pure tensor math; avoids per-element syncs on a shared GPU
    torch.manual_seed(0)
    # 1) density vs an independent covariance formula
    n, m, k = 20, 50, 5
    means = torch.randn(n, 3)
    quats = F.normalize(torch.randn(n, 4), dim=-1)
    scales = torch.rand(n, 3) * 0.5 + 0.1
    opac = torch.rand(n)
    x = torch.randn(m, 3)
    idx = torch.randint(n, (m, k))
    rot = quat_to_rotmat(quats)
    d, _, _ = neighbour_density(x, idx, means, inverse_scaled_rotation(rot, scales), opac, 0.5)
    ref = torch.zeros(m)
    for i in range(m):
        for j in idx[i]:
            cov = rot[j] @ torch.diag(scales[j] ** 2) @ rot[j].T
            diff = x[i] - means[j]
            ref[i] += 0.5 * opac[j] * torch.exp(-0.5 * diff @ torch.linalg.inv(cov) @ diff)
    density_err = float((d - ref).abs().max())
    if density_err > 1e-4:
        raise AssertionError(f"neighbour density differs from Eq. 1: {density_err}")
    # 2) a flat alpha=1 Gaussian, rho=1: f(x) = |<x-mu, n>| along its normal (Eq. 5-7)
    t = torch.linspace(-0.05, 0.05, 11)
    s = torch.tensor([[0.5, 0.5, 0.02]])
    pts = t[:, None] * torch.tensor([[0.0, 0.0, 1.0]])
    d1, _, _ = neighbour_density(pts, torch.zeros(11, 1, dtype=torch.long), torch.zeros(1, 3),
                                 inverse_scaled_rotation(torch.eye(3)[None], s), torch.ones(1), 1.0)
    f = sdf_values(d1, torch.full((11,), 0.02))
    plane_err = float((f - t.abs()).abs().max())
    if plane_err > 1e-4:
        raise AssertionError(f"SDF of a flat Gaussian is not the plane distance: {plane_err}")
    # 3) R on a plane of flat Gaussians vs an independent recomputation (explicit covariance
    #    inverse, rho = 1/16, linear depth ramp so bilinear lookup is exact)
    scene = _plane_scene(device)
    ctx = _ctx_for(scene)
    loss = _build_loss("create_sdf", {"samples": 20000, "start": 0}, ctx)
    out = _outputs(scene, 0.0, device)
    size = out["rgb"].shape[0]
    u_c = torch.arange(size, device=device) + 0.5
    ramp = 4.0 + 0.01 * (u_c - size / 2) / (size / 2)
    out["depth_accumulated"] = ramp[None, :, None].expand(size, size, 1).contiguous()
    value = loss({}, out, None)
    t = ctx.cache["terms"]
    xs, nidx = t["x"], t["nidx"]
    rot_s = quat_to_rotmat(scene["quats"])
    cov_inv = torch.linalg.inv(rot_s @ torch.diag_embed(scene["scales"] ** 2) @ rot_s.transpose(1, 2))
    diff = xs[:, None] - scene["means"][nidx]
    q = torch.einsum("mki,mkij,mkj->mk", diff, cov_inv[nidx], diff)
    dref = (scene["opac"][nidx] * torch.exp(-0.5 * q)).sum(-1) / 16.0
    dref = torch.where(dref >= 1, torch.ones_like(dref), dref)
    beta = scene["scales"].min(-1).values[nidx].mean(-1)
    fref = beta * torch.sqrt(-2 * torch.log(dref))
    u = out["camera"]["K"][0, 0] * xs[:, 0] / xs[:, 2] + out["camera"]["K"][0, 2]
    u = u.clamp(0.5, size - 0.5)
    fhat = 4.0 + 0.01 * (u - size / 2) / (size / 2) - xs[:, 2]
    ref = 0.2 * ((fref - fhat.abs()).abs() / (ctx.extent / 10)).mean()
    r_err = float((value - ref).abs() / ref)
    if r_err > 1e-3:
        raise AssertionError(f"R {float(value)} != independent {float(ref)}")
    return {"density_max_err": density_err, "plane_sdf_max_err": plane_err,
            "R": float(value), "R_independent": float(ref), "R_rel_err": r_err}


def normal_regularizer(method, device):
    device = "cpu"
    torch.manual_seed(0)
    res = {}
    for tilt in (0.0, 0.6):
        scene = _plane_scene(device, tilt=tilt)
        scene["quats"].requires_grad_(True)
        ctx = _ctx_for(scene)
        loss = _build_loss("create_normal", {"samples": 256, "start": 0}, ctx)
        out = _outputs(scene, 0.0, device)
        value = loss({}, out, None)
        res[tilt] = value
        if True:
            t = ctx.cache["terms"]
            # independent loop over a few samples
            rot = quat_to_rotmat(scene["quats"].detach())
            smin, axis = scene["scales"].min(dim=-1)
            normals = rot.gather(2, axis[:, None, None].expand(-1, 3, 1))[..., 0]
            errs = []
            for i in range(len(t["x"])):
                g = int(t["gidx"][i])
                wsum, acc = 0.0, torch.zeros(3, device=device)
                ws = []
                for jj, j in enumerate(t["nidx"][i].tolist()):
                    nj = normals[j] * torch.sign(normals[j] @ normals[g])
                    w = float(t["weights"][i, jj]) * float(
                        ((t["x"][i] - scene["means"][j]) @ nj).abs()) / max(float(smin[j]), 1e-6)**2
                    ws.append((w, nj))
                    wsum += w
                for w, nj in ws:
                    acc = acc + w / max(wsum, 1e-6) * nj
                errs.append(float((normals[g] - acc).square().sum()))
            ref = 0.2 * sum(errs) / len(errs)
            res[f"independent_{tilt}"] = ref
            if abs(ref - float(value)) > 1e-3 * ref + 1e-7:
                raise AssertionError(f"R_Norm {float(value)} != independent {ref}")
            grad = torch.autograd.grad(value, scene["quats"])[0]
            if tilt and (not torch.isfinite(grad).all() or float(grad.norm()) == 0):
                raise AssertionError("R_Norm has no gradient to the quaternions")
            res["grad_norm"] = float(grad.norm())
    # Co-planar is ~0 except where the clamped weight sum (1e-6, official) is not reached.
    if not float(res[0.6]) > 2 * float(res[0.0]):
        raise AssertionError(f"R_Norm not larger for tilted than co-planar normals: {res}")
    return {"coplanar": float(res[0.0]), "tilted": float(res[0.6]),
            "independent_coplanar": res["independent_0.0"],
            "independent_tilted": res["independent_0.6"], "grad_norm": res["grad_norm"]}


def level_set(method, device):
    if not torch.cuda.is_available():
        _unavailable("CUDA is unavailable")
    from splatstudio import ComponentSpec

    from .renderer import Renderer

    renderer = Renderer(ComponentSpec("r", "renderer", "", (), (), {}))
    z0, sn = 4.0, 0.05
    params = {
        "means": torch.tensor([[0.0, 0.0, z0]], device=device),
        "quats": torch.tensor([[1.0, 0, 0, 0]], device=device),
        "scales": torch.tensor([[3.0, 3.0, sn]], device=device).log(),
        "opacities": torch.tensor([8.0], device=device),
        "sh0": torch.zeros(1, 1, 3, device=device),
        "shN": torch.zeros(1, 15, 3, device=device),
    }
    size = 48
    K = torch.tensor([[40.0, 0, size / 2], [0, 40.0, size / 2], [0, 0, 1]], device=device)
    cam = {"viewmat": torch.eye(4, device=device), "K": K, "width": size, "height": size}
    with torch.no_grad():
        out = renderer.render(params, cam, sh_degree=0)
        rot = quat_to_rotmat(params["quats"])

        def knn_fn(x, k):
            return torch.zeros(len(x), k, dtype=torch.long, device=x.device)

        pts, nrm, _ = level_set_points(
            out["depth"][..., 0], out["alpha"][..., 0], out["rgb"], cam, params["means"], rot,
            params["scales"].exp(), params["opacities"].sigmoid(), 400, 0.3, knn_fn=knn_fn, k=1)
    if len(pts) < 100:
        raise AssertionError(f"too few level-set points: {len(pts)}")
    center = pts[(pts[:, 0].abs() < 0.3) & (pts[:, 1].abs() < 0.3)]
    # in-plane factor exp(-r^2/(2*3^2)) ~ 1 near the centre; alpha ~ sigmoid(8)
    a = float(torch.sigmoid(torch.tensor(8.0)))
    expected = z0 - sn * math.sqrt(-2 * math.log(0.3 / a))
    err = float((center[:, 2] - expected).abs().max())
    if err > 0.15 * sn:
        raise AssertionError(f"level-set z {center[:, 2].mean():.4f} vs analytic {expected:.4f}")
    align = float(nrm[:, 2].abs().min())
    if align < 0.99:
        raise AssertionError(f"level-set normals not along the wall axis: {align}")
    return {"points": len(pts), "analytic_z": expected, "max_z_err": err, "min_normal_dot": align}


def poisson(method, device):
    try:
        from .mesh import import_open3d, poisson_mesh

        import_open3d()
    except ImportError as exc:
        _unavailable(f"Open3D unavailable: {exc}")
    import numpy as np

    rng = np.random.default_rng(0)
    p = rng.normal(size=(40000, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    col = np.full_like(p, 0.5)
    verts, faces, vcol = poisson_mesh(p, p.copy(), col, np.zeros(3), 2.0, depth=7, quantile=0.1,
                                      target=3000)
    r = np.linalg.norm(verts, axis=1)
    raw, _, _ = poisson_mesh(p, p.copy(), col, np.zeros(3), 2.0, depth=7, quantile=0.1,
                             target=3000, project=False)
    err = float(np.abs(np.linalg.norm(raw, axis=1).mean() - 1.0))
    if err > 0.02 or len(faces) > 3000 or len(faces) < 1000:
        raise AssertionError(f"Poisson sphere radius err {err}, faces {len(faces)}")
    snapped = float(np.abs(r - 1.0).max())
    if snapped > 1e-5:
        raise AssertionError("vertices were not snapped onto the level-set points")
    return {"faces": int(len(faces)), "vertices": int(len(verts)), "radius_err": err,
            "max_snap_err": snapped}


def mesh_binding(method, device):
    verts = torch.tensor([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], requires_grad=True)
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
    bary = torch.tensor([[1 / 3, 1 / 3, 1 / 3]])
    rot2d = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    plane = torch.tensor([[0.1, 0.2], [0.3, 0.4]]).log()
    eps = 1e-6
    means, quats, scales = bound_gaussians(verts, faces, bary, rot2d, plane, math.log(eps))
    fv = verts.detach()[faces]
    if float((means - fv.mean(1)).abs().max()) > 1e-6:
        raise AssertionError("bound means are not barycentres")
    R = quat_to_rotmat(quats.detach())
    rot_err = 0.0
    for f in range(2):
        nrm = F.normalize(torch.cross(fv[f, 1] - fv[f, 0], fv[f, 2] - fv[f, 0], dim=0), dim=0)
        b1 = F.normalize(fv[f, 0] - fv[f, 1], dim=0)
        th = math.atan2(float(rot2d[f, 1]), float(rot2d[f, 0]))
        # Rodrigues rotation of b1 about the normal by theta
        r1 = b1 * math.cos(th) + torch.cross(nrm, b1, dim=0) * math.sin(th)
        rot_err = max(rot_err, float((R[f, :, 0] - nrm).abs().max()),
                      float((R[f, :, 1] - r1).abs().max()))
    if rot_err > 1e-5:
        raise AssertionError(f"bound rotation differs from the supp. parameterisation: {rot_err}")
    expect_s = torch.tensor([[eps, 0.1, 0.2], [eps, 0.3, 0.4]])
    if float((scales - expect_s).abs().max() / 0.4) > 1e-5:
        raise AssertionError("bound scales are not (thickness, s1, s2)")
    means.sum().backward()
    if verts.grad is None or float(verts.grad.abs().sum()) == 0:
        raise AssertionError("no gradient reaches the mesh vertices")
    pairs = edge_face_pairs(faces, 4)
    flat = float(normal_consistency(verts.detach(), pairs))
    folded_v = verts.detach().clone()
    folded_v[3, 2] = 1.0  # fold face 2 up by 45 deg about the diagonal
    angle = math.acos(float(F.cosine_similarity(
        torch.cross(fv[0, 1] - fv[0, 0], fv[0, 2] - fv[0, 0], dim=0),
        torch.cross(folded_v[2] - folded_v[0], folded_v[3] - folded_v[0], dim=0), dim=0)))
    folded = float(normal_consistency(folded_v, pairs))
    if len(pairs) != 1 or flat > 1e-6 or abs(folded - (1 - math.cos(angle))) > 1e-5:
        raise AssertionError(f"normal consistency wrong: pairs {len(pairs)} {flat} {folded}")
    lr = [expon_lr(s, 1.6e-4, 1.6e-6, 30000) for s in (0, 15000, 30000)]
    if abs(lr[0] - 1.6e-4) > 1e-12 or abs(lr[1] - 1.6e-5) > 1e-11 or abs(lr[2] - 1.6e-6) > 1e-12:
        raise AssertionError(f"vertex LR schedule wrong: {lr}")
    return {"rotation_err": rot_err, "flat_consistency": flat, "folded_consistency": folded,
            "fold_expected": 1 - math.cos(angle), "lr_at_0_15k_30k": lr}


def stage_schedule(method, device):
    if not torch.cuda.is_available():
        _unavailable("CUDA is unavailable")
    try:
        from .mesh import import_open3d

        import_open3d()
    except ImportError as exc:
        _unavailable(f"Open3D unavailable: {exc}")
    from splatstudio.components import build
    from splatstudio.state import check_state, initialize

    # Tiny schedule: 3DGS 1-2, entropy (2, 4), prune at 4, SDF from 4, extract after 6.
    comps = []
    for c in method.components:
        cfg = dict(c.config)
        if c.name == "sugar-density":
            cfg.update(coarse_start=2, prune_at=4, extract_at=6, refine_stop_iter=2, knn_every=2,
                       n_total_points=60000, poisson_depth=6, decimation_target=2000)
        elif c.name == "sugar-entropy":
            cfg.update(start=2, end=4)
        elif c.name in ("sugar-sdf", "sugar-normal"):
            cfg.update(start=4, samples=5000)
        comps.append(replace(c, config=cfg))
    m = replace(method, components=tuple(comps))
    renderer, density, losses = build(m)
    ctx = renderer.ctx
    # opaque flat Gaussians on a 1.6 x 1.6 wall at z = 4 plus 20 transparent ones
    g = torch.Generator(device="cpu").manual_seed(0)
    xs = torch.linspace(-0.8, 0.8, 24)
    gx, gy = torch.meshgrid(xs, xs, indexing="ij")
    wall = torch.stack((gx.reshape(-1), gy.reshape(-1), torch.full((576,), 4.0)), -1)
    pts = torch.cat((wall, torch.rand(20, 3, generator=g) - 0.5 + torch.tensor([0, 0, 3.0])))
    pts = pts.to(device)
    params, opts = initialize(m, pts, torch.full_like(pts, 0.5))
    with torch.no_grad():
        params["scales"][:576] = torch.tensor([0.06, 0.06, 0.004], device=device).log()
        params["opacities"][:576] = 4.0
        params["opacities"][576:] = -3.0
    density.initialize(params, opts, 4.0)
    cams = []
    for dx in (-1.0, 0.0, 1.0):
        vm = torch.eye(4, device=device)
        vm[0, 3] = dx
        K = torch.tensor([[60.0, 0, 32], [0, 60.0, 32], [0, 0, 1]], device=device)
        cams.append({"viewmat": vm, "K": K, "width": 64, "height": 64})
    density.set_training_cameras(cams)
    target = torch.full((64, 64, 3), 0.5, device=device)
    record = {}
    for step in range(1, 9):
        out = renderer.render(params, cams[step % 3], sh_degree=0)
        out["step"] = step
        density.before_backward(params, opts, step, out)
        vals = {n: f(params, out, target) for n, f in losses}
        sum(vals.values()).backward()
        if step == 7:
            vg = ctx.mesh.verts.grad
            if vg is None or not torch.isfinite(vg).all() or float(vg.abs().sum()) == 0:
                raise AssertionError("refinement loss does not reach the mesh vertices")
            v_before = ctx.mesh.verts.detach().clone()
        for o in opts.values():
            o.step()
            o.zero_grad(set_to_none=True)
        n_before = len(params["means"])
        density.after_backward(params, opts, step, out)
        check_state(m, params, opts)
        record[step] = {k: float(v) for k, v in vals.items()}
        record[step]["n"] = len(params["means"])
        if step == 4 and record[step]["n"] != n_before - 20:
            raise AssertionError(f"prune at 4 removed {n_before - record[step]['n']} (expected 20)")
        if step == 7 and float((ctx.mesh.verts.detach() - v_before).abs().max()) == 0:
            raise AssertionError("the vertex optimizer did not move the mesh")
    ent = {s: record[s]["sugar-entropy"] for s in record}
    if not (ent[3] > 0 and ent[1] == 0 and ent[2] == 0 and ent[4] == 0 and ent[6] == 0):
        raise AssertionError(f"entropy window wrong: {ent}")
    if not (record[5]["sugar-sdf"] > 0 and record[4]["sugar-sdf"] == 0):
        raise AssertionError("SDF regularizer not active only after prune_at")
    if ctx.mesh is None or record[6]["n"] != len(ctx.mesh.faces):
        raise AssertionError("extraction did not bind one Gaussian per triangle")
    if not record[8]["sugar-normal-consistency"] >= 0 or record[5]["sugar-normal-consistency"]:
        raise AssertionError("normal consistency active outside refinement")
    # The checkpointed base fields must render exactly like the mesh-bound path (ss-eval).
    with torch.no_grad():
        live = renderer.render(params, cams[0], sh_degree=0)["rgb"]
        fresh, _, _ = build(m)
        plain = fresh.render({k: v.detach() for k, v in params.items()}, cams[0],
                             sh_degree=0)["rgb"]
    diff = float((live - plain).abs().max())
    if diff > 1e-4:
        raise AssertionError(f"checkpoint fields do not reproduce the mesh render: {diff}")
    return {"entropy_by_step": ent, "gaussians_by_step": {s: record[s]["n"] for s in record},
            "faces": len(ctx.mesh.faces), "eval_render_max_diff": diff}
