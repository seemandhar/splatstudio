"""Mechanism probes for the SparseGS contributions. Each raises if its mechanism is removed."""

from types import SimpleNamespace

import numpy as np
import torch


def _unavailable(msg):
    from splatstudio.verification import ProbeUnavailable

    raise ProbeUnavailable(msg)


def _camera(device, W=40, H=32, f=40.0):
    return {"viewmat": torch.eye(4, device=device),
            "K": torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=device),
            "width": W, "height": H}


def _params(points, scales, opac, device):
    n = len(points)
    q = torch.zeros(n, 4, device=device)
    q[:, 0] = 1
    return {"means": points, "scales": scales.log(), "quats": q,
            "opacities": torch.logit(opac), "sh0": torch.rand(n, 1, 3, device=device),
            "shN": torch.zeros(n, 15, 3, device=device)}


# ------------------------------------------------------------------ C1
def reference_depths(m2d, con, op, dep, W, H, beta):
    """Independent per-pixel torch reference (float64): all Gaussians sorted by depth, the
    official forward.cu rules (power>0 skip, alpha=min(.99,o G), alpha<1/255 skip, T<1e-4 stop)."""
    order = torch.argsort(dep.detach())
    m2d, con, op, dep = m2d[order], con[order], op[order], dep[order]
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float64, device=m2d.device) + 0.5,
                            torch.arange(W, dtype=torch.float64, device=m2d.device) + 0.5,
                            indexing="ij")
    soft = torch.zeros(H, W, dtype=torch.float64, device=m2d.device)
    alpha_d = torch.zeros_like(soft)
    mode = torch.zeros_like(soft)
    for y in range(H):
        for x in range(W):
            dx = m2d[:, 0] - xs[y, x]
            dy = m2d[:, 1] - ys[y, x]
            power = -0.5 * (con[:, 0] * dx * dx + con[:, 2] * dy * dy) - con[:, 1] * dx * dy
            a = torch.clamp(op * torch.exp(power), max=0.99)
            T = torch.ones((), dtype=torch.float64, device=m2d.device)
            D = Z = AD = AZ = torch.zeros((), dtype=torch.float64, device=m2d.device)
            top_w, top_d = -1.0, 0.0
            for i in range(len(a)):
                if float(power[i]) > 0 or float(a[i]) < 1 / 255:
                    continue
                nT = T * (1 - a[i])
                if float(nT) < 1e-4:
                    break
                w = a[i] * T
                e = torch.exp(beta * (w - 0.3))
                D = D + dep[i] * w * e
                Z = Z + w * e
                AD = AD + dep[i] * w
                AZ = AZ + w
                if float(w) > top_w:
                    top_w, top_d = float(w), float(dep[i])
                T = nT
            Zs = Z if float(Z) != 0 else torch.ones_like(Z)
            AZs = AZ if float(AZ) != 0 else torch.ones_like(AZ)
            soft[y, x] = torch.log(D / Zs + 1e-8)
            alpha_d[y, x] = AD / (AZs + 1e-8)
            mode[y, x] = top_d
    return soft, alpha_d, mode


def softmax_depth(method, device):
    if not torch.cuda.is_available():
        _unavailable("CUDA is unavailable")
    from .depth_raster import render_depths
    from .renderer import rasterize

    g = torch.Generator(device="cpu").manual_seed(3)
    n = 40
    pts = torch.rand(n, 3, generator=g) * torch.tensor([1.6, 1.2, 2.0]) + torch.tensor(
        [-0.8, -0.6, 2.0])
    params = _params(pts.to(device), (torch.rand(n, 3, generator=g) * 0.08 + 0.04).to(device),
                     (torch.rand(n, generator=g) * 0.9 + 0.05).to(device), device)
    cam = _camera(device)
    _, _, info = rasterize(params, cam, 0)
    leaves = {k: info[k].detach().clone().requires_grad_(True)
              for k in ("means2d", "conics", "opacities", "depths")}
    info2 = dict(info, **leaves)
    beta = 5.0
    out = render_depths(info2, beta)
    gout = torch.randn(cam["height"], cam["width"], generator=g).to(device)
    (out["softmax"] * gout).sum().backward()
    ref_in = {k: leaves[k][0].detach().double().clone().requires_grad_(True)
              for k in ("means2d", "conics", "opacities", "depths")}
    ref_s, ref_a, ref_m = reference_depths(ref_in["means2d"], ref_in["conics"],
                                           ref_in["opacities"], ref_in["depths"],
                                           cam["width"], cam["height"], beta)
    (ref_s * gout.double()).sum().backward()
    err = {"softmax": float((out["softmax"].double() - ref_s).abs().max()),
           "alpha": float((out["alpha"].double() - ref_a).abs().max()),
           "mode": float((out["mode"].double() - ref_m).abs().max())}
    gerr = {}
    for k, ref in ref_in.items():
        a, b = leaves[k].grad[0].double(), ref.grad
        gerr[k] = float((a - b).norm() / (b.norm() + 1e-12))
    # beta -> 0: softmax depth = log(alpha depth) where covered
    out0 = render_depths(info2, 0.0)
    cov = out0["alpha"] > 0
    lim = float((out0["softmax"][cov] - out0["alpha"][cov].log()).abs().max())
    # softmax must differ from log alpha depth at beta = 5 (mechanism present)
    sep = float((out["softmax"][cov] - out["alpha"][cov].clamp_min(1e-8).log()).abs().max())
    if max(err.values()) > 1e-3:
        raise AssertionError(f"depth forward mismatch {err}")
    if max(gerr.values()) > 1e-2:
        raise AssertionError(f"depth gradient mismatch (relative) {gerr}")
    if lim > 1e-4:
        raise AssertionError(f"beta->0 limit violated: {lim}")
    if sep < 1e-3:
        raise AssertionError("softmax depth equals log alpha depth at beta=5")
    return {"forward_abs_err": err, "grad_rel_err": gerr, "beta0_limit_err": lim,
            "softmax_vs_alpha": sep, "covered_pixels": int(cov.sum())}


# ------------------------------------------------------------------ C2
def pearson(method, device):
    from .losses import local_pearson_loss, n_patches, pearson_loss, pearson_rows

    g = torch.Generator().manual_seed(0)
    a = torch.rand(64, 96, generator=g)
    b = torch.rand(64, 96, generator=g)
    m = {
        "affine": float(pearson_loss(3.0 * a + 2.0, a)),
        "negated": float(pearson_loss(-a, a)),
        "vs_corrcoef": abs(float(pearson_loss(a, b))
                           - (1 - float(torch.corrcoef(torch.stack([a.flatten(),
                                                                     b.flatten()]))[0, 1]))),
    }
    rows = pearson_rows(a.reshape(8, -1), b.reshape(8, -1))
    ref = torch.stack([1 - torch.corrcoef(torch.stack([x, y]))[0, 1]
                       for x, y in zip(a.reshape(8, -1), b.reshape(8, -1))])
    m["rows_vs_corrcoef"] = float((rows - ref).abs().max())
    m["n_patches"] = n_patches(64, 96, 16, 0.5)
    m["local_affine"] = float(local_pearson_loss(5 * a - 1, a, 16, 0.5,
                                                 torch.Generator().manual_seed(1)))
    m["local_random"] = float(local_pearson_loss(a, b, 16, 0.5, torch.Generator().manual_seed(1)))
    # official numerics: unbiased std inside mean(z_a z_b) -> perfect correlation gives (n-1)/n
    n_full, n_patch = a.numel(), 16 * 16
    if abs(m["affine"] - 1 / n_full) > 1e-4 or abs(m["negated"] - (2 - 1 / n_full)) > 1e-4:
        raise AssertionError(f"Pearson invariances violated {m}")
    if m["vs_corrcoef"] > 1e-3 or m["rows_vs_corrcoef"] > 1e-3:
        raise AssertionError(f"Pearson differs from corrcoef {m}")
    if m["n_patches"] != int(0.5 * 4 * 6) or abs(m["local_affine"] - 1 / n_patch) > 1e-4:
        raise AssertionError(f"local Pearson wrong {m}")
    if m["local_random"] < 0.5:
        raise AssertionError("local Pearson does not penalise uncorrelated depth")
    return m


# ------------------------------------------------------------------ C4
def warp(method, device):
    from scipy.spatial.transform import Rotation

    from .context import CTX, construct_rotation_matrix, forward_warp, warp_viewmats
    from .losses import create_pearson, mask_l1, warp_photometric

    g = torch.Generator().manual_seed(0)
    H, W, f = 24, 32, 30.0
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    img = torch.rand(H, W, 3, generator=g)
    depth = torch.rand(H, W, generator=g)
    eye = torch.eye(4)
    out, mask = forward_warp(img, depth, K, eye, eye, 1.0, 3.0)
    m = {"identity_err": float((out - img).abs().max()), "identity_mask": float(mask.mean())}
    # constant depth Z, target camera translated by +t in x (w2c t_x = -t): shift = -f t / Z
    flat = torch.zeros(H, W)
    flat[0, 0] = 1.0  # min-max renormalisation needs a range; pixel (0,0) is the far one
    Z, t = 2.0, 2.0 * 4 / f  # 4 pixel shift
    trg = torch.eye(4)
    trg[0, 3] = -t
    out, mask = forward_warp(img, flat, K, eye, trg, Z, 5.0)
    m["shift_err"] = float((out[2:-2, 2:W - 6] - img[2:-2, 6:W - 2]).abs().max())
    m["shift_mask_hole"] = float(mask[:, W - 4:].sum())
    axis = np.array([0.2, -0.9, 0.3])
    R = construct_rotation_matrix(axis, 0.3)
    m["rodrigues_err"] = float(np.abs(
        R - Rotation.from_rotvec(-axis / np.linalg.norm(axis) * 0.3).as_matrix()).max())
    vms = [torch.eye(4)]
    vm, src = warp_viewmats(vms, [2.5])[0]
    m["warp_cam_rot_deg"] = float(np.degrees(np.arccos((np.trace(vm[:3, :3].numpy()) - 1) / 2)))
    rgb = torch.rand(H, W, 3, generator=g)
    msk = (torch.rand(H, W, generator=g) > 0.5).float()
    ref = (rgb - img).abs().mul(msk[..., None]).sum() / msk.sum()
    m["mask_l1_err"] = float((mask_l1(rgb, img, msk) - ref).abs())
    m["warp_loss"] = float(warp_photometric(rgb, img, msk))
    # Pearson is skipped on warp iterations (official `not pick_warp_cam`)
    saved = (dict(CTX.draw_cfg), CTX._draw)
    CTX._draw = (7, (False, True))
    p = {"means": torch.zeros(3, 3, requires_grad=True)}
    skipped = float(create_pearson(SimpleNamespace(config={}))(p, {"step": 7}, None))
    CTX.draw_cfg, CTX._draw = saved
    m["pearson_on_warp_step"] = skipped
    if m["identity_err"] > 1e-6 or m["identity_mask"] != 1.0:
        raise AssertionError(f"identity warp wrong {m}")
    if m["shift_err"] > 1e-6 or m["shift_mask_hole"] != 0.0:
        raise AssertionError(f"translated warp wrong {m}")
    if m["rodrigues_err"] > 1e-9 or abs(m["warp_cam_rot_deg"] - 2.5) > 1e-3 or src != 0:
        raise AssertionError(f"warp camera rotation wrong {m}")
    if m["mask_l1_err"] > 1e-6 or skipped != 0.0:
        raise AssertionError(f"warp loss / schedule wrong {m}")
    return m


# ------------------------------------------------------------------ C3
class _StubVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=0.18215)
        self.w = torch.nn.Parameter(torch.randn(4, 3, 8, 8) * 0.1, requires_grad=False)

    def encode(self, x):
        lat = torch.nn.functional.conv2d(x, self.w, stride=8)
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: lat))


class _StubUNet(torch.nn.Module):
    def forward(self, x, t, encoder_hidden_states=None):
        return SimpleNamespace(sample=torch.tanh(x) * (1 + torch.arange(len(x)).view(-1, 1, 1, 1)))


def sds(method, device):
    from .context import Ellipse
    from .sds import StableDiffusionGuidance, fixed_timestep

    sched = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000),
                            alphas_cumprod=torch.linspace(0.999, 0.01, 1000))
    sched.add_noise = lambda lat, noise, t: (sched.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1) * lat
                                             + (1 - sched.alphas_cumprod[t]).sqrt()
                                             .view(-1, 1, 1, 1) * noise)
    guide = StableDiffusionGuidance(None, None, "cpu", fp16=False,
                                    modules=(_StubVAE(), _StubUNet(), sched, torch.zeros(2, 4, 8)))
    img = torch.rand(1, 3, 40, 56, requires_grad=True)
    torch.manual_seed(0)
    loss = guide.train_step(img, 0.99, 100.0)
    loss.backward()
    # independent recomputation of the latent-space gradient
    torch.manual_seed(0)
    with torch.no_grad():
        lat = guide.encode_imgs(torch.nn.functional.interpolate(img, (512, 512), mode="bilinear",
                                                                align_corners=False))
        noise = torch.randn_like(lat)
        t = torch.tensor([20])
        noisy = sched.add_noise(lat, noise, t)
        pred = _StubUNet()(torch.cat([noisy] * 2), None).sample
        u, c = pred.chunk(2)
        eps = u + 100.0 * (c - u)
        grad_ref = (1 - sched.alphas_cumprod[t]).view(1, 1, 1, 1) * (eps - noise)
    lat_req = lat.clone().requires_grad_(True)
    from .sds import sds_target_loss

    sds_target_loss(lat_req, eps, noise, (1 - sched.alphas_cumprod[t]).view(1, 1, 1, 1)).backward()
    m = {"t_099": fixed_timestep(0.99), "t_095": fixed_timestep(0.95), "t_0": fixed_timestep(0.0),
         "latent_grad_err": float((lat_req.grad - grad_ref).abs().max()),
         "image_grad_norm": float(img.grad.norm())}
    # elliptic cylinder: 12 cameras on a circle of radius 3 at height ~0.5 looking at the origin
    vms = []
    for k in range(12):
        a = 2 * np.pi * k / 12
        c = np.array([3 * np.cos(a), 3 * np.sin(a), 0.5 + 0.1 * np.sin(3 * a)])
        z = -c / np.linalg.norm(c)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])  # rows: camera axes in world (OpenCV, y down)
        v = np.eye(4)
        v[:3, :3] = R
        v[:3, 3] = -R @ c
        vms.append(torch.tensor(v, dtype=torch.float32))
    ell = Ellipse(vms)
    rng = np.random.default_rng(0)
    cosines, heights = [], []
    for _ in range(20):
        w2c = ell.sample(rng).double().numpy()
        center = -w2c[:3, :3].T @ w2c[:3, 3]
        fwd = w2c[2, :3]
        cosines.append(float(fwd @ (-center) / np.linalg.norm(center)))
        heights.append(float(center[2]))
        if abs(np.linalg.det(w2c[:3, :3]) - 1) > 1e-6:
            raise AssertionError("ellipse pose is not a rotation")
    m["min_lookat_cos"] = min(cosines)
    m["height_range"] = [min(heights), max(heights)]
    if (m["t_099"], m["t_095"], m["t_0"]) != (20, 50, 980):
        raise AssertionError(f"fixed timestep wrong {m}")
    if m["latent_grad_err"] > 1e-4 or not m["image_grad_norm"] > 0:
        raise AssertionError(f"SDS gradient wrong {m}")
    if m["min_lookat_cos"] < 0.97 or m["height_range"][0] < 0.3 or m["height_range"][1] > 0.7:
        raise AssertionError(f"elliptic views wrong {m}")
    return m


# ------------------------------------------------------------------ C5
def floater_prune(method, device):
    if not torch.cuda.is_available():
        _unavailable("CUDA is unavailable")
    from .density import SparseGSDensity, calc_diff, prune_percentile
    from .renderer import Renderer, rasterize

    W, H = 64, 48
    cam = _camera(device, W, H, 50.0)
    gx, gy = torch.meshgrid(torch.linspace(-3.4, 3.4, 40), torch.linspace(-2.6, 2.6, 30),
                            indexing="ij")
    wall = torch.stack([gx.flatten(), gy.flatten(), torch.full((gx.numel(),), 5.0)], -1)
    fl = torch.stack(torch.meshgrid(torch.linspace(-0.12, 0.12, 2),
                                    torch.linspace(-0.1, 0.1, 2), indexing="ij"), -1).reshape(-1, 2)
    floater = torch.cat([fl, torch.full((len(fl), 1), 1.0)], -1)
    pts = torch.cat([wall, floater]).to(device)
    sc = torch.cat([torch.full((len(wall), 3), 0.12), torch.full((len(floater), 3), 0.03)])
    op = torch.cat([torch.full((len(wall),), 0.9), torch.full((len(floater),), 0.85)])
    base = _params(pts, sc.to(device), op.to(device), device)
    n_wall = len(wall)

    def run(mode_range, prune=True):
        params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in base.items()})
        opts = {k: torch.optim.Adam([p], lr=1e-3) for k, p in params.items()}
        spec = SimpleNamespace(config={"mode_range": mode_range, "prune": prune})
        dens = SparseGSDensity(spec)
        dens.initialize(params, opts, 1.0)
        dens.set_training_cameras([cam])
        rnd = Renderer(SimpleNamespace(config={}))
        res = dens.prune_floaters(params, opts, rnd)
        kept = params["means"].detach()
        floaters_left = int((kept[:, 2] < 2).sum())
        walls_left = int((kept[:, 2] > 2).sum())
        return res, floaters_left, walls_left

    res_p, fl_p, wl_p = run("paper")
    res_o, fl_o, wl_o = run("official")
    # official index formula on the kernel output
    rnd = Renderer(SimpleNamespace(config={}))
    params = {k: v for k, v in base.items()}
    out = rnd.render(params, cam, 0, with_depth=True)
    _, _, info = rasterize(params, cam, 0)
    offs = info["isect_offsets"][0].long()
    ids, off = out["mode_ids"].long(), out["mode_ids_official"].long()
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device),
                            indexing="ij")
    lo = offs[ys // 16, xs // 16]
    rank = (ys % 16) * 16 + xs % 16
    valid = ids[0] >= 0
    pred = lo + ((ids - lo) // 256) * 256 + rank
    idx_err = int((pred[:, valid] != off[:, valid]).sum())
    diff = calc_diff(out["depth_mode"], out["depth_alpha"])
    m = {"paper": {**res_p, "floaters_left": fl_p, "walls_left": wl_p},
         "official": {**res_o, "floaters_left": fl_o, "walls_left": wl_o},
         "n_wall": n_wall, "n_floater": len(floater), "official_index_mismatch": idx_err,
         "max_diff": float(diff.max()), "perc_at_dip0": prune_percentile(0.0, 0.97, 7.5, 80)}
    if fl_p != 0 or wl_p < 0.5 * n_wall:
        raise AssertionError(f"paper-mode pruning did not remove the floater only {m}")
    if idx_err:
        raise AssertionError(f"official start/mode index formula mismatch {m}")
    if abs(m["perc_at_dip0"] - 97.0) > 1e-9 or prune_percentile(1.0, 0.97, 7.5, 80) != 80.0:
        raise AssertionError("percentile formula wrong")
    return m
