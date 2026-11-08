"""Per-scene iNGP geometric proxy (ConeGS Sec. 4.1 / 4.4), ported from the official code.

Official: `scene/nerf_model.py`, `utils/ngp.py` (nerfacc `examples/radiance_fields/ngp.py`),
`utils/nerf_utils.py`, `utils/ray_utils.py` of https://github.com/baranowskibrt/conegs, using
nerfacc's `PropNetEstimator` and tiny-cuda-nn. Neither nerfacc nor tiny-cuda-nn is installed in
the shared env, so this module re-implements, in PyTorch plus the hash-grid CUDA kernel of
`hashgrid.py`:

- `NGPRadianceField` / `NGPDensityField` (bias-free 64-wide MLPs like tcnn FullyFusedMLP,
  trunc_exp(x - 1) density, SH-degree-4 view encoding, unbounded unisphere contraction),
- nerfacc's batched `importance_sampling` CUDA kernel (u = u0 + (i + bias) step, bias 0.5 or
  U(0,1) per ray; upper-bound inverse CDF; midpoint intervals clamped to the input range),
  `_transform_stot` (lindisp), `render_transmittance_from_density`, `_pdf_loss`,
  `get_proposal_requires_grad_fn`,
- the pretraining loop (`NeRFModel.pretrain_model`): 4096 uniform rays over all training pixels,
  smooth-L1, loss x1024 (the GradScaler is never unscaled), Adam(1e-2, eps 1e-15, wd 1e-5),
  LinearLR(0.01, 100) + MultiStepLR(n/2, 3n/4, 9n/10; 0.33),
- the median-depth query (`select_for_densification`): first sample whose inclusive
  transmittance is < 0.5, colour = rendered pixel colour with zeroed view directions,
  std = (|d_x - d| + |d_y - d|)/2 * t / |d| / scale.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from .hashgrid import HashGrid

SH_C0 = 0.28209479177387814


class _TruncExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * torch.exp(torch.clamp(x, max=15))


def trunc_exp(x):
    return _TruncExp.apply(x)


def contract_to_unisphere(x):
    """nerfacc/official `contract_to_unisphere` with aabb [-1,1]^3, ord=2 -> [0,1]^3."""
    mag = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    y = torch.where(mag < 1, x, (2 - 1 / mag) * (x / mag))
    return (y + 2.0) / 4.0


def sh4_encoding(d):
    """tcnn SphericalHarmonics degree 4 (16 outputs) of directions in [-1,1]."""
    x, y, z = d.unbind(-1)
    xy, xz, yz, x2, y2, z2 = x * y, x * z, y * z, x * x, y * y, z * z
    return torch.stack(
        [
            torch.full_like(x, SH_C0),
            -0.48860251190291987 * y,
            0.48860251190291987 * z,
            -0.48860251190291987 * x,
            1.0925484305920792 * xy,
            -1.0925484305920792 * yz,
            0.94617469575755997 * z2 - 0.31539156525251999,
            -1.0925484305920792 * xz,
            0.54627421529603959 * x2 - 0.54627421529603959 * y2,
            0.59004358992664352 * y * (-3.0 * x2 + y2),
            2.8906114426405538 * xy * z,
            0.45704579946446572 * y * (1.0 - 5.0 * z2),
            0.3731763325901154 * z * (5.0 * z2 - 3.0),
            0.45704579946446572 * x * (1.0 - 5.0 * z2),
            1.4453057213202769 * z * (x2 - y2),
            0.59004358992664352 * x * (-x2 + 3.0 * y2),
        ],
        dim=-1,
    )


def mlp(n_in, n_out, hidden=64, n_hidden=1):
    """tcnn FullyFusedMLP: bias-free, ReLU hidden, linear output, Xavier-uniform init."""
    dims = [n_in] + [hidden] * n_hidden + [n_out]
    layers = []
    for i in range(len(dims) - 1):
        lin = torch.nn.Linear(dims[i], dims[i + 1], bias=False)
        torch.nn.init.xavier_uniform_(lin.weight)
        layers.append(lin)
        if i < len(dims) - 2:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def per_level(base, finest, levels):
    return float(np.exp((np.log(finest) - np.log(base)) / (levels - 1)))


class NGPRadianceField(torch.nn.Module):
    def __init__(self, n_levels=16, log2_hashmap_size=19, base_resolution=16,
                 max_resolution=4096, geo_feat_dim=15):
        super().__init__()
        self.geo_feat_dim = geo_feat_dim
        self.encoding = HashGrid(n_levels, log2_hashmap_size, base_resolution,
                                 per_level(base_resolution, max_resolution, n_levels))
        self.mlp_base = mlp(self.encoding.n_output_dims, 1 + geo_feat_dim, 64, 1)
        self.mlp_head = mlp(16 + geo_feat_dim, 3, 64, 2)

    def query_density(self, x):
        xc = contract_to_unisphere(x)
        selector = ((xc > 0.0) & (xc < 1.0)).all(dim=-1)
        h = self.mlp_base(self.encoding(xc.reshape(-1, 3))).reshape(*x.shape[:-1], -1)
        density = trunc_exp(h[..., :1] - 1) * selector[..., None]
        return density, h[..., 1:]

    def forward(self, x, dirs):
        density, geo = self.query_density(x)
        h = torch.cat([sh4_encoding(dirs), geo], dim=-1)
        rgb = torch.sigmoid(self.mlp_head(h.reshape(-1, h.shape[-1]))).reshape(*x.shape[:-1], 3)
        return rgb, density


class NGPDensityField(torch.nn.Module):
    def __init__(self, n_levels=5, log2_hashmap_size=17, base_resolution=16, max_resolution=128):
        super().__init__()
        self.encoding = HashGrid(n_levels, log2_hashmap_size, base_resolution,
                                 per_level(base_resolution, max_resolution, n_levels))
        self.mlp_base = mlp(self.encoding.n_output_dims, 1, 64, 1)

    def forward(self, x):
        xc = contract_to_unisphere(x)
        selector = ((xc > 0.0) & (xc < 1.0)).all(dim=-1)
        h = self.mlp_base(self.encoding(xc.reshape(-1, 3))).reshape(*x.shape[:-1], 1)
        return trunc_exp(h - 1) * selector[..., None]


# ----------------------------------------------------------------------------- nerfacc ports


def importance_sampling(edges, cdfs, n, stratified):
    """nerfacc `importance_sampling` (batched): returns new edges [R, n+1] and samples [R, n]."""
    rays = edges.shape[0]
    u_floor, u_ceil = cdfs[:, :1], cdfs[:, -1:]
    step = (u_ceil - u_floor) / n
    if stratified:
        bias = torch.rand(rays, 1, device=edges.device)
    else:
        bias = torch.full((rays, 1), 0.5, device=edges.device)
    u = u_floor + (torch.arange(n, device=edges.device, dtype=edges.dtype) + bias) * step
    last = edges.shape[1] - 1
    p = torch.searchsorted(cdfs[:, :-1].contiguous(), u.contiguous(), right=True)
    p0 = (p - 1).clamp(0, last)
    p1 = p.clamp(0, last)
    u_lo, u_hi = cdfs.gather(1, p0), cdfs.gather(1, p1)
    t_lo, t_hi = edges.gather(1, p0), edges.gather(1, p1)
    denom = u_hi - u_lo
    flat = denom < 1e-10
    t = torch.where(flat, (t_lo + t_hi) * 0.5,
                    (u - u_lo) * (t_hi - t_lo) / torch.where(flat, torch.ones_like(denom), denom)
                    + t_lo)
    t_min, t_max = edges[:, :1], edges[:, -1:]
    first = torch.maximum(t[:, :1] - (t[:, 1:2] - t[:, :1]) * 0.5, t_min)
    middle = (t[:, 1:] + t[:, :-1]) * 0.5
    final = torch.minimum(t[:, -1:] + (t[:, -1:] - t[:, -2:-1]) * 0.5, t_max)
    return torch.cat([first, middle, final], dim=1), t


def s_to_t(s, near, far):
    """nerfacc `_transform_stot('lindisp', ...)`."""
    s_min, s_max = 1.0 / near, 1.0 / far
    return 1.0 / (s * s_max + (1 - s) * s_min)


def transmittance_from_density(t_starts, t_ends, sigmas):
    """nerfacc `render_transmittance_from_density` (batched): exclusive trans, alphas."""
    sigmas_dt = sigmas * (t_ends - t_starts)
    # exclusive prefix sum without cumsum - x (the opaque last sample would cancel catastrophically)
    excl = torch.cat([torch.zeros_like(sigmas_dt[..., :1]), torch.cumsum(sigmas_dt[..., :-1], -1)],
                     dim=-1)
    return torch.exp(-excl), 1.0 - torch.exp(-sigmas_dt)


def pdf_loss(q_edges, q_cdfs, k_edges, k_cdfs, eps=1e-7):
    """nerfacc `_pdf_loss` with `searchsorted(segments_key, segments_query)` (batched)."""
    last = k_edges.shape[1] - 1
    p = torch.searchsorted(k_edges[:, :-1].contiguous(), q_edges.contiguous(), right=True)
    ids_left = (p - 1).clamp(0, last)[:, :-1]
    ids_right = p.clamp(0, last)[:, 1:]
    w = q_cdfs[:, 1:] - q_cdfs[:, :-1]
    w_outer = k_cdfs.gather(-1, ids_right) - k_cdfs.gather(-1, ids_left)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + eps)


def proposal_requires_grad_fn(target=5.0, num_steps=1000):
    """nerfacc `get_proposal_requires_grad_fn`."""
    state = {"since": 0}

    def fn(step):
        requires = state["since"] > min(step / num_steps, 1.0) * target
        if requires:
            state["since"] = 0
        state["since"] += 1
        return requires

    return fn


OPAQUE_SIGMA = 1e10  # official: sigmas[..., -1, :] = inf (opaque_bkgd); finite avoids inf*0


# ----------------------------------------------------------------------------- normalisation


def gs2nerf_transform(c2w_opencv):
    """Official `get_gs2nerf_transform(c2w, strict_scaling=False)` (nerf-factory)."""
    c2w = np.asarray(c2w_opencv, dtype=np.float64)
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)
    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array([[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]],
                     [-cross[1], cross[0], 0.0]])
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
    translate = -np.median(nearest, axis=0)
    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align
    scale = 1.0 / np.median(np.linalg.norm(t + translate, axis=-1))
    return transform, float(scale)


class RayCaster:
    """Pixel rays in the normalised (nerf) frame; OpenCV cameras (K, world-to-camera viewmat)."""

    def __init__(self, transform, scale, device):
        self.T = torch.as_tensor(transform, dtype=torch.float32, device=device)
        self.scale = float(scale)

    def camera(self, K, viewmat):
        c2w = torch.linalg.inv(viewmat.float())
        R = self.T[:3, :3] @ c2w[:3, :3]
        o = self.scale * (self.T[:3, :3] @ c2w[:3, 3] + self.T[:3, 3])
        return torch.linalg.inv(K.float()), R, o

    @staticmethod
    def rays(K_inv, R, o, px, py):
        """px, py integer pixel coords [M]; K_inv/R [M,3,3] or [3,3]; o [M,3] or [3].

        Returns origins, unit dirs, |d| of the z=1 direction, and the Mip-NeRF radius factor
        (|d_x - d| + |d_y - d|)/2 (official `pixels_to_rays`)."""
        px, py = px.float(), py.float()
        one = torch.ones_like(px)
        pix = torch.stack([
            torch.stack([px + 0.5, py + 0.5, one], -1),
            torch.stack([px + 1.5, py + 0.5, one], -1),
            torch.stack([px + 0.5, py + 1.5, one], -1),
        ])  # [3, M, 3]
        M = (R @ K_inv) if R.dim() == 2 else torch.bmm(R, K_inv)
        if M.dim() == 2:
            dirs = pix @ M.T
        else:
            dirs = torch.einsum("mij,kmj->kmi", M, pix)
        d, dx, dy = dirs
        radii = 0.5 * (torch.linalg.vector_norm(dx - d, dim=-1)
                       + torch.linalg.vector_norm(dy - d, dim=-1))
        norm = torch.linalg.vector_norm(d, dim=-1)
        origins = o.expand_as(d) if o.dim() == 1 else o
        return origins, d / norm[:, None], norm, radii

    def to_world(self, x):
        return (x / self.scale - self.T[:3, 3]) @ self.T[:3, :3]


# ----------------------------------------------------------------------------- model


DEFAULTS = {
    "iterations": 20000,
    "rays": 4096,
    "prop_samples": (256, 96),
    "num_samples": 48,
    "near": 0.2,
    "far": 1e6,
    "lr": 0.01,
    "eps": 1e-15,
    "weight_decay": 1e-5,
    "loss_scale": 1024.0,
}


class INGP(torch.nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.field = NGPRadianceField()
        self.proposals = torch.nn.ModuleList(
            [NGPDensityField(max_resolution=128), NGPDensityField(max_resolution=256)]
        )

    # nerfacc PropNetEstimator.sampling
    def sample(self, origins, dirs, stratified, requires_grad=False, cache=None):
        cfg = self.cfg
        rays = origins.shape[0]
        edges = torch.cat([torch.zeros(rays, 1, device=origins.device),
                           torch.ones(rays, 1, device=origins.device)], dim=-1)
        cdfs = edges
        for net, n in zip(self.proposals, cfg["prop_samples"]):
            with torch.no_grad():
                edges, _ = importance_sampling(edges, cdfs, n, stratified)
            t = s_to_t(edges, cfg["near"], cfg["far"])
            ts, te = t[:, :-1], t[:, 1:]
            with torch.set_grad_enabled(requires_grad):
                mid = ((ts + te) / 2)[..., None]
                sigmas = net(origins[:, None] + dirs[:, None] * mid).clone()
                sigmas[:, -1, :] = OPAQUE_SIGMA
                trans, _ = transmittance_from_density(ts, te, sigmas[..., 0])
                cdfs = 1.0 - torch.cat([trans, torch.zeros_like(trans[:, :1])], dim=-1)
            if requires_grad and cache is not None:
                cache.append((edges, cdfs))
            if not requires_grad:
                cdfs = cdfs.detach()
        with torch.no_grad():
            edges, _ = importance_sampling(edges, cdfs.detach(), cfg["num_samples"], stratified)
        if requires_grad and cache is not None:
            cache.append((edges, None))
        t = s_to_t(edges, cfg["near"], cfg["far"])
        return t[:, :-1], t[:, 1:]

    def render(self, origins, dirs, ts, te, view_dirs=True):
        mid = ((ts + te) / 2)[..., None]
        points = origins[:, None] + dirs[:, None] * mid
        d = dirs[:, None].expand_as(points) if view_dirs else torch.zeros_like(points)
        rgb, sigmas = self.field(points, d)
        sigmas = sigmas.clone()
        sigmas[:, -1, :] = OPAQUE_SIGMA
        trans, alphas = transmittance_from_density(ts, te, sigmas[..., 0])
        weights = trans * alphas
        color = (weights[..., None] * rgb).sum(1)  # black background (opaque last sample)
        return color, {"trans": trans, "alphas": alphas, "mid": mid[..., 0]}

    @torch.no_grad()
    def median_points(self, origins, dirs, norm, radii, batch=8192):
        """Official select_for_densification (cumsum threshold 0.5, min_opacity 0).

        Returns (points in the nerf frame, colours, t, nerf-frame std, valid mask)."""
        outs = []
        for s in range(0, len(origins), batch):
            o, d = origins[s : s + batch], dirs[s : s + batch]
            ts, te = self.sample(o, d, stratified=False)
            color, ex = self.render(o, d, ts, te, view_dirs=False)
            incl = torch.cumprod(1 - ex["alphas"], dim=-1)
            idx = torch.argmax((incl < 0.5).int(), dim=1)
            ar = torch.arange(len(idx), device=idx.device)
            valid = ex["alphas"][ar, idx] > 0
            t = ex["mid"][ar, idx]
            outs.append((o + d * t[:, None], color, t, valid))
        pts, col, t, valid = (torch.cat(x) for x in zip(*outs))
        std = radii * t / norm
        return pts, col, t, std, valid


def train_ingp(model, images, K_inv, R, o, iterations=None, log=print, log_every=1000,
               seed=0):
    """Official `NeRFModel.pretrain_model` training phase. images [N,H,W,3] on device."""
    cfg = model.cfg
    iterations = iterations or cfg["iterations"]
    device = images.device
    gen = torch.Generator(device=device).manual_seed(seed)
    n_img, H, W, _ = images.shape
    model_opt = torch.optim.Adam(model.field.parameters(), lr=cfg["lr"], betas=(0.9, 0.999),
                                 eps=cfg["eps"], weight_decay=cfg["weight_decay"])
    prop_opt = torch.optim.Adam(model.proposals.parameters(), lr=cfg["lr"], betas=(0.9, 0.999),
                                eps=cfg["eps"], weight_decay=cfg["weight_decay"])
    milestones = [iterations // 2, iterations * 3 // 4, iterations * 9 // 10]

    def scheduler(opt):
        return torch.optim.lr_scheduler.ChainedScheduler([
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=100),
            torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.33),
        ])

    model_sched, prop_sched = scheduler(model_opt), scheduler(prop_opt)
    requires_fn = proposal_requires_grad_fn()
    model.train()
    history = []
    for it in range(1, iterations + 1):
        ids = torch.randint(0, n_img, (cfg["rays"],), device=device, generator=gen)
        px = torch.randint(0, W, (cfg["rays"],), device=device, generator=gen)
        py = torch.randint(0, H, (cfg["rays"],), device=device, generator=gen)
        gt = images[ids, py, px]
        origins, dirs, _, _ = RayCaster.rays(K_inv[ids], R[ids], o[ids], px, py)
        requires = requires_fn(it)
        cache = []
        ts, te = model.sample(origins, dirs, stratified=True, requires_grad=requires, cache=cache)
        color, ex = model.render(origins, dirs, ts, te, view_dirs=True)
        loss = F.smooth_l1_loss(color, gt)
        # estimator.update_every_n_steps(trans, requires_grad, loss_scaler=1024)
        if requires:
            final_edges, _ = cache.pop()
            cdfs = (1.0 - torch.cat([ex["trans"], torch.zeros_like(ex["trans"][:, :1])],
                                    dim=-1)).detach()
            prop_loss = 0.0
            while cache:
                p_edges, p_cdfs = cache.pop()
                prop_loss = prop_loss + pdf_loss(final_edges, cdfs, p_edges, p_cdfs).mean()
            prop_opt.zero_grad()
            (prop_loss * cfg["loss_scale"]).backward()
            prop_opt.step()
        prop_sched.step()
        (loss * cfg["loss_scale"]).backward()
        model_opt.step()
        model_sched.step()
        model_opt.zero_grad(set_to_none=True)
        prop_opt.zero_grad(set_to_none=True)
        if it % log_every == 0 or it == iterations:
            with torch.no_grad():
                psnr = -10 * math.log10(max(F.mse_loss(color, gt).item(), 1e-12))
            history.append({"iteration": it, "loss": loss.item(), "ray_psnr": psnr})
            log(history[-1])
    model.eval()
    return history
