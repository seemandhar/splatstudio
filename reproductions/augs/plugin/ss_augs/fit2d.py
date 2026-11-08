"""Error-driven 2D Gaussian training over a rendered image (paper §3.2; official train_2d.py and
gaussian_2d_renderer/__init__.py; rasterizer diff-gaussian-2d-rasterization/forward.cu).

Image-space kernels: xy = (row, col) pixel coordinates (official order), log scales (2), rotation
angle, logit opacity, RGB colour, and a depth (sorting key) bilinearly resampled from the depth map
after every step. Rasterization uses gsplat's differentiable `rasterize_to_pixels` on gsplat tile
lists; the rendered image is the per-pixel background: out = C + T * background.
"""

import math

import torch
import torch.nn.functional as F

H_VAR = 0.3  # Inria antialiasing low-pass (forward.cu preprocessCUDA)
LAMBDA_DSSIM = 0.2


def ssim_map(x, y):
    """3DGS loss_utils._ssim map with zero 'same' padding; inputs CHW -> [C,H,W]."""
    coords = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].expand(x.shape[0], 1, 11, 11).contiguous()

    def conv(t):
        return F.conv2d(t[None], w, padding=5, groups=x.shape[0])[0]

    mx, my = conv(x), conv(y)
    vx, vy, cxy = conv(x * x) - mx * mx, conv(y * y) - my * my, conv(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    return ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


def ssim_value(img, gt):
    """Mean same-padded SSIM of CHW images. CUDA: the vendored fused-ssim kernel the official
    code uses (fused_ssim); CPU: the equivalent 3DGS conv formulation `ssim_map` (the two agree
    to float rounding; checked by the error-driven-2d probe)."""
    if img.is_cuda:
        from .kernels import fused_ssim_map

        return fused_ssim_map(img[None], gt[None]).mean()
    return ssim_map(img, gt).mean()


def image_loss(img, gt):
    """(1 - l) L1 + l (1 - SSIM), CHW (official loss in optimization_process / view budget)."""
    return (1 - LAMBDA_DSSIM) * (img - gt).abs().mean() + LAMBDA_DSSIM * (1 - ssim_value(img, gt))


def loss_map(img, gt):
    """Official loss_2d: per-pixel (1-l) mean_c |I-G| + l (1 - mean_c SSIM), clamped >= 0. [H,W]
    (official ssim_2D is the conv formulation, used here on every device)."""
    l1 = (img - gt).abs().mean(0)
    return ((1 - LAMBDA_DSSIM) * l1 + LAMBDA_DSSIM * (1 - ssim_map(img, gt).mean(0))).clamp(min=0)


def cov2d(scale, rot):
    """utils.general_utils.compute_cov2D: (xx, xy, yy) with x = column axis; scale [N,2], rot [N]."""
    s, c = torch.sin(rot), torch.cos(rot)
    a2, b2 = scale[:, 0] ** 2, scale[:, 1] ** 2
    return torch.stack((a2 * c * c + b2 * s * s, (a2 - b2) * s * c, a2 * s * s + b2 * c * c), -1)


def project2d(xy, log_scale, rot, opac_logit, antialias=True):
    """preprocessCUDA: conic of cov + 0.3 I, opacity x sqrt(max(2.5e-5, det/det_h)) and the
    3-sigma radius from the eigenvalues (sqrt clamp 0.1). Returns gsplat-convention means2d
    (col + 0.5, row + 0.5) -- the Inria 2D rasterizer samples pixel (x, y) at integer coordinates,
    gsplat at (x + 0.5, y + 0.5) -- conics (xx, xy, yy), opacities and int32 radii [N,2]."""
    cov = cov2d(log_scale.exp(), rot)
    cxx, cxy, cyy = cov[:, 0] + H_VAR, cov[:, 1], cov[:, 2] + H_VAR
    det_cov = cov[:, 0] * cov[:, 2] - cov[:, 1] ** 2
    det = cxx * cyy - cxy * cxy
    comp = torch.sqrt(torch.clamp(det_cov / det, min=0.000025)) if antialias else torch.ones_like(det)
    conics = torch.stack((cyy / det, -cxy / det, cxx / det), -1)
    with torch.no_grad():
        mid = 0.5 * (cxx + cyy)
        lam = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
        radius = torch.ceil(3.0 * torch.sqrt(lam))
        radii = torch.stack((radius, radius), -1).int()
    means2d = torch.stack((xy[:, 1] + 0.5, xy[:, 0] + 0.5), -1)
    return means2d, conics, torch.sigmoid(opac_logit) * comp, radii


def sample_depth(depth_map, xy, mode="bilinear"):
    """Gaussian2DModel.sample_depth_map / projector grid_sample (align_corners=False)."""
    H, W = depth_map.shape
    gx = (xy[:, 1] + 0.5) / W * 2 - 1
    gy = (xy[:, 0] + 0.5) / H * 2 - 1
    grid = torch.stack((gx, gy), -1).view(1, -1, 1, 2)
    kw = {"padding_mode": "border"} if mode == "nearest" else {}
    return F.grid_sample(depth_map[None, None], grid, mode=mode, align_corners=False, **kw).view(-1)


class TinyAdam:
    """torch.optim.Adam semantics (bias correction, eps 1e-15) for the five 2D parameter groups,
    with the official cat/prune behaviour: moments are extended with zeros / masked and the step
    count of each group is kept (gaussian_2d_renderer cat_tensors_to_optimizer/_prune_optimizer)."""

    def __init__(self, lrs, eps=1e-15, betas=(0.9, 0.999)):
        self.lrs, self.eps, self.b1, self.b2 = lrs, eps, betas[0], betas[1]
        self.state = {}  # name -> (step, m, v)

    def step(self, params):
        with torch.no_grad():
            for name, p in params.items():
                if p.grad is None:
                    continue
                g = p.grad
                step, m, v = self.state.get(name, (0, torch.zeros_like(p), torch.zeros_like(p)))
                step += 1
                m.mul_(self.b1).add_(g, alpha=1 - self.b1)
                v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                bc1, bc2 = 1 - self.b1**step, 1 - self.b2**step
                denom = (v.sqrt() / math.sqrt(bc2)).add_(self.eps)
                p.addcdiv_(m, denom, value=-self.lrs[name] / bc1)
                self.state[name] = (step, m, v)
                p.grad = None

    def cat(self, n_new):
        for name, (step, m, v) in self.state.items():
            pad = (n_new,) + m.shape[1:]
            self.state[name] = (step, torch.cat((m, m.new_zeros(pad))), torch.cat((v, v.new_zeros(pad))))

    def prune(self, keep):
        for name, (step, m, v) in self.state.items():
            self.state[name] = (step, m[keep], v[keep])


NAMES = ("xy", "color", "opacity", "scale", "rotation")


class Gaussians2D:
    def __init__(self, depth_map, lrs, H, W):
        dev = depth_map.device
        self.depth_map = depth_map
        self.H, self.W = H, W
        self.p = {
            "xy": torch.zeros(0, 2, device=dev),
            "color": torch.zeros(0, 3, device=dev),
            "opacity": torch.zeros(0, device=dev),
            "scale": torch.zeros(0, 2, device=dev),
            "rotation": torch.zeros(0, device=dev),
        }
        for t in self.p.values():
            t.requires_grad_(True)
        self.depth = torch.zeros(0, device=dev)
        self.opt = TinyAdam(lrs)
        # Clamp max scale to reduce clustering time (end_optimizer_step).
        self.max_log_scale = math.log(min(50, (H + W) * 0.01))

    def size(self):
        return self.p["xy"].shape[0]

    def add(self, xy, color, opacity_logit, log_scale, rotation):
        with torch.no_grad():
            self.depth = torch.cat((self.depth, sample_depth(self.depth_map, xy)))
            new = {"xy": xy, "color": color, "opacity": opacity_logit, "scale": log_scale,
                   "rotation": rotation}
            for k in NAMES:
                self.p[k] = torch.cat((self.p[k].detach(), new[k].detach())).requires_grad_(True)
        self.opt.cat(xy.shape[0])

    def prune(self, keep):
        with torch.no_grad():
            for k in NAMES:
                self.p[k] = self.p[k].detach()[keep].clone().requires_grad_(True)
            self.depth = self.depth[keep]
        self.opt.prune(keep)

    def end_step(self):
        with torch.no_grad():
            self.p["color"].clamp_(0.0, 1.0)
            self.depth = sample_depth(self.depth_map, self.p["xy"])
            self.p["xy"][:, 0].clamp_(0.0, self.H - 1)
            self.p["xy"][:, 1].clamp_(0.0, self.W - 1)
            self.p["scale"].clamp_(max=self.max_log_scale)


def render2d(g, background, with_counts=False):
    """CHW render over the per-pixel background; optionally per-kernel pix_covered counts
    (Inria semantics: alpha <= 0.99, inclusive termination)."""
    from gsplat.cuda._wrapper import rasterize_to_pixels

    from .kernels import contrib_stats, tile_lists

    H, W = g.H, g.W
    if g.size() == 0:
        return background, (torch.zeros(0, dtype=torch.int32, device=background.device)
                            if with_counts else None)
    means2d, conics, opac, radii = project2d(g.p["xy"], g.p["scale"], g.p["rotation"],
                                             g.p["opacity"])
    offsets, flatten_ids = tile_lists(means2d.detach(), radii, g.depth, W, H)
    rgb, alpha = rasterize_to_pixels(means2d[None], conics[None], g.p["color"][None], opac[None],
                                     W, H, 16, offsets[None], flatten_ids)
    out = rgb[0].permute(2, 0, 1) + (1 - alpha[0].permute(2, 0, 1)) * background
    counts = None
    if with_counts:
        _, counts = contrib_stats(means2d.detach(), conics.detach(), opac.detach(), offsets,
                                  flatten_ids, W, H, 16, t_threshold=2.0, alpha_max=0.99,
                                  inclusive_stop=True)
    return out, counts


def optimize(g, background, gt, steps, prune):
    """optimization_process: Adam steps on the 3DGS loss; prune opacity <= 0.005 or covering
    <= 25 pixels using the statistics of the last forward pass."""
    if g.size() == 0:
        return
    counts = None
    for i in range(steps):
        last = i == steps - 1
        img, c = render2d(g, background, with_counts=last and prune)
        loss = image_loss(img, gt)
        loss.backward()
        g.opt.step(g.p)
        g.end_step()
        if last:
            counts = c
    if prune:
        keep = (torch.sigmoid(g.p["opacity"].detach()) > 0.005) & (counts > 25)
        g.prune(keep)


def fit_view(background, gt, depth_map, target, cfg, generator=None):
    """train_2d_kernels inner loop for one view. background/gt CHW, depth_map HW."""
    H, W = depth_map.shape
    lrs = {"xy": cfg["xy_lr"] * (H + W) / 2, "color": cfg["color_lr"], "opacity": cfg["opacity_lr"],
           "scale": cfg["scale_lr"], "rotation": cfg["rotation_lr"]}
    g = Gaussians2D(depth_map, lrs, H, W)
    with torch.no_grad():
        refined = render2d(g, background)[0]
        lmap = loss_map(refined, gt)
    done = False
    rounds = 0
    while not done:
        with torch.no_grad():
            # Official: min(round(0.2 target), target - size); >= 1 so tiny targets terminate.
            n = max(1, min(round(target * cfg["add_fraction"]), target - g.size()))
            weights = lmap.flatten() ** 2
            n = min(n, int((weights > 0).sum()))
            if n > 0:
                flat = torch.multinomial(weights, n, replacement=False, generator=generator)
                coords = torch.stack((flat // W, flat % W), 1)
                xy = coords.float()
                log_scale = torch.full((n, 2), math.log(min(W, H) * cfg["init_scale"]), device=xy.device)
                rot = torch.zeros(n, device=xy.device)
                opac = torch.logit(torch.full((n,), cfg["init_opacity"], device=xy.device), eps=1e-6)
                color = refined[:, coords[:, 0], coords[:, 1]].clamp(1 / 255, 244 / 255).t().contiguous()
                g.add(xy, color, opac, log_scale, rot)
            if g.size() >= target or n == 0:
                done = True
        optimize(g, background, gt, cfg["coarse_steps"], prune=True)
        rounds += 1
        with torch.no_grad():
            refined = render2d(g, background)[0]
            lmap = loss_map(refined, gt)
    optimize(g, background, gt, cfg["finetune_steps"], prune=True)
    with torch.no_grad():
        final = image_loss(render2d(g, background)[0], gt).item()
    return g, {"rounds": rounds, "kernels": g.size(), "loss_after": final}
