"""Deblurring 3DGS density control, xyz LR, extra points and the blur-MLP bank.

Port of train.py and scene/gaussian_model.py of
github.com/benhenryL/Deblurring-3D-Gaussian-Splatting (3DGS-based):

* statistics: every render of the step adds ||dL/d means2D|| (Inria NDC units = gsplat pixel
  gradient * (W/2, H/2)) where visible, denom += 1 / n_renders (add_densification_stats; 1 for
  defocus, 1/5 for motion), while step < densify_until_iter.
* densify when step > 500 and step % 100 == 0 (< 15000): clone (max scale <= 0.01 extent) ->
  split (N=2, /1.6) -> depth-based opacity prune (Sec. 3.3): sigmoid(o) < t_p / w(z),
  w(z) = (z - z_min) / (z_max - z_min) * (prune_range - 1) + 1 on world z. No opacity reset and
  no screen/world-size prune (train.py never calls reset_opacity, size_threshold = None).
* extra points at step pts_iter = 2500 (Alg. 1, supp. N_p), after that step's densification:
  N_p = int(min(prod(initial SfM bbox) / rate^3, 200000)) uniform samples in the per-axis
  [sorted[bound], sorted[-bound]] box of the current Gaussians; colour (all SH) = sum of the K=4
  nearest Gaussians' SH weighted by distance / sum(distance) over neighbours closer than t_d;
  samples without such a neighbour are dropped. Then ALL Gaussians are re-initialised
  (scales = log sqrt(mean 3-NN d^2), identity rotation, opacity 0.1; positions and SH kept) and
  every Adam (Gaussians and MLP) starts from scratch.
* xyz lr = expon(1.6e-3 -> 1.6e-5) * extent over position_lr_max_steps = 30000 (no delay),
  set for every step (update_learning_rate).
"""

import math
from collections import defaultdict

import torch
from gsplat.strategy.ops import duplicate, remove, split

from . import bank as bank_mod
from .bank import BANK_DEFAULTS

REQUIRES = ("gradient.screen.signed",)
DEFAULTS = {
    "densify_from_iter": 500,
    "densify_until_iter": 15000,
    "densification_interval": 100,
    "densify_grad_threshold": 2e-4,      # real defocus (motion: 5e-4)
    "densify_prune_threshold": 5e-3,     # t_p, real defocus (motion: 1e-2)
    "percent_dense": 0.01,
    "densify_with_depth": True,          # arguments densify_with_depth = 1
    "prune_range": 3,                    # w_p (paper Sec. 4.1)
    "pts_iter": 2500,                    # N_st
    "pts_dist": 2.0,                     # t_d
    "pts_N_intpl": 4,                    # K
    "pts_N_max": 200000,                 # N_max
    "pts_add_bound": 10,
    "pts_rate": 1.1,                     # c
    "add_points": True,
    "knn_chunk": 1024,
    "position_lr_init": 1.6e-3,
    "position_lr_final": 1.6e-5,
    "position_lr_max_steps": 30000,
    **{f"bank_{k}": v for k, v in BANK_DEFAULTS.items()},
}


def expon_lr(step, init, final, max_steps):
    """utils.general_utils.get_expon_lr_func with lr_delay_steps = 0."""
    if step < 0 or (init == 0.0 and final == 0.0):
        return 0.0
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(init) * (1 - t) + math.log(final) * t)


def depth_prune_threshold(z, t_p, prune_range):
    """densify_and_prune(prune_depth=1): t_p / ((z - min) / (max - min) * (range - 1) + 1)."""
    zmin, zmax = z.amin(), z.amax()
    norm = (z - zmin) / (zmax - zmin) * (prune_range - 1) + 1
    return t_p / norm


@torch.no_grad()
def densify_and_prune(params, optimizers, grads, cfg, extent):
    """clone -> split (padded grads) -> (depth-scaled) opacity prune. Returns counts."""
    thr = cfg["densify_grad_threshold"]
    max_scale = params["scales"].exp().max(dim=1).values
    clone = (grads >= thr) & (max_scale <= cfg["percent_dense"] * extent)
    n_clone = int(clone.sum())
    if n_clone:
        duplicate(params, optimizers, {}, clone)
    n = len(params["means"])
    padded = torch.zeros(n, device=grads.device)
    padded[: grads.shape[0]] = grads
    max_scale = params["scales"].exp().max(dim=1).values
    sel = (padded >= thr) & (max_scale > cfg["percent_dense"] * extent)
    n_split = int(sel.sum())
    if n_split:
        split(params, optimizers, {}, sel)
    t_p = cfg["densify_prune_threshold"]
    if cfg["densify_with_depth"]:
        t_p = depth_prune_threshold(params["means"][:, 2], t_p, cfg["prune_range"])
    prune = params["opacities"].sigmoid() < t_p
    n_prune = int(prune.sum())
    if n_prune:
        remove(params, optimizers, {}, prune)
    return n_clone, n_split, n_prune


@torch.no_grad()
def knn_log_scales(points, k=3, chunk=1024):
    """simple-knn distCUDA2 init: log sqrt(clamp(mean of the 3 nearest squared distances, 1e-7)),
    exact distances (no matmul expansion)."""
    out = torch.empty(len(points), device=points.device)
    for s in range(0, len(points), chunk):
        d2 = torch.cdist(points[s:s + chunk], points,
                         compute_mode="donot_use_mm_for_euclid_dist").square()
        d2[torch.arange(d2.shape[0]), torch.arange(s, s + d2.shape[0])] = float("inf")
        out[s:s + chunk] = d2.topk(k, dim=1, largest=False).values.mean(dim=1)
    return out.clamp_min(1e-7).sqrt().log()[:, None].expand(-1, 3)


def num_extra_points(pts_min, pts_max, cfg):
    """train.py: volume of the initial point bbox / rate^3, at most N_max (int())."""
    vol = float((pts_max - pts_min).prod())
    if cfg["pts_rate"] > 0.0:
        return int(min(vol / (cfg["pts_rate"] ** 3), cfg["pts_N_max"]))
    return int(cfg["pts_N_max"])


@torch.no_grad()
def interpolate_points(existing, features, samples, k, dist, chunk=1024):
    """Alg. 1 for every sample: K nearest existing points, keep those with distance < t_d,
    feature = sum(f_i * d_i / sum d) (official add_points weights). Returns (keep, features)."""
    keep = torch.zeros(len(samples), dtype=torch.bool, device=samples.device)
    out = torch.zeros((len(samples), *features.shape[1:]), device=samples.device)
    flat = features.reshape(len(features), -1)
    for s in range(0, len(samples), chunk):
        d = torch.cdist(samples[s:s + chunk], existing,
                        compute_mode="donot_use_mm_for_euclid_dist")  # exact, like torch.norm
        dk, ik = d.topk(k, dim=1, largest=False)
        valid = dk < dist
        w = torch.where(valid, dk, torch.zeros_like(dk))
        wsum = w.sum(dim=1, keepdim=True)
        ok = valid.any(dim=1)
        w = w / torch.where(wsum > 0, wsum, torch.ones_like(wsum))
        out[s:s + chunk] = (flat[ik] * w[..., None]).sum(dim=1).reshape(-1, *features.shape[1:])
        keep[s:s + chunk] = ok
    return keep, out


def _replace_all(params, optimizers, new):
    """Swap every parameter and start each Adam from an empty state (fresh optimizer)."""
    for name, value in new.items():
        p = torch.nn.Parameter(value.contiguous(), requires_grad=params[name].requires_grad)
        params[name] = p
        opt = optimizers[name]
        opt.param_groups[0]["params"] = [p]
        opt.state = defaultdict(dict)


@torch.no_grad()
def add_points(params, optimizers, cfg, pts_min, pts_max, generator=None):
    """gaussian_model.add_points (official semantics, vectorised KNN). Returns #added."""
    means = params["means"].detach()
    colors = torch.cat((params["sh0"], params["shN"]), dim=1).detach()  # [N, 16, 3]
    num = num_extra_points(pts_min, pts_max, cfg)
    srt = means.sort(dim=0).values
    b = int(cfg["pts_add_bound"])
    lo, hi = srt[b], srt[-b]
    samples = torch.rand((num, 3), device=means.device, generator=generator) * (hi - lo) + lo
    keep, feats = interpolate_points(means, colors, samples, int(cfg["pts_N_intpl"]),
                                     float(cfg["pts_dist"]), int(cfg["knn_chunk"]))
    xyz = torch.cat([means, samples[keep]])
    feats = torch.cat([colors, feats[keep]])
    n = len(xyz)
    quats = torch.zeros(n, 4, device=xyz.device)
    quats[:, 0] = 1
    new = {
        "means": xyz,
        "sh0": feats[:, :1],
        "shN": feats[:, 1:],
        "scales": knn_log_scales(xyz, chunk=int(cfg["knn_chunk"])).contiguous(),  # distCUDA2: mean of 3-NN squared dists
        "quats": quats,
        "opacities": torch.full((n,), math.log(0.1 / 0.9), device=xyz.device),
    }
    extra = set(params) - set(new)
    if extra:
        raise RuntimeError(f"add_points cannot re-initialise fields {sorted(extra)}")
    _replace_all(params, optimizers, new)
    return int(keep.sum())


class Density:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown deblurring-3dgs density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = set(REQUIRES)
        self.state = {}
        self.bank = None
        self.counts = {"densify": 0, "clone": 0, "split": 0, "prune": 0, "added_points": 0}

    @property
    def bank_cfg(self):
        return {k[5:]: v for k, v in self.cfg.items() if k.startswith("bank_")}

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.extent = float(scene_scale)  # recipe 3dgs: getNerfppNorm radius (cameras_extent)
        self.pts_min = params["means"].detach().amin(dim=0)
        self.pts_max = params["means"].detach().amax(dim=0)
        self.bank = bank_mod.BlurBank(self.bank_cfg, params["means"].device)
        bank_mod.ACTIVE = self.bank
        optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(1)
        self._reset_stats(params)
        self.state = {"counts": self.counts, "extent": self.extent, "bank": self.bank.export()}

    def xyz_lr(self, step):
        c = self.cfg
        return expon_lr(step, c["position_lr_init"] * self.extent,
                        c["position_lr_final"] * self.extent, c["position_lr_max_steps"])

    def _reset_stats(self, params):
        n, dev = len(params["means"]), params["means"].device
        self.grad_accum = torch.zeros(n, device=dev)
        self.denom = torch.zeros(n, device=dev)

    def before_backward(self, params, optimizers, step, outputs):
        if self.bank is None or bank_mod.ACTIVE is not self.bank:
            raise RuntimeError("deblurring-3dgs density was not initialised")
        if "renders" not in outputs:
            raise RuntimeError("training step did not render through the blur MLP")
        outputs["info"]["means2d"].retain_grad()

    def accumulate(self, info, n_renders):
        grad = info["means2d"].grad
        if grad is None:
            raise RuntimeError("means2d gradient missing (retain_grad not applied)")
        g = grad.reshape(-1, grad.shape[-2], 2).clone()  # [renders, N, 2]
        g[..., 0] *= info["width"] / 2.0
        g[..., 1] *= info["height"] / 2.0
        radii = info["radii"].reshape(g.shape[0], g.shape[1], -1)
        vis = (radii > 0).all(dim=-1)
        self.grad_accum += (g.norm(dim=-1) * vis).sum(dim=0)
        self.denom += vis.float().sum(dim=0) / n_renders

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        self.bank.step()
        if step < c["densify_until_iter"]:
            self.accumulate(outputs["info"], outputs["n_renders"])
            if step > c["densify_from_iter"] and step % c["densification_interval"] == 0:
                grads = self.grad_accum / self.denom
                grads[grads.isnan()] = 0.0
                nc, ns, npr = densify_and_prune(params, optimizers, grads, c, self.extent)
                self.counts["densify"] += 1
                self.counts["clone"] += nc
                self.counts["split"] += ns
                self.counts["prune"] += npr
                self._reset_stats(params)  # densification_postfix zeroes the statistics
            if c["add_points"] and step == c["pts_iter"]:
                self.counts["added_points"] += add_points(params, optimizers, c, self.pts_min,
                                                          self.pts_max)
                self.bank.reset_optimizer()
                self.bank.counts["optimizer_resets"] += 1
                self._reset_stats(params)
                torch.cuda.empty_cache()
        optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(step + 1)
        if step == c["pts_iter"] or step % 1000 == 0:
            self.state["bank"] = self.bank.export()


def create_density(spec):
    return Density(spec)
