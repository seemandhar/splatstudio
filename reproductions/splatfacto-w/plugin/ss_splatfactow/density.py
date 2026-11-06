"""Splatfacto-W density: the splatfacto adaptive density control of the official code
(``SplatfactoWModel.after_train`` / ``refinement_after`` / ``cull_gaussians``), the code's
initialisation (log mean 3-NN distance scales, random quaternions) and every learning-rate schedule
(``splatfactow_config.py``), plus the learned bank (embeddings, appearance MLP, background MLP).

Thresholds are in Nerfstudio scene units (NerfW dataparser: orient 'up', centre on poses, scale to
+-1 then x3); they are converted with the bank's scale s (x_ns = s R (x - c)). Topology changes use
gsplat's ``duplicate`` / ``split`` (2 samples, scales / 1.6) / ``remove`` / ``reset_opa``, which keep
optimizer moments aligned (new Gaussians start with zero moments, as the code's ``dup_in_optim``).
"""

import math

import torch
from gsplat.strategy.ops import duplicate, remove, reset_opa, split

from . import bank as bank_mod
from .bank import BANK_DEFAULTS, SfwBank
from .schedule import lr_at

REFINE_DEFAULTS = {
    "warmup_length": 1000,
    "refine_every": 100,
    "reset_alpha_every": 25,
    "densify_grad_thresh": 0.0008,
    "densify_size_thresh": 0.01,
    "cull_alpha_thresh": 0.1,
    "cull_scale_thresh": 0.5,
    "cull_screen_size": 0.15,
    "split_screen_size": 0.05,
    "stop_screen_size_at": 15000,
    "stop_split_at": 20000,
    "continue_cull_post_densification": True,
    "reinit_scales_quats": True,
}
DEFAULTS = {**REFINE_DEFAULTS, **{f"bank_{k}": v for k, v in BANK_DEFAULTS.items()}}
REQUIRES = ("gradient.screen.absolute",)
GAUSSIAN_LR_FIELDS = ("opacities", "scales", "quats", "appearance_features")


def random_quat_tensor(n, generator=None):
    """splatfacto random_quat_tensor (uniform random rotations, wxyz)."""
    u, v, w = (torch.rand(n, generator=generator) for _ in range(3))
    return torch.stack([torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
                        torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
                        torch.sqrt(u) * torch.sin(2 * math.pi * w),
                        torch.sqrt(u) * torch.cos(2 * math.pi * w)], dim=-1)


def knn_mean_log_scale(points, k=3, chunk=4096):
    """splatfacto init: log(mean distance to the 3 nearest neighbours)."""
    out = torch.empty(len(points), device=points.device)
    for s in range(0, len(points), chunk):
        d = torch.cdist(points[s: s + chunk], points)
        d[torch.arange(d.shape[0]), torch.arange(s, s + d.shape[0])] = float("inf")
        out[s: s + chunk] = d.topk(k, dim=1, largest=False).values.mean(dim=1)
    return out.clamp_min(1e-7).log()[:, None].expand(-1, 3)


def is_reset_step(k, cfg):
    interval = int(cfg["reset_alpha_every"]) * int(cfg["refine_every"])
    return (k > int(cfg["warmup_length"]) and k % int(cfg["refine_every"]) == 0
            and k < int(cfg["stop_split_at"]) and k % interval == int(cfg["refine_every"]))


def is_densify_step(k, cfg, n_train):
    interval = int(cfg["reset_alpha_every"]) * int(cfg["refine_every"])
    return (k > int(cfg["warmup_length"]) and k % int(cfg["refine_every"]) == 0
            and k < int(cfg["stop_split_at"])
            and k % interval > n_train + int(cfg["refine_every"]))


class Density:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown splatfacto-w density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = set(REQUIRES)
        self.bank = None
        self.n_train = None
        self.s = 1.0
        self.reset_accumulators()
        self.counters = {"densify": 0, "split": 0, "dup": 0, "cull": 0, "reset": 0,
                         "alpha_active_steps": 0, "alpha_selected_pixels": 0,
                         "alpha_last_active_step": 0}

    @property
    def bank_cfg(self):
        return {k[5:]: v for k, v in self.cfg.items() if k.startswith("bank_")}

    @property
    def state(self):
        s = {"counters": dict(self.counters), "ns_scale": float(self.s)}
        if self.bank is not None:
            s["bank"] = self.bank.export()
        return s

    def reset_accumulators(self):
        self.grad_norm = None
        self.vis_counts = None
        self.max_2d = None
        self.last_size = None

    def initialize(self, params, optimizers, scene_scale=1.0):
        if self.cfg["reinit_scales_quats"]:
            g = torch.Generator().manual_seed(int(self.cfg["bank_seed"]))
            with torch.no_grad():
                params["scales"].copy_(knn_mean_log_scale(params["means"].detach()))
                params["quats"].copy_(random_quat_tensor(len(params["means"]), g)
                                      .to(params["quats"].device))
        self.optimizers = optimizers
        self.set_lrs(optimizers, 0)

    def set_lrs(self, optimizers, k):
        for name in GAUSSIAN_LR_FIELDS:
            if name in optimizers:
                optimizers[name].param_groups[0]["lr"] = lr_at(name, k)
        optimizers["means"].param_groups[0]["lr"] = lr_at("means", k) / self.s

    def set_training_cameras(self, cameras):
        dev = cameras[0]["viewmat"].device
        self.bank = SfwBank([c["viewmat"] for c in cameras], [int(c["width"]) for c in cameras],
                            [c["K"] for c in cameras], self.bank_cfg, dev)
        self.bank.train()
        bank_mod.ACTIVE = self.bank
        self.n_train = len(cameras)
        self.s = self.bank.ns_scale
        self.set_lrs(self.optimizers, 0)

    def before_backward(self, params, optimizers, step, outputs):
        if self.bank is None:
            raise RuntimeError("splatfacto-w density needs set_training_cameras (native ss-train)")
        outputs["info"]["means2d"].retain_grad()

    # -- splatfacto after_train -------------------------------------------------------------
    @torch.no_grad()
    def accumulate(self, info, masked_view=False):
        radii = info["radii"][0]
        if radii.dim() == 2:  # gsplat >= 1.5: per-axis radii
            visible = (radii > 0).all(dim=-1)
            r = radii.max(dim=-1).values
        else:
            visible, r = radii > 0, radii
        n = len(r)
        grads = info["means2d"].absgrad[0][visible].norm(dim=-1)
        if self.grad_norm is None:
            self.grad_norm = torch.zeros(n, device=r.device)
            self.vis_counts = torch.ones(n, device=r.device)
        self.vis_counts[visible] += 1
        self.grad_norm[visible] += grads
        h, w = int(info["height"]), int(info["width"])
        if masked_view:  # left-half crop: the code renders (and normalises by) the full image
            w = 2 * w
        if self.max_2d is None:
            self.max_2d = torch.zeros(n, device=r.device)
        self.max_2d[visible] = torch.maximum(self.max_2d[visible],
                                             r[visible].float() / float(max(h, w)))
        self.last_size = (h, w)

    # -- splatfacto cull_gaussians ----------------------------------------------------------
    @torch.no_grad()
    def cull_mask(self, params, k, max_2d=None):
        c = self.cfg
        culls = torch.sigmoid(params["opacities"]).flatten() < float(c["cull_alpha_thresh"])
        if k > int(c["refine_every"]) * int(c["reset_alpha_every"]):
            toobig = params["scales"].exp().max(dim=-1).values > float(c["cull_scale_thresh"]) / self.s
            if k < int(c["stop_screen_size_at"]) and max_2d is not None:
                toobig |= max_2d > float(c["cull_screen_size"])
            culls |= toobig
        return culls

    # -- splatfacto refinement_after --------------------------------------------------------
    @torch.no_grad()
    def refine(self, params, optimizers, k):
        c = self.cfg
        if k <= int(c["warmup_length"]):
            return
        culls = None
        if is_densify_step(k, c, self.n_train):
            avg = self.grad_norm / self.vis_counts * 0.5 * max(self.last_size)
            high = avg > float(c["densify_grad_thresh"])
            smax = params["scales"].exp().max(dim=-1).values
            small = smax <= float(c["densify_size_thresh"]) / self.s
            splits = ~small
            if k < int(c["stop_screen_size_at"]):
                splits |= self.max_2d > float(c["split_screen_size"])
            splits &= high
            # Official split_gaussians shrinks the split Gaussians' scales by 1.6 *in place*
            # (splatfactow_model.py:778-780) before dups are computed from self.scales
            # (:632-637): a split Gaussian whose shrunk scale is <= thr is also duplicated
            # (3 children) and every duplicate of a split Gaussian carries the shrunk scale.
            shrink = math.log(1.6)
            smax_after = torch.where(splits, smax / 1.6, smax)
            dups = (smax_after <= float(c["densify_size_thresh"]) / self.s) & high
            nd, ns = int(dups.sum()), int(splits.sum())
            n0 = len(smax)
            max_2d = torch.cat([self.max_2d, torch.zeros(nd, device=smax.device)])
            if nd:
                duplicate(params, optimizers, {}, dups)
                shrunk = splits[dups]  # appended copies (in order) whose source is split
                if bool(shrunk.any()):
                    tail = params["scales"].data[n0:]
                    tail[shrunk] -= shrink
            split_mask = torch.cat([splits, torch.zeros(nd, dtype=torch.bool, device=smax.device)])
            if ns:
                split(params, optimizers, {}, split_mask)
            max_2d = torch.cat([max_2d[~split_mask], torch.zeros(2 * ns, device=smax.device)])
            culls = self.cull_mask(params, k, max_2d)
            self.counters["densify"] += 1
            self.counters["dup"] += nd
            self.counters["split"] += ns
        elif k >= int(c["stop_split_at"]) and c["continue_cull_post_densification"]:
            culls = self.cull_mask(params, k, None)
        if culls is not None and bool(culls.any()):
            if bool(culls.all()):
                raise RuntimeError("splatfacto-w culling would remove every Gaussian")
            self.counters["cull"] += int(culls.sum())
            remove(params, optimizers, {}, culls)
        if is_reset_step(k, c):
            reset_opa(params, optimizers, {}, value=float(c["cull_alpha_thresh"]) * 2.0)
            self.counters["reset"] += 1
        self.reset_accumulators()

    def after_backward(self, params, optimizers, step, outputs):
        k = int(step) - 1  # Nerfstudio 0-based step of the iteration that just ran
        self.bank.after_step(step)  # bank LRs for the next iteration (0-based step = step)
        self.set_lrs(optimizers, step)
        sel = int(outputs.get("sfw_alpha_selected", 0))
        if sel:
            self.counters["alpha_active_steps"] += 1
            self.counters["alpha_selected_pixels"] += sel
            self.counters["alpha_last_active_step"] = int(step)
        if k < int(self.cfg["stop_split_at"]):
            self.accumulate(outputs["info"], bool(outputs.get("masked_view", False)))
        if k % int(self.cfg["refine_every"]) == 0:
            self.refine(params, optimizers, k)


def create_density(spec):
    return Density(spec)
