"""HGS density: Scaffold-GS anchor init / statistics / growing / pruning plus the two HGS
hard-Gaussian criteria.

Scaffold-GS references (github.com/city-super/Scaffold-GS @59c833b): scene/gaussian_model.py
(voxelize_sample, create_from_pcd, training_setup, training_statis, anchor_growing,
adjust_anchor, _prune_anchor_optimizer), train.py (loop), arguments/__init__.py (defaults).
Engineering (gsplat-ops alignment, NDC gradient scaling, guards) follows the matched HAC native
plugin (reproductions/hac/plugin/ss_hac/density.py).

HGS (arXiv 2412.04826) additions, no official code:
  PGHGS (Eq. 5-6): per neural Gaussian, keep the k largest per-view positional-gradient norms of
      the current growth interval; hard if the k-th largest >= lambda * tau.
  REHGS (Eq. 7-8): per training view, |S_i| = #pixels whose max-weight Gaussian is i; over-large if
      |S_i| > tau_large |P|; hard if additionally SSIM(I, I^) at its projected centre < tau_SSIM;
      grown only when detected in >= 2 views of the interval.
Integration into Scaffold's 3-level growing (paper.md §7, a reproduction choice): at level i with
threshold tau_i = tau 2^i a neural Gaussian is a candidate iff
      (avg >= tau_i and denom > 40)  or  (g'_k >= lambda tau_i)  or  REHGS-hard,
then Scaffold's random keep mask and voxel de-duplication apply unchanged. Hard statistics are
reset at every growth step ("during the M iterations").
"""

import copy
import os
import sys

import torch
import torch.nn.functional as F
from gsplat.strategy.ops import _update_param_with_optimizer

from . import kernels, model
from .renderer import BLOB, BLOB_WIDTH, pack_blob

DEFAULTS = {
    "total_steps": 30000,
    "feat_dim": 32,
    "n_offsets": 10,
    "voxel_size": 0.001,  # train_mip360.sh (train_tnt.sh 0.01, train_db.sh 0.005)
    "update_depth": 3,
    "update_init_factor": 16,
    "update_hierachy_factor": 4,
    "start_stat": 500,
    "update_from": 1500,
    "update_interval": 100,
    "update_until": 15000,
    "min_opacity": 0.005,
    "success_threshold": 0.8,
    "densify_grad_threshold": 0.0002,
    "pghgs": {"enabled": True, "k": 3, "lambda": 1.0},  # §6.1
    "rehgs": {"enabled": True, "tau_large": 2e-4, "tau_ssim": 0.7, "min_views": 2},  # §6.1, §5.2
    "lr": {
        "offset": [0.01, 0.0001],  # x spatial_lr_scale
        "feat": 0.0075,
        "scaling": 0.007,
        "opacity": 0.02,  # anchor _opacity: unused by the renderer (no gradient)
        "rotation": 0.002,  # anchor _rotation: unused (no gradient)
        "mlp_opacity": [0.002, 0.00002],
        "mlp_cov": [0.004, 0.004],
        "mlp_color": [0.008, 0.00005],
    },
}


def merge(defaults, config):
    out = copy.deepcopy(defaults)
    for k, v in config.items():
        if k not in out:
            raise ValueError(f"Unknown hgs-density option {k}")
        if isinstance(out[k], dict):
            unknown = set(v) - set(out[k])
            if unknown:
                raise ValueError(f"Unknown hgs-density option {k}.{sorted(unknown)}")
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- SSIM map (Eq. 8)
def _gauss_window(size=11, sigma=1.5, device=None):
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None].repeat(3, 1, 1, 1)


@torch.no_grad()
def ssim_map(img, gt, size=11):
    """3DGS utils/loss_utils.py:_ssim map (11x11 Gaussian window, sigma 1.5, same padding,
    C1 = 0.01^2, C2 = 0.03^2), averaged over RGB. Inputs HWC in [0,1]; returns [H,W]."""
    a = img.permute(2, 0, 1)[None].float()
    b = gt.permute(2, 0, 1)[None].float()
    w = _gauss_window(size, device=a.device)
    p = size // 2
    mu1 = F.conv2d(a, w, padding=p, groups=3)
    mu2 = F.conv2d(b, w, padding=p, groups=3)
    s11 = F.conv2d(a * a, w, padding=p, groups=3) - mu1**2
    s22 = F.conv2d(b * b, w, padding=p, groups=3) - mu2**2
    s12 = F.conv2d(a * b, w, padding=p, groups=3) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1**2 + mu2**2 + c1) * (s11 + s22 + c2))
    return m[0].mean(0)


@torch.no_grad()
def push_topk(buf, idx, values):
    """Insert `values` into the per-row descending top-k buffer rows `idx` (Eq. 5 sort)."""
    if len(idx) == 0:
        return
    cur = torch.cat([buf[idx], values[:, None]], dim=1)
    buf[idx] = cur.topk(buf.shape[1], dim=1).values


@torch.no_grad()
def rehgs_hard(area_max, means2d, radii_ok, ssim, width, height, tau_large, tau_ssim):
    """Eq. 7 and Eq. 8 for the rasterized Gaussians of one view -> bool [M]."""
    over = area_max.float() > tau_large * width * height
    u = torch.floor(means2d[:, 0]).long()
    v = torch.floor(means2d[:, 1]).long()
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    s = torch.ones_like(over, dtype=torch.float32)
    s[inside] = ssim[v[inside], u[inside]]
    return over & inside & radii_ok & (s < tau_ssim)


# ------------------------------------------------------------------------ population helpers
def new_anchor_fields(xyz, log_scaling, feat, n_offsets):
    n = len(xyz)
    dev = xyz.device
    quats = torch.zeros(n, 4, device=dev)
    quats[:, 0] = 1
    return {
        "means": xyz,
        "scales": log_scaling[:, :3].clone(),
        "quats": quats,
        "opacities": torch.full((n,), float(torch.logit(torch.tensor(0.1))), device=dev),
        "sh0": torch.zeros(n, 1, 3, device=dev),
        "shN": torch.zeros(n, 15, 3, device=dev),
        "hgs_feat": feat,
        "hgs_offset": torch.zeros(n, n_offsets, 3, device=dev),
        "hgs_scaling": log_scaling,
        BLOB: torch.zeros(n, BLOB_WIDTH, device=dev),
    }


@torch.no_grad()
def append_anchors(params, optimizers, new):
    """cat_tensors_to_optimizer: append rows; Adam moments of new rows start at zero."""
    for name in list(params.keys()):
        extra = new[name]

        def param_fn(n, p, extra=extra):
            return torch.nn.Parameter(torch.cat([p.detach(), extra.to(p.dtype)]),
                                      requires_grad=True)

        def opt_fn(key, v, extra=extra):
            return torch.cat([v, torch.zeros_like(extra)])

        _update_param_with_optimizer(param_fn, opt_fn, params, optimizers, [name])


@torch.no_grad()
def keep_anchors(params, optimizers, keep):
    sel = torch.where(keep)[0]
    _update_param_with_optimizer(
        lambda n, p: torch.nn.Parameter(p.detach()[sel].clone(), requires_grad=True),
        lambda k, v: v[sel], params, optimizers)


def replace_population(params, optimizers, values, lrs):
    for name, v in values.items():
        params[name] = torch.nn.Parameter(v.detach().contiguous().float(), requires_grad=True)
        optimizers[name] = torch.optim.Adam([params[name]], lr=lrs[name], eps=1e-15)


def _ss_train_num_gaussians():
    argv = sys.argv
    prog = os.path.basename(argv[0]) if argv else ""
    if not (prog == "ss-train" or (len(argv) > 1 and argv[1] == "train")):
        return None
    for i, arg in enumerate(argv):
        if arg == "--num-gaussians" and i + 1 < len(argv):
            return int(argv[i + 1])
        if arg.startswith("--num-gaussians="):
            return int(arg.split("=", 1)[1])
    return 10000


def assert_trainable(params):
    """TODO-runtime pitfall guard: every field must remain a trainable leaf after density ops."""
    for k, p in params.items():
        if not (isinstance(p, torch.nn.Parameter) and p.requires_grad and p.is_leaf):
            raise RuntimeError(f"field {k} is no longer a trainable leaf after a density op")


class HGSDensity:
    requires = {"gradient.screen.signed"}

    def __init__(self, spec):
        self.c = merge(DEFAULTS, spec.config)
        self.bank = None
        self.state = {"hgs": {"events": []}}
        self.spatial_lr_scale = 1.0
        self.interval = {"og": 0, "pg": 0, "re": 0}

    def lr_table(self):
        lr = self.c["lr"]
        return {"means": 0.0, "scales": 0.0, "quats": lr["rotation"], "opacities": lr["opacity"],
                "sh0": 0.0, "shN": 0.0, "hgs_feat": lr["feat"], "hgs_scaling": lr["scaling"],
                "hgs_offset": lr["offset"][0] * self.spatial_lr_scale, BLOB: 1e-12}

    # ------------------------------------------------------------------ initialisation
    @torch.no_grad()
    def initialize(self, params, optimizers, scene_scale=1.0):
        """create_from_pcd (ratio 1, voxelize_sample) + training_setup."""
        self.spatial_lr_scale = float(scene_scale)
        c = self.c
        points = params["means"].detach()
        cap = _ss_train_num_gaussians()
        if cap is not None and len(points) >= cap:
            raise ValueError(
                f"Scaffold-GS voxelizes the full SfM cloud but the runtime kept {len(points)} "
                f"points (--num-gaussians {cap}): pass e.g. --num-gaussians 10000000")
        dev = points.device
        perm = torch.randperm(len(points), device=dev)
        anchors = torch.unique(torch.round(points[perm] / c["voxel_size"]), dim=0)
        anchors = anchors * c["voxel_size"]
        if dev.type == "cuda":
            dist2 = kernels.dist_cuda2(anchors)
        else:  # CPU unit tests: exact 3-NN mean squared distance
            d2 = torch.cdist(anchors, anchors).square()
            d2.fill_diagonal_(float("inf"))
            dist2 = d2.topk(min(3, len(anchors) - 1), dim=1, largest=False).values.mean(1)
        log_scaling = torch.log(torch.sqrt(dist2.clamp_min(1e-7)))[:, None].repeat(1, 6)
        feat = torch.zeros(len(anchors), c["feat_dim"], device=dev)
        replace_population(params, optimizers,
                           new_anchor_fields(anchors, log_scaling, feat, c["n_offsets"]),
                           self.lr_table())
        self.offset_sched = model.expon_lr(c["lr"]["offset"][0] * self.spatial_lr_scale,
                                           c["lr"]["offset"][1] * self.spatial_lr_scale,
                                           c["total_steps"])
        self.bank = model.MLPBank(c, dev)
        model.ACTIVE["bank"] = self.bank
        self.update_learning_rate(params, optimizers, 1)
        self.reset_stats(len(anchors), dev)
        self.pack(params)
        self.state["hgs"]["initial_anchors"] = len(anchors)
        self.state["hgs"]["spatial_lr_scale"] = self.spatial_lr_scale

    def set_training_cameras(self, cameras):
        """Guard: the offset LR scales with 3DGS's cameras_extent, i.e. --recipe 3dgs."""
        centers = torch.stack([torch.linalg.inv(c["viewmat"])[:3, 3] for c in cameras])
        extent = 1.1 * float(torch.linalg.vector_norm(centers - centers.mean(0), dim=1).max())
        if abs(max(extent, 0.01) - self.spatial_lr_scale) > 1e-4 * max(extent, 1e-6):
            raise ValueError(
                f"Scaffold-GS needs the 3DGS camera-extent scene scale ({extent:.4f}); got "
                f"{self.spatial_lr_scale:.4f}: run ss-train with --recipe 3dgs")
        self.state["hgs"]["camera_extent"] = extent

    def reset_stats(self, n, dev):
        k = self.c["n_offsets"]
        self.opacity_accum = torch.zeros(n, 1, device=dev)
        self.anchor_demon = torch.zeros(n, 1, device=dev)
        self.offset_gradient_accum = torch.zeros(n * k, 1, device=dev)
        self.offset_denom = torch.zeros(n * k, 1, device=dev)
        self.reset_hard_stats(n * k, dev)

    def reset_hard_stats(self, nk, dev):
        self.grad_topk = torch.zeros(nk, int(self.c["pghgs"]["k"]), device=dev)
        self.re_count = torch.zeros(nk, device=dev, dtype=torch.int32)

    def update_learning_rate(self, params, optimizers, step):
        optimizers["means"].param_groups[0]["lr"] = 0.0  # position_lr_init = 0
        optimizers["hgs_offset"].param_groups[0]["lr"] = self.offset_sched(step)
        self.bank.update_learning_rate(step)

    def pack(self, params):
        c = self.c
        params[BLOB].data.copy_(pack_blob(self.bank.mlps, len(params["means"]), c["feat_dim"],
                                          c["n_offsets"], params[BLOB].device))

    # ----------------------------------------------------------------------- per step
    def before_backward(self, params, optimizers, step, outputs):
        means2d = outputs["info"]["means2d"]
        if means2d.requires_grad:
            means2d.retain_grad()

    def after_backward(self, params, optimizers, step, outputs):
        c = self.c
        if step < c["total_steps"]:  # official: optimizer.step() only if iteration < iterations
            self.bank.optimizer_step()
        else:
            self.bank.optimizer.zero_grad(set_to_none=True)
        if c["start_stat"] < step < c["update_until"]:
            self.training_statis(outputs)
            if step % c["update_interval"] == 0:
                if step > c["update_from"]:
                    self.adjust_anchor(params, optimizers, step)  # resets the hard stats
                    assert_trainable(params)
                else:  # hard statistics cover one growth interval of M its ("during the M
                    # iterations"), also before update_from (OG averages keep Scaffold's rule)
                    self.reset_hard_stats(len(params["means"]) * c["n_offsets"],
                                          params["means"].device)
        self.update_learning_rate(params, optimizers, step + 1)
        self.pack(params)

    @torch.no_grad()
    def training_statis(self, outputs):
        c = self.c
        k = c["n_offsets"]
        info = outputs["info"]
        hgs = info["hgs"]
        vis, sel = hgs["visible"], hgs["selection"]
        means2d = info["means2d"]
        if means2d.grad is None:
            return
        temp = hgs["neural_opacity"].detach().view(-1).clone()
        temp[temp < 0] = 0
        temp = temp.view(-1, k)
        self.opacity_accum[vis] += temp.sum(dim=1, keepdim=True)
        self.anchor_demon[vis] += 1
        radii = info["radii"][0]
        update_filter = (radii > 0).all(-1) if radii.dim() == 2 else radii > 0
        # global neural-Gaussian index of every rasterized Gaussian (Scaffold combined_mask)
        vis_ids = torch.where(vis)[0]
        nid = (vis_ids[:, None] * k + torch.arange(k, device=vis.device)[None]).reshape(-1)[sel]
        # gsplat means2d gradients are in pixels; 3DGS screenspace gradients are NDC-scaled.
        g = means2d.grad[0].clone()
        g[:, 0] *= info["width"] / 2.0
        g[:, 1] *= info["height"] / 2.0
        gnorm = g[update_filter].norm(dim=-1)
        ids = nid[update_filter]
        self.offset_gradient_accum[ids] += gnorm[:, None]
        self.offset_denom[ids] += 1
        if c["pghgs"]["enabled"]:
            push_topk(self.grad_topk, ids, gnorm)  # Eq. 5 (per-view gradient norms)
        if c["rehgs"]["enabled"]:
            target = outputs.get("hgs_target")
            if target is None:
                raise RuntimeError("REHGS needs the training image (hgs-scaling-reg loss taps it)")
            rgb = outputs["rgb"].detach().clamp(0, 1)
            smap = ssim_map(rgb, target.clamp(0, 1))
            area_max, _ = kernels.max_contrib_counts(info)
            hard = rehgs_hard(area_max, info["means2d"][0].detach(), update_filter, smap,
                              int(info["width"]), int(info["height"]),
                              c["rehgs"]["tau_large"], c["rehgs"]["tau_ssim"])
            self.re_count[nid[hard]] += 1

    @torch.no_grad()
    def candidate_mask(self, level, grads, offset_mask, kth, re_hard):
        """Level-i growth candidates (see module docstring)."""
        c = self.c
        thr = c["densify_grad_threshold"] * ((c["update_hierachy_factor"] // 2) ** level)
        og = (grads >= thr) & offset_mask
        pg = torch.zeros_like(og)
        if c["pghgs"]["enabled"]:
            pg = kth >= c["pghgs"]["lambda"] * thr  # Eq. 6
        cand = og | pg | re_hard
        self.interval["og"] += int(og.sum())
        self.interval["pg"] += int((pg & ~og).sum())
        self.interval["re"] += int((re_hard & ~og & ~pg).sum())
        return cand

    @torch.no_grad()
    def anchor_growing(self, params, optimizers, grads, offset_mask, kth, re_hard):
        c = self.c
        k = c["n_offsets"]
        init_length = len(params["means"]) * k
        added = 0
        for i in range(c["update_depth"]):
            cand = self.candidate_mask(i, grads, offset_mask, kth, re_hard)
            cand = cand & (torch.rand_like(cand.float()) > (0.5 ** (i + 1)))
            length_inc = len(params["means"]) * k - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                cand = torch.cat([cand, torch.zeros(length_inc, dtype=torch.bool,
                                                    device=cand.device)])
            anchor = params["means"].detach()
            scaling = params["hgs_scaling"].detach().exp()
            all_xyz = anchor.unsqueeze(1) + params["hgs_offset"].detach() * scaling[:, :3].unsqueeze(1)
            cur_size = c["voxel_size"] * (c["update_init_factor"] // (c["update_hierachy_factor"] ** i))
            grid_coords = torch.round(anchor / cur_size).int()
            selected = all_xyz.view(-1, 3)[cand]
            sel_coords = torch.round(selected / cur_size).int()
            uniq, inverse = torch.unique(sel_coords, return_inverse=True, dim=0)
            if len(uniq) == 0:
                continue
            both = torch.cat([grid_coords, uniq])
            _, ids = torch.unique(both, return_inverse=True, dim=0)
            existing = torch.zeros(int(ids.max()) + 1, dtype=torch.bool, device=ids.device)
            existing[ids[: len(grid_coords)]] = True
            keep = ~existing[ids[len(grid_coords):]]
            cand_anchor = uniq[keep].float() * cur_size
            if len(cand_anchor) == 0:
                continue
            new_scaling = torch.log(torch.ones_like(cand_anchor).repeat(1, 2) * cur_size)
            feat_rep = params["hgs_feat"].detach().unsqueeze(1).repeat(1, k, 1).view(-1, c["feat_dim"])
            feat_rep = feat_rep[cand]
            new_feat = torch.full((len(uniq), c["feat_dim"]), float("-inf"), device=feat_rep.device)
            new_feat = new_feat.scatter_reduce(
                0, inverse.unsqueeze(1).expand(-1, c["feat_dim"]), feat_rep, "amax",
                include_self=False)[keep]  # torch_scatter.scatter_max
            append_anchors(params, optimizers,
                           new_anchor_fields(cand_anchor, new_scaling, new_feat, k))
            n_new = len(cand_anchor)
            added += n_new
            dev = cand_anchor.device
            self.anchor_demon = torch.cat([self.anchor_demon, torch.zeros(n_new, 1, device=dev)])
            self.opacity_accum = torch.cat([self.opacity_accum, torch.zeros(n_new, 1, device=dev)])
        return added

    @torch.no_grad()
    def adjust_anchor(self, params, optimizers, step):
        c = self.c
        k = c["n_offsets"]
        check_interval = c["update_interval"]
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval * c["success_threshold"] * 0.5).squeeze(1)
        kth = self.grad_topk[:, -1]
        re_hard = self.re_count >= c["rehgs"]["min_views"]
        if not c["rehgs"]["enabled"]:
            re_hard = torch.zeros_like(re_hard)
        self.interval = {"og": 0, "pg": 0, "re": 0}
        added = self.anchor_growing(params, optimizers, grads_norm, offset_mask, kth, re_hard)
        n = len(params["means"])
        dev = grads.device
        self.offset_denom[offset_mask] = 0
        self.offset_denom = torch.cat([self.offset_denom,
                                       torch.zeros(n * k - len(self.offset_denom), 1, device=dev)])
        self.offset_gradient_accum[offset_mask] = 0
        self.offset_gradient_accum = torch.cat(
            [self.offset_gradient_accum,
             torch.zeros(n * k - len(self.offset_gradient_accum), 1, device=dev)])
        prune = (self.opacity_accum < c["min_opacity"] * self.anchor_demon).squeeze(1)
        anchors_mask = (self.anchor_demon > check_interval * c["success_threshold"]).squeeze(1)
        prune = prune & anchors_mask
        keep = ~prune
        self.offset_denom = self.offset_denom.view(-1, k)[keep].view(-1, 1)
        self.offset_gradient_accum = self.offset_gradient_accum.view(-1, k)[keep].view(-1, 1)
        if anchors_mask.sum() > 0:
            self.opacity_accum[anchors_mask] = 0
            self.anchor_demon[anchors_mask] = 0
        self.opacity_accum = self.opacity_accum[keep]
        self.anchor_demon = self.anchor_demon[keep]
        keep_anchors(params, optimizers, keep)
        # _prune_anchor_optimizer clamps raw scaling[:, 3:] at 0.05 on every prune call.
        params["hgs_scaling"].data[:, 3:] = params["hgs_scaling"].data[:, 3:].clamp(max=0.05)
        self.reset_hard_stats(len(params["means"]) * k, dev)
        self.state["hgs"]["events"].append(  # every growth step (~135 rows per run)
            {"step": step, "grown": int(added), "pruned": int(prune.sum()),
                 "anchors": len(params["means"]),
                 "cand_og": self.interval["og"], "cand_pg_only": self.interval["pg"],
                 "cand_re_only": self.interval["re"],
                 "re_hard": int(re_hard.sum()), "pg_hard_l0": int(
                     (kth >= c["pghgs"]["lambda"] * c["densify_grad_threshold"]).sum())})
        return added


def create_density(spec):
    return HGSDensity(spec)
