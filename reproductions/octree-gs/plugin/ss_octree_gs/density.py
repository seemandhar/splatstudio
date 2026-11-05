"""Octree-GS density component: octree anchor initialisation, Scaffold-GS statistics, LOD-aware anchor growing
(same level + next level + LOD-bias update + view-frequency weed-out), opacity pruning, field / MLP learning-rate
schedules, the bank's optimiser and the per-step blob packing.

Official references (github.com/city-super/Octree-GS @ 9db64fe): scene/gaussian_model.py (set_level,
create_from_pcd, weed_out, training_setup, update_learning_rate, training_statis, anchor_growing, adjust_anchor,
_prune_anchor_optimizer), scene/__init__.py (unique points, set_level before create_from_pcd), train.py (loop
order), arguments/__init__.py + train_mipnerf360.sh (defaults). Field helpers, statistics, membership test and
pruning are adapted from the ContextGS / HAC native ports (reproductions/contextgs/plugin/ss_contextgs/density.py).
"""

import copy
import os
import sys

import torch
from gsplat.strategy.ops import _update_param_with_optimizer

from . import kernels, model, octree
from .renderer import BLOB, BLOB_WIDTH, blob_floats, pack_blob_flat, write_blob

DEFAULTS = {
    "total_steps": 40000,  # train.sh iterations=40_000
    "feat_dim": 32,
    "n_offsets": 10,
    "fork": 2,
    "base_layer": 10,  # train_mipnerf360.sh / train_tandt.sh (DB: 12); < 0 -> scene-based (0.02 voxel)
    "visible_threshold": -1.0,  # train_mipnerf360.sh: -1 -> adaptive mean view frequency (T&T / DB: 0.9)
    "dist2level": "round",
    "update_ratio": 0.2,  # beta
    "extra_ratio": 0.25,
    "extra_up": 0.01,  # epsilon
    "progressive": True,
    "dist_ratio": 0.999,  # r_d
    "levels": -1,
    "init_level": -1,
    "extend": 1.1,
    "coarse_iter": 10000,
    "coarse_factor": 1.5,  # omega
    # Official `if ~self.progressive or iteration > coarse_intervals[-1]`: ~True == -2 is truthy, so next-level
    # growing and Delta L updates run during progressive training too. True = the paper's text instead.
    "disable_next_level_during_progressive": False,
    "start_stat": 500,
    "update_from": 1500,
    "update_interval": 100,
    "update_until": 25000,
    "min_opacity": 0.005,
    "success_threshold": 0.8,
    "densify_grad_threshold": 0.0002,
    "lr": {
        "offset": [0.01, 0.0001],  # x spatial_lr_scale
        "feat": 0.0075,
        "scaling": 0.007,
        "opacity": 0.02,
        "rotation": 0.002,
        "mlp_opacity": [0.002, 0.00002],
        "mlp_cov": [0.004, 0.004],
        "mlp_color": [0.008, 0.00005],
    },
}

FIELD_NAMES = ("ogs_feat", "ogs_offset", "ogs_scaling", "ogs_level", "ogs_extra_level", BLOB)


def merge(defaults, config):
    out = copy.deepcopy(defaults)
    for k, v in config.items():
        if k not in out:
            raise ValueError(f"Unknown ogs-density option {k}")
        out[k] = {**out[k], **v} if isinstance(out[k], dict) else v
    if out["dist2level"] != "round":
        raise ValueError("Only dist2level='round' (the official script setting) is implemented")
    return out


def new_anchor_fields(xyz, log_scaling, feat, level, n_offsets):
    """Per-anchor values of create_from_pcd / anchor_growing for every native field."""
    n = len(xyz)
    dev = xyz.device
    quats = torch.zeros(n, 4, device=dev)
    quats[:, 0] = 1
    return {
        "means": xyz.float(),
        "scales": log_scaling[:, :3].clone(),
        "quats": quats,
        "opacities": torch.full((n,), float(torch.logit(torch.tensor(0.1))), device=dev),
        "sh0": torch.zeros(n, 1, 3, device=dev),
        "shN": torch.zeros(n, 15, 3, device=dev),
        "ogs_feat": feat,
        "ogs_offset": torch.zeros(n, n_offsets, 3, device=dev),
        "ogs_scaling": log_scaling,
        "ogs_level": level.reshape(n, 1).float(),
        "ogs_extra_level": torch.zeros(n, 1, device=dev),
        BLOB: torch.zeros(n, BLOB_WIDTH, device=dev),
    }


@torch.no_grad()
def append_anchors(params, optimizers, new):
    """cat_tensors_to_optimizer: append rows, Adam moments of the new rows start at zero."""
    def param_fn(name, p):
        return torch.nn.Parameter(torch.cat([p, new[name].to(p.dtype)]), requires_grad=True)

    for name in list(params.keys()):
        extra = new[name]

        def opt_fn(key, v, extra=extra):
            return torch.cat([v, torch.zeros_like(extra)])

        _update_param_with_optimizer(param_fn, opt_fn, params, optimizers, [name])


@torch.no_grad()
def keep_anchors(params, optimizers, keep):
    sel = torch.where(keep)[0]
    _update_param_with_optimizer(lambda n, p: torch.nn.Parameter(p[sel], requires_grad=True),
                                 lambda k, v: v[sel], params, optimizers)


def replace_population(params, optimizers, values, lrs):
    """Swap every field for a new population with fresh Adam state (initialisation)."""
    for name, v in values.items():
        params[name] = torch.nn.Parameter(v.contiguous().float(), requires_grad=True)
        optimizers[name] = torch.optim.Adam([params[name]], lr=lrs[name], eps=1e-15)


def membership(query, table):
    """Exact set membership of integer grid cells (official get_remove_duplicates: chunked all()/any())."""
    both = torch.cat([table, query])
    _, ids = torch.unique(both, return_inverse=True, dim=0)
    existing = torch.zeros(int(ids.max()) + 1, dtype=torch.bool, device=ids.device)
    existing[ids[: len(table)]] = True
    return existing[ids[len(table):]]


def scatter_max(src, index, n):
    """torch_scatter.scatter_max(src, index, dim=0)[0] for n groups (every group non-empty)."""
    out = torch.full((n, src.shape[1]), float("-inf"), device=src.device)
    return out.scatter_reduce(0, index.unsqueeze(1).expand(-1, src.shape[1]), src, "amax", include_self=False)


def _ss_train_num_gaussians():
    """--num-gaussians of an `ss-train` command line (runtime default 10000); None outside that CLI."""
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


class OctreeGSDensity:
    requires = {"gradient.screen.signed"}

    def __init__(self, spec):
        self.c = merge(DEFAULTS, spec.config)
        self.bank = None
        self.points = None
        self.state = {"octree_gs": {"events": []}}
        self.spatial_lr_scale = 1.0

    # ------------------------------------------------------------------ initialisation
    def lr_table(self):
        lr = self.c["lr"]
        return {"means": 0.0, "scales": 0.0, "quats": lr["rotation"], "opacities": lr["opacity"], "sh0": 0.0,
                "shN": 0.0, "ogs_feat": lr["feat"], "ogs_scaling": lr["scaling"],
                "ogs_offset": lr["offset"][0] * self.spatial_lr_scale, "ogs_level": 0.0,
                "ogs_extra_level": 0.0, BLOB: 0.0}

    @torch.no_grad()
    def initialize(self, params, optimizers, scene_scale=1.0):
        """Keeps the SfM points; the octree needs the training cameras and is built in set_training_cameras
        (the runtime calls it right after initialize, before step 1)."""
        self.spatial_lr_scale = float(scene_scale)
        points = params["means"].detach()
        cap = _ss_train_num_gaussians()
        if cap is not None and len(points) >= cap:
            raise ValueError(
                f"Octree-GS voxelizes the full SfM cloud but the runtime kept {len(points)} points "
                f"(--num-gaussians {cap}): pass --num-gaussians larger than the SfM cloud, e.g. 10000000")
        self.points = torch.unique(points.clone(), dim=0)  # scene/__init__.py: torch.unique(points, dim=0)
        self.bank = model.OctreeBank(self.c, points.device)
        model.ACTIVE["bank"] = self.bank
        self.params, self.optimizers = params, optimizers

    def set_training_cameras(self, cameras):
        centers = torch.stack([torch.linalg.inv(c["viewmat"])[:3, 3] for c in cameras])
        extent = 1.1 * float(torch.linalg.vector_norm(centers - centers.mean(0), dim=1).max())
        if abs(max(extent, 0.01) - self.spatial_lr_scale) > 1e-4 * max(extent, 1e-6):
            raise ValueError(
                f"Octree-GS needs the 3DGS camera-extent scene scale ({extent:.4f}); got "
                f"{self.spatial_lr_scale:.4f}: run ss-train with --recipe 3dgs")
        self.state["octree_gs"]["camera_extent"] = extent
        self.build(self.params, self.optimizers, centers)

    @torch.no_grad()
    def build(self, params, optimizers, cam_centers):
        """set_level + create_from_pcd (+ training_setup)."""
        c, bank = self.c, self.bank
        pts = self.points
        dev = pts.device
        cam_centers = cam_centers.to(dev).float()
        std, levels, init_level = octree.set_level(pts, cam_centers, c["dist_ratio"], c["fork"], c["levels"],
                                                   c["init_level"])
        box_min = torch.min(pts) * c["extend"]
        box_d = torch.max(pts) * c["extend"] - box_min
        base_layer = c["base_layer"]
        if base_layer < 0:
            base_layer = torch.round(torch.log2(box_d / 0.02)).int().item() - (levels // 2) + 1
        voxel_size, init_pos = octree.octree_box(pts, c["extend"], c["fork"], base_layer)
        bank.levels, bank.init_level, bank.standard_dist = levels, init_level, std
        bank.voxel_size, bank.init_pos = voxel_size.to(dev), init_pos.to(dev)
        bank.cam_centers = cam_centers
        bank.coarse_intervals = octree.coarse_intervals(levels, init_level, c["coarse_iter"], c["coarse_factor"])
        pos, lev = octree.octree_sample(pts, bank.init_pos, bank.voxel_size, levels, c["fork"])
        n_sampled = len(pos)
        vt = float(c["visible_threshold"])
        if vt < 0:
            keep, mean = octree.weed_out(pos, lev, cam_centers, std, levels, 0.0, c["fork"])
            pos, lev, vt = pos[keep], lev[keep], mean
        keep, _ = octree.weed_out(pos, lev, cam_centers, std, levels, vt, c["fork"])
        pos, lev = pos[keep], lev[keep]
        bank.visible_threshold = vt
        dist2 = torch.clamp_min(kernels.dist_cuda2(pos), 0.0000001)
        log_scaling = torch.log(torch.sqrt(dist2))[:, None].repeat(1, 6)
        feat = torch.zeros(len(pos), c["feat_dim"], device=dev)
        values = new_anchor_fields(pos, log_scaling, feat, lev, c["n_offsets"])
        replace_population(params, optimizers, values, self.lr_table())
        self.offset_sched = model.expon_lr(c["lr"]["offset"][0] * self.spatial_lr_scale,
                                           c["lr"]["offset"][1] * self.spatial_lr_scale, c["total_steps"])
        self.update_learning_rate(params, optimizers, 1)
        self.reset_stats(len(pos), dev)
        self.pack(params, 0)
        self.state["octree_gs"].update({
            "levels": levels, "init_level": init_level, "standard_dist": std, "base_layer": int(base_layer),
            "voxel_size": float(voxel_size), "init_pos": float(init_pos[0]), "visible_threshold": vt,
            "coarse_intervals": bank.coarse_intervals, "octree_sampled": n_sampled,
            "initial_anchors": len(pos), "initial_per_level": self.per_level(params),
            "spatial_lr_scale": self.spatial_lr_scale, "sfm_points": len(pts)})
        print({"octree_gs_init": {k: v for k, v in self.state["octree_gs"].items() if k != "events"}}, flush=True)

    def per_level(self, params):
        lev = params["ogs_level"][:, 0]
        return [int((lev == i).sum()) for i in range(self.bank.levels)]

    def reset_stats(self, n, dev):
        k = self.c["n_offsets"]
        self.opacity_accum = torch.zeros(n, 1, device=dev)
        self.anchor_demon = torch.zeros(n, 1, device=dev)
        self.offset_gradient_accum = torch.zeros(n * k, 1, device=dev)
        self.offset_denom = torch.zeros(n * k, 1, device=dev)

    def update_learning_rate(self, params, optimizers, step):
        """update_learning_rate(iteration) at the top of each official iteration."""
        optimizers["means"].param_groups[0]["lr"] = 0.0  # position_lr_init = 0
        optimizers["ogs_offset"].param_groups[0]["lr"] = self.offset_sched(step)
        self.bank.update_learning_rate(step)

    @torch.no_grad()
    def pack(self, params, step):
        need = blob_floats(self.c["feat_dim"], self.c["n_offsets"])
        if len(params["means"]) * BLOB_WIDTH < need:
            raise RuntimeError(f"{len(params['means'])} anchors cannot hold the {need}-float decoder blob")
        b = self.bank
        oct_ = {"levels": b.levels, "fork": b.fork, "voxel_size": float(b.voxel_size),
                "standard_dist": b.standard_dist, "init_pos": float(b.init_pos[0]), "init_level": b.init_level}
        write_blob(params[BLOB], pack_blob_flat(b.mlps, oct_, self.c["feat_dim"], self.c["n_offsets"], step,
                                                params[BLOB].device))

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
        ogs = outputs["info"]["ogs"]
        if c["start_stat"] < step < c["update_until"]:
            self.training_statis(outputs["info"], ogs)
            if step > c["update_from"] and step % c["update_interval"] == 0:
                self.adjust_anchor(params, optimizers, step)
        self.bank.step = step + 1
        self.update_learning_rate(params, optimizers, step + 1)
        self.pack(params, step)
        if step % 1000 == 0:
            self.state["octree_gs"]["events"].append(
                {"step": step, "anchors": len(params["means"]), "per_level": self.per_level(params),
                 "rendered": int(ogs["selection"].sum()), "max_level": ogs["max_level"],
                 "extra_level_mean": float(params["ogs_extra_level"].mean())})

    @torch.no_grad()
    def training_statis(self, info, ogs):
        k = self.c["n_offsets"]
        vis = ogs["visible"]
        sel = ogs["selection"]
        means2d = info["means2d"]
        if means2d.grad is None:
            return
        temp = ogs["neural_opacity"].detach().view(-1).clone()
        temp[temp < 0] = 0
        temp = temp.view(-1, k)
        self.opacity_accum[vis] += temp.sum(dim=1, keepdim=True)
        self.anchor_demon[vis] += 1
        radii = info["radii"][0]
        update_filter = (radii > 0).all(-1) if radii.dim() == 2 else radii > 0
        vis_rep = vis.unsqueeze(1).repeat(1, k).view(-1)
        combined = torch.zeros(len(self.offset_gradient_accum), dtype=torch.bool, device=vis.device)
        combined[vis_rep] = sel
        temp_mask = combined.clone()
        combined[temp_mask] = update_filter
        # gsplat means2d gradients are in pixels; Inria screenspace gradients are NDC-scaled (x W/2, x H/2).
        g = means2d.grad[0].clone()
        g[:, 0] *= info["width"] / 2.0
        g[:, 1] *= info["height"] / 2.0
        self.offset_gradient_accum[combined] += g[update_filter].norm(dim=-1, keepdim=True)
        self.offset_denom[combined] += 1

    def _weed(self, cand, level):
        b = self.bank
        lev = torch.full((len(cand),), float(level), device=cand.device)
        keep, _ = octree.weed_out(cand, lev, b.cam_centers, b.standard_dist, b.levels, b.visible_threshold,
                                  b.fork)
        return keep

    @torch.no_grad()
    def anchor_growing(self, params, optimizers, step, grads, offset_mask):
        """anchor_growing: per level, same-level growth for th <= g < th*2^beta, next-level growth for
        g >= th*2^beta (finer voxel, zero feature), Delta L += extra_up where the anchor gradient exceeds
        0.25 th (all anchors, every level iteration), weed_out of candidates."""
        c, b = self.c, self.bank
        k, fd = c["n_offsets"], c["feat_dim"]
        threshold, fork = c["densify_grad_threshold"], b.fork
        init_length = len(params["means"])
        grads = grads.clone()
        grads[~offset_mask] = 0.0
        anchor_grads = grads.view(-1, k).sum(-1) / (offset_mask.view(-1, k).sum(-1) + 1e-6)
        enabled = (not (c["progressive"] and c["disable_next_level_during_progressive"])
                   or step > (b.coarse_intervals[-1] if b.coarse_intervals else 0))
        stats = {"same": 0, "next": 0, "weeded": 0, "extra_bumps": 0}
        dev = grads.device
        for cur_level in range(b.levels):
            update_value = fork ** c["update_ratio"]
            lev = params["ogs_level"][:, 0]
            level_mask = lev == cur_level
            level_ds_mask = lev == cur_level + 1
            if int(level_mask.sum()) == 0:
                continue
            cur_size = octree.level_size(b.voxel_size, fork, cur_level)
            ds_size = cur_size / fork
            cur_threshold = threshold * (update_value**cur_level)
            ds_threshold = cur_threshold * update_value
            extra_threshold = cur_threshold * c["extra_ratio"]
            cand = (grads >= cur_threshold) & (grads < ds_threshold)
            cand_ds = grads >= ds_threshold
            cand_extra = anchor_grads >= extra_threshold
            length_inc = len(params["means"]) - init_length
            if length_inc > 0:
                zk = torch.zeros(length_inc * k, dtype=torch.bool, device=dev)
                cand, cand_ds = torch.cat([cand, zk]), torch.cat([cand_ds, zk])
                cand_extra = torch.cat([cand_extra, torch.zeros(length_inc, dtype=torch.bool, device=dev)])
            rep = level_mask.repeat_interleave(k)
            cand, cand_ds = cand & rep, cand_ds & rep
            if enabled:
                params["ogs_extra_level"].data[:, 0] += c["extra_up"] * cand_extra.float()
                stats["extra_bumps"] += int(cand_extra.sum())
            anchor = params["means"]
            all_xyz = anchor.unsqueeze(1) + params["ogs_offset"] * params["ogs_scaling"][:, :3].exp().unsqueeze(1)
            all_xyz = all_xyz.view(-1, 3)
            new_parts = []

            def grow(mask_c, grid_mask, size, level):
                grid = torch.round((anchor[grid_mask] - b.init_pos) / size).int()
                sel = torch.round((all_xyz[mask_c] - b.init_pos) / size).int()
                uniq, inverse = torch.unique(sel, return_inverse=True, dim=0)
                if len(uniq) == 0 or len(grid) == 0:
                    return None
                keep = ~membership(uniq, grid)
                cand_anchor = uniq[keep] * size + b.init_pos
                w = self._weed(cand_anchor, level)
                stats["weeded"] += int((~w).sum())
                keep_c = keep.clone()
                keep[keep_c] = w
                return cand_anchor[w], uniq, inverse, keep

            same = grow(cand, level_mask, cur_size, cur_level)
            if same is not None and len(same[0]):
                cand_anchor, uniq, inverse, keep = same
                feat_rep = params["ogs_feat"].unsqueeze(1).repeat(1, k, 1).view(-1, fd)[cand]
                new_feat = scatter_max(feat_rep, inverse, len(uniq))[keep]
                new_parts.append((cand_anchor, new_feat, cur_size, cur_level))
                stats["same"] += len(cand_anchor)
            if enabled and cur_level < b.levels - 1:
                nxt = grow(cand_ds, level_ds_mask, ds_size, cur_level + 1)
                if nxt is not None and len(nxt[0]):
                    new_parts.append((nxt[0], torch.zeros(len(nxt[0]), fd, device=dev), ds_size, cur_level + 1))
                    stats["next"] += len(nxt[0])
            if new_parts:
                xyz = torch.cat([p[0] for p in new_parts])
                feat = torch.cat([p[1] for p in new_parts])
                log_scaling = torch.cat([torch.log(torch.ones(len(p[0]), 6, device=dev) * p[2]) for p in new_parts])
                level = torch.cat([torch.full((len(p[0]),), float(p[3]), device=dev) for p in new_parts])
                append_anchors(params, optimizers, new_anchor_fields(xyz, log_scaling, feat, level, k))
                z = torch.zeros(len(xyz), 1, device=dev)
                self.anchor_demon = torch.cat([self.anchor_demon, z])
                self.opacity_accum = torch.cat([self.opacity_accum, z])
        return stats

    @torch.no_grad()
    def adjust_anchor(self, params, optimizers, step):
        c = self.c
        k = c["n_offsets"]
        check_interval = c["update_interval"]
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval * c["success_threshold"] * 0.5).squeeze(1)
        stats = self.anchor_growing(params, optimizers, step, grads_norm, offset_mask)
        n = len(params["means"])
        dev = grads.device
        self.offset_denom[offset_mask] = 0
        self.offset_denom = torch.cat([self.offset_denom, torch.zeros(n * k - len(self.offset_denom), 1, device=dev)])
        self.offset_gradient_accum[offset_mask] = 0
        self.offset_gradient_accum = torch.cat(
            [self.offset_gradient_accum, torch.zeros(n * k - len(self.offset_gradient_accum), 1, device=dev)])
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
        # _prune_anchor_optimizer clamps the raw scaling[:, 3:] at 0.05 on every prune call.
        params["ogs_scaling"].data[:, 3:] = params["ogs_scaling"].data[:, 3:].clamp(max=0.05)
        stats["pruned"] = int(prune.sum())
        if step % 1000 == 0:
            self.state["octree_gs"]["events"].append({"step": step, "adjust": stats, "anchors": len(params["means"])})
        return stats


def create_density(spec):
    return OctreeGSDensity(spec)

