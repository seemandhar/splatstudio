"""HAC density component: anchor initialisation, Scaffold-GS anchor growing / pruning, field
learning-rate schedules, the HAC bank's optimiser, and the final encode -> decode -> bake.

Official references (github.com/YihangChen-ee/HAC): scene/gaussian_model.py
(create_from_pcd, training_setup, training_statis, anchor_growing, adjust_anchor,
conduct_encoding, conduct_decoding), train.py (loop order, skip of densification in
[3000, 4000), update_from/interval/until), arguments/__init__.py (defaults).
"""

import copy
import os
import sys
import time

import torch
from gsplat.strategy.ops import _update_param_with_optimizer

from . import codec, model
from .renderer import BLOB, BLOB_WIDTH, blob_floats, pack_blob

BIT2MB = 8 * 1024 * 1024

DEFAULTS = {
    "total_steps": 30000,
    "feat_dim": 50,
    "n_offsets": 10,
    "voxel_size": 0.001,  # run_shell_mip360.py (tandt 0.01, db 0.005)
    "update_depth": 3,
    "update_init_factor": 16,
    "update_hierachy_factor": 4,
    "n_features": 4,
    "log2": 13,
    "log2_2d": 15,
    "resolutions": [18, 24, 33, 44, 59, 80, 108, 148, 201, 275, 376, 514],
    "resolutions_2d": [130, 258, 514, 1026],
    "q0": {"feat": 1.0, "scaling": 0.001, "offsets": 0.2},
    "noise_start": 3000,  # uniform noise with fixed Q for 3000 < step <= 10000
    "entropy_start": 10000,  # bound update at 10000, context + rate for step > 10000
    "rate_sample": 0.05,
    "start_stat": 500,
    "update_from": 1500,
    "update_interval": 100,
    "update_until": 15000,
    "densify_pause": [3000, 4000],  # `iteration not in range(3000, 4000)`
    "min_opacity": 0.005,
    "success_threshold": 0.8,
    "densify_grad_threshold": 0.0002,
    "encode_batch": 3000,  # MAX_batch_size of conduct_encoding
    "lr": {
        "offset": [0.01, 0.0001],  # x spatial_lr_scale
        "mask": [0.01, 0.0001],  # x spatial_lr_scale
        "feat": 0.0075,
        "scaling": 0.007,
        "opacity": 0.02,
        "rotation": 0.002,
        "mlp_opacity": [0.002, 0.00002],
        "mlp_cov": [0.004, 0.004],
        "mlp_color": [0.008, 0.00005],
        "encoding_xyz": [0.005, 0.00001],
        "mlp_grid": [0.005, 0.00001],
    },
}


def merge(defaults, config):
    out = copy.deepcopy(defaults)
    for k, v in config.items():
        if k not in out:
            raise ValueError(f"Unknown hac-density option {k}")
        out[k] = {**out[k], **v} if isinstance(out[k], dict) else v
    return out


def knn_dist2(points, k=3, chunk=4096):
    """simple_knn distCUDA2: mean squared distance to the 3 nearest neighbours."""
    out = torch.empty(len(points), device=points.device)
    for s in range(0, len(points), chunk):
        d2 = torch.cdist(points[s : s + chunk], points).square()
        d2[torch.arange(d2.shape[0]), torch.arange(s, s + d2.shape[0])] = float("inf")
        out[s : s + chunk] = d2.topk(min(k, len(points) - 1), dim=1, largest=False).values.mean(1)
    return out


def new_anchor_fields(xyz, log_scaling, feat, n_offsets, blob_width):
    """Per-anchor values of create_from_pcd / anchor_growing for every native field."""
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
        "hac_feat": feat,
        "hac_offset": torch.zeros(n, n_offsets, 3, device=dev),
        "hac_scaling": log_scaling,
        "hac_mask": torch.ones(n, n_offsets, 1, device=dev),
        BLOB: torch.zeros(n, blob_width, device=dev),
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
    """Swap every field for a new population with fresh Adam state (initialisation / bake)."""
    for name, v in values.items():
        params[name] = torch.nn.Parameter(v.contiguous().float(), requires_grad=True)
        optimizers[name] = torch.optim.Adam([params[name]], lr=lrs[name], eps=1e-15)


def _ss_train_num_gaussians():
    """--num-gaussians of an `ss-train` / `splatstudio train` command line (runtime default
    10000 when absent); None when not running under that CLI (probes, scripts)."""
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


class HACDensity:
    requires = {"gradient.screen.signed"}

    def __init__(self, spec):
        self.c = merge(DEFAULTS, spec.config)
        self.bank = None
        self.state = {"hac": {"events": []}}
        self.spatial_lr_scale = 1.0

    # ------------------------------------------------------------------ initialisation
    def lr_table(self):
        lr = self.c["lr"]
        s = self.spatial_lr_scale
        return {"means": 0.0, "scales": 0.0, "quats": lr["rotation"], "opacities": lr["opacity"],
                "sh0": 0.0, "shN": 0.0, "hac_feat": lr["feat"], "hac_scaling": lr["scaling"],
                "hac_offset": lr["offset"][0] * s, "hac_mask": lr["mask"][0] * s, BLOB: 1e-12}

    @torch.no_grad()
    def initialize(self, params, optimizers, scene_scale=1.0):
        """create_from_pcd + training_setup + update_anchor_bound (train.py order)."""
        self.spatial_lr_scale = float(scene_scale)
        c = self.c
        points = params["means"].detach()
        cap = _ss_train_num_gaussians()
        if cap is not None and len(points) >= cap:
            raise ValueError(
                f"HAC voxelizes the full SfM cloud but the runtime kept {len(points)} points "
                f"(--num-gaussians {cap}): pass --num-gaussians larger than the SfM cloud, "
                "e.g. 10000000")
        dev = points.device
        perm = torch.randperm(len(points), device=dev)  # voxelize_sample shuffles first
        anchors = torch.unique(torch.round(points[perm] / c["voxel_size"]), dim=0)
        anchors = anchors * c["voxel_size"]
        dist2 = knn_dist2(anchors).clamp_min(1e-7)
        log_scaling = torch.log(torch.sqrt(dist2))[:, None].repeat(1, 6)
        feat = torch.zeros(len(anchors), c["feat_dim"], device=dev)
        values = new_anchor_fields(anchors, log_scaling, feat, c["n_offsets"], BLOB_WIDTH)
        replace_population(params, optimizers, values, self.lr_table())
        self.offset_sched = model.expon_lr(c["lr"]["offset"][0] * self.spatial_lr_scale,
                                           c["lr"]["offset"][1] * self.spatial_lr_scale,
                                           c["total_steps"])
        self.mask_sched = model.expon_lr(c["lr"]["mask"][0] * self.spatial_lr_scale,
                                         c["lr"]["mask"][1] * self.spatial_lr_scale,
                                         c["total_steps"])
        self.bank = model.HACBank(c, dev)
        self.bank.update_anchor_bound(params["means"])
        model.ACTIVE["bank"] = self.bank
        self.update_learning_rate(params, optimizers, 1)
        self.reset_stats(len(anchors), dev)
        self.state["hac"]["initial_anchors"] = len(anchors)
        self.state["hac"]["spatial_lr_scale"] = self.spatial_lr_scale

    def set_training_cameras(self, cameras):
        """Guard: the offset/mask LRs scale with 3DGS's cameras_extent, i.e. --recipe 3dgs."""
        centers = torch.stack([torch.linalg.inv(c["viewmat"])[:3, 3] for c in cameras])
        extent = 1.1 * float(torch.linalg.vector_norm(centers - centers.mean(0), dim=1).max())
        if abs(max(extent, 0.01) - self.spatial_lr_scale) > 1e-4 * max(extent, 1e-6):
            raise ValueError(
                f"HAC needs the 3DGS camera-extent scene scale ({extent:.4f}); got "
                f"{self.spatial_lr_scale:.4f}: run ss-train with --recipe 3dgs")
        self.state["hac"]["camera_extent"] = extent

    def reset_stats(self, n, dev):
        k = self.c["n_offsets"]
        self.opacity_accum = torch.zeros(n, 1, device=dev)
        self.anchor_demon = torch.zeros(n, 1, device=dev)
        self.offset_gradient_accum = torch.zeros(n * k, 1, device=dev)
        self.offset_denom = torch.zeros(n * k, 1, device=dev)

    def update_learning_rate(self, params, optimizers, step):
        """update_learning_rate(iteration) at the top of each official iteration."""
        optimizers["means"].param_groups[0]["lr"] = 0.0  # position_lr_init = 0
        optimizers["hac_offset"].param_groups[0]["lr"] = self.offset_sched(step)
        optimizers["hac_mask"].param_groups[0]["lr"] = self.mask_sched(step)
        self.bank.update_learning_rate(step)

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
        hac = outputs["info"]["hac"]
        if c["start_stat"] < step < c["update_until"]:
            self.training_statis(outputs["info"], hac)
            lo, hi = c["densify_pause"]
            if not (lo <= step < hi) and step > c["update_from"] and step % c["update_interval"] == 0:
                self.adjust_anchor(params, optimizers, step)
        if step == c["total_steps"]:
            self.bake(params, optimizers)
            return
        self.bank.step = step + 1
        self.update_learning_rate(params, optimizers, step + 1)
        bits = hac.get("bits")
        if bits is not None and step % 1000 == 0:
            self.state["hac"]["events"].append(
                {"step": step, "anchors": len(params["means"]),
                 "bit_per_param": float(bits["bit_per_param"].item())})

    @torch.no_grad()
    def training_statis(self, info, hac):
        k = self.c["n_offsets"]
        vis = hac["visible"]
        sel = hac["selection"]
        means2d = info["means2d"]
        if means2d.grad is None:
            return
        temp = hac["neural_opacity"].detach().view(-1).clone()
        temp[temp < 0] = 0
        temp = temp.view(-1, k)
        self.opacity_accum[vis] += temp.sum(dim=1, keepdim=True)
        self.anchor_demon[vis] += 1
        radii = info["radii"][0]
        update_filter = (radii > 0).all(-1) if radii.dim() == 2 else radii > 0
        vis_rep = vis.unsqueeze(1).repeat(1, k).view(-1)
        combined = torch.zeros(len(self.offset_gradient_accum), dtype=torch.bool,
                               device=vis.device)
        combined[vis_rep] = sel
        temp_mask = combined.clone()
        combined[temp_mask] = update_filter
        # gsplat means2d gradients are in pixels; 3DGS screenspace gradients are NDC-scaled.
        g = means2d.grad[0].clone()
        g[:, 0] *= info["width"] / 2.0
        g[:, 1] *= info["height"] / 2.0
        self.offset_gradient_accum[combined] += g[update_filter].norm(dim=-1, keepdim=True)
        self.offset_denom[combined] += 1

    @torch.no_grad()
    def anchor_growing(self, params, optimizers, grads, threshold, offset_mask):
        c = self.c
        k = c["n_offsets"]
        init_length = len(params["means"]) * k
        added = 0
        for i in range(c["update_depth"]):
            cur_threshold = threshold * ((c["update_hierachy_factor"] // 2) ** i)
            cand = (grads >= cur_threshold) & offset_mask
            cand = cand & (torch.rand_like(cand.float()) > (0.5 ** (i + 1)))
            length_inc = len(params["means"]) * k - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                cand = torch.cat([cand, torch.zeros(length_inc, dtype=torch.bool,
                                                    device=cand.device)])
            anchor = self.bank.quantize_anchor(params["means"])[0]
            scaling = params["hac_scaling"].exp()
            all_xyz = anchor.unsqueeze(1) + params["hac_offset"] * scaling[:, :3].unsqueeze(1)
            cur_size = c["voxel_size"] * (c["update_init_factor"] // (c["update_hierachy_factor"] ** i))
            grid_coords = torch.round(anchor / cur_size).int()
            selected = all_xyz.view(-1, 3)[cand]
            sel_coords = torch.round(selected / cur_size).int()
            uniq, inverse = torch.unique(sel_coords, return_inverse=True, dim=0)
            if len(uniq) == 0:
                continue
            # Exact set-membership of integer grid cells (official: chunked all()/any()).
            both = torch.cat([grid_coords, uniq])
            _, ids = torch.unique(both, return_inverse=True, dim=0)
            existing = torch.zeros(int(ids.max()) + 1, dtype=torch.bool, device=ids.device)
            existing[ids[: len(grid_coords)]] = True
            keep = ~existing[ids[len(grid_coords):]]
            cand_anchor = uniq[keep].float() * cur_size
            if len(cand_anchor) == 0:
                continue
            new_scaling = torch.log(torch.ones_like(cand_anchor).repeat(1, 2) * cur_size)
            feat_rep = params["hac_feat"].unsqueeze(1).repeat(1, k, 1).view(-1, c["feat_dim"])
            feat_rep = feat_rep[cand]
            new_feat = torch.full((len(uniq), c["feat_dim"]), float("-inf"),
                                  device=feat_rep.device)
            new_feat = new_feat.scatter_reduce(
                0, inverse.unsqueeze(1).expand(-1, c["feat_dim"]), feat_rep, "amax",
                include_self=False)[keep]  # torch_scatter.scatter_max
            values = new_anchor_fields(cand_anchor, new_scaling, new_feat, k, BLOB_WIDTH)
            append_anchors(params, optimizers, values)
            n_new = len(cand_anchor)
            added += n_new
            dev = cand_anchor.device
            self.anchor_demon = torch.cat([self.anchor_demon, torch.zeros(n_new, 1, device=dev)])
            self.opacity_accum = torch.cat([self.opacity_accum,
                                            torch.zeros(n_new, 1, device=dev)])
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
        added = self.anchor_growing(params, optimizers, grads_norm,
                                    c["densify_grad_threshold"], offset_mask)
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
        # _prune_anchor_optimizer clamps the raw scaling[:, 3:] at 0.05 on every prune call.
        params["hac_scaling"].data[:, 3:] = params["hac_scaling"].data[:, 3:].clamp(max=0.05)
        if step % 1000 == 0 or added or int(prune.sum()):
            self.state["hac"]["events"].append(
                {"step": step, "grown": int(added), "pruned": int(prune.sum()),
                 "anchors": len(params["means"])})

    # --------------------------------------------------------- encode -> decode -> bake
    @torch.no_grad()
    def bake(self, params, optimizers):
        """conduct_encoding + conduct_decoding, then replace the fields by the decoded model."""
        c = self.c
        bank = self.bank
        k = c["n_offsets"]
        t0 = time.perf_counter()
        mask_sig = torch.sigmoid(params["hac_mask"])
        mask_all = (mask_sig > 0.01).float()
        mask_anchor = mask_all.sum(dim=1)[:, 0] > 0
        anchor, qv = bank.quantize_anchor(params["means"])
        anchor, qv = anchor[mask_anchor], qv[mask_anchor]
        feat_all = params["hac_feat"][mask_anchor]
        offs_all = params["hac_offset"][mask_anchor]
        scal_all = params["hac_scaling"].exp()[mask_anchor]
        masks = mask_all[mask_anchor]
        n = len(anchor)
        bits = {"anchor": n * 3 * model.ANCHOR_ROUND_DIGITS, "feat": 0, "scaling": 0,
                "offsets": 0}
        chunks, enc_values = [], []
        for s in range(0, n, c["encode_batch"]):
            e = min(s + c["encode_batch"], n)
            ctx = bank.context(anchor[s:e])
            mf, sf, qf = ctx["feat"]
            ms, ss, qs = ctx["scaling"]
            mo, so, qo = ctx["offsets"]
            qf = qf.repeat(1, mf.shape[-1])
            qs = qs.repeat(1, ms.shape[-1])
            qo = qo.repeat(1, mo.shape[-1])
            feat = model.STEMultistep.apply(feat_all[s:e], qf, feat_all.mean())
            pf, bf = codec.encode_gaussian(feat, mf, sf.clamp_min(1e-9), qf)
            scal = model.STEMultistep.apply(scal_all[s:e], qs, scal_all.mean())
            ps, bs = codec.encode_gaussian(scal, ms, ss.clamp_min(1e-9), qs)
            m = masks[s:e].repeat(1, 1, 3).view(-1, 3 * k).bool()
            offs = model.STEMultistep.apply(offs_all[s:e].reshape(-1, 3 * k), qo,
                                            offs_all.mean())
            offs = torch.where(m, offs, torch.zeros_like(offs))
            po, bo = codec.encode_gaussian(offs[m], mo[m], so[m].clamp_min(1e-9), qo[m])
            bits["feat"] += bf
            bits["scaling"] += bs
            bits["offsets"] += bo
            chunks.append((pf, ps, po))
            enc_values.append((feat, scal, offs))
        hash_payload, bits["hash"] = codec.encode_binary((bank.hash_embeddings() + 1) / 2)
        mask_payload, bits["masks"] = codec.encode_binary(masks)
        bits["mlps"] = bank.mlp_bits()
        t_enc = time.perf_counter() - t0

        # ---- decode (conduct_decoding) from the payloads only
        t1 = time.perf_counter()
        dev = anchor.device
        masks_dec = codec.decode_binary(mask_payload, n * k, dev).float().view(n, k, 1)
        n_hash = sum(g.params.numel() for g in bank.encoding.grids())
        hash_dec = codec.decode_binary(hash_payload, n_hash, dev).float() * 2 - 1
        i = 0
        for g in bank.encoding.grids():
            m = g.params.numel()
            g.params.data.copy_(hash_dec[i : i + m].view_as(g.params))
            i += m
        interval = (bank.x_bound_max - bank.x_bound_min) * model.Q_ANCHOR + 1e-6
        anchor_dec = qv * interval + bank.x_bound_min
        feats, scals, offsets = [], [], []
        for idx, s in enumerate(range(0, n, c["encode_batch"])):
            e = min(s + c["encode_batch"], n)
            pf, ps, po = chunks[idx]
            ctx = bank.context(anchor_dec[s:e])
            mf, sf, qf = ctx["feat"]
            ms, ss, qs = ctx["scaling"]
            mo, so, qo = ctx["offsets"]
            qf = qf.repeat(1, mf.shape[-1])
            qs = qs.repeat(1, ms.shape[-1])
            qo = qo.repeat(1, mo.shape[-1])
            feats.append(codec.decode_gaussian(pf, mf, sf.clamp_min(1e-9), qf).view(e - s, -1))
            scals.append(codec.decode_gaussian(ps, ms, ss.clamp_min(1e-9), qs).view(e - s, 6))
            m = masks_dec[s:e].repeat(1, 1, 3).view(-1, 3 * k).bool()
            o = torch.zeros_like(mo)
            o[m] = codec.decode_gaussian(po, mo[m], so[m].clamp_min(1e-9), qo[m])
            offsets.append(o.view(e - s, k, 3))
        t_dec = time.perf_counter() - t1
        feat_dec = torch.cat(feats) if feats else feat_all
        scal_dec = torch.cat(scals) if scals else scal_all
        offs_dec = torch.cat(offsets) if offsets else offs_all
        # Lossless-coder check: decoded symbols equal the quantized values that were encoded.
        enc_f = torch.cat([v[0] for v in enc_values])
        enc_s = torch.cat([v[1] for v in enc_values])
        enc_o = torch.cat([v[2] for v in enc_values]).view(n, k, 3)
        mismatch = {
            "feat": float((enc_f - feat_dec).abs().max()),
            "scaling": float((enc_s - scal_dec).abs().max()),
            "offsets": float((enc_o - offs_dec).abs().max()),
            "masks": float((masks - masks_dec).abs().max()),
        }
        if max(mismatch.values()) > 1e-3:
            raise RuntimeError(f"HAC codec round trip failed: {mismatch}")

        need = blob_floats(c["feat_dim"], k)
        if n * BLOB_WIDTH < need:
            raise RuntimeError(f"{n} anchors cannot hold the {need}-float decoder blob")
        log_scaling = torch.log(scal_dec.clamp_min(1e-12))
        values = new_anchor_fields(anchor_dec, log_scaling, feat_dec, k, BLOB_WIDTH)
        values["hac_offset"] = offs_dec
        values["hac_mask"] = masks_dec
        values["hac_scaling"] = scal_dec  # decoded (linear) values: decoded_version semantics
        values[BLOB] = pack_blob(bank.mlps, n, c["feat_dim"], k, dev)
        replace_population(params, optimizers, values, self.lr_table())
        bank.baked = True
        mb = {key: v / BIT2MB for key, v in bits.items()}
        mb["total"] = sum(bits.values()) / BIT2MB
        self.state["hac"]["encoded"] = {
            "anchors_total": int(len(mask_anchor)), "anchors_coded": n,
            "bits": {key: int(v) for key, v in bits.items()}, "size_mb": mb,
            "encode_seconds": t_enc, "decode_seconds": t_dec, "roundtrip_max_error": mismatch,
        }
        print({"hac_encoded_size_mb": mb}, flush=True)


def create_density(spec):
    return HACDensity(spec)
