"""ContextGS density component: anchor initialisation, Scaffold-GS anchor growing / pruning, field learning-rate
schedules, the bank's optimiser, and the final encode -> decode -> bake.

Official references (github.com/wyf0912/ContextGS @ eff8d19): scene/gaussian_model.py (create_from_pcd,
training_setup, training_statis, anchor_growing, adjust_anchor, conduct_encoding, conduct_decoding), train.py (loop
order, densification pause in [3000, 4000), update_from/interval/until), arguments/__init__.py (defaults).
The initialisation / statistics / growing / pruning / blob code is copied from the HAC native port
(reproductions/hac/plugin/ss_hac/density.py; HAC and ContextGS share these Scaffold-GS routines verbatim) and
extended with the hyperprior latent; the bake is new (level-ordered context coding, hyperprior stream).
"""

import copy
import io
import os
import sys
import time

import torch
from gsplat.strategy.ops import _update_param_with_optimizer

from . import codec, levels, model
from .renderer import BLOB, BLOB_WIDTH, blob_floats, pack_blob

BIT2MB = 8 * 1024 * 1024

DEFAULTS = {
    "total_steps": 30000,
    "feat_dim": 50,
    "n_offsets": 10,
    "voxel_size": 0.001,  # train_scripts/train_mlp360.py (tandt 0.01, db 0.005)
    "update_depth": 3,
    "update_init_factor": 16,
    "update_hierachy_factor": 4,
    "level_num": 3,  # K (train.py --level_num, §5.1)
    "target_ratio": 0.2,  # tau (scripts --target_ratio 0.2, §5.1)
    "hyper_divisor": 4,  # h_c: z in R^{50/4}
    "use_hyper": True,  # official --disable_hyper (ablation) when False
    "q0": {"feat": 1.0, "scaling": 0.001, "offsets": 0.2},
    "noise_start": 3000,  # uniform noise with fixed Q for 3000 < step <= 10000
    "entropy_start": 10000,  # bound update at 10000, context + rate for step > 10000
    "rate_sample": 0.15,  # chosse_random_thresh in multi_scale_generating
    "start_stat": 500,
    "update_from": 1500,
    "update_interval": 100,
    "update_until": 15000,
    "densify_pause": [3000, 4000],  # `iteration not in range(3000, 4000)`
    "min_opacity": 0.005,
    "success_threshold": 0.8,
    "densify_grad_threshold": 0.0002,
    "encode_batch": 1000,  # MAX_batch_size of conduct_encoding (per-batch symbol ranges)
    "lr": {
        "offset": [0.01, 0.0001],  # x spatial_lr_scale
        "mask": [0.01, 0.0001],  # x spatial_lr_scale
        "feat": 0.0075,
        "hyper": 0.0075,  # hyper_latent_lr
        "scaling": 0.007,
        "opacity": 0.02,
        "rotation": 0.002,
        "mlp_opacity": [0.002, 0.00002],
        "mlp_cov": [0.004, 0.004],
        "mlp_color": [0.008, 0.00005],
        "latent_codec": [0.005, 0.00001],
        "mlp_grid": [0.005, 0.00001],
    },
}


def merge(defaults, config):
    out = copy.deepcopy(defaults)
    for k, v in config.items():
        if k not in out:
            raise ValueError(f"Unknown cgs-density option {k}")
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


def new_anchor_fields(xyz, log_scaling, feat, n_offsets, blob_width, hyper):
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
        "cgs_feat": feat,
        "cgs_offset": torch.zeros(n, n_offsets, 3, device=dev),
        "cgs_scaling": log_scaling,
        "cgs_mask": torch.ones(n, n_offsets, 1, device=dev),
        "cgs_hyper": hyper,
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


class ContextGSDensity:
    requires = {"gradient.screen.signed"}

    def __init__(self, spec):
        self.c = merge(DEFAULTS, spec.config)
        self.bank = None
        self.state = {"contextgs": {"events": []}}
        self.spatial_lr_scale = 1.0

    # ------------------------------------------------------------------ initialisation
    def lr_table(self):
        lr = self.c["lr"]
        s = self.spatial_lr_scale
        return {"means": 0.0, "scales": 0.0, "quats": lr["rotation"], "opacities": lr["opacity"],
                "sh0": 0.0, "shN": 0.0, "cgs_feat": lr["feat"], "cgs_scaling": lr["scaling"], "cgs_hyper": lr["hyper"],
                "cgs_offset": lr["offset"][0] * s, "cgs_mask": lr["mask"][0] * s, BLOB: 1e-12}

    @torch.no_grad()
    def initialize(self, params, optimizers, scene_scale=1.0):
        """create_from_pcd + training_setup + update_anchor_bound (train.py order)."""
        self.spatial_lr_scale = float(scene_scale)
        c = self.c
        points = params["means"].detach()
        cap = _ss_train_num_gaussians()
        if cap is not None and len(points) >= cap:
            raise ValueError(
                f"ContextGS voxelizes the full SfM cloud but the runtime kept {len(points)} points "
                f"(--num-gaussians {cap}): pass --num-gaussians larger than the SfM cloud, "
                "e.g. 10000000")
        dev = points.device
        perm = torch.randperm(len(points), device=dev)  # voxelize_sample shuffles first
        anchors = torch.unique(torch.round(points[perm] / c["voxel_size"]), dim=0)
        anchors = anchors * c["voxel_size"]
        dist2 = knn_dist2(anchors).clamp_min(1e-7)
        log_scaling = torch.log(torch.sqrt(dist2))[:, None].repeat(1, 6)
        feat = torch.zeros(len(anchors), c["feat_dim"], device=dev)
        hyper = torch.zeros(len(anchors), c["feat_dim"] // c["hyper_divisor"], device=dev)
        values = new_anchor_fields(anchors, log_scaling, feat, c["n_offsets"], BLOB_WIDTH, hyper)
        replace_population(params, optimizers, values, self.lr_table())
        self.offset_sched = model.expon_lr(c["lr"]["offset"][0] * self.spatial_lr_scale,
                                           c["lr"]["offset"][1] * self.spatial_lr_scale,
                                           c["total_steps"])
        self.mask_sched = model.expon_lr(c["lr"]["mask"][0] * self.spatial_lr_scale,
                                         c["lr"]["mask"][1] * self.spatial_lr_scale,
                                         c["total_steps"])
        self.bank = model.ContextBank(c, dev)
        self.bank.update_anchor_bound(params["means"])
        model.ACTIVE["bank"] = self.bank
        self.update_learning_rate(params, optimizers, 1)
        self.reset_stats(len(anchors), dev)
        self.state["contextgs"]["initial_anchors"] = len(anchors)
        self.state["contextgs"]["spatial_lr_scale"] = self.spatial_lr_scale

    def set_training_cameras(self, cameras):
        """Guard: the offset/mask LRs scale with 3DGS's cameras_extent, i.e. --recipe 3dgs."""
        centers = torch.stack([torch.linalg.inv(c["viewmat"])[:3, 3] for c in cameras])
        extent = 1.1 * float(torch.linalg.vector_norm(centers - centers.mean(0), dim=1).max())
        if abs(max(extent, 0.01) - self.spatial_lr_scale) > 1e-4 * max(extent, 1e-6):
            raise ValueError(
                f"ContextGS needs the 3DGS camera-extent scene scale ({extent:.4f}); got "
                f"{self.spatial_lr_scale:.4f}: run ss-train with --recipe 3dgs")
        self.state["contextgs"]["camera_extent"] = extent

    def reset_stats(self, n, dev):
        k = self.c["n_offsets"]
        self.opacity_accum = torch.zeros(n, 1, device=dev)
        self.anchor_demon = torch.zeros(n, 1, device=dev)
        self.offset_gradient_accum = torch.zeros(n * k, 1, device=dev)
        self.offset_denom = torch.zeros(n * k, 1, device=dev)

    def update_learning_rate(self, params, optimizers, step):
        """update_learning_rate(iteration) at the top of each official iteration."""
        optimizers["means"].param_groups[0]["lr"] = 0.0  # position_lr_init = 0
        optimizers["cgs_offset"].param_groups[0]["lr"] = self.offset_sched(step)
        optimizers["cgs_mask"].param_groups[0]["lr"] = self.mask_sched(step)
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
        cgs = outputs["info"]["cgs"]
        if c["start_stat"] < step < c["update_until"]:
            self.training_statis(outputs["info"], cgs)
            lo, hi = c["densify_pause"]
            if not (lo <= step < hi) and step > c["update_from"] and step % c["update_interval"] == 0:
                self.adjust_anchor(params, optimizers, step)
        if step == c["total_steps"]:
            self.bake(params, optimizers)
            return
        self.bank.step = step + 1
        self.update_learning_rate(params, optimizers, step + 1)
        bits = cgs.get("bits")
        if bits is not None and step % 1000 == 0:
            self.state["contextgs"]["events"].append(
                {"step": step, "anchors": len(params["means"]),
                 "bit_per_param": float(bits["bit_per_param"].item())})

    @torch.no_grad()
    def training_statis(self, info, cgs):
        k = self.c["n_offsets"]
        vis = cgs["visible"]
        sel = cgs["selection"]
        means2d = info["means2d"]
        if means2d.grad is None:
            return
        temp = cgs["neural_opacity"].detach().view(-1).clone()
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
            scaling = params["cgs_scaling"].exp()
            all_xyz = anchor.unsqueeze(1) + params["cgs_offset"] * scaling[:, :3].unsqueeze(1)
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
            feat_rep = params["cgs_feat"].unsqueeze(1).repeat(1, k, 1).view(-1, c["feat_dim"])
            feat_rep = feat_rep[cand]
            new_feat = torch.full((len(uniq), c["feat_dim"]), float("-inf"),
                                  device=feat_rep.device)
            new_feat = new_feat.scatter_reduce(
                0, inverse.unsqueeze(1).expand(-1, c["feat_dim"]), feat_rep, "amax",
                include_self=False)[keep]  # torch_scatter.scatter_max
            hd = params["cgs_hyper"].shape[1]
            hyp_rep = params["cgs_hyper"].unsqueeze(1).repeat(1, k, 1).view(-1, hd)[cand]
            new_hyper = torch.full((len(uniq), hd), float("-inf"), device=hyp_rep.device)
            new_hyper = new_hyper.scatter_reduce(
                0, inverse.unsqueeze(1).expand(-1, hd), hyp_rep, "amax",
                include_self=False)[keep]  # scatter_max of _hyper_latent (anchor_growing)
            values = new_anchor_fields(cand_anchor, new_scaling, new_feat, k, BLOB_WIDTH,
                                       new_hyper)
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
        params["cgs_scaling"].data[:, 3:] = params["cgs_scaling"].data[:, 3:].clamp(max=0.05)
        if step % 1000 == 0 or added or int(prune.sum()):
            self.state["contextgs"]["events"].append(
                {"step": step, "grown": int(added), "pruned": int(prune.sum()),
                 "anchors": len(params["means"])})

    # --------------------------------------------------------- encode -> decode -> bake
    @torch.no_grad()
    def code_levels(self, anchor, hyper_feat, masks, values=None, payloads=None):
        """conduct_encoding (values=(feat, scaling, offsets)) or conduct_decoding (payloads) over the levels
        K-1 .. 0; each level's Gaussians are predicted from the hyperprior and the already (de)coded coarser
        level. Returns the quantized attributes, the payloads and per-level bits / counts."""
        bank, c = self.bank, self.c
        k, f = c["n_offsets"], c["feat_dim"]
        n = len(anchor)
        dev = anchor.device
        lv = levels.divide_levels(anchor, c["voxel_size"], bank.level_scale, bank.level_num, None)
        _, inverse_list, mapping_list, _ = lv
        feat_q = torch.zeros(n, f, device=dev)
        scal_q = torch.zeros(n, 6, device=dev)
        offs_q = torch.zeros(n, k, 3, device=dev)
        already = torch.zeros(n, dtype=torch.bool, device=dev)
        out, bits, counts = {}, {"feat": 0, "scaling": 0, "offsets": 0}, {}
        content = None
        for i in reversed(range(bank.level_num)):
            to_code, orig, feat_in = levels.level_inputs(lv, i, content, hyper_feat, n)
            counts[i] = int(to_code.sum())
            if counts[i] > 0:
                pred = bank.predict(i, feat_in)
                mf, sf, qf = pred["feat"]
                ms, ss, qs = pred["scaling"]
                mo, so, qo = pred["offsets"]
                qf, qs, qo = qf.repeat(1, f), qs.repeat(1, 6), qo.repeat(1, 3 * k)
                sf, ss, so = sf.clamp_min(1e-9), ss.clamp_min(1e-9), so.clamp_min(1e-9)
                m = masks[orig].repeat(1, 1, 3).view(-1, 3 * k).bool()
                nb = int(c["encode_batch"])
                fqs, sqs, oqs, chunks = [], [], [], []
                for bi, b0 in enumerate(range(0, len(orig), nb)):
                    sl = slice(b0, b0 + nb)
                    mb = m[sl]
                    if values is not None:
                        feat, scal, offs = values
                        ob = orig[sl]
                        fq = model.STEMultistep.apply(feat[ob], qf[sl])
                        sq = model.STEMultistep.apply(scal[ob], qs[sl])
                        oq = model.STEMultistep.apply(offs[ob].reshape(-1, 3 * k), qo[sl])
                        oq = torch.where(mb, oq, torch.zeros_like(oq))
                        pf, bf = codec.encode_gaussian(fq, mf[sl], sf[sl], qf[sl])
                        ps, bs = codec.encode_gaussian(sq, ms[sl], ss[sl], qs[sl])
                        po, bo = codec.encode_gaussian(oq[mb], mo[sl][mb], so[sl][mb], qo[sl][mb])
                        chunks.append((pf, ps, po))
                        bits["feat"] += bf
                        bits["scaling"] += bs
                        bits["offsets"] += bo
                    else:
                        pf, ps, po = payloads[i][bi]
                        fq = codec.decode_gaussian(pf, mf[sl], sf[sl], qf[sl]).view(-1, f)
                        sq = codec.decode_gaussian(ps, ms[sl], ss[sl], qs[sl]).view(-1, 6)
                        oq = torch.zeros_like(mo[sl])
                        oq[mb] = codec.decode_gaussian(po, mo[sl][mb], so[sl][mb], qo[sl][mb])
                    fqs.append(fq)
                    sqs.append(sq)
                    oqs.append(oq)
                out[i] = chunks
                fq, sq, oq = torch.cat(fqs), torch.cat(sqs), torch.cat(oqs)
                feat_q[orig] = fq
                scal_q[orig] = sq
                offs_q[orig] = oq.view(-1, k, 3)
                already[orig] = True
            if i != 0:
                content = levels.extract_context_feat(anchor, feat_q, scal_q, already,
                                                      inverse_list, mapping_list, i)
        return feat_q, scal_q, offs_q, out, bits, counts

    @torch.no_grad()
    def bake(self, params, optimizers):
        """conduct_encoding + conduct_decoding, then replace the fields by the decoded model."""
        c = self.c
        bank = self.bank
        k = c["n_offsets"]
        t0 = time.perf_counter()
        mask_all = (torch.sigmoid(params["cgs_mask"]) > 0.01).float()
        mask_anchor = mask_all.sum(dim=1)[:, 0] > 0
        anchor, qv = bank.quantize_anchor(params["means"])
        anchor, qv = anchor[mask_anchor], qv[mask_anchor]
        feat_all = params["cgs_feat"][mask_anchor]
        offs_all = params["cgs_offset"][mask_anchor]
        scal_all = params["cgs_scaling"].exp()[mask_anchor]
        hyper_all = params["cgs_hyper"][mask_anchor]
        masks = mask_all[mask_anchor]
        n = len(anchor)
        dev = anchor.device
        if bank.level_scale is None:
            bank.level_scale = levels.find_divide_scale(anchor, c["voxel_size"], bank.x_bound_min,
                                                        bank.x_bound_max, c["target_ratio"],
                                                        bank.level_num)
        # hyperprior (quantize mode="symbols" around the medians, factorized-prior coding)
        sym = bank.prior.symbols(hyper_all)
        hyper_payload, bits_hyper = codec.encode_factorized(sym, bank.prior)
        hyper_feat = sym.float()
        if not c["use_hyper"]:
            hyper_feat = hyper_feat * 0
        feat_e, scal_e, offs_e, streams, bits, counts = self.code_levels(
            anchor, hyper_feat, masks, values=(feat_all, scal_all, offs_all))
        mask_payload, bits["masks"] = codec.encode_binary(masks)
        bits["hyper"] = bits_hyper
        bits["anchor"] = n * 3 * model.ANCHOR_ROUND_DIGITS
        bits["mlps"] = bank.mlp_bits()
        # meta.b analogue: anchor count, per-level counts, mask probability and the byte length of every
        # stream (the symbol ranges are already counted inside the coder's per-call +32-bit headers)
        meta = [len(mask_anchor), counts, float(mask_payload[0]),
                {i: [[[len(pl[3]) for pl in st] for st in ch] for ch in s] for i, s in streams.items()},
                [len(p[3]) if p else 0 for p in hyper_payload]]
        buf = io.BytesIO()
        torch.save(meta, buf)
        bits["meta"] = len(buf.getvalue()) * 8
        t_enc = time.perf_counter() - t0

        # ---- decode (conduct_decoding) from the payloads only
        t1 = time.perf_counter()
        hyper_dec = bank.prior.dequantize(codec.decode_factorized(hyper_payload, n, bank.prior, dev))
        interval = (bank.x_bound_max - bank.x_bound_min) * model.Q_ANCHOR + 1e-6
        anchor_dec = qv * interval + bank.x_bound_min
        masks_dec = codec.decode_binary(mask_payload, n * k, dev).float().view(n, k, 1)
        hyper_in = hyper_dec if c["use_hyper"] else hyper_dec * 0
        feat_dec, scal_dec, offs_dec, _, _, _ = self.code_levels(anchor_dec, hyper_in, masks_dec,
                                                                 payloads=streams)
        t_dec = time.perf_counter() - t1
        mismatch = {
            "hyper": float((hyper_dec - sym.float()).abs().max()) if n else 0.0,
            "feat": float((feat_e - feat_dec).abs().max()) if n else 0.0,
            "scaling": float((scal_e - scal_dec).abs().max()) if n else 0.0,
            "offsets": float((offs_e - offs_dec).abs().max()) if n else 0.0,
            "masks": float((masks - masks_dec).abs().max()) if n else 0.0,
        }
        if max(mismatch.values()) > 1e-3:
            raise RuntimeError(f"ContextGS codec round trip failed: {mismatch}")

        need = blob_floats(c["feat_dim"], k)
        if n * BLOB_WIDTH < need:
            raise RuntimeError(f"{n} anchors cannot hold the {need}-float decoder blob")
        log_scaling = torch.log(scal_dec.clamp_min(1e-12))
        values = new_anchor_fields(anchor_dec, log_scaling, feat_dec, k, BLOB_WIDTH, hyper_dec)
        values["cgs_offset"] = offs_dec
        values["cgs_mask"] = masks_dec
        values["cgs_scaling"] = scal_dec  # decoded (linear) values: decoded_version semantics
        values[BLOB] = pack_blob(bank.mlps, n, c["feat_dim"], k, dev)
        replace_population(params, optimizers, values, self.lr_table())
        bank.baked = True
        mb = {key: v / BIT2MB for key, v in bits.items()}
        mb["total"] = sum(bits.values()) / BIT2MB
        self.state["contextgs"]["encoded"] = {
            "anchors_total": int(len(mask_anchor)), "anchors_coded": n,
            "level_counts": {str(i): v for i, v in counts.items()},
            "level_scale": [float(v) for v in bank.level_scale],
            "bits": {key: int(v) for key, v in bits.items()}, "size_mb": mb,
            "encode_seconds": t_enc, "decode_seconds": t_dec, "roundtrip_max_error": mismatch,
        }
        print({"contextgs_encoded_size_mb": mb}, flush=True)


def create_density(spec):
    return ContextGSDensity(spec)
