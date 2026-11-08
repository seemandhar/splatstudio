"""OMG density control: the Mini-Splatting pipeline (our matched port, ms_density.py) plus the OMG
events of the official train.py.

Per training step s (the runtime calls `after_backward` after the per-Gaussian Adam step):
- s > net_itr and s < total_steps: step the MLP Adam (`optimizer_net.step`) and set the learning
  rate of the next step (`scheduler_net.step`).
- s >= svq_itr and s < total_steps: step the codebook Adam (`optimizer_code.step`). As in the
  official loop, the codebook gradients are never zeroed.
- Mini-Splatting events (ms_density): densification and depth reinit before 15k, then
  importance sampling plus reinit at restart_step.
- s == simp_iteration2: intersection preserving with the live MLP opacity, then local
  distinctiveness (C1) and a CDF prune at importance_thresh τ. This replaces Mini-Splatting's 1 %
  prune.
- s == net_itr: construct_net. T is set to the current SH DC, V = 0, and a fresh Gaussian Adam is
  used (C2).
- s == svq_itr - 1: apply_svq (C3). The official call is at the start of iteration svq_itr, which
  is the same state.
- s == total_steps: encode to bytes, decode, and replace the state with the decoded plain 3DGS
  model (C4).
"""

import json
import math
import time
from dataclasses import replace

import torch
from gsplat.strategy import ops

from . import codec, core, kernels, model, ms_core
from .ms_density import MiniSplattingDensity
from .renderer import STATIC, VIEW, live_attributes

OMG_DEFAULTS = {
    "net_itr": 15000,
    "svq_itr": 29000,
    "total_steps": 30000,
    # C1 local distinctiveness
    "ld_scoring": True,
    "importance_thresh": 0.96,  # OMG-XS (README default); S 0.98, M 0.99, L 0.999, XL 0.9999
    "lambda_ld": 2.0,
    "ld_neighbors": "official",
    # C2 space feature + MLPs
    "net": dict(model.NET_DEFAULTS),
    "net_lr": 0.01,
    "net_warmup_iters": 100,
    "net_warmup_start": 0.01,
    "net_lr_milestones": [1000, 3500, 6000],
    "net_lr_gamma": 0.33,
    # C3 SVQ
    "svq_slices": {"scale": 1, "rotation": 2, "app": 2},
    "svq_clusters": {"scale": 64, "rotation": 512, "app": 1024},
    "codebook_lr": 1e-8,
    "kmeans_max_iter": 1000,
    "kmeans_tol": 1e-4,
    "kmeans_seed": 0,
    # C4
    "encode": True,
}


class OMGDensity(MiniSplattingDensity):
    def __init__(self, spec):
        cfg = dict(spec.config)
        omg = {k: cfg.pop(k) for k in list(cfg) if k in OMG_DEFAULTS}
        super().__init__(replace(spec, config=cfg))
        self.omg = {**OMG_DEFAULTS, **omg}
        o = self.omg
        if not self.cfg["simplify"]:
            raise ValueError("OMG builds on Mini-Splatting with simplification")
        if not (self.cfg["restart_step"] <= o["net_itr"] < self.cfg["simp_iteration2"]
                < o["svq_itr"] <= o["total_steps"]):
            raise ValueError("need restart_step <= net_itr < simp_iteration2 < svq_itr <= total")
        if o["ld_neighbors"] not in ("official", "morton"):
            raise ValueError("ld_neighbors must be 'official' or 'morton'")
        self.bank = None
        self.lr0 = None

    # ------------------------------------------------------------------ runtime hooks
    def initialize(self, params, optimizers, scene_scale=1.0):
        for f in (STATIC, VIEW):
            if f not in params:
                raise ValueError(f"OMG density needs the '{f}' Gaussian field")
        super().initialize(params, optimizers, scene_scale)
        o = self.omg
        self.bank = model.OMGBank(o["net"], o["svq_itr"], o["net_itr"]).to(params["means"].device)
        self.bank.build_optimizer(self.net_lr(0))
        self.bank.step = 1
        self.lr0 = self.lr_init
        self.state["omg"] = {"encoded": None}
        model.ACTIVE["bank"] = self.bank

    def net_lr(self, epoch):
        o = self.omg
        return core.net_lr(o["net_lr"], epoch, o["net_warmup_iters"], o["net_warmup_start"],
                           tuple(o["net_lr_milestones"]), o["net_lr_gamma"])

    def check_run_length(self, step, lr):
        """From our Compact-3DGS / Reduced-3DGS ports: before after_backward the runtime sets the
        means LR to lr0 * 0.01 ** (step / steps), so the run length can be recovered at step 1.
        The OMG events are absolute step numbers, so a run of a different length is refused."""
        if step != 1 or self.lr0 is None:
            return
        if not 0 < lr < self.lr0:
            print("omg-density: WARNING cannot infer the run length from the means LR; OMG "
                  f"events run at the absolute steps up to {self.omg['total_steps']}", flush=True)
            return
        steps = math.log(0.01) / math.log(lr / self.lr0)
        if abs(steps - self.omg["total_steps"]) > 0.5:
            raise ValueError(f"run has ~{round(steps)} steps but OMG total_steps="
                             f"{self.omg['total_steps']}; the OMG schedule would not complete")

    def after_backward(self, params, optimizers, step, outputs):
        self.check_run_length(step, optimizers["means"].param_groups[0]["lr"])
        o, bank = self.omg, self.bank
        last = step >= o["total_steps"]
        if bank.net_enabled:
            if not last:  # official: optimizer_net.step(); zero_grad; scheduler_net.step()
                bank.optimizer.step()
                bank.set_lr(self.net_lr(step - o["net_itr"]))
            bank.optimizer.zero_grad(set_to_none=True)
        if bank.vq_enabled and not last:
            bank.code_optimizer.step()  # official: no zero_grad for optimizer_code
        super().after_backward(params, optimizers, step, outputs)
        if step == o["net_itr"]:
            self._construct_net(params, optimizers, step)
        if step == o["svq_itr"] - 1:
            self._apply_svq(params, step)
        if o["encode"] and step == o["total_steps"]:
            self._encode(params, optimizers, step)
        bank.step = step + 1

    # ------------------------------------------------------------------ Mini-Splatting hooks
    def _reinit(self, params, optimizers, means, sh0):
        """Mini-Splatting reinitial_pts + training_setup (ms_density._reinit) with the OMG
        feature fields zero (they are unused until construct_net)."""
        if self.bank is not None and self.bank.net_enabled:
            raise RuntimeError("OMG: re-initialisation after construct_net is not part of OMG")
        n = len(means)
        dist2 = kernels.knn_dist2(means).clamp_min(1e-7)
        quats = torch.zeros(n, 4, device=means.device)
        quats[:, 0] = 1
        p = torch.tensor(self.cfg["reinit_opacity"])
        values = {
            "means": means.float(),
            "scales": dist2.sqrt().log()[:, None].repeat(1, 3),
            "quats": quats,
            "opacities": torch.full((n,), float(torch.log(p / (1 - p))), device=means.device),
            "sh0": sh0.float().reshape(n, 1, 3),
            "shN": torch.zeros(n, 15, 3, device=means.device),
            STATIC: torch.zeros(n, 3, device=means.device),
            VIEW: torch.zeros(n, 3, device=means.device),
        }
        extra = set(params) - set(values)
        if extra:
            raise ValueError(f"OMG reinit does not know fields {sorted(extra)}")
        self._replace_all(params, optimizers, values)

    def _importance(self, params):
        """Mini-Splatting intersection_preserving; with the net enabled the blending weights
        use the MLP opacity (official render_imp inside intersection_preserving)."""
        bank = self.bank
        if bank is None or not bank.net_enabled:
            return super()._importance(params)
        a = live_attributes(bank, params, use_svq=False)
        n = len(params["means"])
        device = params["means"].device
        imp = torch.zeros(n, device=device)
        accum_area_max = torch.zeros(n, device=device)
        for cam in self.cameras:
            meta = kernels.project_tensors(a["means"], a["quats"], a["scales"], a["opacities"],
                                           cam)
            view = kernels.blend_stats(meta, weights=True)
            imp, accum_area_max = ms_core.accumulate_importance(
                imp, accum_area_max, view, self.cfg["imp_metric"])
        imp[accum_area_max == 0] = 0
        return imp

    def _simplify_prune(self, params, optimizers, step):
        """Official simp_iteration2: intersection_preserving -> ld_scoring (C1) -> CDF prune with
        importance_thresh -> training_setup (fresh Gaussian Adam; the MLP Adam is kept)."""
        o = self.omg
        n0 = len(params["means"])
        imp = self._importance(params)
        if o["ld_scoring"]:
            feats = params[STATIC] if self.bank.net_enabled else params["sh0"][:, 0]
            imp = core.ld_importance(imp, feats.detach(), params["means"].detach(),
                                     o["lambda_ld"], o["ld_neighbors"])
        keep = ms_core.cdf_keep_mask(imp, o["importance_thresh"])
        self._replace_all(params, optimizers, {k: v.detach()[keep] for k, v in params.items()})
        self._log(step, "ld_prune" if o["ld_scoring"] else "cdf_prune", n_before=n0,
                  n_after=len(params["means"]))
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ OMG events
    @torch.no_grad()
    def _construct_net(self, params, optimizers, step):
        n = len(params["means"])
        values = {STATIC: params["sh0"].detach()[:, 0, :].clone(),
                  VIEW: torch.zeros(n, 3, device=params["means"].device)}
        self._replace_all(params, optimizers, values)
        for opt in optimizers.values():  # official training_setup: new Gaussian optimiser
            opt.state.clear()
        self.bank.net_enabled = True
        self._log(step, "construct_net", gaussians=n, mlp_params=self.bank.param_count())

    @torch.no_grad()
    def _apply_svq(self, params, step):
        o = self.omg
        t0 = time.perf_counter()
        data = {"scale": params["scales"].detach(), "rotation": params["quats"].detach(),
                "app": torch.cat([params[STATIC].detach(), params[VIEW].detach()], dim=-1)}
        self.bank.apply_svq(data, o["svq_slices"], o["svq_clusters"], o["codebook_lr"],
                            o["kmeans_max_iter"], o["kmeans_tol"], o["kmeans_seed"])
        self._log(step, "apply_svq", gaussians=len(params["means"]),
                  milliseconds=round(1000 * (time.perf_counter() - t0)))

    @torch.no_grad()
    def _encode(self, params, optimizers, step):
        bank = self.bank
        if not bank.vq_enabled:
            raise RuntimeError("OMG encode needs SVQ (svq_itr <= total_steps)")
        t0 = time.perf_counter()
        n = len(params["means"])
        dev = params["means"].device
        u16 = core.float16_to_uint16(params["means"].detach())
        order = core.position_order(u16)
        u16s = u16[order]
        stream, official_xyz = codec.gpcc_encode(u16s.cpu().numpy())
        sections = {"xyz": stream}
        meta = {"n": n, "streams": {}, "mlp_shapes": {}}
        official = {"xyz": official_xyz, "scale": 0, "rotation": 0, "app": 0, "MLPs": 0}
        sorted_idx = {}
        for attr in ("scale", "rotation", "app"):
            sorted_idx[attr] = []
            for m, (code, idx) in enumerate(zip(bank.codes[attr], bank.indices[attr])):
                key = f"{attr}_{m}"
                sections[f"code_{key}"] = codec.f16_bytes(code)
                official[attr] += code.numel() * 2
                si = idx[order]
                sorted_idx[attr].append(si)
                h = codec.huffman_encode(si.cpu().numpy())
                sections[f"index_{key}"] = h["payload"]
                meta["streams"][key] = {"n": h["n"], "count": h["count"],
                                        "symbols": h["symbols"], "code_shape": list(code.shape)}
                official[attr] += h["official_bytes"]
        for name, ws in bank.mlps.items():
            flat = torch.cat([w.detach().reshape(-1) for w in ws])
            sections[f"mlp_{name}"] = codec.f16_bytes(flat)
            meta["mlp_shapes"][name] = [list(w.shape) for w in ws]
            official["MLPs"] += flat.numel() * 2
        blob = codec.write_container(sections, meta)
        t1 = time.perf_counter()
        dec = self.decode(blob, dev)
        t2 = time.perf_counter()
        if not torch.equal(dec["u16"].to(dev), u16s.to(dec["u16"].dtype).to(dev)):
            raise RuntimeError("G-PCC round trip changed the positions")
        for attr in ("scale", "rotation", "app"):
            for a, b in zip(dec["indices"][attr], sorted_idx[attr]):
                if not torch.equal(a.to(dev), b):
                    raise RuntimeError(f"decoded {attr} indices differ")
        # decoded model (official decode): fp16 positions, codebook geometry, MLP appearance
        # at the decoded positions with the fp16 MLP weights
        means = dec["means"].to(dev)
        app = dec["app"].to(dev)
        weights = {k: [w.to(dev) for w in v] for k, v in dec["mlp"].items()}
        sh0, shn, opacity = bank.appearance(means, app[:, 0:3], app[:, 3:6], weights)
        # diagnostic: decoded appearance vs the live fp32 model (positions, codebooks, MLPs)
        live_app = bank.svq_values("app")[order]
        l_dc, l_sh, l_op = bank.appearance(params["means"].detach()[order], live_app[:, 0:3],
                                           live_app[:, 3:6])
        drift = {"sh0_mean_abs": float((sh0 - l_dc).abs().mean()),
                 "shN_mean_abs": float((shn - l_sh).abs().mean()),
                 "opacity_mean_abs": float((opacity.sigmoid() - l_op.sigmoid()).abs().mean())}
        new = {"means": means, "scales": dec["scale"].to(dev), "quats": dec["rotation"].to(dev),
               "opacities": opacity, "sh0": sh0, "shN": shn, STATIC: app[:, 0:3],
               VIEW: app[:, 3:6]}
        missing = set(params) - set(new)
        if missing:
            raise ValueError(f"OMG encode does not know fields {sorted(missing)}")

        def param_fn(name, p):
            return torch.nn.Parameter(new[name].to(p.dtype).reshape(p.shape).contiguous(),
                                      requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.zeros_like(v)

        ops._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        bank.baked = True
        total_official = sum(official.values())
        report = {
            "gaussians": n,
            "official_bytes": {k: int(v) for k, v in official.items()},
            "official_total_mb": round(total_official / codec.MiB, 4),
            "container_bytes": len(blob),
            "container_mb": round(len(blob) / codec.MiB, 4),
            "gpcc_stream_bytes": len(stream),
            "gpcc_official_flag_bytes": int(official_xyz),
            "mlp_params": int(bank.param_count()),
            "encode_seconds": round(t1 - t0, 2),
            "decode_seconds": round(t2 - t1, 2),
            "decode_drift_vs_live_fp32": drift,
        }
        self.state["omg"]["encoded"] = report
        self._log(step, "encode", gaussians=n, container_bytes=len(blob),
                  official_bytes=int(total_official))
        print(json.dumps({"omg_storage": report}), flush=True)

    @staticmethod
    def decode(blob, device="cpu"):
        """Container bytes -> decoded tensors (positions in the coder's order)."""
        meta, s = codec.read_container(blob)
        pts = torch.from_numpy(codec.gpcc_decode(s["xyz"]))
        pts = pts[core.position_order(pts)]
        out = {"u16": pts, "means": core.uint16_to_float16(pts).float(), "indices": {}}
        for attr in ("scale", "rotation", "app"):
            parts, idxs = [], []
            m = 0
            while f"{attr}_{m}" in meta["streams"]:
                st = meta["streams"][f"{attr}_{m}"]
                idx = torch.from_numpy(codec.huffman_decode(
                    {**st, "payload": s[f"index_{attr}_{m}"]}))
                code = codec.f16_tensor(s[f"code_{attr}_{m}"], st["code_shape"])
                parts.append(code[idx])
                idxs.append(idx)
                m += 1
            out[attr] = torch.cat(parts, dim=-1)
            out["indices"][attr] = idxs
        out["mlp"] = {}
        for name, shapes in meta["mlp_shapes"].items():
            flat = codec.f16_tensor(s[f"mlp_{name}"], (-1,))
            ws, i = [], 0
            for a, b in shapes:
                ws.append(flat[i:i + a * b].view(a, b).clone())
                i += a * b
            out["mlp"][name] = ws
        return out


def create_density(spec):
    return OMGDensity(spec)
