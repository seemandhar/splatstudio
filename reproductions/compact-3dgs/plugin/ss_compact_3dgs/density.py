"""Compact-3DGS density: 3DGS densification + mask pruning + neural-field optimiser + final codec.

Official train.py loop per iteration: render (R-VQ for itr > rvq_iter) -> loss -> backward ->
[itr == iterations: final_prune(compress) + precompute] -> densification (< densify_until:
densify_and_prune every 100 after 500, with sigma(mask) <= 0.01 in the prune mask; opacity reset
every 3000) else mask_prune every mask_prune_iter -> optimizer.step / optimizer_net.step /
scheduler_net.step (skipped at the last iteration).

Native order: the runtime steps the per-Gaussian Adam first, then calls after_backward, which:
1. steps the net Adam (hash grid + MLP; skipped at total_steps) and sets the lr of the next step
   (closed-form ChainedScheduler);
2. runs the SplatStudio 3DGS density (gsplat DefaultStrategy + opacity-reset fix);
3. mask-prunes sigma(m) <= eps at refine steps (500 < step < 15000, step % 100 == 0) and every
   1000 steps from 15000;
4. at total_steps: final prune -> encode (fp16 xyz, R-VQ indices + fp32 codebooks, 8-bit +
   Huffman opacity / pruned hash under PP, fp16 MLP) into bytes -> decode the bytes -> replace the
   parameters with the decoded model and bake the eval-time state (hash features at the decoded
   positions, MLP blob). Encode -> decode -> replace is the Reduced-3DGS port's final step
   (reproductions/reduced-3dgs/plugin/ss_reduced_3dgs/density.py, copied pattern).
"""

import json
import math
import time
from dataclasses import replace

import torch
from gsplat.strategy import ops
from splatstudio.backends.gsplat import Density

from . import codec, model
from .renderer import BLOB, FEATURE, MASK, pack_blob

DEFAULTS = {
    "base": {"absgrad": False, "grow_grad2d": 0.0002},
    "total_steps": 30000,
    "densify_from": 500,
    "densify_every": 100,
    "densify_until": 15000,
    "mask_threshold": 0.01,
    "mask_prune_every": 1000,
    "rvq_iter": 29000,
    "rvq_size": 64,
    "rvq_num": 6,
    "kmeans_iters": 1,
    "codebook_lr": 1e-4,
    "net": dict(model.NET_DEFAULTS),
    "net_lr": 0.01,
    "net_warmup_iters": 100,
    "net_warmup_start": 0.01,
    "net_lr_milestones": [5000, 15000, 25000],
    "net_lr_gamma": 0.33,
    "comp": True,
    "hash_prune": 0.1,
    "encode": True,
}


def schedule_events(step, cfg):
    """Compact-3DGS events after optimizer step `step` (1-based), excluding 3DGS refine/reset."""
    events = []
    if cfg["densify_from"] < step < cfg["densify_until"]:
        if step % cfg["densify_every"] == 0:
            events.append("mask_prune")
    elif step >= cfg["densify_until"] and step % cfg["mask_prune_every"] == 0:
        events.append("mask_prune")
    if cfg["encode"] and step == cfg["total_steps"]:
        events.append("encode")
    return events


class CompactDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown compact-density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        base = dict(self.cfg["base"])
        base.setdefault("refine_start_iter", self.cfg["densify_from"])
        base.setdefault("refine_every", self.cfg["densify_every"])
        base.setdefault("refine_stop_iter", self.cfg["densify_until"])
        self.base = Density(replace(spec, config=base))
        self.requires = self.base.requires
        self.bank = None
        self.history = []
        self.encoded = None
        self.stream = None  # encoded bytes (kept in memory; sizes are in `state`)
        self.lr0 = None

    @property
    def state(self):
        return {"strategy": self.base.state,
                "compact3dgs": {"events": self.history, "encoded": self.encoded}}

    def lr_at(self, epoch):
        c = self.cfg
        return model.net_lr(c["net_lr"], epoch, c["net_warmup_iters"], c["net_warmup_start"],
                            tuple(c["net_lr_milestones"]), c["net_lr_gamma"])

    def initialize(self, params, optimizers, scene_scale=1.0):
        for f in (MASK, FEATURE, BLOB):
            if f not in params:
                raise ValueError(f"compact-density needs the '{f}' Gaussian field")
        self.base.initialize(params, optimizers, scene_scale)
        c = self.cfg
        device = params["means"].device
        self.bank = model.CompactBank(c["net"], c["rvq_size"], c["rvq_num"], c["kmeans_iters"],
                                      c["codebook_lr"], c["rvq_iter"]).to(device)
        self.bank.build_optimizer(self.lr_at(0))
        self.bank.step = 1
        self.lr0 = optimizers["means"].param_groups[0]["lr"]
        model.ACTIVE["bank"] = self.bank

    def before_backward(self, params, optimizers, step, outputs):
        self.base.before_backward(params, optimizers, step, outputs)

    def check_run_length(self, step, lr):
        """Copied from the Reduced-3DGS port: the runtime sets the means LR to
        lr0 * 0.01 ** (step / steps) before after_backward, so the run length is recoverable."""
        if not self.cfg["encode"] or step != 1 or self.lr0 is None:
            return
        if not 0 < lr < self.lr0:
            print("compact-density: WARNING cannot infer the run length from the means LR; the "
                  f"encode step runs only at step {self.cfg['total_steps']}", flush=True)
            return
        steps = math.log(0.01) / math.log(lr / self.lr0)
        if abs(steps - self.cfg["total_steps"]) > 0.5:
            raise ValueError(
                f"run has ~{round(steps)} steps but compact-density total_steps="
                f"{self.cfg['total_steps']}: the final encode -> decode would never run. "
                "Train for total_steps or set the density config accordingly.")

    def after_backward(self, params, optimizers, step, outputs):
        self.check_run_length(step, optimizers["means"].param_groups[0]["lr"])
        bank = self.bank
        if step < self.cfg["total_steps"]:  # official: no optimizer step at the last iteration
            bank.step_optimizer()
            bank.set_lr(self.lr_at(step))
        else:
            bank.optimizer.zero_grad(set_to_none=True)
        self.base.after_backward(params, optimizers, step, outputs)
        for event in schedule_events(step, self.cfg):
            start = time.perf_counter()
            record = {"compact3dgs": event, "step": step, "before": len(params["means"])}
            if event == "mask_prune":
                self._mask_prune(params, optimizers)
            elif event == "encode":
                self._mask_prune(params, optimizers)  # final_prune
                record.update(self._encode(params, optimizers))
            record.update(after=len(params["means"]), seconds=round(time.perf_counter() - start, 2))
            if event == "encode" or record["after"] != record["before"]:
                self.history.append(record)
            if event == "encode" or step % 1000 == 0:
                print(json.dumps(record), flush=True)
        bank.step = step + 1

    def prune_mask(self, params):
        return torch.sigmoid(params[MASK].detach()) <= self.cfg["mask_threshold"]

    @torch.no_grad()
    def _mask_prune(self, params, optimizers):
        mask = self.prune_mask(params)
        if bool(mask.any()):
            ops.remove(params, optimizers, self.base.state, mask)

    @torch.no_grad()
    def _encode(self, params, optimizers):
        bank = self.bank
        n = len(params["means"])
        t0 = time.perf_counter()
        rvq_trained = bank.vq_scale.initted and bank.vq_rot.initted
        means16 = params["means"].detach().half().float()
        scales = params["scales"].detach().exp()
        quats = torch.nn.functional.normalize(params["quats"].detach(), dim=-1)
        _, sidx = bank.vq_scale(scales, train=False)  # official final_prune: layers.training=False
        _, ridx = bank.vq_rot(quats, train=False)
        attrs = {
            "means": means16, "scale_idx": sidx, "rot_idx": ridx,
            "scale_books": torch.stack([la.embed.detach() for la in bank.vq_scale.layers]),
            "rot_books": torch.stack([la.embed.detach() for la in bank.vq_rot.layers]),
            "opacity": torch.sigmoid(params["opacities"].detach()),
            "hash": bank.hash.params.detach(),
            "mlp": [w.detach() for w in bank.mlp],
        }
        data, report = codec.encode(attrs, pp=self.cfg["comp"], hash_prune=self.cfg["hash_prune"],
                                    rvq_bits=bank.rvq_bits)
        t1 = time.perf_counter()
        dec = codec.decode(data, params["means"].device)
        t2 = time.perf_counter()
        if not (torch.equal(dec["scale_idx"], sidx) and torch.equal(dec["rot_idx"], ridx)):
            raise RuntimeError("decoded R-VQ indices differ from the encoded ones")
        if not torch.equal(dec["means"], means16):
            raise RuntimeError("decoded positions differ from the fp16 positions")
        features = bank.features(dec["means"], dec["hash"].contiguous())  # official precompute()
        blob = pack_blob(dec["mlp"], n, bank.n_in, bank.net["sh_encoding_degree"],
                         bank.net["tcnn_direction_quirk"], params["means"].device)
        opacity = dec["opacity"].clamp(1e-7, 1 - 1e-7)
        new = {
            "means": dec["means"],
            # R-VQ stores activated scales; the rasterizer uses s^2, so log|s| is exact.
            "scales": dec["scales"].abs().clamp_min(1e-12).log(),
            "quats": dec["quats"],
            "opacities": torch.log(opacity / (1 - opacity)),
            FEATURE: features,
            BLOB: blob,
        }

        def param_fn(name, p):
            value = new.get(name, p.detach())
            return torch.nn.Parameter(value.to(p.dtype).reshape(p.shape).contiguous(),
                                      requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.zeros_like(v)

        ops._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        bank.baked = True
        self.stream = data
        self.encoded = {
            "gaussians": n,
            "pp": bool(self.cfg["comp"]),
            **report,
            "hash_params": int(attrs["hash"].numel()),
            "mlp_params_tcnn": int(sum(w.numel() for w in attrs["mlp"])),
            "rvq_trained_before_encode": bool(rvq_trained),
            "encode_seconds": round(t1 - t0, 2),
            "decode_seconds": round(t2 - t1, 2),
        }
        key = "ours_pp" if self.cfg["comp"] else "ours"
        return {"paper_size_mb": round(report["paper_size_mb"][key]["total"], 3),
                "paper_size_mb_no_pp": round(report["paper_size_mb"]["ours"]["total"], 3),
                "actual_mb": round(report["actual_mb"], 3)}


def create_density(spec):
    return CompactDensity(spec)
