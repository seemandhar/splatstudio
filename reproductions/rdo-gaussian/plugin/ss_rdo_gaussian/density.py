"""RDO-Gaussian density: 3DGS densification + RD-training schedule + final encode/decode.

Official train.py iteration: [it == 15001: k-means init of the six codebooks, VQ / masks on]
[it == 20001: rate-constrained assignment] -> render -> loss -> backward -> 3DGS, VQ (+ its
MultiStepLR) and mask optimizer steps (all skipped at the last iteration) -> [last iteration:
apply SH mask, remove masked Gaussians, encode `bitstream/`] -> [it % 1000 == 0 and it < 20001:
reactivate dead codewords] -> densification (< 15000).

Native order: the runtime steps the per-Gaussian Adam (including the two mask fields, which
receive no gradient before activation and are therefore not stepped, as official), then
`after_backward`:
1. bank Adam step (codebooks lr 2e-4, logits lr 2e-3, x0.1 after 12000 bank steps) while VQ is
   active and step < total_steps;
2. the SplatStudio 3DGS density (gsplat DefaultStrategy + opacity-reset fix), unchanged;
3. codeword reactivation (vq active, step % 1000 == 0, step < rate_constrain_start);
4. at step == start - 1: k-means initialisation / activation for the next render;
5. at step == rate_constrain_start - 1: rate-constrained assignment from the next render;
6. at total_steps: remove masked Gaussians -> SH mask -> negative scale codewords -> 1e-8 ->
   rate-constrained assignment -> encode the official bitstream -> decode the bytes -> replace
   the parameters with the decoded model (the encode -> decode -> replace pattern of the
   Reduced-3DGS port, reproductions/reduced-3dgs/plugin/ss_reduced_3dgs/density.py), so the
   checkpoint, validation.json and ss-eval measure the decoded model.
"""

import copy
import json
import math
import time
from dataclasses import replace

import numpy as np
import torch
from gsplat.strategy import ops
from splatstudio.backends.gsplat import Density

from . import bank as bank_mod
from . import codec
from .bank import KEYS, SH_KEYS, SH_SLICES, RDOBank
from .renderer import GS_MASK, SH_MASK

DEFAULTS = {
    "base": {"absgrad": False, "grow_grad2d": 0.0002},
    "total_steps": 30000,
    "gs_prune_start": 15001,
    "sh_prune_start": 15001,
    "vq_start": 15001,
    "rate_constrain_start": 20001,
    "reactivate_every": 1000,
    "reactivate_prob": 1e-5,
    "gs_threshold": 0.1,
    "sh_threshold": 0.1,
    "codebook_size": {"scale": 8192, "rot": 8192, "dc": 8192, "sh1": 4096, "sh2": 4096,
                      "sh3": 4096},
    "lmbda": {"scale": 32768.0, "rot": 256.0, "dc": 256.0, "sh1": 256.0, "sh2": 256.0,
              "sh3": 256.0},
    "codebook_lr": 2e-4,
    "logits_lr": 2e-3,
    "lr_milestone": 12000,
    "lr_gamma": 0.1,
    "index_update_every": 10,
    "patch_size": 65536,
    "kmeans_iters": 10,
    "rans_precision": 16,
    "opacity_bits": 8,
    "encode": True,
    "evaluate": "decoded",  # "official": fp32 xyz + unquantised opacity, as official render.py
    "log_every": 1000,
}
KEPT_LOGIT = 10.0  # mask fields of the decoded model (sigmoid > threshold: kept)


def schedule_events(step, cfg):
    """RDO events after optimizer step `step` (1-based), excluding 3DGS refine/reset."""
    events = []
    if (cfg["vq_start"] <= step < cfg["rate_constrain_start"]
            and step % cfg["reactivate_every"] == 0):
        events.append("reactivate")
    if step == cfg["vq_start"] - 1:
        events.append("vq_on")
    if step == cfg["gs_prune_start"] - 1:
        events.append("gs_on")
    if step == cfg["sh_prune_start"] - 1:
        events.append("sh_on")
    if step == cfg["rate_constrain_start"] - 1:
        events.append("rate_on")
    if cfg["encode"] and step == cfg["total_steps"]:
        events.append("encode")
    return events


def merged(defaults, overrides):
    out = copy.deepcopy(defaults)
    for k, v in overrides.items():
        out[k] = {**out[k], **v} if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


class RDODensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown rdo-density options: {sorted(unknown)}")
        self.cfg = merged(DEFAULTS, spec.config)
        c = self.cfg
        if c["evaluate"] not in ("decoded", "official"):
            raise ValueError("evaluate must be 'decoded' or 'official'")
        self.base = Density(replace(spec, config=dict(c["base"])))
        stop = self.base.strategy.refine_stop_iter
        if min(c["vq_start"], c["gs_prune_start"], c["sh_prune_start"]) <= stop:
            raise ValueError("RD training must start after densification (official 15001 > "
                             f"15000); refine_stop_iter={stop}")
        self.requires = self.base.requires
        self.bank = None
        self.history = []
        self.encoded = None
        self.stream = None
        self.lr0 = None

    @property
    def state(self):
        return {"strategy": self.base.state,
                "rdo": {"events": self.history, "encoded": self.encoded,
                        "bitstream": self.stream}}

    def initialize(self, params, optimizers, scene_scale=1.0):
        for f in (GS_MASK, SH_MASK):
            if f not in params:
                raise ValueError(f"rdo-density needs the '{f}' Gaussian field")
        self.base.initialize(params, optimizers, scene_scale)
        self.bank = RDOBank(self.cfg).to(params["means"].device)
        self.bank.build_optimizer()
        self.lr0 = optimizers["means"].param_groups[0]["lr"]
        bank_mod.ACTIVE["bank"] = self.bank

    def before_backward(self, params, optimizers, step, outputs):
        self.base.before_backward(params, optimizers, step, outputs)

    def check_run_length(self, step, lr):
        """Copied from the Reduced-3DGS port: the runtime sets the means LR to
        lr0 * 0.01 ** (step / steps) before after_backward, so the run length is recoverable."""
        if not self.cfg["encode"] or step != 1 or self.lr0 is None:
            return
        if not 0 < lr < self.lr0:
            print("rdo-density: WARNING cannot infer the run length from the means LR; the "
                  f"encode step runs only at step {self.cfg['total_steps']}", flush=True)
            return
        steps = math.log(0.01) / math.log(lr / self.lr0)
        if abs(steps - self.cfg["total_steps"]) > 0.5:
            raise ValueError(
                f"run has ~{round(steps)} steps but rdo-density total_steps="
                f"{self.cfg['total_steps']}: the final encode -> decode would never run. "
                "Train for total_steps or set the density config accordingly.")

    def after_backward(self, params, optimizers, step, outputs):
        self.check_run_length(step, optimizers["means"].param_groups[0]["lr"])
        b = self.bank
        if b.vq_active and step < self.cfg["total_steps"]:
            b.step_optimizer()
        else:
            b.optimizer.zero_grad(set_to_none=True)
        self.base.after_backward(params, optimizers, step, outputs)
        if b.any_active and step % self.cfg["log_every"] == 0 and "rdo" in outputs:
            self._log(step, outputs["rdo"])
        for event in schedule_events(step, self.cfg):
            start = time.perf_counter()
            record = {"rdo": event, "step": step, "gaussians": len(params["means"])}
            if event == "vq_on":
                record.update(self._kmeans(params))
                b.vq_active = True
            elif event == "gs_on":
                b.gs_active = True
            elif event == "sh_on":
                b.sh_active = True
            elif event == "rate_on":
                b.rate_constrain = True
            elif event == "reactivate":
                record["reactivated"] = {k: b.reactivate(k, self.cfg["reactivate_prob"])
                                         for k in KEYS}
            elif event == "encode":
                record.update(self._encode(params, optimizers))
            record["seconds"] = round(time.perf_counter() - start, 2)
            self.history.append(record)
            print(json.dumps(record), flush=True)
        b.render_step = step + 1

    def _log(self, step, rdo):
        row = {"rdo": "stats", "step": step,
               "rate_loss": float(rdo["rate"]), "vq_loss": float(rdo["vq"]),
               "bits_MB": {k: float(v) / 8e6 for k, v in rdo["bits"].items()}}
        for k in ("gs_mask_percent", "sh_mask_percent"):
            if k in rdo:
                row[k] = float(rdo[k])
        if self.bank.vq_active:
            row["live_codewords"] = {k: int(self.bank.live_mask(k, self.cfg["reactivate_prob"])
                                            .sum()) for k in KEYS}
        self.history.append(row)
        print(json.dumps(row), flush=True)

    @torch.no_grad()
    def vector_dict(self, params, shn=None):
        """Official GaussianModel.vector_dict (scale activated, raw rotation / DC / SH)."""
        shn = params["shN"].detach() if shn is None else shn
        out = {"scale": params["scales"].detach().exp(), "rot": params["quats"].detach(),
               "dc": params["sh0"].detach().reshape(-1, 3)}
        for k in SH_KEYS:
            a, b = SH_SLICES[k]
            out[k] = shn[:, a:b, :].reshape(len(shn), -1)
        return out

    @torch.no_grad()
    def _kmeans(self, params):
        vectors = self.vector_dict(params)
        for k in KEYS:
            self.bank.kmeans_initialize(k, vectors[k])
        return {"kmeans_live": {k: int(self.bank.live_mask(k, self.cfg["reactivate_prob"]).sum())
                                for k in KEYS}}

    @torch.no_grad()
    def _encode(self, params, optimizers):
        c, b = self.cfg, self.bank
        t0 = time.perf_counter()
        n_before = len(params["means"])
        report = {"gaussians_before": n_before}
        # official: gs_mask_percent / sh_mask_percent of the last render (hard masks)
        if b.gs_active:
            keep = torch.sigmoid(params[GS_MASK].detach()) > c["gs_threshold"]
            if not bool(keep.any()):
                raise RuntimeError("the Gaussian mask pruned every Gaussian")
            ops.remove(params, optimizers, self.base.state, ~keep)
        n = len(params["means"])
        report["gaussian_prune_ratio"] = 1.0 - n / n_before
        if b.sh_active:
            bits3 = torch.sigmoid(params[SH_MASK].detach()) > c["sh_threshold"]
        else:
            bits3 = torch.ones(n, 3, dtype=torch.bool, device=params["means"].device)
        coeff = torch.cat([bits3[:, i : i + 1].expand(-1, m) for i, m in enumerate((3, 5, 7))],
                          dim=1).float()
        report["sh_prune_ratio"] = 1.0 - float(coeff.mean())
        shn = params["shN"].detach() * coeff[..., None]  # official apply_sh_mask hard values
        sh_bits = {k: bits3[:, i] for i, k in enumerate(SH_KEYS)}
        cb = b.codebooks["scale"].data
        cb[cb < 0] = 1e-8  # official remove_negatives on the scale codebook
        vectors = self.vector_dict(params, shn)
        index = {}
        for k in KEYS:
            x = vectors[k][sh_bits[k]] if k in SH_KEYS else vectors[k]
            _, _, idx = b.quantize(k, x)  # no_grad: unclamped bits, as official
            index[k] = idx.cpu().numpy()
        attrs = {
            "xyz": params["means"].detach().cpu().numpy().astype(np.float32),
            "opacity": params["opacities"].detach().cpu().numpy().astype(np.float32)[:, None],
            "sh_bits": {k: v.cpu().numpy() for k, v in sh_bits.items()},
            "index": index,
            "codebook": {k: b.codebooks[k].detach().cpu().numpy()[None] for k in KEYS},
            "logits": {k: b.logits[k].detach().cpu().numpy()[None] for k in KEYS},
        }
        files, info = codec.encode(attrs, c["rans_precision"], c["opacity_bits"])
        t1 = time.perf_counter()
        dec = codec.decode(files, c["rans_precision"])
        t2 = time.perf_counter()
        for k in codec.INDEX_KEYS:
            ref = info["q_opacity"].reshape(-1) if k == "opa" else info["index"][k]
            if not np.array_equal(dec["index"][k].reshape(-1), ref.reshape(-1).astype(np.int64)):
                raise RuntimeError(f"{k}: rANS decode does not reproduce the encoded indices")
        dev = params["means"].device
        if c["evaluate"] == "official":
            xyz = attrs["xyz"][info["order"]]
            opacity = info["opacity_sorted"].reshape(-1)
        else:
            xyz, opacity = dec["xyz"], dec["opacity"].reshape(-1)
        scale = torch.from_numpy(dec["scale"]).to(dev)
        new = {
            "means": torch.from_numpy(np.ascontiguousarray(xyz)).to(dev),
            "opacities": torch.from_numpy(np.ascontiguousarray(opacity)).to(dev),
            # official decode: scaling_inverse_activation(codeword); exact zeros clamped
            "scales": scale.clamp_min(1e-12).log(),
            "quats": torch.from_numpy(dec["rot"]).to(dev),
            "sh0": torch.from_numpy(dec["dc"]).to(dev),
            "shN": torch.from_numpy(dec["shN"]).to(dev),
            GS_MASK: torch.full((n,), KEPT_LOGIT, device=dev),
            SH_MASK: torch.full((n, 3), KEPT_LOGIT, device=dev),
        }

        def param_fn(name, p):
            return torch.nn.Parameter(new[name].to(p.dtype).reshape(p.shape).contiguous(),
                                      requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.zeros_like(v)

        ops._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        for k, v in list(self.base.state.items()):
            if isinstance(v, torch.Tensor) and v.dim() > 0 and len(v) == n:
                self.base.state[k] = torch.zeros_like(v)
        b.baked = True
        self.stream = {name: torch.frombuffer(bytearray(data), dtype=torch.uint8)
                       for name, data in files.items()}
        size = codec.size_report(files, info["index_bytes"])
        self.encoded = {
            **report,
            "gaussians": n,
            **size,
            "used_codewords": info["used_codewords"],
            "boundaries": [int(v) for v in info["boundaries"]],
            "evaluate": c["evaluate"],
            "uncompressed_3dgs_ply_MB_same_count": n_before * 248 / 1e6,
            "encode_seconds": round(t1 - t0, 2),
            "decode_seconds": round(t2 - t1, 2),
        }
        return {"size_MB": round(size["size_MB"], 3), "gaussians_after": n,
                "gaussian_prune_ratio": round(report["gaussian_prune_ratio"], 4),
                "sh_prune_ratio": round(report["sh_prune_ratio"], 4)}


def create_density(spec):
    return RDODensity(spec)
