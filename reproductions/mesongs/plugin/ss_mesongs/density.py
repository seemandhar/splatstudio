"""MesonGS density component: 3DGS training, then the official post-training codec in one run.

One native run of `total_steps` (38000 = 3DGS 30000 + 8000 fine-tune iterations) expresses the
official workflow (3DGS `train.py` 30k, then `mesongs.py`):

* steps 1..base_steps: the unchanged SplatStudio 3DGS density (gsplat DefaultStrategy with the
  opacity-reset fix), means LR of a `--steps base_steps` run (c3dgs / pup-3dgs pattern), so this
  phase is the shared `baseline3dgs-*-30000` recipe.
* after step base_steps (`compress`): C1 importance over all training cameras and pruning, C2
  octree voxelisation with mean merge, RAHT tables, block quantisers, C6 SH VQ, then the params
  are replaced by the voxelised model (fresh Adam, official fine-tune LRs, means frozen) and a
  `MesonBank` is activated.
* steps base_steps+1..total_steps: C8 fine-tune through the normal runtime loop; the renderer
  renders C3-C5 (Euler, RAHT, block fake quantisation) from the bank; the codebook Adam lives in
  the bank; no optimizer step at the last iteration (official `iteration < iterations`).
* at total_steps (`encode`): C7 official bitstream (zip of six npz) in memory, decoded into the
  base fields so checkpoint, validation and `ss-eval` measure the decoded model (pattern of the
  reduced-3dgs / rdo-gaussian / c3dgs plugins).
"""

import json
import time
from dataclasses import replace

import numpy as np
import torch
from splatstudio.backends.gsplat import Density

from . import bank as bank_mod
from . import codec, importance, octree, raht, schedule, vq

DEFAULTS = {
    "base": {"absgrad": False, "grow_grad2d": 0.0002},
    "base_steps": 30000,
    "total_steps": 38000,
    "lr_final_ratio": 0.01,
    "scene": None,
    # per-scene table (official mesongs.py config3 / beta_list / lr_scale_list)
    "prune_percent": None,
    "codebook_size": None,
    "depth": None,
    "n_block": None,
    "beta": 0.1,
    "lr_scale": None,  # official lr_scale_list[scene] * 4
    "num_bits": 8,
    "vq_batch_size": 262144,
    "vq_steps": 1000,
    "vq_decay": 0.8,
    "vq_importance": "official",  # official: pre-merge importance indexed by post-merge row
    "lrs": {"opacities": 0.05, "sh0": 0.0025, "shN_codebook": 0.0025 / 20.0, "quats": 0.001,
            "scales": 0.001},  # official OptimizationParams; scales also x spatial_lr_scale
    "scales_lr_times_extent": True,
    "seed": 0,
    "keep_bitstream": True,
}
REQUIRED = ("scene", "prune_percent", "codebook_size", "depth", "n_block", "lr_scale")


class MesonDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown mesongs-density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        missing = [k for k in REQUIRED if self.cfg[k] is None]
        if missing:
            raise ValueError(f"mesongs-density needs per-scene values for {missing}")
        if not 0 < self.cfg["base_steps"] <= self.cfg["total_steps"]:
            raise ValueError("mesongs needs 0 < base_steps <= total_steps")
        if self.cfg["vq_importance"] not in ("official", "aligned"):
            raise ValueError("vq_importance must be 'official' or 'aligned'")
        self.base = Density(replace(spec, config=dict(self.cfg["base"])))
        self.requires = self.base.requires
        self.cameras = None
        self.lr0 = None
        self.scene_scale = None
        self.bank = None
        self.geometry = None
        self.history = []
        self.encoded = None
        self.bitstream = None

    @property
    def state(self):
        m = {"events": self.history, "encoded": self.encoded}
        if self.bitstream is not None:
            m["bitstream"] = self.bitstream
        return {"strategy": self.base.state, "mesongs": m}

    def set_training_cameras(self, cameras):
        self.cameras = cameras

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.base.initialize(params, optimizers, scene_scale)
        self.lr0 = optimizers["means"].param_groups[0]["lr"]
        self.scene_scale = float(scene_scale)
        bank_mod.ACTIVE["bank"] = None

    def check_run_length(self, step, lr):
        """Raise if the native run length differs from total_steps (the encode would not run)."""
        if step != 1 or self.lr0 is None:
            return
        steps = schedule.inferred_run_length(self.lr0, lr)
        if steps is None:
            print("mesongs-density: WARNING cannot infer the run length from the means LR",
                  flush=True)
            return
        if abs(steps - self.cfg["total_steps"]) > 0.5:
            raise ValueError(
                f"run has ~{round(steps)} steps but mesongs-density total_steps="
                f"{self.cfg['total_steps']}: the compressed model would never be encoded. "
                "Train for total_steps or change the density config.")

    def before_backward(self, params, optimizers, step, outputs):
        if step <= self.cfg["base_steps"]:
            self.base.before_backward(params, optimizers, step, outputs)

    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        self.check_run_length(step, optimizers["means"].param_groups[0]["lr"])
        events = schedule.events(step, c["base_steps"], c["total_steps"])
        if "base" in events:
            self.base.after_backward(params, optimizers, step, outputs)
        if step < c["base_steps"]:
            optimizers["means"].param_groups[0]["lr"] = schedule.pretrain_means_lr(
                self.lr0, step, c["base_steps"], c["lr_final_ratio"])
        if "compress" in events:
            self.compress(params, optimizers, step)
        if "bank_step" in events:
            self.bank.step()
        if step >= c["base_steps"]:
            self.set_finetune_lrs(optimizers, frozen="freeze" in events or
                                  step >= c["total_steps"] - 1)
        if "encode" in events:
            self.encode(params, optimizers, step)

    def finetune_lrs(self):
        """Official finetuning_setup LRs (x lr_scale; scaling also x spatial_lr_scale)."""
        c = self.cfg
        s = c["lr_scale"]
        lrs = {k: v * s for k, v in c["lrs"].items()}
        if c["scales_lr_times_extent"]:
            lrs["scales"] *= self.scene_scale
        lrs["means"] = 0.0  # octree positions are fixed (requires_grad False in the official code)
        lrs["shN"] = 0.0  # the trainable SH rest is the bank's codebook
        return lrs

    def set_finetune_lrs(self, optimizers, frozen=False):
        lrs = self.finetune_lrs()
        for name, opt in optimizers.items():
            opt.param_groups[0]["lr"] = 0.0 if frozen else lrs[name]
        if self.bank is not None:
            self.bank.optimizer.param_groups[0]["lr"] = 0.0 if frozen else lrs["shN_codebook"]

    # --- C1, C2, C4-C6 ---------------------------------------------------------------------------
    @torch.no_grad()
    def compress(self, params, optimizers, step):
        if not self.cameras:
            raise RuntimeError("mesongs-density needs set_training_cameras() before compression")
        c = self.cfg
        device = params["means"].device
        record = {"mesongs": "compress", "step": step, "scene": c["scene"],
                  "gaussians_before_prune": len(params["means"])}

        t0 = time.perf_counter()
        imp = importance.importance(params, self.cameras, c["beta"])
        pmask = importance.prune_mask(c["prune_percent"], imp)
        keep = ~pmask
        imp = imp[keep]
        record.update(importance_seconds=round(time.perf_counter() - t0, 2),
                      pruned=int(pmask.sum()))

        t0 = time.perf_counter()
        names = ("opacities", "sh0", "shN", "scales", "quats")
        feats = torch.cat([params["opacities"].detach()[keep][:, None],
                           params["sh0"].detach()[keep].flatten(1),
                           params["shN"].detach()[keep].flatten(1),
                           params["scales"].detach()[keep],
                           params["quats"].detach()[keep]], -1).cpu().numpy()
        xyz, merged, V, keys, oct_params, inverse = octree.voxelize(
            params["means"].detach()[keep].cpu().numpy(), feats, c["depth"])
        n = len(xyz)
        merged = torch.from_numpy(merged).to(device)
        new = {"means": torch.from_numpy(xyz).float().to(device)}
        sizes = {"opacities": 1, "sh0": 3, "shN": 45, "scales": 3, "quats": 4}
        o = 0
        for k in names:
            new[k] = merged[:, o:o + sizes[k]].reshape((n,) + tuple(params[k].shape[1:]))
            o += sizes[k]
        record.update(gaussians_after_prune=int(keep.sum()), gaussians=n,
                      octree_seconds=round(time.perf_counter() - t0, 2))

        t0 = time.perf_counter()
        tables = raht.RAHTTables(V, c["depth"], device)
        record["raht_tables_seconds"] = round(time.perf_counter() - t0, 2)

        t0 = time.perf_counter()
        gen = torch.Generator(device=device)
        gen.manual_seed(int(c["seed"]))
        if c["vq_importance"] == "official":
            # official vq_fe(imp, ...): imp keeps the pruned (pre-merge) order and length while the
            # features are voxel-sorted; importance/importance.max() is indexed by feature row
            vq_imp = (imp / imp.max())[:n]
        else:
            vq_imp = torch.zeros(n, device=device).index_add_(
                0, torch.from_numpy(inverse).to(device), imp)
            vq_imp = vq_imp / vq_imp.max()
        features = new["shN"].flatten(1)
        codebook, indices = vq.vq_features(features, vq_imp, c["codebook_size"],
                                           c["vq_batch_size"], c["vq_steps"], c["vq_decay"],
                                           generator=gen)
        record["vq_seconds"] = round(time.perf_counter() - t0, 2)
        new["shN"] = codebook[indices].reshape(n, 15, 3)

        self._replace(params, optimizers, new)
        lrs = self.finetune_lrs()
        self.bank = bank_mod.MesonBank(tables, c["n_block"], c["num_bits"], codebook, indices,
                                       lrs["shN_codebook"], device)
        bank_mod.ACTIVE["bank"] = self.bank
        self.geometry = {"keys": keys, "params": oct_params, "depth": c["depth"]}
        record.update(n_block=c["n_block"], codebook=c["codebook_size"], depth=c["depth"],
                      lrs=lrs)
        torch.cuda.empty_cache()
        self._log(record)

    def _replace(self, params, optimizers, new):
        """Swap every field for the voxelised model; fresh Adam state (official finetuning_setup
        builds a new optimizer)."""
        for name, value in new.items():
            p = torch.nn.Parameter(value.contiguous().float(), requires_grad=True)
            params[name] = p
            opt = optimizers[name]
            opt.state.clear()
            opt.param_groups[0]["params"] = [p]
        n = len(new["means"])
        for k, v in list(self.base.state.items()):
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                self.base.state[k] = torch.zeros((n,) + tuple(v.shape[1:]), dtype=v.dtype,
                                                 device=v.device)

    # --- C7 --------------------------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, params, optimizers, step):
        t0 = time.perf_counter()
        bank = self.bank
        bank.zero_grad()
        g = self.geometry
        data, members = codec.encode(bank, params, g["keys"], g["params"], g["depth"])
        encode_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        decoded = codec.decode(data, params["means"].device)
        decode_seconds = time.perf_counter() - t0
        comp = next(e for e in self.history if e["mesongs"] == "compress")
        report = codec.size_report(data, members, comp["gaussians_before_prune"],
                                   comp["gaussians_after_prune"], bank.n)
        report.update(bank.summary(), encode_seconds=round(encode_seconds, 2),
                      decode_seconds=round(decode_seconds, 2))
        lrs = {name: opt.param_groups[0]["lr"] for name, opt in optimizers.items()}
        self._replace(params, optimizers, decoded)
        for name, opt in optimizers.items():
            opt.param_groups[0]["lr"] = lrs[name]
        bank.decoded = True
        if bank_mod.ACTIVE["bank"] is bank:
            bank_mod.ACTIVE["bank"] = None
        self.encoded = report
        if self.cfg["keep_bitstream"]:
            self.bitstream = torch.from_numpy(np.frombuffer(data, dtype=np.uint8).copy())
        self._log({"mesongs": "encode", "step": step, **report})

    def _log(self, record):
        self.history.append(record)
        print(json.dumps(record), flush=True)


def create_density(spec):
    return MesonDensity(spec)
