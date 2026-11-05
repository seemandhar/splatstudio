"""c3dgs density component: 3DGS training, then the official compression pipeline in one run.

One native run of `total_steps` (35000 = 3DGS 30000 + 5000 QA fine-tune iterations) expresses the
official workflow (3DGS `train.py` 30k, then c3dgs `compress.py`):

* steps 1..base_steps: the unchanged SplatStudio 3DGS density (gsplat DefaultStrategy with the
  opacity-reset fix); the means LR follows the schedule of a `--steps base_steps` run, so this
  phase is the shared `baseline3dgs-*-30000` recipe (pattern of the pup-3dgs plugin).
* after step base_steps: C1 sensitivity over all training views and pruning of Gaussians with
  zero colour sensitivity, C2 colour (48-dim SH) and shape (normalised covariance) clustering,
  then a `CompressedBank` is activated (official `run_vq` up to `finetune`).
* steps base_steps+1..total_steps: C3 QA fine-tuning through the normal runtime loop: fresh Adam
  (runtime moments cleared, bank Adam new), official LRs, means frozen (the official model has
  spatial_lr_scale = 0 after loading a .ply, so the xyz LR is 0), no densification; the bank's
  optimizer skips the last iteration as `finetune.py` does.
* at total_steps: C4 Morton sort + npz (official layout, DEFLATE), size = len / 2**20, decode the
  npz bytes into the base fields so checkpoint, validation and `ss-eval` measure the decoded
  model (pattern of the reduced-3dgs plugin).
"""

import json
import time
from dataclasses import replace

import torch
from gsplat.strategy import ops
from splatstudio.backends.gsplat import Density

from . import bank as bank_mod
from . import codec, schedule, sensitivity, vq
from .quant import fake_quantizers

DEFAULTS = {
    "base": {"absgrad": False, "grow_grad2d": 0.0002},
    "base_steps": 30000,
    "total_steps": 35000,
    "lr_final_ratio": 0.01,
    "prune_threshold": 0.0,
    "color_codebook_size": 4096,
    "color_importance_include": 0.6e-6,
    "color_cluster_iterations": 100,
    "color_batch_size": 2**18,
    "color_compress_non_dir": True,
    "gaussian_codebook_size": 4096,
    "gaussian_importance_include": 0.3e-5,
    "gaussian_cluster_iterations": 800,
    "gaussian_batch_size": 2**20,
    "decay": 0.8,
    "lrs": dict(bank_mod.DEFAULT_LRS),
    "freeze_means": True,
    "sort_morton": True,
    "sensitivity_white_background": False,
    "seed": 0,
    "keep_npz": True,
}


class C3DGSDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown c3dgs-density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        if not 0 < self.cfg["base_steps"] < self.cfg["total_steps"]:
            raise ValueError("c3dgs needs 0 < base_steps < total_steps")
        if not self.cfg["color_compress_non_dir"]:
            raise ValueError("only the official default colour vector (DC + rest) is implemented")
        self.base = Density(replace(spec, config=dict(self.cfg["base"])))
        self.requires = self.base.requires
        self.cameras = None
        self.lr0 = None
        self.bank = None
        self.history = []
        self.encoded = None
        self.npz = None

    @property
    def state(self):
        c3 = {"events": self.history, "encoded": self.encoded}
        if self.npz is not None:
            c3["npz"] = self.npz
        return {"strategy": self.base.state, "c3dgs": c3}

    def set_training_cameras(self, cameras):
        self.cameras = cameras

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.base.initialize(params, optimizers, scene_scale)
        # Runtime has already multiplied the means LR by the scene scale (spatial_lr_scale).
        self.lr0 = optimizers["means"].param_groups[0]["lr"]
        bank_mod.ACTIVE["bank"] = None

    def check_run_length(self, step, lr):
        """Raise if the native run length differs from total_steps (the encode would not run)."""
        if step != 1 or self.lr0 is None:
            return
        steps = schedule.inferred_run_length(self.lr0, lr)
        if steps is None:
            print("c3dgs-density: WARNING cannot infer the run length from the means LR; "
                  f"compression runs at step {self.cfg['base_steps']}, encode at "
                  f"{self.cfg['total_steps']}", flush=True)
            return
        if abs(steps - self.cfg["total_steps"]) > 0.5:
            raise ValueError(
                f"run has ~{round(steps)} steps but c3dgs-density total_steps="
                f"{self.cfg['total_steps']}: the compressed model would never be encoded. "
                "Train for total_steps (35000 for the paper) or set the density config.")

    def before_backward(self, params, optimizers, step, outputs):
        if step <= self.cfg["base_steps"]:
            self.base.before_backward(params, optimizers, step, outputs)

    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        self.check_run_length(step, optimizers["means"].param_groups[0]["lr"])
        events = schedule.events(step, c["base_steps"], c["total_steps"])
        if "base" in events:
            self.base.after_backward(params, optimizers, step, outputs)
        if "compress" in events:
            self.compress(params, optimizers, step)
        if "bank_step" in events:
            self.bank.step()
        if "encode" in events:
            self.encode(params, optimizers, step)
        self.set_learning_rates(optimizers, step)

    def set_learning_rates(self, optimizers, step):
        """Means LR for step + 1 (the runtime set its own 35000-step schedule just before)."""
        c = self.cfg
        if step < c["base_steps"]:
            lr = schedule.pretrain_means_lr(self.lr0, step, c["base_steps"], c["lr_final_ratio"])
        elif c["freeze_means"]:
            lr = 0.0
        else:  # paper text: positions optimised; 3DGS lr_final * spatial_lr_scale after 30k
            lr = self.lr0 * c["lr_final_ratio"]
        optimizers["means"].param_groups[0]["lr"] = lr

    # --- C1 + C2 -------------------------------------------------------------------------------
    @torch.no_grad()
    def compress(self, params, optimizers, step):
        if not self.cameras:
            raise RuntimeError("c3dgs-density needs set_training_cameras() before compression")
        c = self.cfg
        device = params["means"].device
        gen = torch.Generator(device=device)
        gen.manual_seed(int(c["seed"]))
        q = fake_quantizers(device)
        record = {"c3dgs": "compress", "step": step, "gaussians_before_prune": len(params["means"])}
        t0 = time.perf_counter()
        with torch.enable_grad():
            color_s, shape_s, pixels = sensitivity.compute(
                params, q, self.cameras, c["sensitivity_white_background"])
        color_imp = color_s.amax(-1)
        shape_imp = shape_s.amax(-1)
        record.update(sensitivity_seconds=round(time.perf_counter() - t0, 2), pixels=pixels)
        torch.cuda.empty_cache()

        t0 = time.perf_counter()
        n0 = len(params["means"])
        if c["prune_threshold"] >= 0:  # official compress_gaussians: keep importance > threshold
            keep = color_imp > c["prune_threshold"]
            if not bool(keep.all()):
                ops.remove(params, optimizers, self.base.state, ~keep)
            color_imp, shape_imp = color_imp[keep], shape_imp[keep]
        record["pruned"] = n0 - len(params["means"])
        # fresh optimizer for the fine-tune (official training_setup builds a new Adam)
        for opt in optimizers.values():
            opt.state.clear()

        # colour: the official calls get_features twice (shape, then values)
        sh0, shN = params["sh0"].detach(), params["shN"].detach()
        q["features_dc"](sh0), q["features_rest"](shN)
        features = torch.cat((q["features_dc"](sh0), q["features_rest"](shN)), dim=1)
        n_coefs = features.shape[1]
        color_features = features.flatten(-2)
        keep_c = color_imp > c["color_importance_include"]
        cb, idx = self._cluster(color_features[~keep_c], color_imp[~keep_c],
                                c["color_codebook_size"], c["color_batch_size"],
                                c["color_cluster_iterations"], False, gen)
        colors, color_idx = vq.join_features(color_features, keep_c, cb, idx)
        colors = colors.reshape(-1, n_coefs, 3)

        # shape: normalised covariance of the (fake-quantised) normalised scale and rotation
        s_hat, log_eta = sensitivity.reparameterize(params)
        scaling = q["scaling"](torch.nn.functional.normalize(torch.nn.functional.relu(s_hat)))
        rotation = torch.nn.functional.normalize(q["rotation"](params["quats"].detach()))
        cov = vq.covariance(scaling, rotation)
        keep_g = shape_imp > c["gaussian_importance_include"]
        cb, idx = self._cluster(cov[~keep_g], shape_imp[~keep_g], c["gaussian_codebook_size"],
                                c["gaussian_batch_size"], c["gaussian_cluster_iterations"], True,
                                gen)
        cov_cb, gauss_idx = vq.join_features(cov, keep_g, cb, idx)
        rot_cb, scale_cb = vq.extract_rot_scale(vq.to_full_cov(cov_cb))

        self.bank = bank_mod.CompressedBank(
            colors[:, :1], colors[:, 1:], color_idx, scale_cb, rot_cb, gauss_idx, log_eta, q,
            c["lrs"])
        bank_mod.ACTIVE["bank"] = self.bank
        record.update(
            gaussians=len(params["means"]),
            color_keep_fraction=float(keep_c.float().mean()),
            shape_keep_fraction=float(keep_g.float().mean()),
            color_codebook=len(colors), shape_codebook=len(cov_cb),
            clustering_seconds=round(time.perf_counter() - t0, 2),
        )
        torch.cuda.empty_cache()
        self._log(record)

    def _cluster(self, x, importance, size, batch, steps, normalize, gen):
        if len(x) == 0:
            return (torch.empty((0, x.shape[1]), device=x.device),
                    torch.empty((0,), dtype=torch.long, device=x.device))
        return vq.vq_features(x, importance, size, batch, steps, self.cfg["decay"],
                              scale_normalize=normalize, generator=gen)

    # --- C4 --------------------------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, params, optimizers, step):
        t0 = time.perf_counter()
        bank = self.bank
        bank.zero_grad()
        data, _ = codec.encode_npz(bank, params, self.cfg["sort_morton"])
        encode_seconds = time.perf_counter() - t0
        decoded = codec.decode_npz(data, params["means"].device)
        n_before = next(e["gaussians_before_prune"] for e in self.history
                        if e["c3dgs"] == "compress")
        report = codec.size_report(data, n_before, bank.n)
        report.update(bank.summary(), encode_seconds=round(encode_seconds, 2),
                      sort_morton=bool(self.cfg["sort_morton"]))

        # Morton order differs from the training order; training is over, so the Adam moments
        # are reset alongside the new parameter tensors.
        def param_fn(name, p):
            return torch.nn.Parameter(decoded[name].to(p.dtype).reshape(p.shape),
                                      requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.zeros_like(v)

        ops._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        for k, v in list(self.base.state.items()):
            if isinstance(v, torch.Tensor) and v.dim() > 0 and len(v) == bank.n:
                self.base.state[k] = torch.zeros_like(v)
        bank.decoded = True
        if bank_mod.ACTIVE["bank"] is bank:
            bank_mod.ACTIVE["bank"] = None
        self.encoded = report
        if self.cfg["keep_npz"]:
            self.npz = torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()
        self._log({"c3dgs": "encode", "step": step, **{k: v for k, v in report.items()
                                                        if k != "qparams"}})

    def _log(self, record):
        self.history.append(record)
        print(json.dumps(record), flush=True)


def create_density(spec):
    return C3DGSDensity(spec)
