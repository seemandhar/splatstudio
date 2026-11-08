"""LightGaussian density component: 3DGS training, then the official compression pipeline.

One native run of `total_steps` (45000 = 3DGS 30000 + co-adaptation 5000 + SH distillation 5000
+ VQ fine-tune 5000) expresses the official multi-script workflow (pattern of the pup-3dgs /
gaussianpop / c3dgs plugins; ss-train cannot resume):

* 1..prune_step: unchanged SplatStudio 3DGS density (gsplat DefaultStrategy + opacity-reset fix)
  with the means LR of a `--steps prune_step` run, i.e. the `baseline3dgs-*-30000` recipe.
* prune_step: C1 global significance over all training views (CUDA port of the official
  `renderCUDA_count`), v-score with beta = 0.1, remove the 66 % least significant
  (`prune_finetune.py`, Adam moments kept as the official restore does).
* ..distill_step: co-adaptation, photometric loss, no densification, non-means LR x0.95/400 its.
* distill_step: C2 teacher = frozen SH-3 copy; SH truncated to degree 2 (shN[:, 8:] kept 0);
  fresh optimizer state; opacity frozen; AdamW decay 0.01; LR x0.90/500 its; pseudo views.
* vq_step: C3 v-score on the student, official significance-weighted VQ of the 27-d SH of the
  60 % least significant Gaussians; the model is set to the official dequantised values.
* ..total_steps: VQ fine-tune (paper §3.4): codebook + all attributes, fixed assignment.
* total_steps: official `extreme_saving` zip, size, decode into the params (decoded-model
  evaluation).
"""

import json
import time
from dataclasses import replace

import torch
from gsplat.strategy import ops
from splatstudio.backends.gsplat import Density

from . import codec, schedule, significance, vq
from .shared import STATE

DEFAULTS = {
    "base": {"absgrad": False, "grow_grad2d": 0.0002},
    "prune_step": 30000,
    "distill_step": 35000,
    "vq_step": 40000,
    "total_steps": 45000,
    "lr_final_ratio": 0.01,
    # C1 (scripts/run_prune_finetune.sh, prune.py)
    "prune_percent": 0.66,
    "v_pow": 0.1,
    "coadapt_gamma": 0.95,
    "coadapt_every": 400,
    "coadapt_weight_decay": 0.0,
    # C2 (scripts/run_distill_finetune.sh, distill_train.py)
    "new_max_sh": 2,
    "pseudo_views": True,
    "pseudo_every": 3,
    "pseudo_translation_std": 0.05,
    "distill_gamma": 0.90,
    "distill_every": 500,
    "distill_weight_decay": 0.01,
    "distill_opacity": False,
    "distill_covariance": True,
    # C3 (scripts/run_vectree_quantize.sh, vectree/vectree.py)
    "vq_ratio": 0.6,
    "codebook_size": 8192,
    "vq_iterations": 1000,
    "vq_batch": 80000,
    "vq_decay": 0.8,
    "vq_k_expire": 10,
    # VQ fine-tune (paper §3.4 / supp.; no released code: choices documented in plan.md)
    "vqft_gamma": 0.95,
    "vqft_every": 400,
    "vqft_weight_decay": 0.01,
    "codebook_lr_dc": 0.0025,
    "codebook_lr_rest": 0.0025 / 20,
    "seed": 0,
    "keep_zip": True,
}

SH_REST = {0: 0, 1: 3, 2: 8, 3: 15}


class LightGaussianDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown lightgaussian-density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        schedule.validate(self.cfg)
        if self.cfg["new_max_sh"] not in SH_REST:
            raise ValueError("new_max_sh must be 0..3")
        self.base = Density(replace(spec, config=dict(self.cfg["base"])))
        self.requires = self.base.requires
        self.cameras = None
        self.lr0 = None
        self.base_lrs = None
        self.lr_used = {}
        self.history = []
        self.opacity_snapshot = None
        self.vq = None  # dict(mask, rows, idx, codebook tensors, optimizer, hooks)
        self.frozen_snapshots = None
        self.encoded = None
        self.zip = None

    # --- runtime interface -----------------------------------------------------------------------
    @property
    def state(self):
        lg = {"events": self.history, "encoded": self.encoded}
        if self.zip is not None:
            lg["zip"] = self.zip
        return {"strategy": self.base.state, "lightgaussian": lg}

    def set_training_cameras(self, cameras):
        self.cameras = cameras

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.base.initialize(params, optimizers, scene_scale)
        # The runtime has already multiplied the means LR by the scene scale (spatial_lr_scale).
        self.lr0 = optimizers["means"].param_groups[0]["lr"]
        self.base_lrs = {k: o.param_groups[0]["lr"] for k, o in optimizers.items()}
        self.lr_used = dict(self.base_lrs)
        STATE.reset()
        STATE.active = True
        STATE.cfg = self.cfg
        STATE.next_step = 1
        STATE.generator = torch.Generator().manual_seed(int(self.cfg["seed"]))

    def check_run_length(self, step, lr):
        """Raise if the native run length differs from total_steps (the stages would not run)."""
        if step != 1 or self.lr0 is None:
            return
        steps = schedule.inferred_run_length(self.lr0, lr)
        if steps is None:
            print("lightgaussian-density: WARNING cannot infer the run length from the means LR",
                  flush=True)
            return
        if abs(steps - self.cfg["total_steps"]) > 0.5:
            raise ValueError(
                f"run has ~{round(steps)} steps but lightgaussian-density total_steps="
                f"{self.cfg['total_steps']}: the compression stages would not run as the "
                "official pipeline. Train for total_steps (45000 for the paper).")

    def before_backward(self, params, optimizers, step, outputs):
        if step <= self.cfg["prune_step"]:
            self.base.before_backward(params, optimizers, step, outputs)

    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        self.check_run_length(step, optimizers["means"].param_groups[0]["lr"])
        stage = schedule.stage(step, c)
        if stage == "3dgs":
            self.base.after_backward(params, optimizers, step, outputs)
        else:
            self.post_step(params, step, stage)
        for event in schedule.events(step, c):
            getattr(self, f"on_{event}")(params, optimizers, step)
        self.set_learning_rates(optimizers, step + 1)
        STATE.next_step = step + 1
        if step == c["total_steps"]:
            STATE.active = False

    # --- per-step stage behaviour ----------------------------------------------------------------
    @torch.no_grad()
    def post_step(self, params, step, stage):
        """Effects of step `step` that the runtime's plain Adam cannot express."""
        wd = schedule.weight_decay(step, self.cfg)
        frozen = set()
        if stage == "distill":
            if not self.cfg["distill_opacity"]:
                frozen.add("opacities")
            if not self.cfg["distill_covariance"]:
                frozen.update(("scales", "quats"))
        for name, snap in (self.frozen_snapshots or {}).items():
            if name in frozen:
                params[name].data.copy_(snap)
        if wd > 0:  # AdamW decoupled decay, p <- p (1 - lr wd), with the LR of this step
            for name, p in params.items():
                if name not in frozen:
                    p.data.mul_(1.0 - self.lr_used[name] * wd)
        if stage == "vqft" and self.vq is not None:
            self.codebook_step(params)
        if stage in ("distill", "vqft"):
            params["shN"].data[:, SH_REST[self.cfg["new_max_sh"]]:] = 0.0

    def set_learning_rates(self, optimizers, k):
        """LRs used by optimizer step k (the runtime set its own means LR just before)."""
        c = self.cfg
        if k > c["total_steps"]:
            return
        factor = schedule.other_lr_factor(k, c)
        for name, opt in optimizers.items():
            if name == "means":
                lr = schedule.means_lr(self.lr0, k, c)
            else:
                lr = self.base_lrs[name] * factor
            opt.param_groups[0]["lr"] = lr
            self.lr_used[name] = lr
        if self.vq is not None and "optimizer" in self.vq:
            for g in self.vq["optimizer"].param_groups:
                g["lr"] = g["initial_lr"] * factor
                g["weight_decay"] = schedule.weight_decay(k, c)

    # --- events ----------------------------------------------------------------------------------
    @torch.no_grad()
    def on_prune(self, params, optimizers, step):
        if not self.cameras:
            raise RuntimeError("lightgaussian-density needs set_training_cameras() before pruning")
        t0 = time.perf_counter()
        n0 = len(params["means"])
        v, count, score = significance.global_significance(params, self.cameras, self.cfg["v_pow"])
        mask = significance.prune_mask(v, self.cfg["prune_percent"])
        ops.remove(params, optimizers, self.base.state, mask)
        self.check_leaves(params)
        self._log({"lightgaussian": "prune", "step": step, "before": n0,
                   "after": len(params["means"]), "zero_score": int((score == 0).sum()),
                   "never_hit": int((count == 0).sum()),
                   "seconds": round(time.perf_counter() - t0, 2)})
        del v, count, score, mask
        torch.cuda.empty_cache()

    @torch.no_grad()
    def on_start_distill(self, params, optimizers, step):
        # Teacher G' (Alg. 1): the pruned + co-adapted model with full-degree SH.
        STATE.teacher = {k: p.detach().clone() for k, p in params.items()}
        # Student: onedownSHdegree() keeps (new_max_sh + 1)^2 - 1 rest coefficients.
        params["shN"].data[:, SH_REST[self.cfg["new_max_sh"]]:] = 0.0
        # distill_train.py calls training_setup() after restore(): new optimizer, fresh moments.
        for opt in optimizers.values():
            opt.state.clear()
        self.frozen_snapshots = {k: params[k].detach().clone()
                                 for k in ("opacities", "scales", "quats")}
        self._log({"lightgaussian": "start_distill", "step": step,
                   "gaussians": len(params["means"]), "sh_degree": self.cfg["new_max_sh"]})

    @torch.no_grad()
    def on_vq(self, params, optimizers, step):
        c = self.cfg
        t0 = time.perf_counter()
        self.frozen_snapshots = None
        rest = SH_REST[c["new_max_sh"]]
        v, _, _ = significance.global_significance(params, self.cameras, c["v_pow"])
        feats = vq.sh_features(params["sh0"].detach(), params["shN"].detach(), rest)
        non_vq = vq.split_masks(v, c["vq_ratio"])
        vq_mask = ~non_vq
        gen = torch.Generator(device=feats.device).manual_seed(int(c["seed"]))
        codebook, cluster_size = vq.learn_codebook(
            feats[vq_mask], v[vq_mask], c["codebook_size"], c["vq_iterations"], c["vq_batch"],
            c["vq_decay"], c["vq_k_expire"], gen)
        indices = vq.nearest(feats, codebook)  # calc_vector_quantized_feature (all rows)
        idx = indices[vq_mask]
        codebook = codebook.half().float()
        err_vq = (feats[vq_mask] - codebook[idx]).square().sum(1).mean()
        # Official dequantised model: VQ SH from the fp16 codebook, other SH / opacity / scale /
        # rotation through fp16, xyz fp32.
        q = feats.half().float()
        q[vq_mask] = codebook[idx]
        sh0, sh_rest = vq.from_sh_features(q, rest)
        params["sh0"].data.copy_(sh0)
        params["shN"].data[:, :rest] = sh_rest
        params["shN"].data[:, rest:] = 0.0
        for name in ("opacities", "scales", "quats"):
            params[name].data.copy_(params[name].data.half().float())
        rows = torch.where(vq_mask)[0]
        self.vq = {"non_vq_mask": non_vq, "rows": rows, "idx": idx, "rest": rest,
                   "codebook": codebook}
        self._log({"lightgaussian": "vq", "step": step, "gaussians": len(feats),
                   "vq_gaussians": int(vq_mask.sum()),
                   "IS_percent": float(v[non_vq].sum() / v.sum()),
                   "used_codes": int(torch.unique(idx).numel()),
                   "mse_vq_rows": float(err_vq),
                   "seconds": round(time.perf_counter() - t0, 2)})
        del v, feats, q, cluster_size
        torch.cuda.empty_cache()

    def on_start_vqft(self, params, optimizers, step):
        """Fine-tune with fixed assignment: codebook gradient = sum of member SH gradients."""
        c, d = self.cfg, self.vq
        rest = d["rest"]
        cb_dc, cb_rest = vq.from_sh_features(d["codebook"], rest)
        d["cb_dc"] = cb_dc.clone().requires_grad_(True)
        d["cb_rest"] = cb_rest.clone().requires_grad_(True)
        d["optimizer"] = torch.optim.AdamW(
            [{"params": [d["cb_dc"]], "lr": c["codebook_lr_dc"]},
             {"params": [d["cb_rest"]], "lr": c["codebook_lr_rest"]}],
            eps=1e-15, weight_decay=c["vqft_weight_decay"])
        for g in d["optimizer"].param_groups:
            g["initial_lr"] = g["lr"]
        for opt in optimizers.values():  # fresh optimizer state (new training_setup)
            opt.state.clear()
        rows, idx = d["rows"], d["idx"]

        def hook_for(target, ncoef):
            def hook(p):
                g = p.grad
                if g is None:
                    return
                acc = torch.zeros_like(target).index_add_(0, idx, g[rows, :ncoef])
                target.grad = acc if target.grad is None else target.grad + acc
                g[rows] = 0.0  # VQ rows are driven by the codebook only
            return hook

        d["hooks"] = [
            params["sh0"].register_post_accumulate_grad_hook(hook_for(d["cb_dc"], 1)),
            params["shN"].register_post_accumulate_grad_hook(hook_for(d["cb_rest"], rest)),
        ]
        d["sh0_param"], d["shN_param"] = params["sh0"], params["shN"]
        self.check_leaves(params)
        self._log({"lightgaussian": "start_vqft", "step": step,
                   "codebook_rows": int(cb_dc.shape[0])})

    @torch.no_grad()
    def codebook_step(self, params):
        d = self.vq
        if params["sh0"] is not d["sh0_param"] or params["shN"] is not d["shN_param"]:
            raise RuntimeError("SH parameters were replaced during the VQ fine-tune")
        if d["cb_dc"].grad is not None or d["cb_rest"].grad is not None:
            d["optimizer"].step()
        d["optimizer"].zero_grad(set_to_none=True)
        rows, idx, rest = d["rows"], d["idx"], d["rest"]
        params["sh0"].data[rows] = d["cb_dc"].detach()[idx]
        params["shN"].data[rows, :rest] = d["cb_rest"].detach()[idx]

    @torch.no_grad()
    def on_encode(self, params, optimizers, step):
        t0 = time.perf_counter()
        if self.vq is None:
            raise RuntimeError("encode without a VQ stage")
        d = self.vq
        for h in d.get("hooks", []):
            h.remove()
        rest = d["rest"]
        if "cb_dc" in d:
            codebook = vq.sh_features(d["cb_dc"].detach(), d["cb_rest"].detach(), rest)
        else:
            codebook = d["codebook"]
        feats = vq.sh_features(params["sh0"].detach(), params["shN"].detach(), rest)
        other = torch.cat((params["opacities"].detach()[:, None], params["scales"].detach(),
                           params["quats"].detach()), dim=1)
        data, files = codec.encode(params["means"].detach(), feats, other, codebook, d["idx"],
                                   d["non_vq_mask"])
        dec = codec.decode(data, params["means"].device)
        sh0, sh_rest = vq.from_sh_features(dec["sh_feats"], rest)
        params["means"].data.copy_(dec["xyz"])
        params["sh0"].data.copy_(sh0)
        params["shN"].data.zero_()
        params["shN"].data[:, :rest] = sh_rest
        params["opacities"].data.copy_(dec["other"][:, 0])
        params["scales"].data.copy_(dec["other"][:, 1:4])
        params["quats"].data.copy_(dec["other"][:, 4:8])
        for opt in optimizers.values():  # training is over; moments of the encoded model reset
            opt.state.clear()
        report = codec.size_report(data, files, len(params["means"]), len(d["idx"]))
        report["encode_seconds"] = round(time.perf_counter() - t0, 2)
        self.encoded = report
        if self.cfg["keep_zip"]:
            self.zip = torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()
        self.vq = None
        self.check_leaves(params)
        self._log({"lightgaussian": "encode", "step": step,
                   **{k: v for k, v in report.items() if k != "file_bytes"}})

    # --- helpers ---------------------------------------------------------------------------------
    @staticmethod
    def check_leaves(params):
        for name, p in params.items():
            if not (p.is_leaf and p.requires_grad):
                raise RuntimeError(f"field {name} is no longer a trainable leaf")

    def _log(self, record):
        self.history.append(record)
        print(json.dumps(record), flush=True)


def create_density(spec):
    return LightGaussianDensity(spec)
