"""Denoising-GS density component: the 3DGS-MCMC backbone (gsplat MCMC ops) plus
C1 momentum-biased exploration (Eqs. m2, msgld), C2 spatial gradient-based denoising (Eqs. m3-m5),
C3 Fisher-uncertainty pruning (Eqs. f2-f3) and C4 spatial coherence refinement (Eq. 4.1 + p6/p7).

The runtime calls after_backward *after* the Adam step and zero_grad, so the step's mean and
scale gradients are captured with one-shot tensor hooks registered in before_backward.
The Fisher pass needs the training images, which the runtime does not expose: they are
reloaded from ss-train's --data (pattern from reproductions/dashgaussian)."""

import math
import sys
import time
from pathlib import Path

import torch
from gsplat import rasterization
from gsplat.strategy import ops

from . import core

DEFAULTS = {
    # 3DGS-MCMC backbone (gsplat MCMCStrategy / official 3dgs-mcmc defaults)
    "cap_max": None,  # None -> paper Table 2 MCMC count for the --data scene
    "noise_lr": 5e5,  # sqrt(2 lambda tau) xi realised as MCMC's lambda_t * noise_lr * Sigma xi
    "refine_start_iter": 500,
    "refine_stop_iter": 25000,
    "refine_every": 100,
    "growth": 1.05,
    "min_opacity": 0.005,
    "init_opacity": 0.5,
    # C1 (Sec. 5.1)
    "momentum": True,
    "momentum_mode": "adam",  # "adam": -alpha lr m^/(sqrt(v^)+eps) (default) | "raw": literal
    "alpha": 0.05,
    "beta1": 0.9,
    # C2
    "spatial_denoise": True,
    "beta2": 0.5,
    "denoise_sign": "descent",  # "descent": along the update (Fig. 2) | "literal": Eq. m4 sign
    "dmu_magnitude": "grad",  # "grad": |dS_j| (log-scale grad) | "adam_step": |exp(s') - exp(s)|_j
    # C3
    "fisher_prune": True,
    "prune_fraction": 0.10,
    "fisher_scale_param": "log",  # "log" (stored parameter) or "activated" (exp)
    # C4
    "scr": True,
    "knn_k": 3,
    "densify_fraction": 0.0005,
    "densify_times": 5,
    "densify_every": 500,
    # schedule (guess, Appendix missing): prune at total_steps - prune_offset
    "total_steps": 35000,
    "prune_offset": 5000,
    "data": None,
    "downscale": None,
}


def _argv_value(flag):
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def render_rgb(params, camera, means=None, scales=None, sh_degree=3):
    """Same rasterization as splatstudio.backends.gsplat.Renderer (white background)."""
    means = params["means"] if means is None else means
    scales = params["scales"].exp() if scales is None else scales
    colors = torch.cat((params["sh0"], params["shN"]), dim=1)
    render, alpha, _ = rasterization(
        means=means, quats=params["quats"], scales=scales,
        opacities=params["opacities"].sigmoid(), colors=colors,
        viewmats=camera["viewmat"][None], Ks=camera["K"][None],
        width=camera["width"], height=camera["height"], sh_degree=sh_degree,
        packed=False, render_mode="RGB",
    )
    return render[0] + (1 - alpha[0])


def fisher_uncertainty(params, cameras, images, scale_param="log", sh_degree=3):
    """Eqs. f2-f3: u_i = trace(sum_v g_v g_v^T), g_v = d||I_v - I^_v||^2 / d(M_i, S_i)."""
    u = torch.zeros(len(params["means"]), device=params["means"].device)
    for cam, img in zip(cameras, images):
        m = params["means"].detach().requires_grad_(True)
        if scale_param == "log":
            s_raw = params["scales"].detach().requires_grad_(True)
            s = s_raw.exp()
        else:
            s_raw = params["scales"].detach().exp().requires_grad_(True)
            s = s_raw
        with torch.enable_grad():
            rgb = render_rgb({k: v.detach() for k, v in params.items()}, cam, m, s, sh_degree)
            loss = (rgb - img.to(rgb.device)).square().sum()
            gm, gs = torch.autograd.grad(loss, (m, s_raw))
        u += gm.square().sum(-1) + gs.square().sum(-1)
    return core.uncertainty_from_sq_norms(u)


@torch.no_grad()
def split_selected(params, optimizers, state, idxs, binoms, min_opacity):
    """MCMC densification (Eqs. p6/p7) at chosen sources: like gsplat ops.sample_add but with
    the sources given (sparsest primitives) instead of opacity-sampled."""
    if torch.unique(idxs).numel() != idxs.numel():
        raise ValueError("split_selected expects unique sources (each split into N = 2)")
    opacities = torch.sigmoid(params["opacities"]).flatten()
    eps = torch.finfo(torch.float32).eps
    # Stable N = 2 closed form (gsplat compute_relocation overflows to inf for o < ~1e-7).
    new_o, new_s = core.split_in_two(opacities[idxs], torch.exp(params["scales"])[idxs])
    new_o = torch.clamp(new_o, max=1.0 - eps, min=min_opacity)
    if params["opacities"].dim() == 2:
        new_o = new_o[:, None]

    def param_fn(name, p):
        if name == "opacities":
            p[idxs] = torch.logit(new_o)
        elif name == "scales":
            p[idxs] = torch.log(new_s)
        return torch.nn.Parameter(torch.cat([p, p[idxs]]), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, torch.zeros((len(idxs), *v.shape[1:]), device=v.device)])

    ops._update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for k, v in state.items():
        state[k] = torch.cat((v, torch.zeros((len(idxs), *v.shape[1:]), device=v.device)))


class DenoisingDensity:
    requires = set()

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown denoising-density options: {sorted(unknown)}")
        self.c = {**DEFAULTS, **spec.config}
        if self.c["fisher_scale_param"] not in ("log", "activated"):
            raise ValueError("fisher_scale_param must be 'log' or 'activated'")
        if self.c["momentum_mode"] not in ("adam", "raw"):
            raise ValueError("momentum_mode must be 'adam' or 'raw'")
        if self.c["denoise_sign"] not in ("descent", "literal"):
            raise ValueError("denoise_sign must be 'descent' or 'literal'")
        if self.c["dmu_magnitude"] not in ("grad", "adam_step"):
            raise ValueError("dmu_magnitude must be 'grad' or 'adam_step'")
        self.mom_updates = 0
        self._scales_before = None
        self.prune_step, self.scr_steps = core.schedule(
            int(self.c["total_steps"]), int(self.c["prune_offset"]),
            int(self.c["densify_every"]), int(self.c["densify_times"]))
        self.cameras = None
        self.images = None  # optional injected images (probes); else reloaded from --data
        self.aux = {}
        self.binoms = None
        self.cap = None
        self.history = []
        self._hooks = []
        self._grads = {}

    @property
    def state(self):
        return {"binoms": self.binoms, "cap": self.cap, "history": list(self.history)}

    # ---- set-up ------------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        for key in ("means", "scales", "quats", "opacities"):
            if key not in params or key not in optimizers:
                raise ValueError(f"denoising-density requires the {key} field")
        cap = self.c["cap_max"]
        if cap is None:
            root = self.c["data"] or _argv_value("--data")
            if root is None:
                # Not a training run (contract probes): use gsplat MCMCStrategy's default budget.
                cap, source = 1_000_000, "no-data-default"
            else:
                scene = Path(root).name
                if scene not in core.SCENE_CAPS:
                    raise ValueError(f"No paper MCMC budget for scene {scene!r}; set cap_max")
                cap, source = core.SCENE_CAPS[scene], f"paper-table2-mcmc:{scene}"
        else:
            source = "config"
        self.cap = int(cap)
        steps_arg = _argv_value("--steps")
        if steps_arg is not None and (self.c["fisher_prune"] or self.c["scr"]) and int(
                steps_arg) != int(self.c["total_steps"]):
            raise ValueError(f"denoising-density total_steps={self.c['total_steps']} but ss-train "
                             f"--steps {steps_arg}: the prune/SCR schedule would be skipped")
        self.history.append({"op": "cap", "cap": self.cap, "source": source})
        with torch.no_grad():
            o = float(self.c["init_opacity"])
            params["opacities"].fill_(math.log(o / (1 - o)))
        n_max = 51
        binoms = torch.zeros((n_max, n_max))
        for n in range(n_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)
        self.binoms = binoms.to(params["means"].device)
        self.aux = {"momentum": torch.zeros_like(params["means"]).detach()}

    def set_training_cameras(self, cameras):
        self.cameras = cameras

    def set_training_images(self, images):
        self.images = images

    def _load_images(self):
        if self.images is not None:
            return self.images
        root = self.c["data"] or _argv_value("--data")
        if root is None:
            raise RuntimeError("denoising-density Fisher pass needs the training images: set "
                               "config 'data' or run through ss-train --data")
        from splatstudio.data import load_dataset

        downscale = int(self.c["downscale"] or _argv_value("--downscale") or 1)
        dataset = load_dataset(Path(root), downscale)
        names = dataset["splits"]["train"]
        if self.cameras is None or len(names) != len(self.cameras):
            raise RuntimeError("Reloaded training split does not match the runtime's cameras")
        images = []
        for name, cam in zip(names, self.cameras):
            frame = dataset["frames"][name]
            dev = cam["K"].device
            if not (frame["width"] == cam["width"] and frame["height"] == cam["height"]
                    and torch.allclose(frame["K"].to(dev), cam["K"])
                    and torch.allclose(frame["viewmat"].to(dev), cam["viewmat"])):
                raise RuntimeError(f"Reloaded training view {name} differs from the runtime camera")
            images.append(frame["image"])  # HWC, CPU; moved per view
        return images

    # ---- runtime callbacks -------------------------------------------------------------
    def before_backward(self, params, optimizers, step, outputs):
        self._remove_hooks()
        self._grads = {}
        info = outputs.get("info") if isinstance(outputs, dict) else None
        if info is not None and "means2d" in info and info["means2d"].requires_grad:
            info["means2d"].retain_grad()  # as the built-in MCMCDensity (screen-gradient probe)
        for key in ("means", "scales"):
            p = params[key]
            if p.requires_grad:
                self._hooks.append(p.register_hook(self._store(key)))
        if self.c["spatial_denoise"] and self.c["dmu_magnitude"] == "adam_step":
            self._scales_before = params["scales"].detach().exp().clone()  # before the Adam step

    def _store(self, key):
        def hook(grad):
            self._grads[key] = grad.detach()

        return hook

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def after_backward(self, params, optimizers, step, outputs):
        self._remove_hooks()
        lr = optimizers["means"].param_groups[0]["lr"]
        c = self.c
        with torch.no_grad():
            g_m = self._grads.get("means")
            g_s = self._grads.get("scales")
            if g_m is None:
                g_m = torch.zeros_like(params["means"])
            if g_s is None:
                g_s = torch.zeros_like(params["scales"])
            means = params["means"]
            if c["momentum"]:  # C1, Eqs. m2 / msgld
                denom, bias = None, 1.0
                if c["momentum_mode"] == "adam":
                    denom = self._adam_denom(params, optimizers)
                    if denom is None:  # no Adam moments yet (before the first step): no drift
                        denom = torch.full_like(params["means"], float("inf"))
                    if self.mom_updates > 0:
                        bias = 1.0 - float(c["beta1"]) ** self.mom_updates
                offset, self.aux["momentum"] = core.momentum_drift(
                    self.aux["momentum"], g_m, lr, float(c["alpha"]), float(c["beta1"]),
                    denom, bias)
                self.mom_updates += 1
                means.add_(offset)
            if c["spatial_denoise"]:  # C2, Eqs. m3-m5
                mag = None
                if c["dmu_magnitude"] == "adam_step":
                    mag = params["scales"].detach().exp() - self._scales_before
                world, _ = core.denoise_term(g_m, g_s, params["quats"], c["denoise_sign"], mag)
                means.add_(float(c["beta2"]) * world)
                self._scales_before = None
        self._grads = {}
        if c["refine_start_iter"] < step < c["refine_stop_iter"] and step % c["refine_every"] == 0:
            self._relocate(params, optimizers)
            self._grow(params, optimizers)
            torch.cuda.empty_cache()
        if c["noise_lr"] > 0:  # sqrt(2 lambda_t tau) xi, MCMC's opacity-aware Sigma-shaped noise
            ops.inject_noise_to_position(params, optimizers, {}, scaler=lr * float(c["noise_lr"]))
        if c["fisher_prune"] and step == self.prune_step:
            self.prune(params, optimizers, step, sh_degree=min((step - 1) // 1000, 3))
        if c["scr"] and step in self.scr_steps:
            self.refine(params, optimizers, step)

    # ---- MCMC backbone -----------------------------------------------------------------
    @torch.no_grad()
    def _relocate(self, params, optimizers):
        dead = torch.sigmoid(params["opacities"].flatten()) <= self.c["min_opacity"]
        if dead.any():
            ops.relocate(params, optimizers, self.aux, dead, self.binoms, self.c["min_opacity"])
            self.aux["momentum"][dead] = 0  # dead slots now hold new primitives

    @torch.no_grad()
    def _grow(self, params, optimizers):
        n = len(params["means"])
        n_new = max(0, min(self.cap, int(self.c["growth"] * n)) - n)
        if n_new > 0:
            ops.sample_add(params, optimizers, self.aux, n_new, self.binoms, self.c["min_opacity"])

    # ---- C3 / C4 -----------------------------------------------------------------------
    @staticmethod
    def _adam_denom(params, optimizers):
        """sqrt(v^) + eps of the means Adam (bias-corrected second moment)."""
        opt = optimizers["means"]
        group = opt.param_groups[0]
        st = opt.state.get(params["means"], {})
        if "exp_avg_sq" not in st:
            return None
        t = float(st["step"])
        v_hat = st["exp_avg_sq"] / (1.0 - group["betas"][1] ** t)
        return v_hat.sqrt() + group["eps"]

    def prune(self, params, optimizers, step, sh_degree=3):
        t0 = time.perf_counter()
        images = self._load_images()
        u = fisher_uncertainty(params, self.cameras, images, self.c["fisher_scale_param"],
                               sh_degree)
        mask = core.prune_mask(u, float(self.c["prune_fraction"]))
        n0 = len(params["means"])
        ops.remove(params, optimizers, self.aux, mask)
        if self.images is None:
            del images
        self.history.append({"step": step, "op": "fisher-prune", "before": n0,
                             "removed": int(mask.sum()), "after": len(params["means"]),
                             "seconds": round(time.perf_counter() - t0, 2)})
        print(f"[denoising-gs] {self.history[-1]}", flush=True)
        torch.cuda.empty_cache()

    def refine(self, params, optimizers, step):
        t0 = time.perf_counter()
        d = core.knn_sparsity(params["means"], int(self.c["knn_k"]))
        idxs = core.sparsest(d, float(self.c["densify_fraction"]))
        n0 = len(params["means"])
        split_selected(params, optimizers, self.aux, idxs, self.binoms, self.c["min_opacity"])
        self.history.append({"step": step, "op": "scr-split", "before": n0, "added": len(idxs),
                             "after": len(params["means"]),
                             "seconds": round(time.perf_counter() - t0, 2)})
        print(f"[denoising-gs] {self.history[-1]}", flush=True)


def create_density(spec):
    return DenoisingDensity(spec)
