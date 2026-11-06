"""AdamW-GS density controller for the vanilla-3DGS host (paper Sec. 4.4).

Responsibilities (official `train.py` / `scene/gaussian_model.py`, 3DGS_AdamWGS):
- replace the runtime's per-field Adam optimizers by `SparseAdamWGS` at `initialize` (the
  runtime has no optimizer factory; it steps whatever lives in its `optimizers` dict);
- per-primitive asynchronous step counter `asystep` (int32 in `self.state`, so gsplat's
  duplicate/split/remove inherit/drop it like the official `asystep[mask].repeat(N)`);
- 3DGS adaptive density control with opacity correction on clone (`opacity_correction`);
- position noise from 3DGS-MCMC (`noise_transport`), applied after the optimizer step;
- Re-State Regularisation (`restate_regularization`, Eq. 5) after noise.
"""

import torch
from gsplat import quat_scale_to_covar_preci
from gsplat.strategy.ops import duplicate, remove, reset_opa, split

from .optim import AdamWGSContext, SparseAdamWGS

DEFAULTS = {
    # Sparse Adam / DAR (adam.cu, rasterize_points.cu, mipnerf_adamwgs*/config.yaml)
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 1e-15,
    "lambda_opacity": 1e-3,
    "lambda_scale": 1e-5,
    "opacity_reg_cap": 10.0,
    "scale_reg_cap": 10.0,
    "opacity_reg_begin": 3000,
    "opacity_reg_end": 15000,
    "scale_reg_begin": 1000,
    "scale_reg_end": 15000,
    # Re-State Regularisation (config restate_reg; StSS GS2 for Mip-NeRF 360 outdoor)
    "rsr": True,
    "rsr_begin": 1000,
    "rsr_end": 15000,
    "rsr_interval": 100,
    "rsr_ratios": [0.1, 0.3, 0.5],
    "rsr_shifts": [1000, 7000, 10000],
    "rsr_alpha1": 0.2,
    # code-effective value: config key `rsr_exeavgsq_fading: 0.04` is never read, the code
    # reads `rsr_exeabg_sq_fading` with default 0.01 (gaussian_model.py:121). Paper: 0.04.
    "rsr_alpha2": 0.01,
    # exploration noise (train.py, 3DGS-MCMC); off for Mip-NeRF 360 indoor configs
    "noise": True,
    "noise_begin": 1000,
    "noise_end": 14000,
    "noise_lr": 5e5,
    "total_steps": 30000,
    # 3DGS adaptive density control (config trainer)
    "opacity_correction": True,
    "grow_grad2d": 0.0002,
    "percent_dense": 0.01,
    "refine_start_iter": 500,
    "refine_stop_iter": 15000,
    "refine_every": 100,
    "prune_opa": 0.005,
    # opacity reset: off in every released AdamW-GS config (interval 300000); 3000 = paper GS8
    "opacity_reset_every": 0,
    "reset_value": 0.01,
    "big_prune_world": 0.1,
}


def rsr_ratio(config, step):
    """Milestone StSS: ratio_list[i] from ratio_shift[i] on (official uni_curriculum)."""
    ratio = 0.0
    for r, s in zip(config["rsr_ratios"], config["rsr_shifts"]):
        if step >= s:
            ratio = r
    return ratio


def rsr_active(config, step):
    return (
        bool(config["rsr"])
        and config["rsr_begin"] <= step <= config["rsr_end"]
        and step % config["rsr_interval"] == 0
    )


def noise_active(config, step):
    return (
        bool(config["noise"])
        and config["noise_begin"] <= step <= config["noise_end"]
        and step < config["total_steps"]
    )


def op_sigmoid(x, k=100.0, x0=0.995):
    return 1.0 / (1.0 + torch.exp(-k * (x - x0)))


class AdamWGSDensity:
    requires = {"gradient.screen.signed"}

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown adamw-gs density options: {sorted(unknown)}")
        c = {**DEFAULTS, **spec.config}
        if len(c["rsr_ratios"]) != len(c["rsr_shifts"]):
            raise ValueError("rsr_ratios and rsr_shifts must have the same length")
        if not (0 <= c["rsr_alpha1"] < 1 and 0 <= c["rsr_alpha2"] < 1):
            raise ValueError("RSR requires 0 <= alpha < 1 (Eq. 5)")
        self.config = c
        self.state = None
        self.context = None
        self.last_rsr = 0

    # ------------------------------------------------------------------ setup
    def initialize(self, params, optimizers, scene_scale=1.0):
        n = len(params["means"])
        device = params["means"].device
        self.state = {
            "grad2d": torch.zeros(n, device=device),
            "count": torch.zeros(n, device=device),
            "asystep": torch.zeros(n, dtype=torch.int32, device=device),
            "scene_scale": float(scene_scale),
        }
        self.context = AdamWGSContext(self.config, self.state)
        c = self.config
        for name in list(optimizers):
            old = optimizers[name]
            if len(old.param_groups) != 1 or len(old.param_groups[0]["params"]) != 1:
                raise ValueError(f"{name}: expected one parameter per optimizer")
            if len(old.state):
                raise ValueError(f"{name}: optimizer already has state; replace before training")
            (param,) = old.param_groups[0]["params"]
            if param is not params[name]:
                raise ValueError(f"{name}: optimizer does not own the current parameter")
            optimizers[name] = SparseAdamWGS(
                param, old.param_groups[0]["lr"], self.context, name,
                betas=(c["beta1"], c["beta2"]), eps=c["eps"],
            )

    def set_training_cameras(self, cameras):
        # official optimizer_init: N_I of training_views[0]
        cam = cameras[0]
        self.context.set_image_pixels(int(cam["width"]) * int(cam["height"]))

    # ------------------------------------------------------------- iteration
    def before_backward(self, params, optimizers, step, outputs):
        info = outputs["info"]
        info["means2d"].retain_grad()
        visible = (info["radii"] > 0).all(dim=-1)[0]
        self.context.begin_iteration(step, visible)

    def after_backward(self, params, optimizers, step, outputs):
        c = self.config
        info = outputs["info"]
        if step < c["refine_stop_iter"]:
            self._update_stats(info)
            if step > c["refine_start_iter"] and step % c["refine_every"] == 0:
                self._densify(params, optimizers, step)
            if c["opacity_reset_every"] > 0 and step % c["opacity_reset_every"] == 0:
                reset_opa(params, optimizers, self.state, value=c["reset_value"])
        if noise_active(c, step):
            self.add_noise(params, optimizers["means"].param_groups[0]["lr"])
        if rsr_active(c, step):
            self.last_rsr = self.restate(params, optimizers, rsr_ratio(c, step))

    # ------------------------------------------------------------ mechanisms
    @torch.no_grad()
    def _update_stats(self, info):
        grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        sel = (info["radii"] > 0).all(dim=-1)
        ids = torch.where(sel)[1]
        self.state["grad2d"].index_add_(0, ids, grads[sel].norm(dim=-1))
        self.state["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    @torch.no_grad()
    def correct_clone_opacity(self, params, mask):
        """Official opacity_correction (clone only): o' = clamp(1 - sqrt(1 - o), 0.005, 1-eps)."""
        o = torch.sigmoid(params["opacities"].data[mask])
        new = torch.clamp(1 - torch.sqrt(1 - o), min=0.005, max=1.0 - torch.finfo(torch.float32).eps)
        params["opacities"].data[mask] = torch.logit(new)

    @torch.no_grad()
    def _densify(self, params, optimizers, step):
        c, st = self.config, self.state
        grads = st["grad2d"] / st["count"].clamp_min(1)
        high = grads >= c["grow_grad2d"]
        small = params["scales"].exp().max(dim=-1).values <= c["percent_dense"] * st["scene_scale"]
        clone = high & small
        n_clone = int(clone.sum())
        if n_clone:
            if c["opacity_correction"]:
                self.correct_clone_opacity(params, clone)
            duplicate(params, optimizers, st, clone)
        is_split = torch.cat([high & ~small, torch.zeros(n_clone, dtype=torch.bool,
                                                          device=high.device)])
        if int(is_split.sum()):
            split(params, optimizers, st, is_split)
        prune = torch.sigmoid(params["opacities"]) < c["prune_opa"]
        if c["opacity_reset_every"] > 0 and step > c["opacity_reset_every"]:
            # official prune(): big_points_vs = max_radii2D > 20 never fires, because both
            # densify_and_clone and densify_and_split call densification_postfix, which
            # re-zeros max_radii2D before prune() reads it; only big_points_ws is effective.
            prune |= (params["scales"].exp().max(dim=-1).values
                      > c["big_prune_world"] * st["scene_scale"])
        if int(prune.sum()):
            remove(params, optimizers, st, prune)
        # official densification_postfix re-zeros the accumulators
        for key in ("grad2d", "count"):
            st[key].zero_()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def add_noise(self, params, lr, noise=None):
        """3DGS-MCMC exploration noise: mu += Sigma (n * op_sigmoid(1-o) * noise_lr * lr)."""
        o = torch.sigmoid(params["opacities"])
        covar, _ = quat_scale_to_covar_preci(params["quats"], params["scales"].exp(),
                                             compute_covar=True, compute_preci=False, triu=False)
        if noise is None:
            noise = torch.randn_like(params["means"])
        n = noise * op_sigmoid(1 - o)[:, None] * self.config["noise_lr"] * lr
        params["means"].add_(torch.bmm(covar, n[..., None])[..., 0])

    @torch.no_grad()
    def restate(self, params, optimizers, ratio):
        """RSR (Eq. 5): uniform sample without replacement, m *= alpha1, v *= alpha2."""
        n = len(params["means"])
        k = int(ratio * n)
        if k <= 0:
            return 0
        idx = torch.randperm(n, device=params["means"].device)[:k]
        a1, a2 = self.config["rsr_alpha1"], self.config["rsr_alpha2"]
        for name, opt in optimizers.items():
            state = opt.state.get(params[name])
            if not state:
                continue
            state["exp_avg"][idx] *= a1
            state["exp_avg_sq"][idx] *= a2
        return k


def create_density(spec):
    return AdamWGSDensity(spec)

