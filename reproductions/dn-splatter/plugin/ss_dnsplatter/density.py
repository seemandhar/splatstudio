"""Splatfacto (Nerfstudio 1.1.3) adaptive density control as used by DN-Splatter
(splatfacto.py:after_train/refinement_after/cull_gaussians/split_gaussians/dup_gaussians,
overridden refinement_after in dn_model.py:271-385).

Nerfstudio steps are 0-based and the callbacks run after the optimizer step; the native runtime
is 1-based, so every schedule below uses ``ns = step - 1``.

* statistics every step while ns < stop_split_at: sum of |absgrad| norms of visible Gaussians,
  visibility counts starting at 1, max screen radius / max(H, W). gsplat 1.5 reports per-axis
  radii; the max-2D size uses the gsplat 1.0 eigen radius ceil(3 sqrt(lambda_max)) recomputed
  from the (blurred) conics so the 0.05 / 0.15 thresholds see the same numbers as gsplat 1.0.
* refinement when ns % refine_every == 0 and ns > warmup (stats are NOT cleared on the early
  return during warm-up):
  - densify if ns < stop_split_at and ns % (reset_alpha_every*refine_every) > num_train +
    refine_every: grad = sum/count * 0.5 * max(H, W) > densify_grad_thresh; split if max scale >
    densify_size_thresh (or max2D > split_screen_size before stop_screen_size_at), into
    n_split_samples copies (means + R(s*randn), scales / 1.6); the originals' scales are divided
    by 1.6 in place *before* the duplicate mask is computed (official order); duplicate if
    max scale <= densify_size_thresh; new Adam moments are zero; then cull (with the split
    originals).
  - else if ns >= stop_split_at and continue_cull_post_densification: cull only.
  - cull: sigmoid(opacity) < cull_alpha_thresh; if ns > refine_every*reset_alpha_every also
    max scale > cull_scale_thresh and (ns < stop_screen_size_at) max2D > cull_screen_size.
  - opacity reset when ns < stop_split_at and ns % reset_interval == refine_every: clamp to
    logit(2 * cull_alpha_thresh) and zero both Adam moments of the opacities.
  - statistics are cleared after every refinement call past warm-up.
* initialize: the means LR is set to the official 1.6e-4 (Splatfacto does not multiply it by
  the scene extent; the runtime's exponential decay to 1 % then matches lr_final 1.6e-6), and the
  sidecar log-scales / normal-aligned quaternions are loaded when the means equal the sidecar
  points. Official zero 3-NN distances give log(0) = -inf; ``zero_log_scale`` (-1e4) keeps the
  state finite while exp() still underflows to exactly 0 (identical rendering and zero
  gradients).
* ``sidecar_init``: "oriented" = DN-Splatter (dn_model.py:186-214: log(avg) on x/y, log(avg/10)
  on z, quats rotating z onto the Open3D normal); "isotropic" = vanilla Splatfacto
  (splatfacto.py:214-220: ``k_nearest_sklearn(means, 3)`` -> mean of the 3 non-self distances ->
  log(avg_dist) on all three axes, ``random_quat_tensor(N)``) from the same sidecar points.
  init.npz column 0 is exactly log(avg) (column 2 = log(avg) - log 10), so the isotropic scales
  are column 0 repeated; quats use the official Shoemake formula on the CPU torch RNG.
"""

import math

import torch
from gsplat.strategy.ops import _update_param_with_optimizer
from gsplat.utils import normalized_quat_to_rotmat

from . import sidecar

DEFAULTS = {
    "warmup_length": 500,
    "refine_every": 100,
    "reset_alpha_every": 30,
    "stop_split_at": 15000,
    "densify_grad_thresh": 0.0008,
    "densify_size_thresh": 0.01,
    "n_split_samples": 2,
    "split_screen_size": 0.05,
    "stop_screen_size_at": 4000,
    "cull_alpha_thresh": 0.1,
    "cull_scale_thresh": 0.5,
    "cull_screen_size": 0.15,
    "continue_cull_post_densification": True,
    "means_lr": 1.6e-4,
    "use_sidecar_init": True,
    "sidecar_init": "oriented",
    "zero_log_scale": -1.0e4,
}


def v1_radius(conics):
    """gsplat 1.0 projection radius from the inverse 2D covariance [..., 3] (a, b, c)."""
    a, b, c = conics.unbind(-1)
    det_c = a * c - b * b
    ok = det_c > 0
    det_c = torch.where(ok, det_c, torch.ones_like(det_c))
    cxx, cxy, cyy = c / det_c, -b / det_c, a / det_c
    mid = 0.5 * (cxx + cyy)
    det = cxx * cyy - cxy * cxy
    v1 = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
    return torch.where(ok, torch.ceil(3.0 * torch.sqrt(v1)), torch.zeros_like(v1))


def random_quat_tensor(n, generator=None):
    """splatfacto.py:random_quat_tensor (uniform unit quaternions, same component order)."""
    u, v, w = (torch.rand(n, generator=generator) for _ in range(3))
    return torch.stack([
        torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
        torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
        torch.sqrt(u) * torch.sin(2 * math.pi * w),
        torch.sqrt(u) * torch.cos(2 * math.pi * w),
    ], dim=-1)


def init_scales_quats(init, mode):
    """(log_scales [N,3], quats [N,4]) for the sidecar init mode (may contain -inf scales)."""
    if mode == "oriented":
        return init["log_scales"], init["quats"]
    if mode == "isotropic":
        log_avg = init["log_scales"][:, :1]
        return log_avg.repeat(1, 3), random_quat_tensor(len(log_avg))
    raise ValueError(f"sidecar_init must be 'oriented' or 'isotropic', got {mode!r}")


class Density:
    requires = {"gradient.screen.absolute"}

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown splatfacto density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        if self.cfg["sidecar_init"] not in ("oriented", "isotropic"):
            raise ValueError(f"sidecar_init must be 'oriented' or 'isotropic', "
                             f"got {self.cfg['sidecar_init']!r}")
        self.state = {"num_train": None, "init_from_sidecar": False, "refinements": 0,
                      "sidecar_init": self.cfg["sidecar_init"]}
        self.grad = None
        self.count = None
        self.max2d = None
        self.last_size = None

    # ------------------------------------------------------------------ setup
    def set_training_cameras(self, cameras):
        self.state["num_train"] = len(cameras)

    def initialize(self, params, optimizers, scene_scale=1.0):
        for g in optimizers["means"].param_groups:
            g["lr"] = float(self.cfg["means_lr"])
        if not self.cfg["use_sidecar_init"]:
            return
        tab = sidecar.table()
        init = tab.init() if tab is not None else None
        if init is None:
            return
        pts = init["points"].to(params["means"].device)
        if pts.shape != params["means"].shape or not torch.equal(pts, params["means"].detach()):
            raise ValueError("Sidecar init points differ from the initial means; train with "
                             "--num-gaussians >= the sidecar point count (all points kept)")
        ls, quats = init_scales_quats(init, self.cfg["sidecar_init"])
        with torch.no_grad():
            ls = ls.to(pts.device)
            ls = torch.where(torch.isfinite(ls), ls, torch.full_like(ls, self.cfg["zero_log_scale"]))
            params["scales"].copy_(ls)
            params["quats"].copy_(quats.to(pts.device))
        self.state["init_from_sidecar"] = True

    # --------------------------------------------------------------- per step
    def before_backward(self, params, optimizers, step, outputs):
        outputs["info"]["means2d"].retain_grad()

    def _accumulate(self, params, info):
        radii = info["radii"][0]
        visible = (radii > 0).all(dim=-1) if radii.dim() == 2 else radii > 0
        H, W = int(info["height"]), int(info["width"])
        self.last_size = (H, W)
        absgrad = info["means2d"].absgrad[0]
        n = len(params["means"])
        if self.grad is None:
            self.grad = torch.zeros(n, device=absgrad.device)
            self.count = torch.ones(n, device=absgrad.device)
        if self.max2d is None:
            self.max2d = torch.zeros(n, device=absgrad.device)
        self.count[visible] += 1
        self.grad[visible] += absgrad[visible].norm(dim=-1)
        r = v1_radius(info["conics"][0].detach())[visible]
        self.max2d[visible] = torch.maximum(self.max2d[visible], r / float(max(H, W)))

    def after_backward(self, params, optimizers, step, outputs):
        ns = step - 1
        cfg = self.cfg
        if ns < cfg["stop_split_at"]:
            self._accumulate(params, outputs["info"])
        if ns % cfg["refine_every"] == 0:
            self.refine(params, optimizers, ns)

    # ------------------------------------------------------------- refinement
    @torch.no_grad()
    def refine(self, params, optimizers, ns):
        cfg = self.cfg
        if ns <= cfg["warmup_length"]:
            return
        if self.state["num_train"] is None:
            raise RuntimeError("splatfacto density needs set_training_cameras (num_train)")
        reset_interval = cfg["reset_alpha_every"] * cfg["refine_every"]
        do_densify = (ns < cfg["stop_split_at"]
                      and ns % reset_interval > self.state["num_train"] + cfg["refine_every"])
        removed = None
        if do_densify:
            removed = self._densify_and_cull(params, optimizers, ns)
        elif ns >= cfg["stop_split_at"] and cfg["continue_cull_post_densification"]:
            removed = self._cull(params, optimizers, ns, None)
        if ns < cfg["stop_split_at"] and ns % reset_interval == cfg["refine_every"]:
            self._reset_opacity(params, optimizers)
        self.state["refinements"] += 1
        self.state["last_removed"] = int(removed) if removed is not None else 0
        self.grad = self.count = self.max2d = None

    def _densify_and_cull(self, params, optimizers, ns):
        cfg = self.cfg
        H, W = self.last_size
        avg = (self.grad / self.count) * 0.5 * max(H, W)
        high = avg > cfg["densify_grad_thresh"]
        smax = params["scales"].exp().max(dim=-1).values
        splits = smax > cfg["densify_size_thresh"]
        if ns < cfg["stop_screen_size_at"]:
            splits |= self.max2d > cfg["split_screen_size"]
        splits &= high
        k = int(cfg["n_split_samples"])
        sel = torch.where(splits)[0]
        # split_gaussians: samples around the (pre-shrink) Gaussians
        s = params["scales"][sel].exp().repeat(k, 1)
        q = params["quats"][sel] / params["quats"][sel].norm(dim=-1, keepdim=True)
        rots = normalized_quat_to_rotmat(q.repeat(k, 1))
        samples = torch.bmm(rots, (s * torch.randn_like(s))[..., None]).squeeze(-1)
        new = {name: p[sel].repeat(k, *([1] * (p.dim() - 1))) for name, p in params.items()}
        new["means"] = samples + params["means"][sel].repeat(k, 1)
        shrunk = torch.log(params["scales"][sel].exp() / 1.6)
        new["scales"] = shrunk.repeat(k, 1)
        params["scales"].data[sel] = shrunk  # official in-place update before the dup mask
        dups = (params["scales"].exp().max(dim=-1).values <= cfg["densify_size_thresh"]) & high
        dsel = torch.where(dups)[0]
        dnew = {name: p[dsel] for name, p in params.items()}
        n_new = len(sel) * k + len(dsel)

        def param_fn(name, p):
            return torch.nn.Parameter(torch.cat([p.data, new[name], dnew[name]]),
                                      requires_grad=p.requires_grad)

        def opt_fn(key, v):
            return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), device=v.device,
                                             dtype=v.dtype)])

        _update_param_with_optimizer(param_fn, opt_fn, params, optimizers)
        self.max2d = torch.cat([self.max2d, torch.zeros(n_new, device=self.max2d.device)])
        extra = torch.cat([splits, torch.zeros(n_new, dtype=torch.bool, device=splits.device)])
        self.state["last_split"] = int(len(sel))
        self.state["last_dup"] = int(len(dsel))
        return self._cull(params, optimizers, ns, extra)

    def _cull(self, params, optimizers, ns, extra):
        cfg = self.cfg
        culls = params["opacities"].sigmoid() < cfg["cull_alpha_thresh"]
        if extra is not None:
            culls = culls | extra
        if ns > cfg["refine_every"] * cfg["reset_alpha_every"]:
            toobig = params["scales"].exp().max(dim=-1).values > cfg["cull_scale_thresh"]
            if ns < cfg["stop_screen_size_at"] and self.max2d is not None:
                toobig = toobig | (self.max2d > cfg["cull_screen_size"])
            culls = culls | toobig
        keep = ~culls
        if culls.any():
            def param_fn(name, p):
                return torch.nn.Parameter(p.data[keep], requires_grad=p.requires_grad)

            def opt_fn(key, v):
                return v[keep]

            _update_param_with_optimizer(param_fn, opt_fn, params, optimizers)
        return int(culls.sum())

    def _reset_opacity(self, params, optimizers):
        value = self.cfg["cull_alpha_thresh"] * 2.0
        cap = math.log(value / (1 - value))
        params["opacities"].data.clamp_(max=cap)
        st = optimizers["opacities"].state.get(params["opacities"], {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in st:
                st[key].zero_()
        self.state["last_reset"] = True


def create_density(spec):
    return Density(spec)
