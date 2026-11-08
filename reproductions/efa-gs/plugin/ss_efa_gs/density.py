"""EFA-GS density controller on the Mip-Splatting (GOF densification) base.

Port of the official EFA-GS `Mip-Splatting/train.py` densification block and
`scene/gaussian_model.py` (`add_densification_stats_lff`, `densify_and_prune_lff`,
`densify_lff`, `densify_and_clone`, `densify_and_split`, `densification_postfix_lff`,
`prune_points_lff`, `reset_opacity`, `compute_3D_filter`), commit 57f330f.

Per-Gaussian statistics (all [N], aligned with params):
  g, g_abs, cnt           accumulated since the last densification (official xyz_gradient_accum*)
  lff_g, lff_g_abs, lff_cnt  accumulated since the last LFCF step (official lff_xyz_grad_accum*)
  prev_g, prev_g_abs      mean gradients at the previous LFCF step (official prev_lff_xyz_grad*)
  prev_sel                selected at the previous LFCF step (official prev_selected_pts_mask)

Population changes are written with gsplat's `_update_param_with_optimizer` (new Parameters keep
the original `requires_grad`; new Adam moments are zero). LFCF scale edits are in place on the
log-scale data and keep the Adam moments, like the official `set_attributes`.
"""

import torch
import torch.nn.functional as F
from gsplat.strategy.ops import _update_param_with_optimizer
from gsplat.utils import normalized_quat_to_rotmat

from .lfcf import (
    abs_threshold,
    anneal_fraction,
    anneal_lower_bound,
    enlarge_delta,
    is_lfcf_step,
    lfcf_masks,
    log_enlarging_factor,
    normalize_log_minmax,
    shrink_delta,
    split_probability,
    splitting_lower_bound,
)
from .mip_filters import compute_filter_3d, is_filter_step, is_reset_step, reset_logits

KEYS = {
    "grad_threshold", "densify_from", "densify_until", "densify_every", "percent_dense",
    "prune_opacity", "prune_world", "opacity_reset_every", "opacity_reset_value", "filter_coef",
    "near", "frustum_margin", "filter_interval", "total_steps", "lfcf", "interval_times",
    "c_max", "c_min", "c_last", "pow", "splitting_ub", "split_multiplier", "tolerance",
    "diffscale", "depth_strategy", "lfcf_children",
}
FLOAT_STATS = ("g", "g_abs", "cnt", "lff_g", "lff_g_abs", "lff_cnt", "prev_g", "prev_g_abs")


class EFADensity:
    def __init__(self, spec):
        cfg = dict(spec.config)
        unknown = set(cfg) - KEYS
        missing = KEYS - set(cfg)
        if unknown or missing:
            raise ValueError(f"efa-density config: unknown {sorted(unknown)} missing {sorted(missing)}")
        self.cfg = cfg
        if cfg["c_max"] < cfg["c_min"] or cfg["c_min"] < 1.0:
            raise ValueError("need c_max >= c_min >= 1 (official asserts)")
        if int(cfg["lfcf_children"]) != 1:
            raise ValueError("official LFCF splits with N = 1")
        self.lower = anneal_lower_bound(cfg["c_last"], cfg["c_max"])
        self.requires = {"gradient.screen.signed", "gradient.screen.absolute"}
        self.state = {"lfcf_steps": 0, "gof_steps": 0, "enlarged": 0, "shrunk": 0,
                      "lfcf_children": 0, "cloned": 0, "split": 0, "pruned": 0,
                      "nan_pruned": 0, "resets": 0, "filter_updates": 0}
        self.stats = None
        self.cameras = None
        self.params = None
        self.scene_scale = None
        self.trainable = None
        self.last_lfcf = None  # evidence of the latest LFCF step (used by probes)

    # ------------------------------------------------------------------ setup
    def set_training_cameras(self, cameras):
        cameras = list(cameras)
        if not cameras:
            raise ValueError("EFA-GS needs at least one training camera")
        self.cameras = cameras
        if self.params is not None:
            self.update_filter(self.params)

    def _cams(self, device):
        if not self.cameras:
            raise RuntimeError(
                "EFA-GS needs the training cameras: the runtime must call "
                "density.set_training_cameras(...) before training"
            )
        if self.cameras[0]["viewmat"].device != device:
            self.cameras = [
                {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in c.items()}
                for c in self.cameras
            ]
        return self.cameras

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.params = params
        self.scene_scale = float(scene_scale)
        self.trainable = {k: p.requires_grad for k, p in params.items()}
        n, dev = len(params["means"]), params["means"].device
        self.stats = {k: torch.zeros(n, device=dev) for k in FLOAT_STATS}
        # Official training_setup: at first all Gaussians count as previously selected.
        self.stats["prev_sel"] = torch.ones(n, dtype=torch.bool, device=dev)
        if self.cameras:
            self.update_filter(params)

    def update_filter(self, params):
        cams = self._cams(params["means"].device)
        f = compute_filter_3d(params["means"], cams, self.cfg["filter_coef"], self.cfg["near"],
                              self.cfg["frustum_margin"])
        params["filter_3d"].data.copy_(f.to(params["filter_3d"].dtype))
        self.state["filter_updates"] += 1

    def sampling_interval(self, params):
        """Official compute_3D_interval: min visible depth / max focal (filter without sqrt(s))."""
        cams = self._cams(params["means"].device)
        return compute_filter_3d(params["means"], cams, 1.0, self.cfg["near"],
                                 self.cfg["frustum_margin"])

    # --------------------------------------------------------------- per step
    def before_backward(self, params, optimizers, step, outputs):
        self._cams(params["means"].device)
        outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def accumulate(self, info):
        """Official add_densification_stats_lff (NDC-scaled screen gradients, radii > 0)."""
        xy = info["means2d"]
        if xy.grad is None or getattr(xy, "absgrad", None) is None:
            raise RuntimeError("EFA-GS needs signed and absolute screen-space gradients")
        scale = torch.tensor([info["width"] / 2.0, info["height"] / 2.0], device=xy.device)
        scale = scale * info["n_cameras"]
        grad = xy.grad * scale  # [C, N, 2]
        absg = xy.absgrad * scale
        sel = (info["radii"] > 0).all(dim=-1)  # [C, N]
        ids = torch.where(sel)[1]
        g = grad[sel].norm(dim=-1)
        # Official rasterizer: grad[:, 2] += |dL/dx| + |dL/dy| per pixel (NDC). gsplat's absgrad
        # sums |.| per axis, so the official scalar is the L1 norm of the scaled absgrad.
        ga = absg[sel].abs().sum(dim=-1)
        ones = torch.ones_like(g)
        s = self.stats
        for a, b, v in (("g", "lff_g", g), ("g_abs", "lff_g_abs", ga), ("cnt", "lff_cnt", ones)):
            s[a].index_add_(0, ids, v)
            s[b].index_add_(0, ids, v)

    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        if step < c["densify_until"]:
            self.accumulate(outputs["info"])
            if step > c["densify_from"] and step % c["densify_every"] == 0:
                if c["lfcf"] and is_lfcf_step(step, c["interval_times"], c["densify_every"]):
                    self.lfcf(params, optimizers, step)
                else:
                    self.gof_densify(params, optimizers)
                self.prune(params, optimizers, step)
                self.update_filter(params)
            if is_reset_step(step, c["opacity_reset_every"], c["densify_until"]):
                self.reset_opacity(params, optimizers)
        self.prune_nan(params, optimizers)
        if step > c["densify_until"] and is_filter_step(
            step, c["filter_interval"], c["densify_from"], c["densify_until"], c["total_steps"]
        ):
            self.update_filter(params)

    # ---------------------------------------------------------- population ops
    def _check_trainable(self, params):
        for k, p in params.items():
            if not p.is_leaf or p.requires_grad != self.trainable[k]:
                raise RuntimeError(f"field {k} is no longer a trainable leaf after a density op")

    @torch.no_grad()
    def _append(self, params, optimizers, new, stat_rows):
        """Append rows (dict field -> tensor) and their statistics (dict key -> tensor)."""
        k = len(new["means"])
        if k == 0:
            return

        def param_fn(name, p):
            return torch.nn.Parameter(torch.cat([p.detach(), new[name].to(p.dtype)]),
                                      requires_grad=p.requires_grad)

        def opt_fn(key, v):
            return torch.cat([v, torch.zeros((k, *v.shape[1:]), device=v.device, dtype=v.dtype)])

        _update_param_with_optimizer(param_fn, opt_fn, params, optimizers)
        for key, rows in stat_rows.items():
            self.stats[key] = torch.cat([self.stats[key], rows])
        self._check_trainable(params)

    @torch.no_grad()
    def _remove(self, params, optimizers, mask):
        if not mask.any():
            return
        keep = torch.where(~mask)[0]

        def param_fn(name, p):
            return torch.nn.Parameter(p.detach()[keep], requires_grad=p.requires_grad)

        _update_param_with_optimizer(param_fn, lambda key, v: v[keep], params, optimizers)
        for key in self.stats:
            self.stats[key] = self.stats[key][keep]
        self._check_trainable(params)

    def _child_stats(self, k, device):
        """Official densification_postfix_lff for new Gaussians: zero stats, prev_sel True."""
        rows = {key: torch.zeros(k, device=device) for key in FLOAT_STATS}
        rows["prev_sel"] = torch.ones(k, dtype=torch.bool, device=device)
        return rows

    @staticmethod
    def _mean(acc, cnt):
        m = acc / cnt
        m[m.isnan()] = 0.0  # official grads[grads.isnan()] = 0 (0/0 for unseen Gaussians)
        return m

    # ------------------------------------------------- GOF densify (non-LFCF)
    @torch.no_grad()
    def gof_densify(self, params, optimizers):
        """Official densify_and_clone + densify_and_split (Mip-Splatting GOF densification)."""
        c, s = self.cfg, self.stats
        grads, grads_abs = self._mean(s["g"], s["cnt"]), self._mean(s["g_abs"], s["cnt"])
        q = abs_threshold(grads, grads_abs, c["grad_threshold"])
        hot = (grads >= c["grad_threshold"]) | (grads_abs >= q)
        max_scale = torch.exp(params["scales"]).max(dim=1).values
        limit = c["percent_dense"] * self.scene_scale
        clone = hot & (max_scale <= limit)
        n0, dev = len(grads), grads.device
        # Clone (same parameters).
        sel = torch.where(clone)[0]
        self._append(params, optimizers, {k: p[sel] for k, p in params.items()},
                     self._child_stats(len(sel), dev))
        # Split (N = 2, s / 1.6, originals removed); clones have zero padded gradients.
        pad_g = torch.cat([grads, torch.zeros(len(sel), device=dev)])
        pad_a = torch.cat([grads_abs, torch.zeros(len(sel), device=dev)])
        max_scale = torch.exp(params["scales"]).max(dim=1).values
        split = ((pad_g >= c["grad_threshold"]) | (pad_a >= q)) & (max_scale > limit)
        ids = torch.where(split)[0]
        scales = torch.exp(params["scales"][ids])
        rot = normalized_quat_to_rotmat(F.normalize(params["quats"][ids], dim=-1))
        new = {}
        for k, p in params.items():
            new[k] = p[ids].repeat(2, *[1] * (p.dim() - 1))
        samples = torch.randn(2 * len(ids), 3, device=dev) * scales.repeat(2, 1)
        new["means"] = torch.einsum("nij,nj->ni", rot.repeat(2, 1, 1), samples) + new["means"]
        new["scales"] = torch.log(scales.repeat(2, 1) / (0.8 * 2))
        self._append(params, optimizers, new, self._child_stats(2 * len(ids), dev))
        prune = torch.cat([split, torch.zeros(2 * len(ids), dtype=torch.bool, device=dev)])
        self._remove(params, optimizers, prune)
        # Official postfix resets the per-densification accumulators for everyone.
        for key in ("g", "g_abs", "cnt"):
            self.stats[key].zero_()
        self.state["gof_steps"] += 1
        self.state["cloned"] += len(sel)
        self.state["split"] += len(ids)
        return {"n_before": n0, "cloned": len(sel), "split": len(ids)}

    # ------------------------------------------------------------------- LFCF
    @torch.no_grad()
    def lfcf(self, params, optimizers, step):
        """Official densify_lff (LFCF, Alg. 1 with the depth/scale strategies, App. C 1-3)."""
        c, s = self.cfg, self.stats
        dev = params["means"].device
        grads, grads_abs = self._mean(s["lff_g"], s["lff_cnt"]), self._mean(
            s["lff_g_abs"], s["lff_cnt"]
        )
        q = abs_threshold(grads, grads_abs, c["grad_threshold"])
        selected, enlarged, shrunk, both = lfcf_masks(
            grads, grads_abs, c["grad_threshold"], q, s["prev_sel"], s["prev_g"],
            s["prev_g_abs"], c["tolerance"],
        )
        theta = normalize_log_minmax(self.sampling_interval(params))
        fraction = anneal_fraction(step, c["densify_from"], c["densify_until"], c["pow"],
                                   self.lower)
        log_c = log_enlarging_factor(theta, c["c_min"], c["c_max"], fraction,
                                     c["depth_strategy"])[:, None]
        log_s = params["scales"].data
        log_s[enlarged] += enlarge_delta(log_s[enlarged], log_c[enlarged], c["diffscale"])
        log_s[shrunk] += shrink_delta(log_s[shrunk], log_c[shrunk], c["split_multiplier"],
                                      c["diffscale"])
        # Probabilistic split (strategy 3); shrunk Gaussians that lose the draw stay shrunk.
        lower = splitting_lower_bound(step, c["densify_from"], c["densify_until"])
        thr = split_probability(theta, lower, c["splitting_ub"])
        draw = torch.rand(int(both.sum()), device=dev) <= thr[both]
        split = shrunk.clone()
        split[both] = shrunk[both] & draw
        ids = torch.where(split)[0]
        stds = torch.exp(params["scales"][ids])
        stds[stds.isnan()] = 1e-3
        rot = normalized_quat_to_rotmat(F.normalize(params["quats"][ids], dim=-1))
        new = {k: p[ids] for k, p in params.items()}
        offset = torch.einsum("nij,nj->ni", rot, torch.randn_like(stds) * stds)
        new["means"] = new["means"] + offset
        # Official postfix (islff=True): prev grads <- current LFCF grads (children 0),
        # prev selection <- current selection (children True), LFCF + densify accumulators reset.
        s["prev_g"], s["prev_g_abs"] = grads, grads_abs
        s["prev_sel"] = selected
        n0 = len(grads)
        self._append(params, optimizers, new, self._child_stats(len(ids), dev))
        for key in ("g", "g_abs", "cnt", "lff_g", "lff_g_abs", "lff_cnt"):
            self.stats[key].zero_()
        self.state["lfcf_steps"] += 1
        self.state["enlarged"] += int(enlarged.sum())
        self.state["shrunk"] += int(shrunk.sum())
        self.state["lfcf_children"] += len(ids)
        self.last_lfcf = {"step": step, "n_before": n0, "enlarged": enlarged, "shrunk": shrunk,
                          "split": split, "selected": selected, "log_c": log_c[:, 0],
                          "fraction": fraction, "q_abs": float(q)}
        return self.last_lfcf

    # --------------------------------------------------------------- pruning
    @torch.no_grad()
    def nan_mask(self, params):
        mask = torch.zeros(len(params["means"]), dtype=torch.bool, device=params["means"].device)
        for k in ("means", "scales", "quats", "opacities", "sh0", "shN"):
            p = params[k]
            mask |= p.reshape(len(p), -1).isnan().any(dim=1)
        return mask

    @torch.no_grad()
    def prune(self, params, optimizers, step):
        """Official prune in densify_and_prune_lff.

        sigma(o) < 0.005 (unfiltered opacity) or NaN; after the first reset interval also the
        world-size prune (max scale > 0.1 extent). The official screen-size prune never fires
        (max_radii2D is zeroed by every densification postfix) and is not implemented.
        """
        c = self.cfg
        mask = torch.sigmoid(params["opacities"]) < c["prune_opacity"]
        if step > c["opacity_reset_every"]:
            mask |= torch.exp(params["scales"]).max(dim=1).values > c["prune_world"] * self.scene_scale
        mask |= self.nan_mask(params)
        self.state["pruned"] += int(mask.sum())
        self._remove(params, optimizers, mask)

    @torch.no_grad()
    def prune_nan(self, params, optimizers):
        """Official prune_nan_points (every iteration)."""
        mask = self.nan_mask(params)
        if mask.any():
            self.state["nan_pruned"] += int(mask.sum())
            self._remove(params, optimizers, mask)

    def reset_opacity(self, params, optimizers):
        """Official filter-aware reset_opacity (Adam moments zeroed)."""
        with torch.no_grad():
            logits = reset_logits(params["scales"], params["opacities"], params["filter_3d"],
                                  self.cfg["opacity_reset_value"])

        def param_fn(name, p):
            return torch.nn.Parameter(logits.to(p.dtype), requires_grad=p.requires_grad)

        _update_param_with_optimizer(param_fn, lambda key, v: torch.zeros_like(v), params,
                                     optimizers, ["opacities"])
        self._check_trainable(params)
        self.state["resets"] += 1


def create_density(spec):
    return EFADensity(spec)
