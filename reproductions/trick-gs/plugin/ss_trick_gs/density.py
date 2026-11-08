"""Trick-GS density control.

3DGS adaptive density control (Inria `densify_and_prune`: clone small / split large Gaussians
with screen-gradient >= 2e-4, every 100 steps in (500, 15000), opacity prune 0.005, world-size
prune 0.1 extent after 3000, opacity reset every 3000) plus the Trick-GS tricks:
- SC (§3.2.4c, §4.2): start from the 20 % SfM points with the lowest reprojection error
  (`set_training_cameras`, sidecar from make_sfm_errors.py; 3-NN scales recomputed on the subset
  like distCUDA2 on the loaded cloud) and ABE split for the first 10000 steps (RAIN-GS code,
  github.com/cvlab-kaist/RAIN-GS gaussian_model.densify_and_split: every split candidate also
  emits a copy at 0.3 * extent * mu, B_0 = origin). The low-pass part lives in the renderer.
- GM (§3.2.1, §4.2): Gaussians with sigmoid(m) <= eps_m pruned at every densification step and
  every 500 steps after densification stops.
- SG (§3.2.2, §4.2): significance pruning at `sg_steps` with rate 0.6 * 0.7^k (significance.py).
- DE (§4.2): densification re-enabled every 100 steps in (20000, 20500], statistics restarted at
  20000.
- AT (§3.2.5): `shN` Adam steps on 16-step gradient sums (IntervalAdam, Taming 3DGS).
- SHM post-processing: masked SH bands zeroed at the last step (bands removed after training);
  FP16 storage (§4.1): every parameter rounded to half precision at the last step.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from gsplat.strategy.ops import _update_param_with_optimizer, duplicate, remove, reset_opa
from gsplat.utils import normalized_quat_to_rotmat

from . import schedule, significance
from .optim import IntervalAdam
from .renderer import GS_MASK, SH_MASK, expand_sh, scaled_camera
from .shared import STATE

DEFAULT_ERROR_DIR = str(Path(__file__).resolve().parents[2] / "sfm_errors")
DEFAULTS = {
    "total_steps": 30000,
    # 3DGS densification (arguments/__init__.py, train.py)
    "densify_from": 500,
    "densify_until": 15000,
    "densify_every": 100,
    "grad_threshold": 0.0002,
    "percent_dense": 0.01,
    "split_divisor": 1.6,
    "prune_opacity": 0.005,
    "prune_big_world": 0.1,
    "prune_big_after": 3000,
    "reset_every": 3000,
    "reset_value": 0.01,
    # SC
    "init_fraction": 0.2,
    "sfm_error_dir": DEFAULT_ERROR_DIR,
    "abe_split": True,
    "abe_until": 10000,
    "abe_factor": 0.3,
    "progressive_lowpass": True,
    "lowpass_until": 10000,
    "lowpass_every": 1000,
    "lowpass_min": 0.3,
    "lowpass_max": 300.0,
    # DS / BL
    "ds_start": 0.125,
    "ds_until": 19500,
    "ds_mode": "linear",
    "blur": True,
    "blur_radius": 4,
    "blur_sigma": 2.4,
    "blur_every": 100,
    "blur_until": 19500,
    # GM / SHM
    "gs_threshold": 0.05,
    "sh_threshold": 0.1,
    "mask_prune_every": 500,
    "sh_mask_start": 3500,
    # SG
    "sg_steps": (5000, 10000, 15000, 20000, 22000, 23000),
    "sg_first": 0.6,
    "sg_decay": 0.7,
    "sg_v_pow": 0.1,
    # DE
    "late_densify": True,
    "redensify": (20000, 20500),
    # AT
    "sh_update_every": 16,
    # post-processing
    "fp16_final": True,
}


def means_sha256(means):
    return hashlib.sha256(means.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def knn3_log_scales(points, budget=2**27):
    """3DGS distCUDA2: log sqrt(mean squared distance to the 3 nearest neighbours), isotropic."""
    n = len(points)
    out = torch.empty(n, device=points.device)
    chunk = max(1, budget // max(n, 1))
    for s in range(0, n, chunk):
        d2 = torch.cdist(points[s : s + chunk], points).square()
        d2[torch.arange(d2.shape[0]), torch.arange(s, s + d2.shape[0])] = float("inf")
        out[s : s + chunk] = d2.topk(min(3, n - 1), dim=1, largest=False).values.mean(dim=1)
    return out.clamp_min(1e-7).sqrt().log()[:, None].expand(-1, 3).contiguous()


def lowest_error_subset(errors, fraction):
    """Indices (ascending order) of the ceil(fraction N) points with the smallest SfM error."""
    k = max(2, int(math.ceil(fraction * len(errors))))
    order = np.argsort(np.asarray(errors), kind="stable")[:k]
    return np.sort(order)


@torch.no_grad()
def replace_population(params, optimizers, new):
    """Swap every field for new rows (fresh trainable leaves, empty optimizer state)."""
    for name, value in new.items():
        old = params[name]
        p = torch.nn.Parameter(value.to(old.device, old.dtype).contiguous(), requires_grad=True)
        params[name] = p
        opt = optimizers[name]
        opt.state.pop(old, None)
        opt.param_groups[0]["params"] = [p]


@torch.no_grad()
def abe_append(params, optimizers, state, mask, factor):
    """RAIN-GS ABE-Split third Gaussian: mean * factor (0.3 * extent), other fields copied."""
    sel = torch.where(mask)[0]

    def param_fn(name, p):
        rows = p[sel] * factor if name == "means" else p[sel]
        return torch.nn.Parameter(torch.cat([p.detach(), rows]), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, v.new_zeros((len(sel), *v.shape[1:]))])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for k, v in list(state.items()):
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, v[sel]))


@torch.no_grad()
def split_gaussians(params, optimizers, state, mask, divisor):
    """3DGS densify_and_split: 2 samples from N(mu, Sigma), scale / (0.8 * 2), parent removed."""
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]
    scales = torch.exp(params["scales"][sel])
    quats = torch.nn.functional.normalize(params["quats"][sel], dim=-1)
    rotmats = normalized_quat_to_rotmat(quats)
    samples = torch.einsum("nij,nj,bnj->bni", rotmats, scales,
                           torch.randn(2, len(scales), 3, device=scales.device))

    def param_fn(name, p):
        if name == "means":
            rows = (p[sel] + samples).reshape(-1, 3)
        elif name == "scales":
            rows = torch.log(scales / divisor).repeat(2, 1)
        else:
            rows = p[sel].repeat([2] + [1] * (p.dim() - 1))
        return torch.nn.Parameter(torch.cat([p[rest].detach(), rows]),
                                  requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v[rest], v.new_zeros((2 * len(sel), *v.shape[1:]))])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for k, v in list(state.items()):
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v[rest], v[sel].repeat([2] + [1] * (v.dim() - 1))))


def gs_mask_prune(params, threshold):
    if GS_MASK not in params:
        return torch.zeros(len(params["means"]), dtype=torch.bool, device=params["means"].device)
    return torch.sigmoid(params[GS_MASK].detach()) <= threshold


class TrickDensity:
    requires = {"gradient.screen.signed"}

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown trick-density options: {sorted(unknown)}")
        cfg = {**DEFAULTS, **spec.config}
        cfg["sg_steps"] = tuple(int(s) for s in cfg["sg_steps"])
        cfg["redensify"] = tuple(int(s) for s in cfg["redensify"])
        if cfg["ds_mode"] not in ("linear", "log"):
            raise ValueError("ds_mode must be linear or log")
        self.cfg = cfg
        self.state = None
        self.cameras = None

    # ---------------------------------------------------------------- setup
    def initialize(self, params, optimizers, scene_scale=1.0):
        self.params, self.optimizers = params, optimizers
        self.state = {"grad2d": None, "count": None}
        self.scene_scale = float(scene_scale)
        self.events = []
        if self.cfg["sh_update_every"] > 1 and not isinstance(optimizers["shN"], IntervalAdam):
            old = optimizers["shN"]
            optimizers["shN"] = IntervalAdam([params["shN"]], lr=old.param_groups[0]["lr"],
                                             every=int(self.cfg["sh_update_every"]),
                                             eps=old.param_groups[0]["eps"])
        STATE.activate(self.cfg)

    @torch.no_grad()
    def set_training_cameras(self, cameras):
        self.cameras = list(cameras)
        frac = float(self.cfg["init_fraction"])
        if frac >= 1.0:
            return
        params = self.params
        key = means_sha256(params["means"])
        path = Path(self.cfg["sfm_error_dir"]) / f"{key}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"trick-gs: no SfM error sidecar {path} for this point cloud; run "
                "plugin/make_sfm_errors.py on the scene's COLMAP points3D.bin (or set "
                "init_fraction 1.0, which is not the paper protocol)")
        errors = np.load(path)
        if len(errors) != len(params["means"]):
            raise ValueError("trick-gs: SfM error sidecar length differs from the point cloud")
        keep = torch.as_tensor(lowest_error_subset(errors, frac), device=params["means"].device)
        new = {name: p.detach()[keep].clone() for name, p in params.items()}
        new["scales"] = knn3_log_scales(new["means"])
        replace_population(params, self.optimizers, new)
        self.state.update(grad2d=None, count=None)
        self.events.append({"step": 0, "init_kept": int(len(keep)), "init_total": len(errors),
                            "max_error_kept": float(np.asarray(errors)[keep.cpu().numpy()].max())})

    # ------------------------------------------------------------ per step
    def before_backward(self, params, optimizers, step, outputs):
        trick = outputs.get("trick") or {}
        if trick.get("step") is not None and trick["step"] != step:
            raise RuntimeError(f"trick-gs: renderer schedule step {trick['step']} != {step}")
        outputs["info"]["means2d"].retain_grad()

    def _update_stats(self, params, info):
        grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        n = len(params["means"])
        if self.state["grad2d"] is None or len(self.state["grad2d"]) != n:
            self.state["grad2d"] = torch.zeros(n, device=grads.device)
            self.state["count"] = torch.zeros(n, device=grads.device)
        sel = (info["radii"] > 0.0).all(dim=-1)
        ids = torch.where(sel)[1]
        self.state["grad2d"].index_add_(0, ids, grads[sel].norm(dim=-1))
        self.state["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    def _remove(self, params, optimizers, mask, what):
        n = int(mask.sum())
        if n:
            if n == len(mask):
                raise RuntimeError(f"trick-gs {what} would remove every Gaussian")
            remove(params, optimizers, self.state, mask)
        return n

    @torch.no_grad()
    def densify(self, params, optimizers, step):
        cfg, state = self.cfg, self.state
        extent = self.scene_scale
        n0 = len(params["means"])
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n0, device=params["means"].device)
            state["count"] = torch.zeros(n0, device=params["means"].device)
        grads = state["grad2d"] / state["count"]
        grads[grads.isnan()] = 0.0
        high = grads >= cfg["grad_threshold"]
        big = torch.exp(params["scales"]).max(dim=-1).values > cfg["percent_dense"] * extent
        is_clone, is_split = high & ~big, high & big
        n_clone, n_split = int(is_clone.sum()), int(is_split.sum())
        if n_clone:
            duplicate(params, optimizers, state, is_clone)
        is_split = torch.cat([is_split, is_split.new_zeros(n_clone)])
        n_abe = 0
        if cfg["abe_split"] and step <= cfg["abe_until"] and n_split:
            abe_append(params, optimizers, state, is_split, cfg["abe_factor"] * extent)
            n_abe = n_split
            is_split = torch.cat([is_split, is_split.new_zeros(n_abe)])
        if n_split:
            split_gaussians(params, optimizers, state, is_split, cfg["split_divisor"])
        prune = torch.sigmoid(params["opacities"].flatten()) < cfg["prune_opacity"]
        if step > cfg["prune_big_after"]:
            prune |= torch.exp(params["scales"]).max(dim=-1).values > cfg["prune_big_world"] * extent
        masked = gs_mask_prune(params, cfg["gs_threshold"])
        prune |= masked
        n_masked = int(masked.sum())
        n_prune = self._remove(params, optimizers, prune, "densification prune")
        state["grad2d"].zero_()
        state["count"].zero_()
        return {"step": step, "clone": n_clone, "split": n_split, "abe": n_abe,
                "prune": n_prune, "mask_prune": n_masked, "total": len(params["means"])}

    @torch.no_grad()
    def significance_prune(self, params, optimizers, step):
        cfg = self.cfg
        if self.cameras is None:
            raise RuntimeError("trick-gs significance pruning needs set_training_cameras")
        k = cfg["sg_steps"].index(step)
        rate = schedule.sg_rate(k, cfg["sg_first"], cfg["sg_decay"])
        cams = []
        for cam in self.cameras:
            h, w = schedule.ds_size(step, int(cam["height"]), int(cam["width"]),
                                    cfg["ds_start"], cfg["ds_until"], cfg["ds_mode"])
            cams.append(scaled_camera(cam, h, w))
        scales = params["scales"].detach().exp()
        opac = params["opacities"].detach().sigmoid()
        if GS_MASK in params:
            opac = opac * (torch.sigmoid(params[GS_MASK].detach()) > cfg["gs_threshold"]).float()
        _, score = significance.hit_scores(params["means"].detach(), params["quats"].detach(),
                                           scales, opac, cams, eps2d=STATE.lowpass)
        v = significance.v_importance(score, scales, cfg["sg_v_pow"])
        mask = significance.prune_mask(v, rate)
        n = self._remove(params, optimizers, mask, "significance prune")
        return {"step": step, "sg_round": k, "sg_rate": rate, "sg_prune": n,
                "total": len(params["means"])}

    @torch.no_grad()
    def finalize(self, params):
        """Bake SH masks (masked bands zeroed) and store every parameter in FP16."""
        if SH_MASK in params:
            hard = (torch.sigmoid(params[SH_MASK]) > self.cfg["sh_threshold"]).float()
            params["shN"].mul_(expand_sh(hard))
        if self.cfg["fp16_final"]:
            for p in params.values():
                p.copy_(p.half().float())

    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        if step == cfg["redensify"][0] and cfg["late_densify"] and self.state["grad2d"] is not None:
            self.state["grad2d"].zero_()
            self.state["count"].zero_()
        if schedule.in_stats_window(step, cfg) and (cfg["late_densify"] or step < cfg["densify_until"]):
            self._update_stats(params, outputs["info"])
        densified = False
        if schedule.densify_due(step, cfg):
            self.events.append(self.densify(params, optimizers, step))
            densified = True
        if schedule.reset_due(step, cfg):
            reset_opa(params, optimizers, self.state, value=cfg["reset_value"])
        if not densified and schedule.mask_prune_due(step, cfg):
            with torch.no_grad():
                n = self._remove(params, optimizers, gs_mask_prune(params, cfg["gs_threshold"]),
                                 "mask prune")
            self.events.append({"step": step, "mask_prune": n, "total": len(params["means"])})
        if step in cfg["sg_steps"]:
            self.events.append(self.significance_prune(params, optimizers, step))
        if step == cfg["total_steps"]:
            self.finalize(params)
        STATE.next_step = step + 1
        for name, p in params.items():
            if not (p.is_leaf and p.requires_grad):
                raise RuntimeError(f"trick-gs: field {name} is no longer a trainable leaf")


def create_density(spec):
    return TrickDensity(spec)
