"""DOF-GS density control, Mip 3D filter schedule, xyz LR and the per-view DOF bank.

Port of DOFGS_implementation @cfd635c training.py:266-313 and gaussian_model.py:

  stats       step < densify_until, radii > 0: accum += ||means2D.grad[:, :2]||, denom += 1
              (defocused render; the official kernel's NDC screen gradient)
  densify     step > densify_from, step % 100 == 0: clone (grad >= 2e-4, max s <= 0.03 extent,
              plain copy), split (post-clone, 2 samples, s/1.6), prune sigmoid(o) < 0.005.
              use_size_prune is false in the configs -> no screen-size / world-size prune.
  compactness step % 1000 == 0 (step < densify_until): densify_by_shrink_then_compatness
              (K=3 nearest neighbours incl. self, raw log-scales / 1.5, verbatim official math)
  filter      recompute after any densification, after extra points, and every 100 steps for
              densify_until < step < total - 100 (band_limit only; without it the official
              filter is computed but never used, so it is skipped)
  reset       step % 3000 == 0 (step < densify_until): band_limit -> filtered opacity clamped
              to 0.01 (reset_opacity_with_filter3D); else plain logit(min(sigmoid(o), 0.01))
              (reset_opacity), training.py:286-289
  extra pts   step 2000: 2M uniform samples in the +-25 % bbox, kept if nearest of k=4 SfM
              neighbours < 15, IDW-interpolated SH/opacity, iso scale = log(min dist)
  xyz lr      get_expon_lr_func(5e-4 * extent -> 5e-6 * extent, max_steps 80000)
  bank        f_m/Q_m Adam (lr 0.05 / 0.01) from step 100 to 15000, ILN Adam (5e-4)
Difference to the official loop: the runtime applies the Gaussian optimizer step before this
controller runs. Officially densify/prune/compactness/reset replace every Gaussian parameter
(grad None) before optimizer.step, so on those steps the official run skips the Adam step for
all Gaussians (~245 steps of 40k); the runtime steps them.
"""

import math
import sys

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove, split
from gsplat.utils import normalized_quat_to_rotmat

from . import bank as bank_mod
from .filters import compute_filter_3d, reset_logits

PYLIB = "/home/sejain/models/dofgs-pylib"  # torch_cluster 1.6.3+pt24cu121 (pip --target)

DEFAULTS = {
    "densify_from_iter": 500,
    "densify_until_iter": 25000,
    "densification_interval": 100,
    "densify_grad_threshold": 0.0002,
    "percent_dense": 0.03,
    "min_opacity": 0.005,
    "opacity_reset_interval": 3000,
    "opacity_reset_value": 0.01,
    "band_limit": True,
    "use_compactness": True,
    "compactness_interval": 1000,
    "compactness_K": 3,
    "shrink_factor": 1.5,
    "add_point_iterations": [2000],
    "num_extra_points": 30000,
    "extra_points_factor": 0.25,
    "extra_points_k": 4,
    "extra_points_dist_thres": 15.0,
    "filter_coef": 0.2,
    "filter_near": 0.2,
    "filter_margin": 0.15,
    "filter_interval": 100,
    "total_steps": 40000,
    "position_lr_init": 0.0005,
    "position_lr_final": 0.000005,
    "position_lr_delay_mult": 0.01,
    "position_lr_delay_steps": 0,
    "position_lr_max_steps": 80000,
    "scaling_lr": 0.01,
    # per-view DOF camera + ILN (bank)
    "coc_max": 15.0,
    "fdist_lr": 0.05,
    "kapert_lr": 0.01,
    "mask_nn_lr": 0.0005,
    "iter_defocus_start": 100,
    "iter_stop_cam_gd": 15000,
}
REQUIRES = {"gradient.screen.signed"}


def torch_cluster():
    try:
        import torch_cluster as tc
    except ImportError:
        if PYLIB not in sys.path:
            sys.path.append(PYLIB)
        import torch_cluster as tc
    return tc


def expon_lr(step, lr_init, lr_final, delay_steps=0, delay_mult=1.0, max_steps=30000):
    """utils/general_utils.get_expon_lr_func."""
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        return 0.0
    if delay_steps > 0:
        delay_rate = delay_mult + (1 - delay_mult) * math.sin(
            0.5 * math.pi * min(max(step / delay_steps, 0), 1))
    else:
        delay_rate = 1.0
    t = min(max(step / max_steps, 0), 1)
    return delay_rate * math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


@torch.no_grad()
def append(params, optimizers, new):
    """densification_postfix: concatenate new Gaussians (zero Adam moments)."""
    n_new = len(new["means"])

    def param_fn(name, p):
        return torch.nn.Parameter(torch.cat([p, new[name].to(p.dtype)]),
                                  requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), device=v.device)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


def distance_to_gaussian_surface(mean, svec, rotmat, query):
    """src/utils/ops.py verbatim (svec receives raw log-scales / shrink in the official call)."""
    xyz = query - mean
    xyz = torch.einsum("bij,bj->bi", rotmat.transpose(-1, -2), xyz)
    xyz = torch.nn.functional.normalize(xyz, dim=-1)
    z, y, x = xyz[..., 2], xyz[..., 1], xyz[..., 0]
    r_xy = torch.sqrt(x**2 + y**2 + 1e-10)
    cos_theta, sin_theta = z, r_xy
    cos_phi, sin_phi = x / r_xy, y / r_xy
    d2 = svec[..., 0] ** 2 * cos_phi**2 + svec[..., 1] ** 2 * sin_phi**2
    r2 = svec[..., 2] ** 2 * cos_theta**2 + d2**2 * sin_theta**2
    return torch.sqrt(r2 + 1e-10)


@torch.no_grad()
def compactness_candidates(params, nn_idx, shrink_factor):
    """densify_by_compatnes_with_idx for one neighbour column (official math, raw log svec)."""
    xyz = params["means"]
    svec = params["scales"] / shrink_factor
    rotmat = normalized_quat_to_rotmat(torch.nn.functional.normalize(params["quats"], dim=-1))
    nn_svec, nn_rot, nn_pos = svec[nn_idx], rotmat[nn_idx], xyz[nn_idx]
    nn_surf = distance_to_gaussian_surface(nn_pos, nn_svec, nn_rot, xyz)
    surf = distance_to_gaussian_surface(xyz, svec, rotmat, nn_pos)
    dist = torch.norm(nn_pos - xyz, dim=-1)
    mask = (surf + nn_surf) < dist
    direction = (nn_pos - xyz) / dist[..., None]
    new = {"means": (xyz + direction * (dist + surf - nn_surf)[..., None] / 2.0)[mask]}
    for name in params:
        if name not in ("means", "scales"):
            new[name] = params[name][mask]
    gap = ((dist - surf - nn_surf)[mask][..., None]) / 6.0
    new["scales"] = torch.log(torch.clamp(torch.ones_like(params["scales"][mask]) * gap,
                                          1e-6, 10.0))
    if "filter_3d" in new:
        new["filter_3d"] = torch.zeros_like(new["filter_3d"])
    return new


@torch.no_grad()
def compactness_densify(params, optimizers, K=3, shrink_factor=1.5):
    """densify_by_shrink_then_compatness. The official `self._scaling / shrink` assignment is
    discarded by cat_tensors_to_optimizer (it re-reads the optimizer's parameter), so the shrink
    only enters the candidate computation."""
    xyz = params["means"]
    edge = torch_cluster().knn(xyz, xyz, int(K + 1), cosine=False, num_workers=4)
    idx = edge[1].reshape(-1, int(K + 1))
    blocks = [compactness_candidates(params, idx[:, i], shrink_factor) for i in range(K)]
    new = {k: torch.cat([b[k] for b in blocks]) for k in blocks[0]}
    if len(new["means"]):
        append(params, optimizers, new)
    return len(new["means"])


@torch.no_grad()
def extra_points(params, optimizers, M=30000, k=4, dist_thres=15.0, range_factor=0.25):
    """gaussian_model.add_extra_points."""
    xyz = params["means"]
    lo, hi = xyz.min(dim=0).values, xyz.max(dim=0).values
    lo = lo - range_factor * lo.abs()
    hi = hi + range_factor * hi.abs()
    n = int(M * 2)
    pts = torch.stack([torch.empty(n, device=xyz.device).uniform_(float(lo[i]), float(hi[i]))
                       for i in range(3)], dim=1)
    edge = torch_cluster().knn(xyz, pts, k, cosine=False, num_workers=1)
    nbr = edge[1].reshape(-1, k)
    dist = torch.norm(pts.unsqueeze(1) - xyz[nbr], dim=2)
    valid = dist.min(dim=1).values < dist_thres
    nbr, dist = nbr[valid], dist[valid]
    w = 1 / (dist + 1e-5)
    w = w / w.sum(dim=1, keepdim=True)
    new = {"means": pts[valid],
           "scales": torch.log(torch.clamp(dist.min(dim=1).values, 1e-6, 10.0))[..., None]
           .repeat(1, 3)}
    quats = torch.zeros(len(new["means"]), 4, device=xyz.device)
    quats[:, 0] = 1.0
    new["quats"] = quats
    new["sh0"] = (w[..., None, None] * params["sh0"][nbr]).sum(dim=1)
    new["shN"] = (w[..., None, None] * params["shN"][nbr]).sum(dim=1)
    new["opacities"] = (w * params["opacities"][nbr]).sum(dim=1)
    for name in params:
        if name not in new:
            new[name] = torch.zeros((len(new["means"]), *params[name].shape[1:]),
                                    device=xyz.device)
    append(params, optimizers, new)
    return len(new["means"])


@torch.no_grad()
def clamp_overflowed_moments(optimizers):
    """Finite stand-in for the official run's +inf Adam second moments (float32 g**2 overflow).

    Needle Gaussians (one axis ~1e3, another ~1e-2, far from the camera; cake from step ~12k)
    get per-element quats/scales/means gradients of 1e16..1e19 from the official kernel. Once
    |g| > 1.8e19, g**2 overflows and exp_avg_sq = +inf. The official code (plain torch Adam, no
    state checks) keeps that inf forever (beta2 * inf = inf), so the element's update is
    m / inf = 0 for the rest of training: it is frozen. The SplatStudio runtime rejects non-finite
    moments, so here such entries are pinned to float32 max (sentinel: > max / 2, unreachable
    without overflow) on every step and their exp_avg is zeroed. Each later Adam step then moves
    them by about lr * 0.1 * |g| / 1.8e19 (~1e-4 lr at |g| ~ 1e17): the same freeze to float noise.
    NaN moments are left untouched so a genuine bug still fails check_state.
    Returns the number of pinned entries.
    """
    counts = []
    for opt in optimizers.values():
        for state in opt.state.values():
            v, m = state.get("exp_avg_sq"), state.get("exp_avg")
            if v is None or not v.is_floating_point():
                continue
            big = torch.finfo(v.dtype).max
            frozen = v > 0.5 * big  # includes +inf; False for NaN
            counts.append(frozen.sum())
            v.masked_fill_(frozen, big)
            if m is not None and m.shape == v.shape:
                m.masked_fill_(frozen, 0.0)
    return int(torch.stack(counts).sum()) if counts else 0  # one host sync per step


def plain_reset_logits(opacity_logits, value=0.01):
    """gaussian_model.reset_opacity; logit(min(sigmoid(o), v)) == min(o, logit(v))."""
    return torch.minimum(opacity_logits, torch.logit(torch.tensor(value)).to(opacity_logits))


class Density:
    requires = REQUIRES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown DOF-GS density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.cfg["band_limit"] = bool(self.cfg["band_limit"])
        self.requires = set(REQUIRES)
        self.state = {}
        self.stats = None
        self.extent = None
        self.params = None
        self.cameras = None
        self.bank = None
        self.counts = {"filter_updates": 0, "resets": 0, "densify": 0, "compactness_added": 0,
                       "extra_added": 0, "moments_clamped": 0}

    # -- runtime hooks -------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        self.extent = float(scene_scale)  # 3dgs recipe: getNerfppNorm radius
        self.params = params
        optimizers["scales"].param_groups[0]["lr"] = float(self.cfg["scaling_lr"])
        optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(0)
        self._reset_stats(len(params["means"]), params["means"].device)
        self.state = {"extent": self.extent, "counts": self.counts}
        if self.cameras:
            self._setup()

    def set_training_cameras(self, cameras):
        cameras = list(cameras)
        if not cameras:
            raise ValueError("DOF-GS needs the training cameras")
        self.cameras = cameras
        if self.params is not None:
            self._setup()

    def _setup(self):
        params = self.params
        if self.cfg["band_limit"]:
            self.update_filter(params)  # official compute_3D_filter before iteration 1
        dev = params["means"].device
        centers = torch.stack([torch.linalg.inv(c["viewmat"].float())[:3, 3]
                               for c in self.cameras]).to(dev)
        translate_z = float(-centers.mean(dim=0)[2])  # getNerfppNorm translate = -center
        self.bank = bank_mod.DofBank(self.cameras, params["means"].detach()[:, 2], translate_z,
                                     self.cfg, dev)
        bank_mod.ACTIVE = self.bank
        self.state["bank"] = self.bank.export()

    def before_backward(self, params, optimizers, step, outputs):
        if self.bank is None:
            raise RuntimeError("DOF-GS needs density.set_training_cameras(...) first")
        if step == 1:
            optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(1)

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        # entry-steps pinned (a frozen entry counts once per step)
        self.counts["moments_clamped"] += clamp_overflowed_moments(optimizers)
        self.bank.after_step(step)
        if step < c["densify_until_iter"]:
            self.accumulate(outputs["info"])
            densified = False
            if step > c["densify_from_iter"] and step % c["densification_interval"] == 0:
                self.densify_and_prune(params, optimizers)
                densified = True
            if c["use_compactness"] and step % c["compactness_interval"] == 0:
                added = compactness_densify(params, optimizers, c["compactness_K"],
                                            c["shrink_factor"])
                self.counts["compactness_added"] += added
                self._reset_stats(len(params["means"]), params["means"].device)
                densified = True
            if densified and c["band_limit"]:
                self.update_filter(params)
            if step % c["opacity_reset_interval"] == 0:
                self.reset_opacity(params, optimizers)
        if step in c["add_point_iterations"]:
            added = extra_points(params, optimizers, c["num_extra_points"], c["extra_points_k"],
                                 c["extra_points_dist_thres"], c["extra_points_factor"])
            self.counts["extra_added"] += added
            self._reset_stats(len(params["means"]), params["means"].device)
            if c["band_limit"]:
                self.update_filter(params)
        if (c["band_limit"] and step % c["filter_interval"] == 0
                and c["densify_until_iter"] < step < c["total_steps"] - c["filter_interval"]):
            self.update_filter(params)
        optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(step + 1)
        if step % 1000 == 0 or step == c["total_steps"]:
            self.state["bank"] = self.bank.export()

    # -- pieces ----------------------------------------------------------------------------
    def xyz_lr(self, step):
        c = self.cfg
        return expon_lr(step, c["position_lr_init"] * self.extent,
                        c["position_lr_final"] * self.extent, c["position_lr_delay_steps"],
                        c["position_lr_delay_mult"], c["position_lr_max_steps"])

    def _reset_stats(self, n, dev):
        self.stats = {"grad_accum": torch.zeros(n, device=dev), "denom": torch.zeros(n, device=dev)}

    def accumulate(self, info):
        grad = info["means2d"].grad
        if grad is None:
            raise RuntimeError("DOF-GS density: means2d has no gradient")
        vis = info["radii"] > 0
        if len(vis) != len(self.stats["denom"]):
            raise RuntimeError("DOF-GS density: statistics and population are misaligned")
        self.stats["grad_accum"][vis] += torch.linalg.vector_norm(grad[vis, :2], dim=-1)
        self.stats["denom"][vis] += 1

    def update_filter(self, params):
        dev = params["means"].device
        if self.cameras[0]["viewmat"].device != dev:
            self.cameras = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in cam.items()}
                            for cam in self.cameras]
        c = self.cfg
        f = compute_filter_3d(params["means"], self.cameras, c["filter_coef"], c["filter_near"],
                              c["filter_margin"])
        params["filter_3d"].data.copy_(f.to(params["filter_3d"].dtype))
        self.counts["filter_updates"] += 1

    def reset_opacity(self, params, optimizers):
        value = self.cfg["opacity_reset_value"]
        if self.cfg["band_limit"]:  # reset_opacity_with_filter3D
            logits = reset_logits(params["scales"], params["opacities"], params["filter_3d"],
                                  value)
        else:  # reset_opacity: inverse_sigmoid(min(sigmoid(o), 0.01))
            logits = plain_reset_logits(params["opacities"], value)

        def param_fn(name, p):
            return torch.nn.Parameter(logits.to(p.dtype), requires_grad=p.requires_grad)

        _update_param_with_optimizer(param_fn, lambda key, v: torch.zeros_like(v), params,
                                     optimizers, ["opacities"])
        self.counts["resets"] += 1

    @torch.no_grad()
    def densify_and_prune(self, params, optimizers):
        c, st = self.cfg, self.stats
        grads = st["grad_accum"] / st["denom"]
        grads[grads.isnan()] = 0.0
        thr, dense = c["densify_grad_threshold"], c["percent_dense"] * self.extent
        n0 = len(grads)
        smax = params["scales"].exp().max(dim=1).values
        clone = (grads >= thr) & (smax <= dense)
        sel = torch.where(clone)[0]
        append(params, optimizers, {k: v[sel] for k, v in params.items()})
        n_clone = len(sel)
        g_pad = torch.cat([grads, torch.zeros(n_clone, device=grads.device)])
        smax = params["scales"].exp().max(dim=1).values
        split_mask = (g_pad >= thr) & (smax > dense)
        split(params, optimizers, {}, split_mask)
        prune = params["opacities"].sigmoid() < c["min_opacity"]
        remove(params, optimizers, {}, prune)
        n = len(params["means"])
        self._reset_stats(n, params["means"].device)
        self.counts["densify"] += 1
        self.state["last"] = {"before": n0, "clone": n_clone, "split": int(split_mask.sum()),
                              "prune": int(prune.sum()), "after": n}


def create_density(spec):
    return Density(spec)
