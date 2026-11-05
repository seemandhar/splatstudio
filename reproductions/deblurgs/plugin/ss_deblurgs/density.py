"""DeblurGS density control, xyz LR, opacity parameterisation and the trajectory bank.

Port of scene/gaussian_model.py + train.py:170-200 of github.com/taekkii/deblurgs @e0587475:

* opacity: raw identity value (Clamp activation): init 0.1, reset every 3000 (< 75000) to
  min(clamp(o,0,1), 0.1) with zeroed Adam moments, prune clamp(o) < 0.005 only (no screen- or
  world-size prune).
* statistics (every sub-frame): grad_accum += ||d L / d means2D|| (Inria NDC units = gsplat
  pixels * (W/2, H/2)) where visible, denom += 1 / n_subframes (add_densification_stats).
* Gaussian Densification Annealing (paper Sec. 3.3): clone/split threshold theta(step) =
  expon(4e-4 -> 2e-4, max_steps 25000); densify when step > 500 and step % 200 == 0, while
  step < 75000; clone if max scale <= 0.01 * extent, split (N=2, /1.6) otherwise.
* scene extent (dataset_readers.getNerfppNorm, DeblurGS version):
  min(percentile_10 ||C - mean(SfM)||, 1.1 max ||C - mean(C)||); xyz lr 1.6e-4 -> 1.6e-6
  times extent over 150k (log-linear).
* the bank (Bezier trajectories + nu) is created in ``set_training_cameras`` and stepped after
  every runtime step; exported to ``density_state["bank"]``.
"""

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, duplicate, remove, split

from . import bank as bank_mod
from .bank import BANK_DEFAULTS, expon

REQUIRES = ("gradient.screen.signed",)
DEFAULTS = {
    "total_steps": 150000,
    "densify_from_iter": 500,
    "densification_interval": 200,
    "densify_until_iter": 75000,
    "densify_grad_threshold_init": 4e-4,
    "densify_grad_threshold_final": 2e-4,
    "densify_annealing_until": 25000,
    "anneal_threshold": True,
    "opacity_reset_interval": 3000,
    "reset_opacity_value": 0.1,
    "min_opacity": 0.005,
    "percent_dense": 0.01,
    "init_opacity": 0.1,
    "position_lr_init": 1.6e-4,
    "position_lr_final": 1.6e-6,
    "scaling_lr": 0.005,
    **{f"bank_{k}": v for k, v in BANK_DEFAULTS.items()},
}


def official_extent(camera_centers, points):
    """getNerfppNorm(cam_info, pcd) of the DeblurGS code."""
    d_pcd = torch.linalg.vector_norm(camera_centers - points.mean(dim=0), dim=1)
    radius1 = float(torch.quantile(d_pcd.double(), 0.10))  # np.percentile linear
    d_cam = torch.linalg.vector_norm(camera_centers - camera_centers.mean(dim=0), dim=1)
    radius2 = 1.1 * float(d_cam.max())
    return min(radius1, radius2)


def densify_threshold(step, cfg):
    if not cfg["anneal_threshold"]:
        return cfg["densify_grad_threshold_init"]
    return expon(step, cfg["densify_grad_threshold_init"], cfg["densify_grad_threshold_final"],
                 cfg["densify_annealing_until"])


@torch.no_grad()
def reset_opacity(params, optimizers, value):
    """reset_opacity: raw = clamp(min(clamp(o,0,1), value)); Adam moments of opacity zeroed."""

    def param_fn(name, p):
        return torch.nn.Parameter(torch.minimum(p.clamp(0.0, 1.0), torch.full_like(p, value)),
                                  requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.zeros_like(v)

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers, names=["opacities"])


@torch.no_grad()
def densify_and_prune(params, optimizers, grads, threshold, extent, cfg):
    """densify_and_prune(max_grad, extent): clone -> split (padded grads) -> opacity prune.
    Returns (n_clone, n_split, n_prune)."""
    state = {}
    max_scale = params["scales"].exp().max(dim=1).values
    clone = (grads >= threshold) & (max_scale <= cfg["percent_dense"] * extent)
    n_clone = int(clone.sum())
    if n_clone:
        duplicate(params, optimizers, state, clone)
    n = len(params["means"])
    padded = torch.zeros(n, device=grads.device)
    padded[: grads.shape[0]] = grads
    max_scale = params["scales"].exp().max(dim=1).values
    sel = (padded >= threshold) & (max_scale > cfg["percent_dense"] * extent)
    n_split = int(sel.sum())
    if n_split:
        split(params, optimizers, state, sel)
    prune = params["opacities"].clamp(0.0, 1.0) < cfg["min_opacity"]
    n_prune = int(prune.sum())
    if n_prune:
        remove(params, optimizers, state, prune)
    return n_clone, n_split, n_prune


class Density:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown deblurgs density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = set(REQUIRES)
        self.state = {}
        self.params = None
        self.cameras = None
        self.bank = None
        self.extent = None
        self.grad_accum = None
        self.denom = None
        self.counts = {"densify": 0, "clone": 0, "split": 0, "prune": 0, "resets": 0}

    @property
    def bank_cfg(self):
        return {k[5:]: v for k, v in self.cfg.items() if k.startswith("bank_")}

    # -- runtime hooks -------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        self.params, self.optimizers = params, optimizers
        with torch.no_grad():  # official Clamp activation: raw opacity = 0.1
            params["opacities"].fill_(float(self.cfg["init_opacity"]))
        optimizers["scales"].param_groups[0]["lr"] = float(self.cfg["scaling_lr"])
        self._reset_stats()
        self.state = {"counts": self.counts}
        if self.cameras:
            self._setup()

    def set_training_cameras(self, cameras):
        cameras = list(cameras)
        if not cameras:
            raise ValueError("DeblurGS needs the training cameras")
        self.cameras = cameras
        if self.params is not None:
            self._setup()

    def _setup(self):
        dev = self.params["means"].device
        centers = torch.stack([torch.linalg.inv(c["viewmat"].float())[:3, 3]
                               for c in self.cameras]).to(dev)
        self.extent = official_extent(centers, self.params["means"].detach())
        self.optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(1)
        self.bank = bank_mod.TrajectoryBank(self.cameras, self.bank_cfg, dev)
        bank_mod.ACTIVE = self.bank
        self.state.update(extent=self.extent, bank=self.bank.export())

    def before_backward(self, params, optimizers, step, outputs):
        if self.bank is None:
            raise RuntimeError("DeblurGS needs density.set_training_cameras(...) first")
        if "subframes" not in outputs:
            raise RuntimeError("DeblurGS training step did not render a bank view")
        outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        self.bank.after_step(step)
        if step < c["densify_until_iter"]:
            self.accumulate(outputs["info"])
            if step > c["densify_from_iter"] and step % c["densification_interval"] == 0:
                grads = self.grad_accum / self.denom
                grads[grads.isnan()] = 0.0
                nc, ns, npr = densify_and_prune(params, optimizers, grads,
                                                densify_threshold(step, c), self.extent, c)
                self.counts["densify"] += 1
                self.counts["clone"] += nc
                self.counts["split"] += ns
                self.counts["prune"] += npr
                self._reset_stats()  # densification_postfix zeroes the statistics
                torch.cuda.empty_cache()
            if step % c["opacity_reset_interval"] == 0:
                reset_opacity(params, optimizers, float(c["reset_opacity_value"]))
                self.counts["resets"] += 1
        optimizers["means"].param_groups[0]["lr"] = self.xyz_lr(step + 1)
        if step % 1000 == 0 or step == c["total_steps"]:
            self.state["bank"] = self.bank.export()

    # -- pieces ----------------------------------------------------------------------------
    def xyz_lr(self, step):
        c = self.cfg
        return expon(step, c["position_lr_init"] * self.extent,
                     c["position_lr_final"] * self.extent, c["total_steps"])

    def _reset_stats(self):
        n, dev = len(self.params["means"]), self.params["means"].device
        self.grad_accum = torch.zeros(n, device=dev)
        self.denom = torch.zeros(n, device=dev)

    def accumulate(self, info):
        """add_densification_stats(viewspace, visibility, 1/len(render_pkgs)) per sub-frame."""
        grad = info["means2d"].grad  # [C, N, 2] pixels
        if grad is None:
            raise RuntimeError("means2d gradient missing (retain_grad not applied)")
        if self.grad_accum.shape[0] != grad.shape[1]:
            self._reset_stats()
        g = grad.clone()
        g[..., 0] *= info["width"] / 2.0
        g[..., 1] *= info["height"] / 2.0
        vis = (info["radii"] > 0).all(dim=-1) if info["radii"].dim() == 3 else info["radii"] > 0
        n_sub = grad.shape[0]
        self.grad_accum += (g.norm(dim=-1) * vis).sum(dim=0)
        self.denom += vis.float().sum(dim=0) / n_sub


def create_density(spec):
    return Density(spec)
