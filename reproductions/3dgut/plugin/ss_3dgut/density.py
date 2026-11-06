"""3DGUT adaptive density control (paper §4.4; official threedgrut/strategy/gs.py @ bb203b1).

3DGUT has no 2D screen-space gradients (the projection is not differentiated), so it follows
3DGRT and densifies on the 3D positional gradient scaled by half the distance to the camera:

    accum += || dL/dmu * ||mu - o_cam|| || / 2,   count += 1    (Gaussians with nonzero grad)

densify every 300 iterations (500 < s < 15000) when accum/count >= 2e-4: clone if
max(scale) <= 0.01 * extent, split (N=2, scale / 1.6) otherwise; prune opacity < 0.005 every 100
iterations; reset opacity to min(sigma, 0.01) every 3000 iterations before 15000. No screen- or
world-size pruning (`prune_scale` / `prune_weight` are disabled in the official config).

Optionally (`official_init`, default on) the official COLMAP initialisation is applied once the
training cameras are known: isotropic scale = 0.01 * distance to the nearest camera centre and
random rotations (normalised U[0,1)^4, seed 42), model.py:708 (`use_observation_points: true`).
"""

import torch
from gsplat.strategy.ops import duplicate, remove, reset_opa, split

REQUIRES = frozenset({"gradient.world.means"})

# configs/strategy/gs.yaml, configs/base_gs.yaml, configs/initialization/colmap.yaml.
DEFAULTS = {
    "grad_threshold": 0.0002,  # clone_grad_threshold == split_grad_threshold
    "densify_every": 300,
    "densify_start": 500,
    "densify_stop": 15000,
    "prune_every": 100,
    "prune_start": 500,
    "prune_stop": 15000,
    "prune_opa": 0.005,
    "reset_every": 3000,
    "reset_start": 0,
    "reset_stop": 15000,
    "reset_value": 0.01,
    "percent_dense": 0.01,
    "official_init": True,
    "init_scale_factor": 0.01,
    "init_seed": 42,
}


def step_condition(step, start, end, every):
    """threedgrut/utils/misc.py:check_step_condition."""
    return (start >= 0 and step > start) and (step < end or end == -1) and step % every == 0


def densification_statistic(grad, means, campos):
    """Per-Gaussian increment and mask of the 3DGRT/3DGUT gradient statistic (gs.py:130)."""
    mask = (grad != 0).any(dim=1)
    distance = (means[mask] - campos).norm(dim=1, keepdim=True)
    return mask, torch.norm(grad[mask] * distance, dim=-1) / 2


def official_init_values(means, camera_centers, factor, seed):
    """model.py:default_initialize_from_points with use_observer_pts=True."""
    dist = torch.cdist(means, camera_centers).min(dim=1).values.clamp_min(1e-7)
    log_scales = (dist * factor).log()[:, None].expand(-1, 3).contiguous()
    gen = torch.Generator(device=means.device).manual_seed(seed)
    quats = torch.rand((len(means), 4), generator=gen, device=means.device)
    return log_scales, torch.nn.functional.normalize(quats, dim=-1)


class Density:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown 3DGUT density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = set(REQUIRES)
        self.state = None
        self._params = None
        self._grad = None
        self._hook = None
        self._campos = None

    def initialize(self, params, optimizers, scene_scale=1.0):
        n = len(params["means"])
        device = params["means"].device
        self._params = params
        self.state = {
            "scene_scale": float(scene_scale),
            "accum": torch.zeros(n, device=device),
            "count": torch.zeros(n, device=device, dtype=torch.int32),
        }

    def set_training_cameras(self, cameras):
        if not self.cfg["official_init"]:
            return
        params = self._params
        centers = torch.stack([torch.linalg.inv(c["viewmat"])[:3, 3] for c in cameras])
        log_scales, quats = official_init_values(
            params["means"].detach(), centers.to(params["means"].device),
            float(self.cfg["init_scale_factor"]), int(self.cfg["init_seed"]),
        )
        with torch.no_grad():
            params["scales"].copy_(log_scales)
            params["quats"].copy_(quats)

    def before_backward(self, params, optimizers, step, outputs):
        # The runtime zeroes gradients before after_backward: capture dL/dmu with a one-shot hook.
        self._grad = None
        self._campos = outputs["info"]["campos"].detach()
        if step < self.cfg["densify_stop"]:
            def keep(grad):
                self._grad = grad.detach().clone()

            self._hook = params["means"].register_hook(keep)

    def _reset_buffers(self, n, device):
        self.state["accum"] = torch.zeros(n, device=device)
        self.state["count"] = torch.zeros(n, device=device, dtype=torch.int32)

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        cfg, state = self.cfg, self.state
        self._params = params
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        if step < cfg["densify_stop"] and self._grad is not None:
            mask, inc = densification_statistic(self._grad, params["means"], self._campos)
            state["accum"][mask] += inc
            state["count"][mask] += 1
        self._grad = None
        device = params["means"].device
        if step_condition(step, cfg["densify_start"], cfg["densify_stop"], cfg["densify_every"]):
            grads = state["accum"] / state["count"]
            grads[grads.isnan()] = 0.0
            big = params["scales"].exp().max(dim=-1).values > cfg["percent_dense"] * state[
                "scene_scale"
            ]
            hot = grads >= cfg["grad_threshold"]
            clone_mask = hot & ~big
            split_mask = hot & big
            n_clone = int(clone_mask.sum())
            if n_clone:
                duplicate(params, optimizers, state, clone_mask)
            # Clones are appended at the end and are not split (gs.py split_gaussians padding).
            split_mask = torch.cat((split_mask, split_mask.new_zeros(n_clone)))
            if int(split_mask.sum()):
                split(params, optimizers, state, split_mask)
            self._reset_buffers(len(params["means"]), device)
        if step_condition(step, cfg["prune_start"], cfg["prune_stop"], cfg["prune_every"]):
            low = params["opacities"].sigmoid() < cfg["prune_opa"]
            if int(low.sum()):
                remove(params, optimizers, state, low)
        if step_condition(step, cfg["reset_start"], cfg["reset_stop"], cfg["reset_every"]):
            reset_opa(params, optimizers, state, value=cfg["reset_value"])


def create_density(spec):
    return Density(spec)
