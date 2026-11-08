"""BBSplat initialisation, MCMC density control and optimisation schedule (official
scene/gaussian_model.py + train.py + arguments/__init__.py).

Initialisation (create_from_pcd): FPS subsample of the SfM points to ``max_read_points``
(pytorch3d sample_farthest_points, start point 0), ``sky_points`` Fibonacci points on a sphere of
radius max|xyz| (white), log sqrt(mean d^2 of 3 NN) plane scales, U[0,1)^4 quaternions,
SH dc = RGB2SH(c) - 0.1, textures logit(0.5 G) / logit(0.1).

Densification (500 < it < 25000, every 100): relocate dead billboards (mean T^alpha <= 0.005)
onto multinomial(mean T^alpha) samples of alive ones, rescaling only the alpha map
T' = 1 - (1 - T)^(1/(n+1)) (Eq. 13; scales unchanged), then add min(cap, 1.05 N) - N new ones
the same way. Adam moments are zeroed at the sampled source indices only (official).
MCMC position noise: the official code fixes opacity to 1, so its coefficient
1/(1+exp(100*(0-0.995))) is exactly 0 in float32 -> no noise (see ``noise_coefficient``).

The runtime steps the optimisers before ``after_backward``; learning rates set at step t
therefore apply to step t+1, reproducing the official per-iteration schedule.
"""

import math

import numpy as np
import torch
from gsplat.strategy.ops import _update_param_with_optimizer

from . import texture as tex

SH_C0 = 0.28209479177387814
DEFAULTS = {
    "total_steps": 32000,  # OptimizationParams.iterations
    "cap_max": 60000,
    "max_read_points": 50000,
    "sky_points": 10000,  # --add_sky_box, sphere_point
    "densify_from_iter": 500,
    "densify_until_iter": 25000,
    "densification_interval": 100,
    "dead_opacity": 0.005,
    "growth": 1.05,
    "position_lr_init": 0.00016,
    "position_lr_final": 0.0000016,
    "position_lr_max_steps": 30000,  # also: freeze everything but SH rest after this
    "feature_lr": 0.005,
    "scaling_lr": 0.005,
    "rotation_lr": 0.001,
    "texture_opacity_lr": 0.001,
    "texture_color_lr": 0.0025,
    "texture_from_iter": 500,
    "texture_to_iter": 30000,
    "noise_lr": 5e5,
    "init_third_log_scale": -30.0,  # runtime keeps 3 scale columns; billboards use 2
    # "zero": means lr 0 for every step > 30000 (iteration-0 behaviour). "official": train.py
    # calls update_learning_rate (xyz = expon lr) at the start of every iteration and
    # deactivate_gaussians_training zeroes it only once, so only step 30001 has xyz lr 0.
    "xyz_lr_after_freeze": "zero",
}


def expon_lr(step, lr_init, lr_final, max_steps):
    """get_expon_lr_func with lr_delay_steps = 0 (delay multiplier inert)."""
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


def noise_coefficient(opacity=1.0, k=100.0, x0=0.995):
    """op_sigmoid(1 - opacity) of train.py in float32 (opacity is fixed to 1 there)."""
    x = torch.tensor([1.0 - opacity], dtype=torch.float32)
    return float(1 / (1 + torch.exp(-k * (x - x0))))


def farthest_point_sampling(points, k):
    """pytorch3d.ops.sample_farthest_points(points[None], K=k) with random_start_point=False:
    start at index 0, then repeatedly take the point farthest from the selected set."""
    # float32 numpy loop on the CPU: ~80 us per sample, independent of (shared) GPU load
    p = points.detach().float().cpu().numpy()
    x, y, z = (np.ascontiguousarray(p[:, i]) for i in range(3))
    dist = np.full(len(p), np.inf, dtype=np.float32)
    d, t = np.empty_like(dist), np.empty_like(dist)
    idx = np.empty(k, dtype=np.int64)
    cur = 0
    for i in range(k):
        idx[i] = cur
        np.subtract(x, x[cur], out=t)
        np.multiply(t, t, out=d)
        np.subtract(y, y[cur], out=t)
        d += t * t
        np.subtract(z, z[cur], out=t)
        d += t * t
        np.minimum(dist, d, out=dist)
        cur = int(np.argmax(dist))
    return torch.from_numpy(idx).to(points.device)


def fibonacci_sphere(samples):
    """GaussianModel.fibonacci_sphere (arXiv 0912.4540), unit radius, float64."""
    i = torch.arange(samples, dtype=torch.float64)
    phi = math.pi * (math.sqrt(5.0) - 1.0)
    y = 1 - (i / float(samples - 1)) * 2
    r = torch.sqrt(1 - y * y)
    theta = phi * i
    return torch.stack([torch.cos(theta) * r, y, torch.sin(theta) * r], dim=-1)


def knn_mean_sq(points, k=3, chunk=2048):
    """simple_knn distCUDA2: mean squared distance to the 3 nearest other points."""
    out = torch.empty(len(points), device=points.device)
    for start in range(0, len(points), chunk):
        d2 = torch.cdist(points[start:start + chunk], points).square()
        rows = torch.arange(d2.shape[0], device=points.device)
        d2[rows, rows + start] = float("inf")
        out[start:start + chunk] = d2.topk(k, dim=1, largest=False).values.mean(dim=1)
    return out


class Density:
    requires = set()

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown BBSplat density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.extent = 1.0
        self.counts = {"relocated": 0, "added": 0, "densify_steps": 0, "initial": 0,
                       "sky": 0}
        self.state = {}

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def initialize(self, params, optimizers, scene_scale=1.0):
        c = self.cfg
        self.extent = float(scene_scale)
        dev = params["means"].device
        pts = params["means"].detach().clone()
        rgb = (params["sh0"][:, 0, :].detach() * SH_C0 + 0.5).clamp(0, 1)  # runtime RGB2SH
        if len(pts) >= int(c["max_read_points"]):  # fetchPly: `if len(positions) >= max_points`
            keep = farthest_point_sampling(pts, int(c["max_read_points"]))
            pts, rgb = pts[keep], rgb[keep]
        n_sky = int(c["sky_points"])
        if n_sky > 0:
            radius = float(pts.abs().max())  # np.max(np.abs(pcd_points))
            sky = (fibonacci_sphere(n_sky) * radius).float().to(dev)
            pts = torch.cat([sky, pts])
            rgb = torch.cat([torch.ones_like(sky), rgb])
        n = len(pts)
        scales2 = torch.log(torch.sqrt(knn_mean_sq(pts).clamp_min(1e-7)))[:, None].repeat(1, 2)
        third = torch.full((n, 1), float(c["init_third_log_scale"]), device=dev)
        alpha, color = tex.initial_logits(n, dev)
        values = {
            "means": pts,
            "scales": torch.cat([scales2, third], dim=1),
            "quats": torch.rand(n, 4, device=dev),
            "opacities": torch.zeros(n, device=dev),  # unused: billboards have no opacity
            "sh0": ((rgb - 0.5) / SH_C0 - tex.SH_DC_OFFSET)[:, None, :],
            "shN": torch.zeros(n, 15, 3, device=dev),
            tex.ALPHA: alpha,
            tex.COLOR: color,
        }
        missing = set(params.keys()) - set(values)
        if missing:
            raise RuntimeError(f"BBSplat init does not handle fields {sorted(missing)}")

        def param_fn(name, p):
            return torch.nn.Parameter(values[name].to(p.dtype).contiguous(),
                                      requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.zeros((n, *v.shape[1:]), device=dev, dtype=v.dtype)

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        self.counts.update(initial=n, sky=n_sky)
        self.set_learning_rates(optimizers, 1)
        self._sync_state()

    # ------------------------------------------------------------------ schedule
    def learning_rates(self, step):
        """Learning rates of official iteration ``step`` (1-based)."""
        c = self.cfg
        frozen = step > c["position_lr_max_steps"]  # deactivate_gaussians_training
        tex_on = c["texture_from_iter"] <= step < c["texture_to_iter"]
        if c["xyz_lr_after_freeze"] not in ("zero", "official"):
            raise ValueError("xyz_lr_after_freeze must be 'zero' or 'official'")
        xyz_frozen = frozen and (c["xyz_lr_after_freeze"] == "zero"
                                 or step == c["position_lr_max_steps"] + 1)
        lrs = {
            "means": 0.0 if xyz_frozen else expon_lr(step, c["position_lr_init"] * self.extent,
                                                  c["position_lr_final"] * self.extent,
                                                  c["position_lr_max_steps"]),
            "sh0": 0.0 if frozen else c["feature_lr"],
            "shN": c["feature_lr"] / 20.0,
            "scales": 0.0 if frozen else c["scaling_lr"],
            "quats": 0.0 if frozen else c["rotation_lr"],
            "opacities": 0.0,
            tex.ALPHA: c["texture_opacity_lr"] if tex_on and not frozen else 0.0,
            tex.COLOR: c["texture_color_lr"] if tex_on and not frozen else 0.0,
        }
        if step >= c["total_steps"]:  # `if iteration < opt.iterations: optimizer.step()`
            lrs = {k: 0.0 for k in lrs}
        return lrs

    def set_learning_rates(self, optimizers, step):
        for name, lr in self.learning_rates(step).items():
            if name in optimizers:
                optimizers[name].param_groups[0]["lr"] = lr

    # ------------------------------------------------------------------ hooks
    def before_backward(self, params, optimizers, step, outputs):
        return None

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        if (c["densify_from_iter"] < step < c["densify_until_iter"]
                and step % c["densification_interval"] == 0):
            n = len(params[tex.ALPHA])
            dead = params[tex.ALPHA].sigmoid().view(n, -1).mean(1) <= c["dead_opacity"]
            self.relocate(params, optimizers, dead)
            self.add_new(params, optimizers, int(c["cap_max"]))
            self.counts["densify_steps"] += 1
        self.set_learning_rates(optimizers, step + 1)
        self._sync_state()

    def _sync_state(self):
        self.state = {k: torch.tensor(v) for k, v in self.counts.items()}

    # ------------------------------------------------------------------ MCMC ops
    @staticmethod
    def _sample(probs, num, alive_indices=None):
        """_sample_alives: multinomial with replacement; ratio = bincount of the samples."""
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        idx = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            idx = alive_indices[idx]
        ratio = torch.bincount(idx).unsqueeze(-1)
        return idx, ratio

    @staticmethod
    def rescaled_alpha(alpha_logits, ratio):
        """_update_params: 1 - (1 - T)^(1/(n+1)) per texel (Eq. 13), back to logits."""
        a = alpha_logits.sigmoid()
        n = ratio[:, :, None].to(a.dtype) + 1
        return tex.safe_logit(1.0 - torch.pow(1.0 - a, 1.0 / n))

    @staticmethod
    def _zero_moments(optimizers, params, idx):
        for name in params:
            st = optimizers[name].state.get(params[name])
            if st:
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in st:
                        st[key][idx] = 0

    @torch.no_grad()
    def relocate(self, params, optimizers, dead):
        if int(dead.sum()) == 0:
            return 0
        alive_idx = (~dead).nonzero(as_tuple=True)[0]
        dead_idx = dead.nonzero(as_tuple=True)[0]
        if len(alive_idx) == 0:
            return 0
        a = params[tex.ALPHA]
        probs = a[alive_idx].sigmoid().view(len(alive_idx), -1).mean(1)
        src, ratio = self._sample(probs, len(dead_idx), alive_idx)
        new_alpha = self.rescaled_alpha(a[src], ratio[src])
        for name, p in params.items():  # in place: Parameters keep identity and requires_grad
            p.data[dead_idx] = new_alpha if name == tex.ALPHA else p.data[src]
        a.data[src] = a.data[dead_idx]  # `self._texture_alpha[reinit_idx] = ...[dead_indices]`
        self._zero_moments(optimizers, params, src)
        self.counts["relocated"] += len(dead_idx)
        return len(dead_idx)

    @torch.no_grad()
    def add_new(self, params, optimizers, cap_max):
        n = len(params["means"])
        target = min(cap_max, int(self.cfg["growth"] * n))
        num = max(0, target - n)
        if num <= 0:
            return 0
        a = params[tex.ALPHA]
        probs = a.sigmoid().view(n, -1).mean(1)
        src, ratio = self._sample(probs, num)
        new_alpha = self.rescaled_alpha(a[src], ratio[src])
        a.data[src] = new_alpha
        appended = {name: (new_alpha if name == tex.ALPHA else p.data[src])
                    for name, p in params.items()}

        def param_fn(name, p):
            return torch.nn.Parameter(torch.cat([p.detach(), appended[name]]).contiguous(),
                                      requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.cat([v, torch.zeros((num, *v.shape[1:]), device=v.device,
                                             dtype=v.dtype)])

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
        self._zero_moments(optimizers, params, src)
        self.counts["added"] += num
        return num


def create_density(spec):
    return Density(spec)
