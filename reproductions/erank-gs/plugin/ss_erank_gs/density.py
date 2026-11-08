"""Adaptive density control of the official erank-gs code (scene/gaussian_model.py,
train.py @99fd128) = GOF's densification ("ADC fix", paper App. A.4) on vanilla 3DGS.

  stats    step < densify_until, radii > 0:
             accum     += ||g_ndc||_2              (g_ndc = means2D grad, NDC units)
             accum_abs += sum_pixels |gx| + |gy|   (official kernel channel z, backward.cu:548)
           gsplat's absgrad holds (sum_pix |gx|, sum_pix |gy|) in pixels; scaled by (W/2, H/2)
           its L1 norm equals the official channel exactly.
  densify  step > densify_from and step % interval == 0 (densify_and_prune_v2):
           grads = accum/denom, grads_abs = accum_abs/denom (nan -> 0)
           ratio = mean(grads >= thr); Q = quantile(grads_abs, 1 - ratio)
           clone: (grads >= thr | grads_abs >= Q) & max(s) <= percent_dense * extent,
                  new mean ~ N(mu, R diag(s)^2 R^T)
           split: same criterion on the post-clone set (clones padded with 0) and
                  max(s) > percent_dense * extent; 2 samples, s / 1.6
           prune: sigmoid(o) < min_opacity (0.05); for step > reset_interval also
                  max(s) > 0.1 extent. The screen-size prune (max_radii2D > 20) is a no-op in
                  the official code (densification_postfix zeroes max_radii2D) -> not applied.
           statistics restart at zero after every densification
  reset    step % 3000 == 0 while step < densify_until: opacity = min(opacity, 0.01),
           optimizer moments zeroed.
abs_gradient=False, clone_resample=False, min_opacity=0.005 gives the paper's 3DGS baseline
(densify_and_prune). Difference to the official loop: the runtime applies the optimizer step
before this controller runs.
"""

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove, reset_opa, split
from gsplat.utils import normalized_quat_to_rotmat

DEFAULTS = {
    "densify_from_iter": 500,
    "densify_until_iter": 15000,
    "densification_interval": 100,
    "densify_grad_threshold": 0.0002,
    "abs_gradient": True,
    "clone_resample": True,
    "percent_dense": 0.01,
    "min_opacity": 0.05,
    "opacity_reset_interval": 3000,
    "opacity_reset_value": 0.01,
    "big_world_factor": 0.1,
}
REQUIRES = {"gradient.screen.absolute"}


@torch.no_grad()
def clone_gaussians(params, optimizers, mask, resample=True):
    """densify_and_clone(_v2): append copies; v2 draws the new mean from N(mu, Sigma)."""
    sel = torch.where(mask)[0]
    new_means = params["means"][sel]
    if resample:
        stds = params["scales"][sel].exp()
        rots = normalized_quat_to_rotmat(torch.nn.functional.normalize(params["quats"][sel],
                                                                       dim=-1))
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        new_means = torch.bmm(rots, samples[..., None])[..., 0] + new_means

    def param_fn(name, p):
        new = new_means if name == "means" else p[sel]
        # detach: building from a no_grad view keeps requires_grad from the original leaf
        return torch.nn.Parameter(torch.cat([p.detach(), new.detach()]),
                                  requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=v.device,
                                         dtype=v.dtype)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


def screen_stats(info):
    """Per-Gaussian (signed L2, abs L1) gradient magnitudes in official NDC units + visibility."""
    m2d = info["means2d"]
    if m2d.grad is None:
        raise RuntimeError("erank density: means2d has no gradient (retain_grad missing)")
    absg = getattr(m2d, "absgrad", None)
    if absg is None:
        raise RuntimeError("erank density: renderer did not supply absgrad")
    scale = torch.tensor([info["width"] / 2.0, info["height"] / 2.0], device=m2d.device)
    g = (m2d.grad * scale)[0]
    a = (absg * scale)[0]
    radii = info["radii"][0]
    vis = (radii > 0).all(dim=-1) if radii.dim() == 2 else radii > 0
    return torch.linalg.vector_norm(g, dim=-1), a.abs().sum(dim=-1), vis


class Density:
    requires = REQUIRES

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown erank density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = set(REQUIRES)
        self.state = None
        self.stats = None
        self.extent = None

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.extent = float(scene_scale)  # 3dgs recipe: getNerfppNorm radius
        self._reset_stats(len(params["means"]), params["means"].device)
        self.state = {"extent": self.extent, "counts": {"densify": 0, "resets": 0}}

    def before_backward(self, params, optimizers, step, outputs):
        outputs["info"]["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        if step < c["densify_until_iter"]:
            self.accumulate(outputs["info"])
            if step > c["densify_from_iter"] and step % c["densification_interval"] == 0:
                self.densify_and_prune(params, optimizers, step > c["opacity_reset_interval"])
            if step % c["opacity_reset_interval"] == 0:
                reset_opa(params, optimizers, {}, value=c["opacity_reset_value"])
                self.state["counts"]["resets"] += 1

    def _reset_stats(self, n, dev):
        self.stats = {"grad_accum": torch.zeros(n, device=dev),
                      "grad_accum_abs": torch.zeros(n, device=dev),
                      "denom": torch.zeros(n, device=dev)}

    def accumulate(self, info):
        g, a, vis = screen_stats(info)
        st = self.stats
        st["grad_accum"][vis] += g[vis]
        st["grad_accum_abs"][vis] += a[vis]
        st["denom"][vis] += 1

    @torch.no_grad()
    def densify_and_prune(self, params, optimizers, world_prune):
        c, st = self.cfg, self.stats
        grads = st["grad_accum"] / st["denom"]
        grads[grads.isnan()] = 0.0
        grads_abs = st["grad_accum_abs"] / st["denom"]
        grads_abs[grads_abs.isnan()] = 0.0
        thr, dense = c["densify_grad_threshold"], c["percent_dense"] * self.extent
        n0 = len(grads)
        cand = grads >= thr
        if c["abs_gradient"]:
            q = torch.quantile(grads_abs, 1 - cand.float().mean())
            cand = cand | (grads_abs >= q)
        smax = params["scales"].exp().max(dim=1).values
        clone = cand & (smax <= dense)
        clone_gaussians(params, optimizers, clone, c["clone_resample"])
        n_clone = int(clone.sum())
        pad = torch.zeros(n_clone, device=grads.device)
        cand = torch.cat([grads, pad]) >= thr
        if c["abs_gradient"]:
            cand = cand | (torch.cat([grads_abs, pad]) >= q)
        smax = params["scales"].exp().max(dim=1).values
        split_mask = cand & (smax > dense)
        split(params, optimizers, {}, split_mask)
        prune = params["opacities"].sigmoid() < c["min_opacity"]
        if world_prune:
            prune = prune | (params["scales"].exp().max(dim=1).values
                             > c["big_world_factor"] * self.extent)
        remove(params, optimizers, {}, prune)
        n = len(params["means"])
        self._reset_stats(n, params["means"].device)
        self.state["counts"]["densify"] += 1
        self.state["last"] = {"before": n0, "clone": n_clone, "split": int(split_mask.sum()),
                              "prune": int(prune.sum()), "after": n}


def create_density(spec):
    return Density(spec)
