"""WildGaussians initialisation, densification, Mip-Splatting filter updates, LR schedule and the
appearance/uncertainty bank (official method.py GaussianModel / WildGaussians.train_iteration).

* initialisation (`set_training_cameras`, called by the runtime right after `initialize`):
  features_dc := RGB in [0, 1] (official quirk, RGB2SH commented out), opacity 0.1, plus
  `num_sky_gaussians` Fibonacci sky points (colour (237, 247, 252)/255, opacity 1.0 -> logit 20,
  which is sigmoid == 1.0 in float32 like the official logit(1) = +inf); scales = log sqrt(mean
  3-NN dist^2) over SfM + sky points; Fourier per-Gaussian embeddings; 3D filter; bank.
* statistics (every step while it < densify_until, visible = radii > 0 of the raw pass):
  grad += ||sum_passes dL/dmu'|| and abs += sum_passes sum_px (|gx| + |gy|) (GOF backward.cu
  L1 channel; NDC units = gsplat pixels
  * (W/2, H/2)), denom += 1. **Never reset** (official densification_postfix appends zeros for
  new Gaussians and keeps the accumulators of the others).
* densify (from < it < until, it % every == 0): ratio = mean(g >= thr), Q = quantile(a, 1 - ratio);
  clone (max scale <= 0.01 extent) and then split (N = 2, /1.6; padded statistics of the clones
  are 0) where g >= thr or a >= Q; prune sigmoid(o) < 0.005 and, if it > reset interval, max scale
  > 0.1 extent; recompute the 3D filter.
* opacity reset (it % reset == 0, it < until): official filter-aware reset with the doubled
  filter quirk (geometry.reset_logits), Adam moments zeroed.
* 3D filter every 100 it after densify_until while it < iterations - 100.
* xyz lr log-linear position_lr_init -> position_lr_final (x extent) over `iterations`, evaluated
  at the official 1-based iteration; scales lr per config.
"""

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, duplicate, remove, split

from . import bank as bank_mod
from .appearance import fourier_features
from .bank import BANK_DEFAULTS, WildBank
from .geometry import compute_filter_3d, knn_log_scales, reset_logits, sky_points

REQUIRES = ("gradient.screen.absolute", "gradient.screen.signed")
SKY_COLOR = (237 / 255, 247 / 255, 252 / 255)
SKY_LOGIT = 20.0
DEFAULTS = {
    "iterations": 30000,
    "densify_from_iter": 500,
    "densify_until_iter": 15000,
    "densification_interval": 100,
    "opacity_reset_interval": 3000,
    "densify_grad_threshold": 2e-4,
    "use_abs_gradient": True,
    "percent_dense": 0.01,
    "min_opacity": 0.005,
    "big_points_ws": 0.1,
    "init_opacity": 0.1,
    "position_lr_init": 1.6e-4,
    "position_lr_final": 1.6e-6,
    "scaling_lr": 0.005,
    "rotation_lr": 0.001,
    "opacity_lr": 0.05,
    "feature_lr": 0.0025,
    "embedding_lr": 0.005,
    "num_sky_gaussians": 100000,
    "appearance_n_fourier_freqs": 4,
    "appearance_init_fourier": True,
    "filter_interval": 100,
    **{f"bank_{k}": v for k, v in BANK_DEFAULTS.items()},
}


def expon_lr(step, init, final, max_steps):
    """get_expon_lr_func (lr_delay_steps 0): log-linear, t = clip(step / max_steps, 0, 1)."""
    if step < 0:
        return 0.0
    t = min(max(step / max_steps, 0.0), 1.0)
    return float(torch.exp(torch.log(torch.tensor(init, dtype=torch.float64)) * (1 - t)
                           + torch.log(torch.tensor(final, dtype=torch.float64)) * t))


def _set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g["lr"] = float(lr)


@torch.no_grad()
def densify_and_prune(params, optimizers, stats, thr, extent, cfg, size_prune):
    """densify_and_prune(max_grad, 0.005, extent, enable_size_pruning). `stats` holds grad, abs,
    denom tensors [N]; returns (n_clone, n_split, n_prune)."""
    grads = stats["grad"] / stats["denom"]
    grads[grads.isnan()] = 0.0
    q = None
    grads_abs = None
    if cfg["use_abs_gradient"]:
        ratio = (grads >= thr).float().mean()
        grads_abs = stats["abs"] / stats["denom"]
        grads_abs[grads_abs.isnan()] = 0.0
        q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)
    pd = cfg["percent_dense"] * extent
    # clone
    sel = grads >= thr
    if q is not None:
        sel = sel | (grads_abs >= q)
    sel = sel & (params["scales"].exp().max(dim=1).values <= pd)
    n_clone = int(sel.sum())
    if n_clone:
        n0 = len(params["means"])
        duplicate(params, optimizers, stats, sel)
        for k in stats:  # new Gaussians start with zero statistics
            stats[k][n0:] = 0
    # split (padded statistics: zeros for the clones)
    n = len(params["means"])
    pg = torch.zeros(n, device=grads.device)
    pg[: grads.shape[0]] = grads
    sel = pg >= thr
    if q is not None:
        pa = torch.zeros(n, device=grads.device)
        pa[: grads_abs.shape[0]] = grads_abs
        sel = sel | (pa >= q)
    sel = sel & (params["scales"].exp().max(dim=1).values > pd)
    n_split = int(sel.sum())
    if n_split:
        split(params, optimizers, stats, sel)
        for k in stats:
            stats[k][len(params["means"]) - 2 * n_split:] = 0
    # prune
    prune = params["opacities"].sigmoid() < cfg["min_opacity"]
    if size_prune:
        prune = prune | (params["scales"].exp().max(dim=1).values > cfg["big_points_ws"] * extent)
    n_prune = int(prune.sum())
    if n_prune:
        remove(params, optimizers, stats, prune)
    return n_clone, n_split, n_prune


class Density:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown wildgaussians density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = set(REQUIRES)
        self.params = self.optimizers = self.cameras = self.bank = None
        self.extent = None
        self.stats = None
        self.counts = {"densify": 0, "clone": 0, "split": 0, "prune": 0, "resets": 0,
                       "filter_updates": 0, "sky_points": 0}
        self._state = {}

    @property
    def bank_cfg(self):
        return {k[5:]: v for k, v in self.cfg.items() if k.startswith("bank_")}

    @property
    def state(self):
        """Checkpointed as density_state; the bank is exported on access (always current)."""
        s = dict(self._state)
        s["counts"] = dict(self.counts)
        if self.bank is not None:
            s["bank"] = self.bank.export()
        return s

    # -- runtime hooks ---------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        c = self.cfg
        self.params, self.optimizers = params, optimizers
        self.extent = float(scene_scale)  # recipe 3dgs: 1.1 x max camera distance (official)
        _set_lr(optimizers["scales"], c["scaling_lr"])
        _set_lr(optimizers["quats"], c["rotation_lr"])
        _set_lr(optimizers["opacities"], c["opacity_lr"])
        _set_lr(optimizers["sh0"], c["feature_lr"])
        _set_lr(optimizers["shN"], c["feature_lr"] / 20.0)
        if "wg_embedding" in optimizers:
            _set_lr(optimizers["wg_embedding"], c["embedding_lr"])
        _set_lr(optimizers["means"], self.xyz_lr(1))
        self._state = {"extent": self.extent}
        if self.cameras is not None:
            self._setup()

    def set_training_cameras(self, cameras):
        cameras = list(cameras)
        if not cameras:
            raise ValueError("WildGaussians needs the training cameras")
        self.cameras = cameras
        if self.params is not None:
            self._setup()

    @torch.no_grad()
    def _setup(self):
        c, params, dev = self.cfg, self.params, self.params["means"].device
        self.cameras = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in cam.items()}
                        for cam in self.cameras]
        pts = params["means"].detach().clone()
        rgb = params["sh0"][:, 0, :] * 0.28209479177387814 + 0.5  # runtime stored RGB2SH(rgb)
        opac = torch.full((len(pts),), float(c["init_opacity"]), device=dev).logit()
        if c["num_sky_gaussians"]:
            sky, radius = sky_points(int(c["num_sky_gaussians"]), pts, self.cameras)
            self.counts["sky_points"] = len(sky)
            self._state["sky_radius"] = radius
            pts = torch.cat((pts, sky))
            rgb = torch.cat((rgb, torch.tensor(SKY_COLOR, device=dev).expand(len(sky), 3)))
            opac = torch.cat((opac, torch.full((len(sky),), SKY_LOGIT, device=dev)))
        n = len(pts)
        values = {
            "means": pts, "scales": knn_log_scales(pts), "opacities": opac,
            "quats": torch.cat((torch.ones(n, 1, device=dev), torch.zeros(n, 3, device=dev)), 1),
            "sh0": rgb[:, None, :].clone(),  # official features_dc = rgb (not RGB2SH)
            "shN": torch.zeros(n, 15, 3, device=dev),
        }
        if "wg_embedding" in params:
            emb = fourier_features(pts, int(c["appearance_n_fourier_freqs"]))
            g = torch.Generator(device="cpu").manual_seed(int(c["bank_seed"]))
            emb = emb + (torch.randn(emb.shape, generator=g) * 1e-4).to(dev)
            if not c["appearance_init_fourier"]:
                emb = (torch.randn(emb.shape, generator=g) * 0.01).to(dev)
            values["wg_embedding"] = emb
        values["filter_3d"] = compute_filter_3d(pts, self.cameras)
        for name in list(params.keys()):
            if name not in values:
                raise RuntimeError(f"WildGaussians init does not handle field {name}")

        def param_fn(name, p):
            return torch.nn.Parameter(values[name].to(p.dtype).contiguous(),
                                      requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.zeros((n, *v.shape[1:]), device=dev, dtype=v.dtype)

        _update_param_with_optimizer(param_fn, optimizer_fn, params, self.optimizers)
        self.counts["filter_updates"] += 1
        self._reset_stats()
        vms = [cam["viewmat"] for cam in self.cameras]
        self.bank = WildBank(vms, self.bank_cfg, dev)
        bank_mod.ACTIVE = self.bank

    def before_backward(self, params, optimizers, step, outputs):
        if self.bank is None:
            raise RuntimeError("WildGaussians needs density.set_training_cameras(...) first")
        for info in outputs["infos"]:
            info["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.cfg
        it = step  # runtime step == official 1-based iteration
        self.bank.after_step(step)
        if it < c["densify_until_iter"]:
            self.accumulate(outputs)
            if it > c["densify_from_iter"] and it % c["densification_interval"] == 0:
                nc, ns, npr = densify_and_prune(
                    params, optimizers, self.stats, c["densify_grad_threshold"], self.extent, c,
                    size_prune=it > c["opacity_reset_interval"])
                self.counts["densify"] += 1
                self.counts["clone"] += nc
                self.counts["split"] += ns
                self.counts["prune"] += npr
                self.update_filter(params)
                torch.cuda.empty_cache()
            if it % c["opacity_reset_interval"] == 0:
                self.reset_opacity(params, optimizers)
        if (it % c["filter_interval"] == 0 and it > c["densify_until_iter"]
                and it < c["iterations"] - c["filter_interval"]):
            self.update_filter(params)
        _set_lr(optimizers["means"], self.xyz_lr(it + 1))

    # -- pieces ----------------------------------------------------------------------------
    def xyz_lr(self, iteration):
        c = self.cfg
        return expon_lr(iteration, c["position_lr_init"] * self.extent,
                        c["position_lr_final"] * self.extent, c["iterations"])

    def _reset_stats(self):
        n, dev = len(self.params["means"]), self.params["means"].device
        self.stats = {"grad": torch.zeros(n, device=dev), "abs": torch.zeros(n, device=dev),
                      "denom": torch.zeros(n, device=dev)}

    def accumulate(self, outputs):
        infos = outputs["infos"]
        info = infos[0]
        if self.stats["grad"].shape[0] != info["means2d"].shape[1]:
            raise RuntimeError("density statistics are misaligned with the Gaussians")
        scale = torch.tensor([info["width"] / 2.0, info["height"] / 2.0],
                             device=info["means2d"].device)
        g = sum(i["means2d"].grad[0] for i in infos) * scale
        a = sum(i["means2d"].absgrad[0] for i in infos) * scale
        radii = info["radii"][0]
        vis = (radii > 0).all(dim=-1) if radii.dim() == 2 else radii > 0
        self.stats["grad"][vis] += g[vis].norm(dim=-1)
        self.stats["abs"][vis] += a[vis].sum(dim=-1)  # official: sum_px |gx| + |gy| (L1)
        self.stats["denom"][vis] += 1

    def update_filter(self, params):
        params["filter_3d"].data.copy_(compute_filter_3d(params["means"], self.cameras))
        self.counts["filter_updates"] += 1

    def reset_opacity(self, params, optimizers):
        new = reset_logits(params["scales"].detach(), params["opacities"].detach(),
                           params["filter_3d"].detach())

        def param_fn(name, p):
            return torch.nn.Parameter(new, requires_grad=p.requires_grad)

        def optimizer_fn(key, v):
            return torch.zeros_like(v)

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers,
                                     names=["opacities"])
        self.counts["resets"] += 1


def create_density(spec):
    return Density(spec)
