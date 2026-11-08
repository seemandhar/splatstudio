"""ConeGS error-guided cone densification (paper Sec. 4.2), replacing 3DGS clone/split/reset.

Official `train.py` + `scene/nerf_model.py:cast_rays_during_optimization` +
`scene/gaussian_model.py:initialize_new_points` + `utils/nerf_utils.py:get_num_rays_to_cast`:

- every step 0 < s < densify_until (25000): E = mean_c |I - I*| of the rendered training view;
  sample n_rays pixels without replacement with p ~ E; for each, a new Gaussian at the iNGP
  median depth on the pixel ray, colour = iNGP colour (zeroed view dir), isotropic std = pixel
  cone footprint at that depth (Eq. 7/11), identity rotation, opacity 0.1, SH rest 0; stash it.
- every `interval` (100) steps: prune opacity <= 0.005 (only s < densify_until), then merge the
  stash; with a budget only the first `budget - N` stashed Gaussians (chronological) are added.
- n_rays = max(int(N * growth), int(1.2 * N_added) [budget only]) // 100 + 1, at least 10
  (Eq. 12 / 13; code growth 0.02).
- initialize: per-point 3-NN log scales from `conegs_init.npz` (found by the SHA-256 of the
  initial means, written by `ss-conegs-init`), iNGP from `conegs_ingp.pt`; means LR scaled by
  3DGS's camera extent (the 3DGS recipe's [4096, N] kNN would need ~32 GB for 1M points, so the
  method trains with `--recipe splatstudio` and restores the 3DGS LR scale here).
"""

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove

from .ingp import INGP, SH_C0, RayCaster

DEFAULTS = {
    "init_root": "~/datasets/conegs-init",
    "budget": 1_000_000,  # optimization.max_points; 0 = no budget (Eq. 13)
    "growth": 0.02,  # optimization.gaussian_percentage_increase (paper Eq. 12 text: 0.2)
    "added_mult": 1.2,
    "min_rays": 10,
    "interval": 100,
    "densify_until": 25000,
    "prune_opacity": 0.005,
    "new_opacity": 0.1,
    "size_mult": 1.0,  # densification_gaussian_size (std = 1 x pixel footprint)
    "camera_extent_lr": True,
    "ingp_batch": 8192,
}


def means_sha256(means):
    return hashlib.sha256(
        np.ascontiguousarray(means.detach().float().cpu().numpy()).tobytes()
    ).hexdigest()


def find_sidecar(root, sha):
    root = Path(root).expanduser()
    for path in sorted(root.rglob("conegs_init.npz")):
        with np.load(path, allow_pickle=False) as data:
            if str(data["points_sha256"]) == sha:
                return path
    raise FileNotFoundError(
        f"No conegs_init.npz under {root} matches the initial points (sha256 {sha[:12]}...). "
        "Train on a scene prepared by ss-conegs-init with --num-gaussians >= its point count."
    )


def rays_to_cast(n_gaussians, added, growth, added_mult=1.2, interval=100, min_rays=10):
    """Official get_num_rays_to_cast (nerf_train_during_3dgs=False)."""
    n = int(n_gaussians * growth)
    if added:
        n = max(n, int(added * added_mult))
    return max(int(n // interval + 1), min_rays)


@torch.no_grad()
def append_gaussians(params, optimizers, new):
    """Concatenate new rows; zero Adam moments for them; keep each field's requires_grad."""
    count = len(next(iter(new.values())))

    def param_fn(name, p):
        rows = new[name].to(p.dtype) if name in new else p.new_zeros((count, *p.shape[1:]))
        return torch.nn.Parameter(torch.cat([p.detach(), rows]), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, v.new_zeros((count, *v.shape[1:]))])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


def load_ingp(path, device):
    blob = torch.load(path, map_location=device, weights_only=True)
    model = INGP(blob["cfg"]).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, RayCaster(blob["transform"], blob["scale"], device)


class ConeGSDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown conegs-density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = {"conegs.frame"}
        self.state = {}
        self.init_override = None  # probes: {"log_scales": tensor}
        self.ingp_override = None  # probes: (model, caster)
        self.stash = []

    @torch.no_grad()
    def initialize(self, params, optimizers, scene_scale=1.0):
        cfg = self.cfg
        dev = params["means"].device
        n = len(params["means"])
        if self.init_override is not None:
            log_scales = torch.as_tensor(self.init_override["log_scales"], device=dev)
            self.model, self.caster = self.ingp_override
        else:
            sha = means_sha256(params["means"])
            path = find_sidecar(cfg["init_root"], sha)
            with np.load(path, allow_pickle=False) as data:
                log_scales = torch.as_tensor(data["log_scales"], device=dev)
            self.model, self.caster = load_ingp(path.with_name("conegs_ingp.pt"), dev)
            self.state["init_file"] = str(path)
        if log_scales.shape != (n,):
            raise ValueError("conegs_init log scales do not match the number of Gaussians")
        params["scales"].copy_(log_scales.float()[:, None].expand(-1, 3))
        params["quats"].zero_()
        params["quats"][:, 0] = 1
        params["opacities"].fill_(math.log(cfg["new_opacity"] / (1 - cfg["new_opacity"])))
        if cfg["budget"] and n > cfg["budget"]:
            raise ValueError(f"{n} initial Gaussians exceed the budget {cfg['budget']}")
        self.optimizers = optimizers
        self.runtime_scene_scale = float(scene_scale)
        self.extent_applied = not cfg["camera_extent_lr"]
        self.n_rays = rays_to_cast(n, None, cfg["growth"], cfg["added_mult"], cfg["interval"],
                                   cfg["min_rays"])
        self.stash = []
        self.state.update(initial_gaussians=n, pruned=0, added=0, cast=0, merges=[],
                          n_rays=self.n_rays)

    def set_training_cameras(self, cameras):
        if not self.cfg["camera_extent_lr"]:
            return
        centres = torch.stack([torch.linalg.inv(c["viewmat"])[:3, 3] for c in cameras])
        extent = 1.1 * float(torch.linalg.vector_norm(centres - centres.mean(0), dim=1).max())
        extent = max(extent, 0.01)
        group = self.optimizers["means"].param_groups[0]
        group["lr"] = group["lr"] / self.runtime_scene_scale * extent
        self.state["camera_extent"] = extent
        self.extent_applied = True

    def before_backward(self, params, optimizers, step, outputs):
        pass

    @torch.no_grad()
    def error_map(self, outputs):
        frame = outputs["conegs_frame"]
        return (outputs["rgb"].detach() - frame["image"]).abs().mean(-1).clamp_min(0.0)

    @torch.no_grad()
    def cast(self, outputs, n_rays):
        """Official cast_rays_during_optimization + select_for_densification."""
        frame = outputs["conegs_frame"]
        weights = self.error_map(outputs).reshape(-1)
        n = min(n_rays, int((weights > 0).sum()))
        if n == 0:
            return 0
        sel = torch.multinomial(weights, n, replacement=False)
        width = int(frame["width"])
        py, px = sel // width, sel % width
        K_inv, R, o = self.caster.camera(frame["K"], frame["viewmat"])
        origins, dirs, norm, radii = RayCaster.rays(K_inv, R, o, px, py)
        pts, col, _, std, valid = self.model.median_points(
            origins, dirs, norm, radii, batch=self.cfg["ingp_batch"]
        )
        pts, col, std = pts[valid], col[valid], std[valid]
        means = self.caster.to_world(pts)
        std_world = std / self.caster.scale * self.cfg["size_mult"]
        self.stash.append({
            "means": means,
            "scales": std_world.clamp_min(1e-12).log()[:, None].expand(-1, 3).contiguous(),
            "sh0": ((col - 0.5) / SH_C0)[:, None, :],
        })
        return len(means)

    @torch.no_grad()
    def merge(self, params, optimizers):
        """Official initialize_new_points(use_cone_radius=True, max_points=budget)."""
        cfg = self.cfg
        count = sum(len(s["means"]) for s in self.stash)
        if count <= 1:
            return 0
        new = {k: torch.cat([s[k] for s in self.stash]) for k in ("means", "scales", "sh0")}
        n = len(params["means"])
        if cfg["budget"] and n + count > cfg["budget"]:
            keep = max(cfg["budget"] - n, 0)
            new = {k: v[:keep] for k, v in new.items()}
        self.stash = []
        k = len(new["means"])
        if k == 0:
            return 0
        dev = params["means"].device
        quats = torch.zeros(k, 4, device=dev)
        quats[:, 0] = 1
        new["quats"] = quats
        new["opacities"] = torch.full(
            (k,), math.log(cfg["new_opacity"] / (1 - cfg["new_opacity"])), device=dev
        )
        new["shN"] = torch.zeros(k, *params["shN"].shape[1:], device=dev)
        append_gaussians(params, optimizers, new)
        return k

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        if not self.extent_applied:
            raise RuntimeError("conegs-density needs set_training_cameras() for the means LR")
        if 0 < step < cfg["densify_until"]:
            self.state["cast"] += self.cast(outputs, self.n_rays)
        if step % cfg["interval"] == 0:
            pruned = 0
            if step < cfg["densify_until"]:
                mask = params["opacities"].sigmoid() <= cfg["prune_opacity"]
                pruned = int(mask.sum())
                if pruned and pruned < len(mask):
                    remove(params, optimizers, {}, mask)
                elif pruned:
                    raise RuntimeError("Opacity pruning would remove every Gaussian")
            before = len(params["means"])
            added = self.merge(params, optimizers)
            n = len(params["means"])
            self.n_rays = rays_to_cast(n, added if cfg["budget"] else None, cfg["growth"],
                                       cfg["added_mult"], cfg["interval"], cfg["min_rays"])
            self.state["pruned"] += pruned
            self.state["added"] += added
            self.state["n_rays"] = self.n_rays
            if pruned or added:
                self.state["merges"].append([step, pruned, added, before, n])
            for name, p in params.items():
                if not (p.is_leaf and p.requires_grad):
                    raise RuntimeError(f"{name} is no longer a trainable leaf after densification")


def create_density(spec):
    return ConeGSDensity(spec)
