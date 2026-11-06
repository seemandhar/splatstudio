"""GaussianPro density control: baseline 3DGS densification + progressive propagation (§4.2).

Standard clone/split/prune/opacity reset: ``splatstudio.backends.gsplat.Density`` (exactly our
gsplat baseline, configured like ``make_method``). Added (official train.py:105-187 and
``GaussianModel.densify_from_depth_propagation`` @ 86bb048):

  after_backward(step) for the training view v rendered this step, if
      propagated_iteration_begin < step < propagated_iteration_after, step % interval == 0 and
      v has not been propagated yet (``propagation_dict``):
    1. D_r = rendered accumulated depth of v; (D_p, N_p) = propagate(v + neighbours, D_r)
       (vendored kernel); v's depth cache := D_p; N_p -> camera frame.
    2. error = |D_p - D_r| / D_p > σ(step)  (σ linear max -> min over the window)
    3. for each neighbour s: depth_s = cache[s] or (render s, propagate(s + *v's* neighbour list),
       cache) — official; consistent_s = check_geometric_consistency(D_p, depth_s, 2 px, 0.01)
    4. Q = #consistent >= 2; normal target of v := N_p (-10 outside Q)
    5. mask = (D_p != 300) & error & Q;  D_p[~Q] = D_p[~mask] = 300 *in place* (the cached tensor)
    6. if mask.sum() > 100: back-project mask pixels on the [::8, ::8] grid with D_p; colour =
       GT image of the *previous* step (official variable reuse), SH rest 0, scale
       log sqrt(distCUDA2(new ∪ old)), identity rotation, opacity sigmoid = 1 (official
       inverse_sigmoid(1.0) = +inf; a finite logit 20 gives sigmoid = 1.0 exactly in float32
       and an exactly zero gradient, and passes the runtime's finiteness check);
       then zero all densification statistics (``densification_postfix``).
  Then the baseline statistics update / densify / reset run (official order: propagation at
  the start of an iteration, densification at its end). The geometry pass' screen gradient is
  added to the RGB pass' (official passes share ``screenspace_points``).
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from gsplat.strategy.ops import _update_param_with_optimizer
from splatstudio.backends.gsplat import Density as BaseDensity

from . import geometry as G
from . import kernels
from .link import LINK

DEFAULTS = {
    # baseline (make_method) densification
    "grow_grad2d": 0.0002,
    "opacity_reset": True,
    # propagation (paper §5.1 / demo.sh / arguments defaults)
    "propagation": True,
    "propagation_interval": 50,
    "propagated_iteration_begin": 1000,
    "propagated_iteration_after": 12000,
    "depth_error_max_threshold": 1.0,
    "depth_error_min_threshold": 0.8,
    "patch_size": 20,
    "depth_min": 0.1,
    "depth_max": 20.0,          # dataset '360' (waymo: 80)
    "num_neighbours": 4,
    "geo_pixel_threshold": 2.0,
    "geo_depth_threshold": 0.01,
    "geo_min_views": 2,
    "min_new_pixels": 100,
    "backproject_stride": 8,
    "new_opacity_logit": 20.0,
    "color_from_previous_view": True,
    "add_geometry_gradient": True,
    "cache_device": "cuda",
    "data": None,
    "downscale": None,
    "sparse_dir": None,
}


def _argv_value(flag):
    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


@torch.no_grad()
def append_gaussians(params, optimizers, new):
    """Concatenate new rows (dict name -> tensor) with zero Adam moments."""
    n_new = len(new["means"])

    def param_fn(name, p):
        return torch.nn.Parameter(torch.cat([p, new[name].to(p)]), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, torch.zeros((n_new, *v.shape[1:]), device=v.device, dtype=v.dtype)])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


class Density:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown GaussianPro density options: {sorted(unknown)}")
        self.c = {**DEFAULTS, **spec.config}
        base_cfg = {"absgrad": False, "grow_grad2d": self.c["grow_grad2d"],
                    "opacity_reset": self.c["opacity_reset"]}
        self.base = BaseDensity(SimpleNamespace(config=base_cfg))
        self.requires = set(self.base.requires)
        self.prev_view = None
        self.counts = {"propagations": 0, "insertions": 0, "inserted": 0, "src_propagations": 0,
                       "propagation_seconds": 0.0}
        self.state = {"counts": self.counts, "events": []}
        LINK.reset_views()

    # -- runtime hooks -------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        self.base.initialize(params, optimizers, scene_scale)
        self.state["extent"] = float(scene_scale)

    def set_training_cameras(self, cameras):
        """Runtime hook: reload the training split's images + COLMAP tracks for the graph."""
        if not self.c["propagation"]:
            return
        root = self.c["data"] or _argv_value("--data")
        if root is None:
            raise RuntimeError("gaussianpro-density needs the training images: set density "
                               "config 'data' or run through ss-train --data")
        downscale = int(self.c["downscale"] or _argv_value("--downscale") or 1)
        from splatstudio.data import load_dataset

        t0 = time.perf_counter()
        dataset = load_dataset(Path(root), downscale)
        names = dataset["splits"]["train"]
        if len(names) != len(cameras):
            raise RuntimeError("Reloaded training split does not match the runtime's cameras")
        device = cameras[0]["K"].device
        images = []
        for name, cam in zip(names, cameras):
            frame = dataset["frames"][name]
            same = (frame["width"] == cam["width"] and frame["height"] == cam["height"]
                    and torch.allclose(frame["K"].to(device), cam["K"])
                    and torch.allclose(frame["viewmat"].to(device), cam["viewmat"]))
            if not same:
                raise RuntimeError(f"Reloaded training view {name} differs from the runtime camera")
            images.append(frame["image"])
        sparse = G.find_sparse_dir(dataset["root"], self.c["sparse_dir"])
        obs = G.read_colmap_image_points(sparse / "images.bin")
        keys = [Path(n).name for n in names]
        missing = [k for k in keys if k not in obs]
        if missing:
            raise RuntimeError(f"COLMAP model {sparse} lacks training images {missing[:3]}")
        neighbours = G.shared_point_neighbours(keys, obs, self.c["num_neighbours"])
        del dataset
        self.set_training_frames(cameras, images, neighbours)
        self.state["sparse_dir"] = str(sparse)
        self.state["load_seconds"] = time.perf_counter() - t0

    def set_training_frames(self, cameras, images, neighbours):
        """Register cameras, HWC [0,1] images and neighbour lists (training indices)."""
        u8 = [(im.detach().float().cpu() * 255).round().clamp(0, 255).to(torch.uint8).contiguous()
              for im in images]
        LINK.set_views([dict(c) for c in cameras], u8, neighbours)
        self.state["neighbours"] = {"views": len(cameras),
                                    "mean_count": sum(map(len, neighbours)) / len(cameras),
                                    "without": sum(not n for n in neighbours)}

    def before_backward(self, params, optimizers, step, outputs):
        if self.c["propagation"] and not LINK.active:
            raise RuntimeError("gaussianpro-density: training views not registered")
        self.base.before_backward(params, optimizers, step, outputs)
        if "geo_info" in outputs and self.c["add_geometry_gradient"]:
            outputs["geo_info"]["means2d"].retain_grad()

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        c = self.c
        view = outputs.get("gp_view")
        if self.should_propagate(step, view):
            self.propagate(params, optimizers, step, view)
        geo = outputs.get("geo_info")
        if geo is not None and c["add_geometry_gradient"] and geo["means2d"].grad is not None:
            main = outputs["info"]["means2d"]
            main.grad = geo["means2d"].grad if main.grad is None else main.grad + geo["means2d"].grad
        self.base.after_backward(params, optimizers, step, outputs)
        self.prev_view = view

    # -- propagation -------------------------------------------------------------------------
    def should_propagate(self, step, view):
        c = self.c
        return (c["propagation"] and view is not None
                and c["propagated_iteration_begin"] < step < c["propagated_iteration_after"]
                and step % c["propagation_interval"] == 0 and not LINK.propagated[view])

    def threshold(self, step):
        c = self.c
        return G.threshold(step, c["propagated_iteration_begin"], c["propagated_iteration_after"],
                           c["depth_error_max_threshold"], c["depth_error_min_threshold"])

    def rendered_depth(self, params, index):
        return LINK.renderer.geometry(params, LINK.cameras[index])["depth"]

    def run_kernel(self, ref, depth, sources):
        """graphics_utils.depth_propagation (images as (img * 255).to(uint8), like official)."""
        c = self.c
        views = [ref] + list(sources)
        imgs = torch.stack([((LINK.images[i].float() / 255.0) * 255).to(torch.uint8)
                            for i in views])
        Ks = torch.stack([LINK.cameras[i]["K"].float() for i in views])
        poses = torch.stack([LINK.cameras[i]["viewmat"].float() for i in views])
        lo, hi = float(c["depth_min"]), float(c["depth_max"])
        interval = torch.tensor([lo, (hi - lo) / 192.0, 192.0, hi]).expand(len(views), 4)
        d, n = kernels.propagate(imgs, Ks, poses, depth, interval, c["patch_size"])
        dev = depth.device
        return d.to(dev), n.to(dev)

    def propagate(self, params, optimizers, step, v):
        c = self.c
        t0 = time.perf_counter()
        LINK.propagated[v] = True
        cam = LINK.cameras[v]
        dev = params["means"].device
        rendered = self.rendered_depth(params, v)
        sources = LINK.neighbours[v]
        prop, normal_w = self.run_kernel(v, rendered, sources)
        LINK.depth_cache[v] = prop
        normal = normal_w @ cam["viewmat"][:3, :3].float().T  # world -> camera (R_w2c @ n)
        valid = prop != G.INVALID_DEPTH
        error_mask = (prop - rendered).abs() / prop > self.threshold(step)
        counts = torch.zeros(prop.shape, dtype=torch.int32, device=dev)
        c2w = torch.linalg.inv(cam["viewmat"].float())
        for s in sources:
            src_depth = LINK.depth_cache[s]
            if src_depth is None:
                src_depth, _ = self.run_kernel(s, self.rendered_depth(params, s), sources)
                LINK.depth_cache[s] = src_depth
                self.counts["src_propagations"] += 1
            scam = LINK.cameras[s]
            counts += G.check_geometric_consistency(
                prop, cam["K"].float(), c2w, src_depth, scam["K"].float(),
                torch.linalg.inv(scam["viewmat"].float()),
                c["geo_pixel_threshold"], c["geo_depth_threshold"]).int()
        consistent = counts >= c["geo_min_views"]
        normal[~consistent] = G.INVALID_NORMAL
        LINK.normal_targets[v] = normal.to(c["cache_device"])
        mask = valid & error_mask & consistent
        prop[~consistent] = G.INVALID_DEPTH
        prop[~mask] = G.INVALID_DEPTH
        n_mask = int(mask.sum())
        inserted = 0
        if n_mask > c["min_new_pixels"]:
            inserted = self.insert(params, optimizers, v, prop, mask)
        self.counts["propagations"] += 1
        self.counts["propagation_seconds"] += time.perf_counter() - t0
        event = {"step": step, "view": v, "mask": n_mask, "consistent": int(consistent.sum()),
                 "inserted": inserted, "threshold": self.threshold(step)}
        self.state["events"].append(event)
        return event

    def insert(self, params, optimizers, v, depth, mask):
        c = self.c
        cam = LINK.cameras[v]
        dev = params["means"].device
        pts, sub = G.backproject(depth, cam["K"].float(), cam["viewmat"].float(), mask,
                                 c["backproject_stride"])
        m = len(pts)
        if m == 0:
            return 0
        color_view = self.prev_view if (c["color_from_previous_view"]
                                        and self.prev_view is not None) else v
        image = LINK.image_float(color_view, dev)
        if image.shape[:2] != depth.shape:
            raise RuntimeError("previous-step image size differs from the propagated view")
        s = c["backproject_stride"]
        colors = image[::s, ::s][sub]
        allpts = torch.cat([pts, params["means"].detach()], dim=0)
        dist2 = kernels.dist_cuda2(allpts).clamp_min(1e-7)[:m]
        n_sh = params["shN"].shape[1]
        quats = torch.zeros(m, 4, device=dev)
        quats[:, 0] = 1
        new = {
            "means": pts,
            "scales": torch.log(torch.sqrt(dist2))[:, None].repeat(1, 3),
            "quats": quats,
            "opacities": torch.full((m,), float(c["new_opacity_logit"]), device=dev),
            "sh0": ((colors - 0.5) / G.SH_C0)[:, None, :],
            "shN": torch.zeros(m, n_sh, 3, device=dev),
        }
        extra = set(params) - set(new)
        if extra:
            raise RuntimeError(f"GaussianPro insertion does not know fields {sorted(extra)}")
        append_gaussians(params, optimizers, new)
        self.reset_statistics(len(params["means"]), dev)
        self.counts["insertions"] += 1
        self.counts["inserted"] += m
        return m

    def reset_statistics(self, n, dev):
        """densification_postfix: accumulators and max radii zeroed for every Gaussian."""
        st = self.base.state
        for key in ("grad2d", "count", "radii"):
            if st.get(key) is not None:
                st[key] = torch.zeros(n, device=dev)


def create_density(spec):
    return Density(spec)

