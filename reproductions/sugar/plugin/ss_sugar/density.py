"""SuGaR stage controller: 3DGS densification, coarse pruning / neighbour tables, level-set mesh
extraction and the switch to mesh-bound Gaussians with a learnable mesh."""

import math

import numpy as np
import torch
from gsplat.strategy.ops import remove
from splatstudio.backends.gsplat import Density as BaseDensity
from splatstudio.specs import ComponentSpec

from .context import Mesh, current_context
from .geometry import (
    BARY,
    C0,
    bound_gaussians,
    camera_center,
    edge_face_pairs,
    expon_lr,
    initial_plane_log_scales,
    knn,
    level_set_points,
    quat_to_rotmat,
)
from .mesh import poisson_mesh

DEFAULTS = {
    # Stage boundaries (absolute training steps; official: 3DGS 7k, coarse to 15k).
    "coarse_start": 7000,
    "prune_at": 9000,
    "prune_opacity": 0.5,
    "knn": 16,
    "knn_every": 500,
    "extract_at": 15000,
    # Mesh extraction (coarse_mesh.py, train.py defaults).
    "extract_prune_opacity": 0.5,
    "surface_level": 0.3,
    "n_total_points": 10_000_000,
    "n_points_in_range": 21,
    "range_size": 3.0,
    "depth_alpha_min": 0.5,
    "poisson_depth": 10,
    "density_quantile": 0.1,
    "fg_bbox_factor": 1.0,
    "bg_bbox_factor": 4.0,
    "decimation_target": 1_000_000,
    "project_on_surface_points": True,
    "open3d_path": "~/models/sugar-pylib",
    "cpu_threads": 16,  # Open3D Poisson / cKDTree workers (shared host; not a paper setting)
    # Refinement (refine.py, train.py --high_poly).
    "gaussians_per_triangle": 1,
    "n_vertices_in_fg": 1_000_000,
    "vertex_lr_init": 0.00016,
    "vertex_lr_final": 0.0000016,
    "vertex_lr_max_steps": 30000,
    "refine_opacity": 0.1,
    "thickness_factor": 1e-6,
    "sh_warmup_every": 1000,
}

STRATEGY_KEYS = {"absgrad", "grow_grad2d", "refine_stop_iter", "reset_every", "opacity_reset",
                 "prune_opa", "refine_start_iter", "refine_every", "grow_scale3d",
                 "prune_scale3d", "verbose"}


def replace_population(params, optimizers, values):
    """Swap every field for a new population; fresh Adam state (the official refinement builds a
    new optimizer). Learning rates of the existing optimizers are kept."""
    for name, value in values.items():
        p = torch.nn.Parameter(value.contiguous())
        params[name] = p
        opt = optimizers[name]
        opt.param_groups[0]["params"] = [p]
        opt.state.clear()


class Density:
    requires = {"gradient.screen.signed"}

    def __init__(self, spec):
        config = dict(spec.config)
        strategy = {k: config.pop(k) for k in list(config) if k in STRATEGY_KEYS}
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown sugar density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **config}
        if self.cfg["gaussians_per_triangle"] not in BARY:
            raise ValueError("gaussians_per_triangle must be 1, 3, 4 or 6")
        strategy.setdefault("refine_stop_iter", self.cfg["coarse_start"])
        self.base = BaseDensity(ComponentSpec(spec.name, "density", "", spec.requires, (),
                                              strategy))
        self.ctx = current_context()
        self.state = {"strategy": None, "mesh": None}
        self.log = []

    # -- runtime hooks -------------------------------------------------------------------
    def initialize(self, params, optimizers, scene_scale=1.0):
        self.base.initialize(params, optimizers, scene_scale)
        self.state["strategy"] = self.base.state
        if self.ctx.extent is None:
            self.ctx.extent = float(scene_scale)

    def set_training_cameras(self, cameras):
        self.ctx.cameras = cameras
        centers = torch.stack([camera_center(c["viewmat"]) for c in cameras])
        mean = centers.mean(dim=0)
        # sugar_model.py:853 get_cameras_spatial_extent (= 3DGS getNerfppNorm radius).
        self.ctx.extent = 1.1 * float((centers - mean).norm(dim=-1).max())
        self.ctx.center = mean

    def before_backward(self, params, optimizers, step, outputs):
        if step <= self.cfg["coarse_start"] and self.ctx.mesh is None:
            self.base.before_backward(params, optimizers, step, outputs)

    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        nxt = step + 1
        if self.ctx.mesh is None:
            if step <= cfg["coarse_start"]:
                self.base.after_backward(params, optimizers, step, outputs)
                if step == cfg["coarse_start"]:
                    self.state["strategy"] = None  # 3DGS densification is over
            if step == cfg["prune_at"]:
                # coarse_sdf.py:478-489, applied at the start of it prune_at + 1.
                with torch.no_grad():
                    mask = params["opacities"].sigmoid() < cfg["prune_opacity"]
                if mask.any():
                    remove(params, optimizers, {}, mask)
                self.log.append({"step": step, "pruned": int(mask.sum())})
                self.ctx.knn = None
            if step == cfg["extract_at"]:
                self.extract_and_bind(params, optimizers)
            elif nxt > cfg["coarse_start"] and nxt <= cfg["extract_at"] and (
                nxt == cfg["coarse_start"] + 1 or nxt % cfg["knn_every"] == 0
                or self.ctx.knn is None
            ):
                self.ctx.knn = knn(params["means"], cfg["knn"], workers=cfg["cpu_threads"])
        else:
            self.refine_step(params, step)
        if self.ctx.mesh is not None:
            i = nxt - cfg["extract_at"]
            self.ctx.sh_cap = min(max(i - 1, 0) // cfg["sh_warmup_every"], 3)

    # -- stage C: mesh extraction -------------------------------------------------------
    @torch.no_grad()
    def level_set_cloud(self, params):
        cfg, ctx = self.cfg, self.ctx
        if not ctx.cameras:
            raise RuntimeError("SuGaR mesh extraction needs the training cameras")
        keep = params["opacities"].sigmoid() >= cfg["extract_prune_opacity"]
        sub = {k: v[keep] for k, v in params.items()}
        means = sub["means"]
        rot = quat_to_rotmat(sub["quats"])
        scales = sub["scales"].exp()
        opac = sub["opacities"].sigmoid()
        from scipy.spatial import cKDTree

        tree = cKDTree(means.float().cpu().numpy())

        def knn_fn(x, k):
            _, idx = tree.query(x.float().cpu().numpy(), k=k, workers=cfg["cpu_threads"])
            return torch.from_numpy(np.asarray(idx).astype(np.int64)).to(x.device)

        per_frame = int(cfg["n_total_points"] / len(ctx.cameras)) + 1
        pts, nrm, col = [], [], []
        for cam in ctx.cameras:
            out = ctx.renderer.render(sub, cam, sh_degree=0)
            p, n, c = level_set_points(
                out["depth"][..., 0], out["alpha"][..., 0], out["rgb"].clamp(0, 1), cam, means,
                rot, scales, opac, 2 * per_frame, cfg["surface_level"], knn_fn=knn_fn,
                n_in_range=cfg["n_points_in_range"], range_size=cfg["range_size"], k=cfg["knn"],
                alpha_min=cfg["depth_alpha_min"],
            )
            sel = torch.randperm(len(p), device=p.device)[:per_frame]
            pts.append(p[sel].cpu())
            nrm.append(n[sel].cpu())
            col.append(c[sel].cpu())
        return (torch.cat(pts).numpy(), torch.cat(nrm).numpy(), torch.cat(col).numpy(),
                int(keep.sum()))

    def extract_and_bind(self, params, optimizers):
        cfg, ctx = self.cfg, self.ctx
        ctx.knn = None
        ctx.cache = {}
        points, normals, colors, n_used = self.level_set_cloud(params)
        verts, faces, vcolors = poisson_mesh(
            points, normals, colors, ctx.center.cpu().numpy(), ctx.extent,
            depth=cfg["poisson_depth"], quantile=cfg["density_quantile"],
            target=cfg["decimation_target"], fg_factor=cfg["fg_bbox_factor"],
            bg_factor=cfg["bg_bbox_factor"], project=cfg["project_on_surface_points"],
            open3d_path=cfg["open3d_path"], threads=cfg["cpu_threads"],
        )
        device = params["means"].device
        self.log.append({"extraction": {"gaussians_used": n_used, "level_points": len(points),
                                        "vertices": len(verts), "faces": len(faces)}})
        print({"sugar_mesh": self.log[-1]["extraction"]}, flush=True)
        self.bind(params, optimizers, torch.from_numpy(verts).to(device),
                  torch.from_numpy(faces).to(device), torch.from_numpy(vcolors).to(device))

    # -- stage D: mesh-bound Gaussians ------------------------------------------------------
    def bind(self, params, optimizers, verts, faces, vcolors):
        cfg, ctx = self.cfg, self.ctx
        n = cfg["gaussians_per_triangle"]
        bary = torch.tensor(BARY[n][0], device=verts.device, dtype=verts.dtype)
        total = len(faces) * n
        log_thickness = math.log(ctx.extent * cfg["thickness_factor"])
        plane = initial_plane_log_scales(verts, faces, n)
        rot2d = torch.zeros(total, 2, device=verts.device)
        rot2d[:, 0] = 1.0
        colors = (bary[None, :, :, None] * vcolors[faces][:, None]).sum(dim=-2).reshape(-1, 3)
        means, quats, _ = bound_gaussians(verts, faces, bary, rot2d, plane, log_thickness)
        values = {
            "means": means,
            "scales": torch.cat((torch.full_like(plane[:, :1], log_thickness), plane), dim=-1),
            "quats": quats,
            "opacities": torch.full((total,), math.log(cfg["refine_opacity"] /
                                                       (1 - cfg["refine_opacity"])),
                                    device=verts.device),
            "sh0": ((colors - 0.5) / C0)[:, None, :],
            "shN": torch.zeros(total, params["shN"].shape[1], 3, device=verts.device),
            "sugar_rot2d": rot2d,
        }
        extra = set(params) - set(values)
        if extra:
            raise RuntimeError(f"No refinement initialisation for fields {sorted(extra)}")
        replace_population(params, optimizers, values)
        v = torch.nn.Parameter(verts.clone())
        # refine.py:426: spatial_lr_scale = 10 * bbox_radius / sqrt(n_vertices_in_fg).
        self.lr_scale = 10.0 * ctx.extent / math.sqrt(cfg["n_vertices_in_fg"])
        opt = torch.optim.Adam([v], lr=cfg["vertex_lr_init"] * self.lr_scale, eps=1e-15)
        ctx.mesh = Mesh(v, faces, bary, log_thickness, edge_face_pairs(faces, len(verts)), opt)
        ctx.sh_cap = 0
        self.state["mesh"] = {"vertices": v.data, "faces": faces, "bary": bary,
                              "log_thickness": torch.tensor(log_thickness)}

    def vertex_lr(self, refine_step):
        cfg = self.cfg
        return expon_lr(refine_step, cfg["vertex_lr_init"] * self.lr_scale,
                        cfg["vertex_lr_final"] * self.lr_scale, cfg["vertex_lr_max_steps"])

    def refine_step(self, params, step):
        mesh = self.ctx.mesh
        lr = self.vertex_lr(step - self.cfg["extract_at"])
        mesh.optimizer.param_groups[0]["lr"] = lr
        mesh.optimizer.step()
        mesh.optimizer.zero_grad(set_to_none=True)
        self.sync(params)

    @torch.no_grad()
    def sync(self, params):
        """Write the mesh-derived world means/quats/thickness into the base fields so that the
        checkpoint renders without the mesh (ss-eval builds a fresh renderer)."""
        mesh = self.ctx.mesh
        means, quats, _ = bound_gaussians(mesh.verts, mesh.faces, mesh.bary, params["sugar_rot2d"],
                                          params["scales"][:, 1:], mesh.log_thickness)
        params["means"].copy_(means)
        params["quats"].copy_(quats)
        params["scales"][:, 0] = mesh.log_thickness


def create_density(spec):
    return Density(spec)
