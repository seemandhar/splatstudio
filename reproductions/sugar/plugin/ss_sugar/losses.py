"""SuGaR losses: opacity entropy, SDF estimation R (Eq. 8), normal R_Norm (Eq. 10), mesh normal
consistency. Code forms follow sugar_trainers/coarse_sdf.py and refine.py."""

import torch
import torch.nn.functional as F

from .context import current_context
from .geometry import (
    camera_center,
    inverse_scaled_rotation,
    lookup_depth,
    neighbour_density,
    normal_consistency,
    quat_to_rotmat,
    sample_in_gaussians,
    saturate_density,
    sdf_values,
    smallest_axis,
    std_along,
    world_to_camera,
)

ENTROPY = {"weight": 0.1, "start": 7000, "end": 9000}
SDF = {
    "weight": 0.2, "start": 9000, "samples": 1_000_000, "sampling_scale": 1.5,
    "density_factor": 1.0 / 16.0, "density_threshold": 1.0, "close_threshold": 2.0,
    "znear": 0.01,
}
NORMAL = {"weight": 0.2, "start": 9000}
CONSISTENCY = {"weight": 0.1}


def _zero(outputs):
    return outputs["rgb"].sum() * 0


def _visible(outputs):
    return (outputs["info"]["radii"][0] > 0).all(dim=-1)


def create_entropy(spec):
    cfg = {**ENTROPY, **spec.config}
    ctx = current_context()

    def loss(params, outputs, target):
        # coarse_sdf.py:528-541: 0.1 * binary entropy of visible opacities for start < it < end.
        step = outputs["step"]
        if not cfg["start"] < step < cfg["end"] or ctx.mesh is not None:
            return _zero(outputs)
        o = outputs["opacities"][_visible(outputs)]
        if len(o) == 0:
            return _zero(outputs)
        h = -o * torch.log(o + 1e-10) - (1 - o) * torch.log(1 - o + 1e-10)
        return cfg["weight"] * h.mean()

    return loss


def sdf_terms(ctx, cfg, outputs):
    """Samples and fields shared by R and R_Norm for this step (coarse_sdf.py:560-597)."""
    step = outputs["step"]
    if ctx.cache.get("step") == step:
        return ctx.cache["terms"]
    ctx.cache = {"step": step, "terms": None}
    if ctx.knn is None:
        raise RuntimeError("SuGaR SDF regularization needs the neighbour table (density)")
    means, quats, scales = outputs["gaussians"]
    cam = outputs["camera"]
    viewmat, K, w, h = cam["viewmat"], cam["K"], cam["width"], cam["height"]
    rot = quat_to_rotmat(quats)
    cam_pts = world_to_camera(means, viewmat)
    max_z = cam_pts[:, 2].max().detach()
    # Depth map rendered with colours = camera z and background = max z (gradients kept).
    a = outputs["alpha"][..., 0]
    depth = outputs["depth_accumulated"][..., 0] + (1 - a) * max_z
    visible = _visible(outputs)
    with torch.no_grad():
        to_cam = F.normalize(camera_center(viewmat) - means, dim=-1)
        stds = std_along(rot, scales, to_cam)
        map_z = lookup_depth(depth, cam_pts, K, w, h)
        close = (map_z - cam_pts[:, 2]).abs() < cfg["close_threshold"] * stds
        mask = visible & close
    if not mask.any():
        return None
    x, gidx = sample_in_gaussians(means, rot, scales, mask, int(cfg["samples"]),
                                  cfg["sampling_scale"])
    opac = outputs["opacities"]
    nidx = ctx.knn[gidx]
    inv_sr = inverse_scaled_rotation(rot, scales)
    d, weights, _ = neighbour_density(x, nidx, means, inv_sr, opac, cfg["density_factor"])
    d = saturate_density(d)
    terms = {"x": x, "gidx": gidx, "nidx": nidx, "weights": weights, "d": d, "rot": rot,
             "scales": scales, "means": means, "depth": depth, "n_mask": int(mask.sum())}
    ctx.cache["terms"] = terms
    return terms


def create_sdf(spec):
    cfg = {**SDF, **spec.config}
    ctx = current_context()

    def loss(params, outputs, target):
        if outputs["step"] <= cfg["start"] or ctx.mesh is not None:
            return _zero(outputs)
        t = sdf_terms(ctx, cfg, outputs)
        if t is None:
            return _zero(outputs)
        cam = outputs["camera"]
        # beta = mean of the neighbours' smallest scales ("average" mode, sugar_model.py:1223).
        beta = t["scales"].min(dim=-1).values[t["nidx"]].mean(dim=1)
        f = sdf_values(t["d"], beta, cfg["density_threshold"])
        xc = world_to_camera(t["x"], cam["viewmat"])
        proj = xc[:, 2] > cfg["znear"]
        f_hat = lookup_depth(t["depth"], xc[proj], cam["K"], cam["width"], cam["height"]) - xc[
            proj, 2]
        extent = ctx.extent
        err = (f[proj] - f_hat.abs()).abs() / (extent / 10.0)
        return cfg["weight"] * err.clamp(max=10.0 * extent).mean()

    return loss


def create_normal(spec):
    cfg = {**SDF, **NORMAL, **spec.config}
    ctx = current_context()

    def loss(params, outputs, target):
        if outputs["step"] <= cfg["start"] or ctx.mesh is not None:
            return _zero(outputs)
        t = sdf_terms(ctx, cfg, outputs)
        if t is None:
            return _zero(outputs)
        # coarse_sdf.py:614-640 (sdf_better_normal_gradient_through_normal_only = True).
        normals = smallest_axis(t["rot"], t["scales"])
        nidx, gidx = t["nidx"], t["gidx"]
        min_s = t["scales"].min(dim=-1).values[nidx].detach()
        n_nb = normals[nidx]
        n_g = normals[gidx]
        n_nb = n_nb * torch.sign((n_nb * n_g[:, None]).sum(-1, keepdim=True)).detach()
        dist = ((t["x"][:, None] - t["means"][nidx]) * n_nb).sum(-1).abs().detach()
        w = t["weights"].detach() * dist / min_s.clamp_min(1e-6) ** 2
        w = w / w.sum(dim=-1).detach().unsqueeze(-1).clamp_min(1e-6)
        err = (n_g - (w[..., None] * n_nb).sum(dim=-2)).square().sum(dim=-1)
        return cfg["weight"] * err.mean()

    return loss


def create_consistency(spec):
    cfg = {**CONSISTENCY, **spec.config}
    ctx = current_context()

    def loss(params, outputs, target):
        # refine.py:747-751: pytorch3d mesh_normal_consistency * 0.1 during refinement.
        mesh = ctx.mesh
        if mesh is None:
            return _zero(outputs)
        return cfg["weight"] * normal_consistency(mesh.verts, mesh.pairs)

    return loss
