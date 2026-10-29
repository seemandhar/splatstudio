"""Gaussian state and optimizer alignment checks shared by training and probes."""

import torch
from torch import nn

from .contracts import BASE_FIELDS


def initialize(method, points, colors):
    n = len(points)
    if n < 2 or points.shape != (n, 3) or colors.shape != (n, 3):
        raise ValueError("Initialization requires at least two XYZ/RGB points")
    # A bounded subset avoids an N x N distance allocation on a full SfM point cloud.
    sample = points[: min(n, 1024)]
    distances = torch.cdist(sample, sample)
    distances.fill_diagonal_(float("inf"))
    scale = distances.min(dim=1).values.median().clamp_min(1e-4)
    quats = torch.zeros(n, 4, device=points.device)
    quats[:, 0] = 1
    if method.representation == "surfel2d":
        # Oriented disks need diverse initial normals, unlike isotropic 3D ellipsoids.
        quats = torch.nn.functional.normalize(torch.randn_like(quats), dim=-1)
    params = nn.ParameterDict(
        {
            "means": nn.Parameter(points.clone()),
            "scales": nn.Parameter(torch.full_like(points, float(scale.log()))),
            "quats": nn.Parameter(quats),
            "opacities": nn.Parameter(
                torch.full((n,), torch.logit(torch.tensor(0.1)).item(), device=points.device)
            ),
            "sh0": nn.Parameter(((colors - 0.5) / 0.28209479177387814)[:, None, :]),
            "shN": nn.Parameter(torch.zeros(n, 15, 3, device=points.device)),
        }
    )
    rates = {
        "means": 1.6e-4,
        "scales": 0.005,
        "quats": 0.001,
        "opacities": 0.05,
        "sh0": 0.0025,
        "shN": 0.0025 / 20,
    }
    for f in method.fields:
        params[f.name] = nn.Parameter(
            torch.full((n, *f.shape), f.initial_value, device=points.device)
        )
        rates[f.name] = f.learning_rate
    optimizers = {
        name: torch.optim.Adam([param], lr=rates[name], eps=1e-15) for name, param in params.items()
    }
    check_state(method, params, optimizers)
    return params, optimizers


def check_state(method, params, optimizers):
    expected = BASE_FIELDS | {f.name for f in method.fields}
    if set(params) != expected or set(optimizers) != expected:
        raise ValueError("Parameter, optimizer, and declared field names differ")
    n = len(params["means"])
    shapes = {
        "means": (3,),
        "scales": (3,),
        "quats": (4,),
        "opacities": (),
        "sh0": (1, 3),
        "shN": (15, 3),
    }
    shapes.update({f.name: f.shape for f in method.fields})
    if n == 0:
        raise ValueError("Density control removed all Gaussians")
    for name, p in params.items():
        if tuple(p.shape) != (n, *shapes[name]):
            raise ValueError(f"{name}: field shape or Gaussian indexing is inconsistent")
        if not torch.isfinite(p).all():
            raise ValueError(f"{name}: nonfinite Gaussian state")
        opt = optimizers[name]
        owned = [v for group in opt.param_groups for v in group["params"]]
        if len(owned) != 1 or owned[0] is not p:
            raise ValueError(f"{name}: optimizer references a stale or wrong parameter")
        if any(v is not p for v in opt.state):
            raise ValueError(f"{name}: optimizer retains state for a stale parameter")
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            moment = opt.state.get(p, {}).get(key)
            if moment is not None and (moment.shape != p.shape or not torch.isfinite(moment).all()):
                raise ValueError(f"{name}: invalid optimizer moment {key}")


def sample_scene(method, device="cpu", n=12):
    g = torch.Generator(device=device).manual_seed(42)
    points = torch.rand(n, 3, generator=g, device=device) - 0.5
    points[:, 2] += 3
    colors = torch.rand(n, 3, generator=g, device=device) * 0.6 + 0.2
    return initialize(method, points, colors)
