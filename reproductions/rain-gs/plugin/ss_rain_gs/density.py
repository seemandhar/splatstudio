"""RAIN-GS density control: SLV random initialization, ABE-Split, warm-up schedule.

Port of the official `--ours_new` path (github.com/cvlab-kaist/RAIN-GS, MIT, KU-CVLAB):
- SLV initialization (paper §4.1, Alg. 1; scene/dataset_readers.py): N=10 random points in a
  cube of side 3(hi-lo) centred at (lo+hi)/2 on every axis, where lo/hi are the scalar min/max
  over all COLMAP `tvec` coordinates (world->camera translations); sh0 from U[0,1) (with the
  official PLY uint8 round trip), 3-NN log scales (distCUDA2), identity rotation, opacity 0.1.
  The native runtime initializes from SfM points; this component replaces that population
  in `set_training_cameras` (called once before step 1).
- 3DGS adaptive density control (scene/gaussian_model.py) with the RAIN-GS changes: densify
  until 15000 + warmup 10000, split scale divisor divide_ratio*N = 1.4, and ABE-Split (paper
  §4.3, Alg. 2 l.21-22) while iteration <= warmup: every split candidate also emits a copy at
  mean * 0.3 * scene_extent with its unchanged scale (official code: B_0 = world origin).
- Warm-up (supp. A.1): means LR fixed at its initial value until warmup, then the 3DGS
  exponential schedule evaluated at (iteration - warmup) with max_steps 30000.
"""

import math

import torch
from gsplat.strategy.ops import _update_param_with_optimizer, duplicate, remove, reset_opa
from gsplat.utils import normalized_quat_to_rotmat

SH_C0 = 0.28209479177387814
DEFAULTS = {
    "slv_init": True,
    "num_init": 10,
    "cube_factor": 3.0,
    "init_opacity": 0.1,
    "warmup": 10000,
    "densify_from": 500,
    "densify_until": 25000,
    "densify_every": 100,
    "grad_threshold": 0.0002,
    "percent_dense": 0.01,
    "divide_ratio": 0.7,
    "abe_split": True,
    "abe_factor": 0.3,
    "prune_opacity": 0.005,
    "prune_big_world": 0.1,
    "prune_big_after": 3000,
    "reset_every": 3000,
    "reset_value": 0.01,
    "means_lr_final_ratio": 0.01,  # position_lr_final / position_lr_init
    "means_lr_max_steps": 30000,
}


def slv_cube(viewmats):
    """Official cube: (lo, side) from the scalar min/max over all camera tvec coordinates."""
    tvec = torch.stack([v[:3, 3] for v in viewmats]).double()
    lo, hi = float(tvec.min()), float(tvec.max())
    side = (hi - lo) * 3.0
    # xyz = U * (hi-lo)*3 - (cube_mean - mean_cam_pos), cube_mean = 1.5 (hi-lo)
    offset = 1.5 * (hi - lo) - (lo + hi) / 2.0
    return -offset, side


def knn3_log_scales(points):
    """distCUDA2: mean squared distance to the 3 nearest neighbours -> log sqrt, isotropic."""
    d2 = torch.cdist(points.double(), points.double()).square()
    d2.fill_diagonal_(float("inf"))
    k = min(3, len(points) - 1)
    mean = d2.topk(k, dim=1, largest=False).values.mean(dim=1).clamp_min(1e-7)
    return mean.sqrt().log().float()[:, None].expand(-1, 3).contiguous()


def official_sh0(shs):
    """storePly(SH2RGB(shs)*255 as uint8) -> fetchPly /255 -> RGB2SH."""
    rgb8 = torch.floor((shs.double() * SH_C0 + 0.5) * 255.0).clamp(0, 255)
    return ((rgb8 / 255.0 - 0.5) / SH_C0).float()


def means_lr(step, lr_init, warmup, final_ratio=0.01, max_steps=30000):
    """LR used at `step`: fixed before warmup, then get_expon_lr_func(step - warmup)."""
    if step < warmup:
        return lr_init
    t = min(max((step - warmup) / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_init * final_ratio) * t)


@torch.no_grad()
def replace_population(params, optimizers, new):
    """Swap every field for new rows (fresh trainable leaves, empty Adam state)."""
    for name, value in new.items():
        old = params[name]
        p = torch.nn.Parameter(value.to(old.device, old.dtype).contiguous(), requires_grad=True)
        params[name] = p
        opt = optimizers[name]
        opt.state.pop(old, None)
        opt.param_groups[0]["params"] = [p]


@torch.no_grad()
def abe_append(params, optimizers, state, mask, factor):
    """ABE-Split third Gaussian: mean * factor (factor = 0.3 * extent), other fields copied."""
    sel = torch.where(mask)[0]

    def param_fn(name, p):
        rows = p[sel] * factor if name == "means" else p[sel]
        return torch.nn.Parameter(torch.cat([p.detach(), rows]), requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, v.new_zeros((len(sel), *v.shape[1:]))])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for k, v in list(state.items()):
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, v[sel]))


@torch.no_grad()
def split_ratio(params, optimizers, state, mask, divisor):
    """3DGS split into 2 samples from N(mu, Sigma); scale / divisor (official 0.7 * 2 = 1.4)."""
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]
    scales = torch.exp(params["scales"][sel])
    quats = torch.nn.functional.normalize(params["quats"][sel], dim=-1)
    rotmats = normalized_quat_to_rotmat(quats)
    samples = torch.einsum("nij,nj,bnj->bni", rotmats, scales,
                           torch.randn(2, len(scales), 3, device=scales.device))

    def param_fn(name, p):
        repeats = [2] + [1] * (p.dim() - 1)
        if name == "means":
            rows = (p[sel] + samples).reshape(-1, 3)
        elif name == "scales":
            rows = torch.log(scales / divisor).repeat(2, 1)
        else:
            rows = p[sel].repeat(repeats)
        return torch.nn.Parameter(torch.cat([p[rest].detach(), rows]),
                                  requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v[rest], v.new_zeros((2 * len(sel), *v.shape[1:]))])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for k, v in list(state.items()):
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v[rest], v[sel].repeat([2] + [1] * (v.dim() - 1))))


class RainDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown rain-gs density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.requires = {"gradient.screen.signed"}
        self.state = None
        self.optimizers = None
        self.initialized = False

    def initialize(self, params, optimizers, scene_scale=1.0):
        self.optimizers = optimizers
        self.params = params
        self.state = {
            "grad2d": None,
            "count": None,
            "scene_scale": float(scene_scale),
            "lr_init": float(optimizers["means"].param_groups[0]["lr"]),
            "events": [],
        }
        self.initialized = not self.cfg["slv_init"]

    @torch.no_grad()
    def set_training_cameras(self, cameras):
        """SLV random initialization (Alg. 1), replacing the runtime's SfM population."""
        if not self.cfg["slv_init"]:
            return
        params, cfg = self.params, self.cfg
        device = params["means"].device
        n = int(cfg["num_init"])
        lo, side = slv_cube([c["viewmat"].cpu() for c in cameras])
        side = side / 3.0 * float(cfg["cube_factor"])
        xyz = (torch.rand(n, 3, dtype=torch.float64) * side + lo).float().to(device)
        shs = torch.rand(n, 3, dtype=torch.float64)
        quats = torch.zeros(n, 4, device=device)
        quats[:, 0] = 1
        p0 = float(cfg["init_opacity"])
        new = {
            "means": xyz,
            "scales": knn3_log_scales(xyz),
            "quats": quats,
            "opacities": torch.full((n,), math.log(p0 / (1 - p0)), device=device),
            "sh0": official_sh0(shs)[:, None, :].to(device),
            "shN": torch.zeros(n, *params["shN"].shape[1:], device=device),
        }
        extra = set(params) - set(new)
        if extra:
            raise ValueError(f"rain-gs SLV init does not know how to initialize {sorted(extra)}")
        replace_population(params, self.optimizers, new)
        self.state.update(grad2d=None, count=None, init_cube_lo=lo, init_cube_side=side,
                          init_count=n)
        self.initialized = True

    def before_backward(self, params, optimizers, step, outputs):
        if not self.initialized:
            raise RuntimeError("rain-gs: SLV init needs set_training_cameras before training")
        info = outputs["info"]
        if info.get("rain_step", step) != step:
            raise RuntimeError(f"rain-gs renderer counter {info['rain_step']} != step {step}")
        info["means2d"].retain_grad()

    def _update_state(self, params, info):
        grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        n = len(params["means"])
        if self.state["grad2d"] is None:
            self.state["grad2d"] = torch.zeros(n, device=grads.device)
            self.state["count"] = torch.zeros(n, device=grads.device)
        sel = (info["radii"] > 0.0).all(dim=-1)
        ids = torch.where(sel)[1]
        self.state["grad2d"].index_add_(0, ids, grads[sel].norm(dim=-1))
        self.state["count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    @torch.no_grad()
    def densify(self, params, optimizers, step):
        cfg, state = self.cfg, self.state
        extent = state["scene_scale"]
        grads = state["grad2d"] / state["count"]
        grads[grads.isnan()] = 0.0
        high = grads >= cfg["grad_threshold"]
        big = torch.exp(params["scales"]).max(dim=-1).values > cfg["percent_dense"] * extent
        is_clone = high & ~big
        is_split = high & big
        n_clone, n_split = int(is_clone.sum()), int(is_split.sum())
        if n_clone:
            duplicate(params, optimizers, state, is_clone)
        is_split = torch.cat([is_split, is_split.new_zeros(n_clone)])
        n_abe = 0
        if cfg["abe_split"] and step <= cfg["warmup"] and n_split:
            abe_append(params, optimizers, state, is_split, cfg["abe_factor"] * extent)
            n_abe = n_split
            is_split = torch.cat([is_split, is_split.new_zeros(n_abe)])
        if n_split:
            split_ratio(params, optimizers, state, is_split, cfg["divide_ratio"] * 2)
        prune = torch.sigmoid(params["opacities"].flatten()) < cfg["prune_opacity"]
        if step > cfg["prune_big_after"]:
            prune |= (torch.exp(params["scales"]).max(dim=-1).values
                      > cfg["prune_big_world"] * extent)
        n_prune = int(prune.sum())
        if n_prune:
            if n_prune == len(prune):
                raise RuntimeError("rain-gs density would remove every Gaussian")
            remove(params, optimizers, state, prune)
        state["grad2d"].zero_()
        state["count"].zero_()
        return {"step": step, "clone": n_clone, "split": n_split, "abe": n_abe,
                "prune": n_prune, "total": len(params["means"])}

    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        if step < cfg["densify_until"]:
            self._update_state(params, outputs["info"])
            if step > cfg["densify_from"] and step % cfg["densify_every"] == 0:
                event = self.densify(params, optimizers, step)
                if step % 1000 == 0 or event["abe"]:
                    self.state["events"].append(event)
            if step % cfg["reset_every"] == 0:
                reset_opa(params, optimizers, self.state, value=cfg["reset_value"])
        # LR for the next step (runtime already applied its own schedule; override it).
        optimizers["means"].param_groups[0]["lr"] = means_lr(
            step + 1, self.state["lr_init"], cfg["warmup"], cfg["means_lr_final_ratio"],
            cfg["means_lr_max_steps"])
        for name, p in params.items():
            if not (p.is_leaf and p.requires_grad):
                raise RuntimeError(f"rain-gs: field {name} is no longer a trainable leaf")


def create_density(spec):
    return RainDensity(spec)
