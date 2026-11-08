"""Confidence-aware fixed-budget densification (paper Sec. 3, arXiv 2610.13541).

Keeps the 3DGS-MCMC frame (densify every 100 it in (500, 25000), +5 % per step up to the cap,
gsplat's `compute_relocation` opacity/scale rule) and replaces
  * opacity-proportional source sampling  -> 80 % Eq. pdf (ranked |grad|_abs x confidence factor)
                                             + 20 % SSIM term                      (Sec. 3.1)
  * opacity-threshold relocation          -> every 400 it, lowest max blending weight w_i with
                                             the share schedule of Sec. 3.2
  * continuous position noise             -> disabled; new means ~ N(mu_i, Sigma'_i)  (Sec. 3.3)
  * inherited colour                      -> ground-truth colour lookup in the first view where
                                             the new primitive has radius < 16 px  (Sec. 3.3)
"""

import math

import torch
from gsplat.relocation import compute_relocation
from gsplat.strategy.ops import _update_param_with_optimizer
from gsplat.utils import normalized_quat_to_rotmat

from .kernels import blend_stats
from .losses import SSIM_ERROR_KEY, TARGET_KEY

C0 = 0.28209479177387814

DEFAULTS = {
    "cap_max": 1_000_000,
    "refine_start": 500,
    "refine_stop": 25000,
    "refine_every": 100,
    "growth": 0.05,
    "relocate_every": 400,
    "share": 0.10,
    "share_full_start": 5000,
    "falloff_start": 10000,
    "falloff_sigma": 5000.0,
    "eta_ramp": 5000,
    "grad_fraction": 0.8,
    "conf_offset": 1.2,
    "min_opacity": 0.005,
    "color_init_radius": 16.0,
    "color_init_timeout": 400,
    "init_opacity": 0.5,
    "seed": 0,
}


# ---------------------------------------------------------------- pure functions (unit-tested)
def normalized_rank(x):
    """r_i(x) in [0, 1]: smallest value 0, largest 1; ties share their lowest rank."""
    n = x.numel()
    if n < 2:
        return torch.zeros_like(x, dtype=torch.float32)
    sorted_x = torch.sort(x).values
    r = torch.searchsorted(sorted_x.contiguous(), x.contiguous(), right=False)
    return r.float() / (n - 1)


def eta(step, ramp):
    """eta = min(t / 5000, 1) (Sec. 3.1)."""
    return min(step / ramp, 1.0)


def sampling_pdf(absgrad, confidence, step, ramp=5000, offset=1.2):
    """Eq. pdf: P_i = r(grad_abs) ((1 - eta) + eta (1.2 - r(gamma_bar)))."""
    e = eta(step, ramp)
    return normalized_rank(absgrad) * ((1 - e) + e * (offset - normalized_rank(confidence)))


def relocation_share(step, share=0.1, full_start=5000, falloff_start=10000, sigma=5000.0):
    """Sec. 3.2: (share, zero_only). Before 5k only w_i = 0 (at most `share`); 5k-10k the
    lowest `share`; from 10k scaled by exp(-1/2 ((t - 10k) / 5k)^2)."""
    if step < full_start:
        return share, True
    if step < falloff_start:
        return share, False
    return share * math.exp(-0.5 * ((step - falloff_start) / sigma) ** 2), False


def select_relocation(w, exempt, step, cfg, generator=None):
    """Indices of primitives to relocate (Sec. 3.2)."""
    n = w.numel()
    share, zero_only = relocation_share(
        step, cfg["share"], cfg["share_full_start"], cfg["falloff_start"], cfg["falloff_sigma"]
    )
    k_max = int(share * n)
    if k_max <= 0:
        return torch.empty(0, dtype=torch.long, device=w.device)
    eligible = ~exempt
    if zero_only:
        cand = (eligible & (w <= 0)).nonzero(as_tuple=True)[0]
        if len(cand) > k_max:
            perm = torch.randperm(len(cand), device=w.device, generator=generator)[:k_max]
            cand = cand[perm]
        return cand
    cand = eligible.nonzero(as_tuple=True)[0]
    k = min(k_max, len(cand))
    order = torch.argsort(w[cand])[:k]
    return cand[order]


def draw_sources(pdf, ssim_term, n, grad_fraction, allowed, generator=None):
    """One draw of n sources: round(0.8 n) from pdf, the rest from the SSIM term (Sec. 3.1)."""
    n_grad = int(round(grad_fraction * n))
    n_ssim = n - n_grad
    out = []
    for weights, k, label in ((pdf, n_grad, "Eq. pdf"), (ssim_term, n_ssim, "SSIM term")):
        if k == 0:
            continue
        p = torch.where(allowed, weights.clamp_min(0), torch.zeros_like(weights))
        if not torch.isfinite(p).all() or p.sum() <= 0:
            raise RuntimeError(f"confidence-densify: {label} distribution is empty at a "
                               "densification step (no observed primitives)")
        out.append(torch.multinomial(p, k, replacement=True, generator=generator))
    idx = torch.cat(out)
    return idx[torch.randperm(len(idx), device=idx.device, generator=generator)]


def sample_positions(means, quats, scales, generator=None):
    """mu' ~ N(mu, R diag(s')^2 R^T) (Sec. 3.3, GOF)."""
    rot = normalized_quat_to_rotmat(torch.nn.functional.normalize(quats, dim=-1))
    z = torch.randn(means.shape, device=means.device, generator=generator)
    return means + torch.einsum("nij,nj->ni", rot, scales * z)


# ---------------------------------------------------------------- density component
class ConfidenceDensity:
    requires = {"gradient.screen.absolute", "render.confidence"}

    def __init__(self, spec):
        cfg = {**DEFAULTS, **spec.config}
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown cad-density options: {sorted(unknown)}")
        self.cfg = cfg
        self.state = None
        self.generator = None
        self.binoms = None

    # state: per-primitive running statistics, kept aligned with the parameters
    def _zeros(self, n, device, dtype=torch.float32):
        return torch.zeros(n, device=device, dtype=dtype)

    def initialize(self, params, optimizers, scene_scale=1.0):
        n, device = len(params["means"]), params["means"].device
        with torch.no_grad():
            # 3DGS-MCMC initialisation opacity (paper silent; MCMC code uses 0.5).
            o = float(self.cfg["init_opacity"])
            params["opacities"].fill_(math.log(o / (1 - o)))
        self.generator = torch.Generator(device=device).manual_seed(int(self.cfg["seed"]))
        n_max = 51
        b = torch.zeros((n_max, n_max))
        for i in range(n_max):
            for k in range(i + 1):
                b[i, k] = math.comb(i, k)
        self.binoms = b.to(device)
        self.state = {
            "grad_sum": self._zeros(n, device),
            "ssim_sum": self._zeros(n, device),
            "count": self._zeros(n, device),
            "max_w": self._zeros(n, device),
            "exempt": self._zeros(n, device, torch.bool),
            "pending_age": torch.full((n,), -1, device=device, dtype=torch.int32),
            "log": [],
        }

    def before_backward(self, params, optimizers, step, outputs):
        outputs["info"]["means2d"].retain_grad()

    # ------------------------------------------------------------ per-step statistics
    @torch.no_grad()
    def _accumulate(self, outputs):
        info = outputs["info"]
        st = self.state
        vis = (info["radii"][0] > 0).all(-1)
        absgrad = info["means2d"].absgrad[0]
        st["grad_sum"][vis] += absgrad[vis].norm(dim=-1)
        st["count"][vis] += 1
        max_w, err = blend_stats(info, outputs[SSIM_ERROR_KEY])  # [H,W] 1-SSIM(p)
        torch.maximum(st["max_w"], max_w, out=st["max_w"])
        st["ssim_sum"] += err

    @torch.no_grad()
    def _color_init(self, params, optimizers, outputs):
        st = self.state
        pending = st["pending_age"] >= 0
        if not pending.any():
            return 0
        info = outputs["info"]
        radii = info["radii"][0]
        ok = pending & (radii > 0).all(-1) & (radii.max(-1).values < self.cfg["color_init_radius"])
        idx = ok.nonzero(as_tuple=True)[0]
        if len(idx):
            target = outputs[TARGET_KEY]
            h, w = target.shape[:2]
            uv = info["means2d"][0, idx].detach()
            grid = torch.stack((2 * uv[:, 0] / w - 1, 2 * uv[:, 1] / h - 1), -1)
            c_gt = torch.nn.functional.grid_sample(
                target.permute(2, 0, 1)[None], grid[None, None], mode="bilinear",
                padding_mode="border", align_corners=False,
            )[0, :, 0].T  # [k,3]
            set_dc_to_color(params, idx, c_gt, outputs["camera"]["viewmat"], outputs["sh_degree"])
            reset_moments(optimizers, params, idx)
            st["pending_age"][idx] = -1
        still = st["pending_age"] >= 0
        st["pending_age"][still] += 1
        st["pending_age"][st["pending_age"] > self.cfg["color_init_timeout"]] = -1
        return len(idx)

    # ------------------------------------------------------------ densification step
    @torch.no_grad()
    def _densify(self, params, optimizers, step):
        cfg, st = self.cfg, self.state
        n = len(params["means"])
        n_add = max(0, min(int(cfg["cap_max"]), int((1 + cfg["growth"]) * n)) - n)
        relocating = step % cfg["relocate_every"] == 0
        dead = (
            select_relocation(st["max_w"], st["exempt"], step, cfg, self.generator)
            if relocating else torch.empty(0, dtype=torch.long, device=params["means"].device)
        )
        n_total = n_add + len(dead)
        if n_total > 0:
            allowed = torch.ones(n, dtype=torch.bool, device=dead.device)
            allowed[dead] = False
            cnt = st["count"].clamp_min(1)
            pdf = sampling_pdf(st["grad_sum"] / cnt, params["confidence"].exp().flatten(), step,
                               cfg["eta_ramp"], cfg["conf_offset"])
            src = draw_sources(pdf, st["ssim_sum"] / cnt, n_total, cfg["grad_fraction"], allowed,
                               self.generator)
            place(params, optimizers, st, src, dead, self.binoms, cfg["min_opacity"],
                  self.generator)
        if relocating:
            st["max_w"].zero_()
            st["exempt"].zero_()
            st["exempt"][dead] = True
        st["grad_sum"].zero_()
        st["ssim_sum"].zero_()
        st["count"].zero_()
        st["log"].append({"step": step, "added": n_add, "relocated": len(dead),
                          "gaussians": len(params["means"])})

    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        if step >= cfg["refine_stop"] and not (self.state["pending_age"] >= 0).any():
            return
        if TARGET_KEY not in outputs:
            raise RuntimeError("cad-density needs the confidence-photometric loss outputs")
        if step < cfg["refine_stop"]:
            self._accumulate(outputs)
        self._color_init(params, optimizers, outputs)
        if (cfg["refine_start"] < step < cfg["refine_stop"]
                and step % cfg["refine_every"] == 0):
            self._densify(params, optimizers, step)


# ---------------------------------------------------------------- state operations
@torch.no_grad()
def place(params, optimizers, state, src, dead, binoms, min_opacity, generator=None):
    """Copy sources to `dead` slots (relocation) and to appended slots (growth) in one op.

    MCMC rule (gsplat compute_relocation, ratio = copies of the source + 1) adjusts opacity and
    scale of the source and its copies; copies inherit rotation, SH, confidence; their means are
    sampled from the source Gaussian with the adjusted scale; optimiser moments of sources and
    copies are zeroed.
    """
    n = len(params["means"])
    n_dead = len(dead)
    n_new = len(src) - n_dead
    device = params["means"].device
    opac = params["opacities"].sigmoid()
    counts = torch.bincount(src, minlength=n)
    eps = torch.finfo(torch.float32).eps
    new_o, new_s = compute_relocation(
        opacities=opac[src], scales=params["scales"].exp()[src], ratios=counts[src] + 1,
        binoms=binoms,
    )
    new_o = new_o.clamp(min=min_opacity, max=1 - eps)
    dest = torch.cat((dead, torch.arange(n, n + n_new, device=device)))
    means_new = sample_positions(params["means"][src], params["quats"][src], new_s, generator)

    def param_fn(name, p):
        p = p.data
        if name == "opacities":
            p[src] = torch.logit(new_o)
        elif name == "scales":
            p[src] = torch.log(new_s)
        values = p[src].clone()
        if name == "means":
            values = means_new
        if n_new:
            p = torch.cat((p, torch.empty((n_new, *p.shape[1:]), device=device, dtype=p.dtype)))
        p[dest] = values
        return torch.nn.Parameter(p, requires_grad=True)

    def optimizer_fn(key, v):
        if n_new:
            v = torch.cat((v, torch.zeros((n_new, *v.shape[1:]), device=device, dtype=v.dtype)))
        v[src] = 0
        v[dest] = 0
        return v

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    for k, v in list(state.items()):
        if not isinstance(v, torch.Tensor):
            continue
        if n_new:
            fill = -1 if k == "pending_age" else 0
            v = torch.cat((v, torch.full((n_new,), fill, device=device, dtype=v.dtype)))
        v[dest] = 0  # fresh statistics; pending_age 0 queues the colour initialisation
        state[k] = v
    return dest


@torch.no_grad()
def set_dc_to_color(params, idx, color, viewmat, sh_degree):
    """Set sh0 so that clamp(SH(dir) + 0.5) equals `color` in this view, keeping SH l >= 1."""
    from gsplat import spherical_harmonics

    campos = torch.linalg.inv(viewmat)[:3, 3]
    dirs = params["means"][idx] - campos[None]
    coeffs = torch.cat((torch.zeros_like(params["sh0"][idx]), params["shN"][idx]), dim=1)
    rest = spherical_harmonics(sh_degree, dirs, coeffs)
    params["sh0"].data[idx] = ((color - 0.5 - rest) / C0)[:, None, :]


@torch.no_grad()
def reset_moments(optimizers, params, idx):
    for name, opt in optimizers.items():
        st = opt.state.get(params[name], {})
        for key in ("exp_avg", "exp_avg_sq"):
            if key in st:
                st[key][idx] = 0


def create_density(spec):
    return ConfidenceDensity(spec)
