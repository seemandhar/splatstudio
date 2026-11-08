"""Loss-aware density control (arXiv 2611.02044, Alg. 1): Split -> Clone -> Decay.

Per training step s < densify_until (the runtime calls `after_backward` after the optimizer
step; all statistics use the pre-step parameters snapshotted in `before_backward`):
  for Gaussians visible in this view (radii > 0):
    S      += S_phi(view)           joint position-log-scale splitting matrix (Eq. 5, App. A.4)
    g_pos  += dL/dp, g_scl += dL/dlog s, g2d += ||screen gradient|| (SteepGS 'uncertain' stat)
    g/h_dup, g/h_dec += clone / decay surrogate terms (Eqs. 12, 16) with proxy curvature omega
    denom  += 1
At s > densify_from, s % densify_every == 0:
    split (Eq. 8/9) -> clone (Eq. 13) -> opacity decay (Eq. 17) -> prune (o < 0.005, world size
    after 3000) -> reset all accumulators. No global opacity reset (replaced by decay, Sec. 4.4).
"""

import math

import torch
from gsplat import spherical_harmonics
from gsplat.strategy.ops import _update_param_with_optimizer, remove
from torch import nn

from . import surrogates as sg
from .kernel import joint_split_accumulate, surrogate_stats
from .losses import TARGET_CAPABILITY
from .renderer import CAMERA_CAPABILITY

DEFAULTS = {
    "tau_pos": 1e-4,  # App. A.1
    "tau_scale": 1e-4,  # App. A.1
    "tau_split": -1e-8,  # App. A.1
    "tau_clone": 1e-6,  # App. A.1
    "tau_decay": 1e-4,  # App. A.1
    "eta_max": 1e-2,  # Eqs. 13, 17 upper bound
    "ssim_weight": 0.2,  # lambda_D-SSIM (3DGS)
    "charb_eps": 1e-3,  # App. A.5
    "gamma_ssim": 1e-4,  # App. A.5
    "eps_ssim": 1e-8,  # App. A.5 (value unspecified in the paper)
    "require_uncertain": True,  # 'unresolved view-space variation' = SteepGS uncertain filter
    "grad2d_threshold": 0.0002,  # SteepGS / 3DGS densify_grad_threshold
    "densify_from": 500,  # App. A.1
    "densify_every": 100,
    "densify_until": 15000,
    "prune_opa": 0.005,  # 3DGS
    "big_world_after": 3000,  # 3DGS: size prune once iteration > opacity_reset_interval
    "big_world": 0.1,
    "split": True,  # ablation switches (probes use them; paper = all True)
    "clone": True,
    "decay": True,
    "drop_mixed_term": True,  # Eq. app_Sps_impl
    # Alg. 1 prunes low-opacity Gaussians right after the three operations, which removes most
    # clones (opacity eta*o <= 0.01*o). False = paper-literal; True exempts this interval's new
    # clones from the opacity prune (diagnostic switch for gap analysis, not the paper).
    "exempt_new_clones_from_prune": False,
}
# The target image arrives through this plugin's photometric loss (it provides
# TARGET_CAPABILITY); the runtime only accepts renderer capabilities in density `requires`, so
# its presence is enforced at run time in `view_statistics` instead.
REQUIRES = {"gradient.screen.signed", CAMERA_CAPABILITY}
NEEDS_FROM_LOSS = TARGET_CAPABILITY
BASE_FIELDS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def append_gaussians(params, optimizers, new):
    """3DGS densification_postfix: append rows with zero Adam moments."""
    n_new = len(new["means"])

    def param_fn(name, p):
        return nn.Parameter(torch.cat([p.detach(), new[name].to(p.dtype)]),
                            requires_grad=p.requires_grad)

    def optimizer_fn(key, v):
        return torch.cat([v, v.new_zeros((n_new, *v.shape[1:]))])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)


def post_sh_colors(params, viewmat, degree):
    """gsplat rasterization(): SH -> RGB with clamp_min(+0.5, 0) (Inria convention)."""
    campos = torch.linalg.inv(viewmat)[:3, 3]
    sh = torch.cat((params["sh0"], params["shN"]), dim=1)
    return torch.clamp_min(spherical_harmonics(degree, params["means"] - campos, sh) + 0.5, 0.0)


class LossAwareDensity:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown loss-aware density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        if self.cfg["tau_split"] >= 0:
            raise ValueError("tau_split must be negative")
        self.requires = set(REQUIRES)
        self.state = None
        self._snap = None
        self._hooks = []
        self._grads = {}

    # ------------------------------------------------------------------ runtime hooks
    def initialize(self, params, optimizers, scene_scale=1.0):
        extra = set(params) - set(BASE_FIELDS)
        if extra:
            raise ValueError(f"loss-aware density does not know fields {sorted(extra)}")
        self.extent = float(scene_scale)
        self._reset_stats(len(params["means"]), params["means"].device)
        self.state = {"events": []}

    def before_backward(self, params, optimizers, step, outputs):
        if step >= self.cfg["densify_until"]:
            return
        info = outputs["info"]
        info["means2d"].retain_grad()
        outputs["rgb"].retain_grad()
        cam = info["ss_camera"]
        with torch.no_grad():
            self._snap = {k: params[k].detach().clone() for k in ("means", "scales", "quats")}
            self._snap["colors"] = post_sh_colors(params, cam["viewmat"], info["ss_sh_degree"])
        self._grads = {}
        for k in ("means", "scales"):
            def grab(g, k=k):
                self._grads[k] = g.detach().clone()
            self._hooks.append(params[k].register_hook(grab))

    @torch.no_grad()
    def after_backward(self, params, optimizers, step, outputs):
        cfg = self.cfg
        for h in self._hooks:
            h.remove()
        self._hooks = []
        if step >= cfg["densify_until"]:
            return
        self._accumulate(outputs)
        self._snap, self._grads = None, {}
        if step > cfg["densify_from"] and step % cfg["densify_every"] == 0:
            self._densify(params, optimizers, step)

    # ------------------------------------------------------------------ statistics
    def _reset_stats(self, n, device):
        z = lambda *s: torch.zeros(n, *s, device=device)  # noqa: E731
        self.S, self.denom = z(6, 6), z()
        self.g_pos, self.g_scl, self.g2d = z(3), z(3), z()
        self.sur = z(4)  # g_dup, h_dup, g_dec, h_dec

    def view_statistics(self, outputs):
        """Kernel statistics + per-view joint splitting matrix of the visible Gaussians."""
        cfg, info = self.cfg, outputs["info"]
        rgb, target = outputs["rgb"], outputs.get("target")
        if target is None or rgb.grad is None:
            raise RuntimeError("loss-aware density needs the target (ss_loss_aware_dc.losses) "
                               "and dL/dI (retained rgb grad)")
        cam = info["ss_camera"]
        W, H = int(cam["width"]), int(cam["height"])
        omega = sg.proxy_curvature(rgb.detach(), target, cfg["ssim_weight"], cfg["charb_eps"],
                                   cfg["gamma_ssim"], cfg["eps_ssim"])
        stats = surrogate_stats(info["means2d"][0].detach(), info["conics"][0].detach(),
                                info["opacities"][0].detach(), self._snap["colors"],
                                info["isect_offsets"][0], info["flatten_ids"],
                                rgb.detach(), rgb.grad, omega, W, H, info["tile_size"])
        vis = (info["radii"][0] > 0).all(dim=-1)
        idx = vis.nonzero()[:, 0]
        return stats, vis, idx, cam

    def _accumulate(self, outputs):
        info = outputs["info"]
        stats, vis, idx, cam = self.view_statistics(outputs)
        # S[idx] += S_phi(view) (CUDA hyper-dual implementation; torch reference in
        # surrogates.joint_split_matrix, compared in the lad-joint-split-cuda probe)
        joint_split_accumulate(idx, self._snap["means"], self._snap["scales"], self._snap["quats"],
                               stats, cam["viewmat"], cam["K"], int(cam["width"]),
                               int(cam["height"]), self.cfg["drop_mixed_term"], self.S)
        self.sur[idx] += stats[idx, 20:24]
        g2 = info["means2d"].grad[0].clone()
        g2[:, 0] *= info["width"] / 2.0  # gsplat DefaultStrategy normalisation (Inria NDC)
        g2[:, 1] *= info["height"] / 2.0
        self.g2d[idx] += torch.linalg.vector_norm(g2[idx], dim=-1)
        if "means" in self._grads:
            self.g_pos[idx] += self._grads["means"][idx]
        if "scales" in self._grads:
            self.g_scl[idx] += self._grads["scales"][idx]
        self.denom[idx] += 1

    # ------------------------------------------------------------------ Alg. 1
    def decisions(self, params):
        """Disjoint split / clone / decay sets from the accumulated statistics."""
        cfg = self.cfg
        n = len(params["means"])
        dev = params["means"].device
        seen = self.denom > 0
        den = self.denom.clamp_min(1)[:, None]
        gpos = torch.linalg.vector_norm(self.g_pos / den, dim=-1)
        gscl = torch.linalg.vector_norm(self.g_scl / den, dim=-1)
        uncertain = (self.g2d / den[:, 0]) >= cfg["grad2d_threshold"]
        cand = seen & (gpos <= cfg["tau_pos"]) & (gscl <= cfg["tau_scale"])
        if cfg["require_uncertain"]:
            cand &= uncertain
        lam = torch.full((n,), math.inf, device=dev)
        vmin = torch.zeros(n, 6, device=dev)
        ci = cand.nonzero()[:, 0]
        if cfg["split"] and len(ci):
            l_, v_, _ = sg.normalized_eigmin(self.S[ci] / self.denom[ci, None, None],
                                             params["scales"][ci], params["quats"][ci])
            lam[ci], vmin[ci] = l_, v_
        split = sg.select_split(gpos, gscl, lam, cfg["tau_pos"], cfg["tau_scale"],
                                cfg["tau_split"], uncertain if cfg["require_uncertain"] else None)
        split &= seen & cfg["split"]
        g_dup, h_dup, g_dec, h_dec = self.sur.unbind(-1)
        clone, eta_dup = sg.select_clone(g_dup, h_dup, split, cfg["tau_clone"], cfg["eta_max"])
        clone &= cfg["clone"]
        decay, eta_dec = sg.select_decay(g_dec, h_dec, split | clone, cfg["tau_decay"],
                                         cfg["eta_max"])
        decay &= cfg["decay"]
        return {"split": split, "clone": clone, "decay": decay, "eta_dup": eta_dup,
                "eta_dec": eta_dec, "lam": lam, "vmin": vmin, "candidates": cand,
                "uncertain": uncertain}

    def _densify(self, params, optimizers, step):
        cfg = self.cfg
        n0 = len(params["means"])
        d = self.decisions(params)
        split, clone, decay = d["split"], d["clone"], d["decay"]
        # ---- opacity decay (Eq. 17): o <- (1 - eta) o, in place (Adam moments kept)
        op = params["opacities"]
        o = op.data[decay].sigmoid()
        op.data[decay] = torch.logit((1 - d["eta_dec"][decay]) * o)
        # ---- clone (Eq. 13): exact duplicate with opacity eta * o, appended
        new = {k: params[k][clone].detach() for k in ("means", "scales", "quats", "sh0", "shN")}
        new["opacities"] = torch.logit(d["eta_dup"][clone] * params["opacities"][clone].sigmoid())
        append_gaussians(params, optimizers, new)
        n_clone = int(clone.sum())
        # ---- split (Eq. 9): two children along J v_min, opacity o/2, parent removed
        K = int(split.sum())
        si = split.nonzero()[:, 0]
        J = sg.normalization(params["scales"][si], params["quats"][si])
        means, logs = sg.split_children(params["means"][si], params["scales"][si],
                                        d["vmin"][si], J)
        new = {"means": means, "scales": logs, "quats": params["quats"][si].repeat(2, 1),
               "opacities": torch.logit(params["opacities"][si].sigmoid() * 0.5).repeat(2),
               "sh0": params["sh0"][si].repeat(2, 1, 1), "shN": params["shN"][si].repeat(2, 1, 1)}
        append_gaussians(params, optimizers, new)
        n_now = len(params["means"])
        drop = torch.zeros(n_now, dtype=torch.bool, device=split.device)
        drop[si] = True
        remove(params, optimizers, {}, drop)
        # ---- prune (Alg. 1 line 14; 3DGS thresholds)
        prune = params["opacities"].sigmoid() < cfg["prune_opa"]
        if cfg["exempt_new_clones_from_prune"]:
            keep = len(params["means"]) - 2 * K
            prune[keep - n_clone:keep] = False
        if step > cfg["big_world_after"]:
            prune |= params["scales"].exp().max(dim=1).values > cfg["big_world"] * self.extent
        remove(params, optimizers, {}, prune)
        self._reset_stats(len(params["means"]), params["means"].device)
        row = {"step": int(step), "n_before": n0, "split": K, "clone": n_clone,
               "decay": int(decay.sum()), "candidates": int(d["candidates"].sum()),
               "uncertain": int(d["uncertain"].sum()),
               "neg_curv": int((d["lam"] < cfg["tau_split"]).sum()),
               "pruned": int(prune.sum()), "n_after": len(params["means"])}
        self.state["events"].append(row)
        if step % 500 == 0:
            print({"loss-aware-dc": row}, flush=True)
        torch.cuda.empty_cache()


def create_density(spec):
    return LossAwareDensity(spec)
