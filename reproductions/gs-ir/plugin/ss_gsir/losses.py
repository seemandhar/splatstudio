"""GS-IR losses (official train.py training loop; paper Eq. 9, 16 and supp. Eq. 1-5).

Stage 1: L_c = 0.8 L1 + 0.2 D-SSIM (component name ``photometric``), L_n-p = L1 between the
rendered Gaussian normal and the depth-derived normal on valid pixels (weight 1), TV_normal
(weight 5). Stage 2: L_shade = L1(PBR, gt) (no D-SSIM), TV_mat on [albedo, roughness,
metallic] (weight 1, masked when some pixels are uncovered), roughness/metallic prior 0.001
(code-only), TV_light on the cube map read as a 512 x 1024 lat-long image (weight 0.01).
Inactive terms return an exact zero attached to the graph.
"""

import torch
import torch.nn.functional as F
from splatstudio.losses import ssim

from . import bank as bank_mod
from . import cubemap
from .geometry import masked_tv_loss, tv_loss

COLOR_DEFAULTS = {"ssim_weight": 0.2}
NORMAL_DEFAULTS = {"weight": 1.0, "tv_weight": 5.0}
MATERIAL_DEFAULTS = {"tv_weight": 1.0, "lamb_weight": 0.001}
LIGHT_DEFAULTS = {"tv_weight": 0.01, "height": 512, "width": 1024}


def _cfg(defaults, spec):
    unknown = set(spec.config) - set(defaults)
    if unknown:
        raise ValueError(f"{spec.name}: unknown options {sorted(unknown)}")
    return {**defaults, **spec.config}


def _zero(outputs):
    return outputs["rgb"].sum() * 0.0


def _stage(outputs):
    return int(outputs.get("gsir_stage", 1))


def create_color(spec):
    c = _cfg(COLOR_DEFAULTS, spec)
    w = float(c["ssim_weight"])

    def loss(params, outputs, target):
        rgb = outputs["rgb"]
        if _stage(outputs) == 2:
            return F.l1_loss(rgb, target)  # train.py:330
        return (1 - w) * F.l1_loss(rgb, target) + w * (1 - ssim(rgb, target))

    return loss


def normal_terms(outputs, target):
    n, nd = outputs["normal"], outputs["normal_from_depth"]
    mask = (nd != 0).all(dim=-1)
    if bool(mask.any()):
        lnp = F.l1_loss(n[mask], nd[mask])
    else:
        lnp = n.sum() * 0.0
    tv = tv_loss(target.permute(2, 0, 1), n.permute(2, 0, 1))
    return lnp, tv


def create_normal(spec):
    c = _cfg(NORMAL_DEFAULTS, spec)

    def loss(params, outputs, target):
        if _stage(outputs) != 1 or "normal" not in outputs:
            return _zero(outputs)
        lnp, tv = normal_terms(outputs, target)
        return float(c["weight"]) * lnp + float(c["tv_weight"]) * tv

    return loss


def material_terms(gb, target):
    mask = gb["mask"]  # [H, W, 1] bool
    brdf = torch.cat((gb["albedo"], gb["roughness"], gb["metallic"]), dim=-1).permute(2, 0, 1)
    gt = target.permute(2, 0, 1)
    if bool((~mask).any()):
        tv = masked_tv_loss(mask.permute(2, 0, 1), gt, brdf)
    else:
        tv = tv_loss(gt, brdf)
    m = mask[..., 0]
    if bool(m.any()):
        lamb = (1.0 - gb["roughness"][m]).mean() + gb["metallic"][m].mean()
    else:
        lamb = gb["roughness"].sum() * 0.0
    return tv, lamb


def create_material(spec):
    c = _cfg(MATERIAL_DEFAULTS, spec)

    def loss(params, outputs, target):
        if _stage(outputs) != 2:
            return _zero(outputs)
        tv, lamb = material_terms(outputs["gbuffer"], target)
        return float(c["tv_weight"]) * tv + float(c["lamb_weight"]) * lamb

    return loss


_DIRS = {}


def env_tv(base, h, w):
    key = (h, w, str(base.device))
    if key not in _DIRS:
        _DIRS[key] = cubemap.latlong_dirs(h, w, base.device, centered=True)
    env = cubemap.sample(base, _DIRS[key], "linear")  # [h, w, 3]
    return (env[1:] - env[:-1]).pow(2).mean() + (env[:, 1:] - env[:, :-1]).pow(2).mean()


def create_light(spec):
    c = _cfg(LIGHT_DEFAULTS, spec)

    def loss(params, outputs, target):
        bank = bank_mod.ACTIVE["bank"]
        if _stage(outputs) != 2 or bank is None or bank.base is None:
            return _zero(outputs)
        return float(c["tv_weight"]) * env_tv(bank.base, int(c["height"]), int(c["width"]))

    return loss
