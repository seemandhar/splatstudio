"""RDO-Gaussian renderer: Gaussian mask, adaptive SH mask and ECVQ before gsplat rasterization.

Training (an active, un-baked `RDOBank` in bank.ACTIVE) follows the official
`gaussian_renderer/__init__.py:render(activate_gsprune, activate_shprune, activate_vq)`:

  GS mask (Eq. 2-4):  M = sg(1[sigmoid(phi) > 0.1] - sigmoid(phi)) + sigmoid(phi)
                      s_hat = M exp(scales), alpha_hat = M sigmoid(opacities)
  SH mask (Eq. 6-7):  per degree l, the same STE on theta_l, applied to its 2l+1 coefficients
  ECVQ (Eq. 10):      on Gaussians with the GS bit mask: scale = s_hat, rot = raw quats,
                      dc = sh0; on Gaussians with the SH-l and GS bit masks: masked SH-l;
                      index = argmin ||x - c||_2 (+ bits / lambda once rate-constrained),
                      recomputed when (step - 1) % 10 == 0, else from the global index cache;
                      rendered value sg(x_hat - x) + x (rotation normalised afterwards).

`outputs["rdo"]` carries the official loss terms (normalisation of the official code):
  rate     = sum_key bits_key / lambda_key / N / 55
  vq       = sum_key sum_rows ||x_hat - x||_2 / N / 55
  gs_prune = mean_N sigmoid(phi)                                          (Eq. 5)
  sh_prune = mean over GS-kept rows of the coefficient-expanded sigmoid(theta)   (Eq. 8)
Without an active bank (any fresh process such as ss-eval) or after the final bake, the stored
fields are rendered as they are: they then hold the decoded model.
"""

import torch
from gsplat import rasterization

from . import bank as bank_mod
from .bank import DIMS, KEYS, SH_KEYS, SH_SLICES, VQ_DIM

GS_MASK = "rdo_gs_mask"
SH_MASK = "rdo_sh_mask"
# number of shN coefficients of each SH degree 1..3 (official apply_sh_mask expand 3/5/7)
SH_COUNTS = (3, 5, 7)


def ste_mask(logits, threshold):
    soft = torch.sigmoid(logits)
    return ((soft > threshold).float() - soft).detach() + soft, soft


def expand_sh(mask3):
    """[N, 3] per-degree values -> [N, 15, 1] per coefficient."""
    return torch.cat([mask3[:, i : i + 1].expand(-1, c) for i, c in enumerate(SH_COUNTS)],
                     dim=1)[..., None]


def rdo_attributes(bank, params, train=True):
    """Masked / quantized activated attributes and the RD terms (official render body)."""
    cfg = bank.cfg
    n = len(params["means"])
    device = params["means"].device
    scales = params["scales"].exp()
    opacities = torch.sigmoid(params["opacities"])
    quats = params["quats"]
    sh0 = params["sh0"]
    shn = params["shN"]
    zero = params["means"].sum() * 0.0
    rdo = {"rate": zero, "vq": zero, "gs_prune": zero, "sh_prune": zero, "bits": {}}

    gs_bit = torch.ones(n, dtype=torch.bool, device=device)
    if bank.gs_active:
        hard, soft = ste_mask(params[GS_MASK], cfg["gs_threshold"])
        scales = scales * hard[:, None]
        opacities = opacities * hard
        gs_bit = soft.detach() > cfg["gs_threshold"]
        rdo["gs_prune"] = soft.mean()
        rdo["gs_mask_percent"] = 1.0 - hard.detach().sum() / hard.numel()

    sh_bit = {k: torch.ones(n, dtype=torch.bool, device=device) for k in SH_KEYS}
    if bank.sh_active:
        soft3 = torch.sigmoid(params[SH_MASK])
        soft = expand_sh(soft3).expand(-1, -1, 3)
        hard = ((soft > cfg["sh_threshold"]).float() - soft).detach() + soft
        shn = shn * hard
        for i, k in enumerate(SH_KEYS):
            sh_bit[k] = soft3[:, i].detach() > cfg["sh_threshold"]
        kept_soft, kept_hard = soft[gs_bit], hard[gs_bit]
        rdo["sh_prune"] = kept_soft.mean() if kept_soft.numel() else zero
        if kept_hard.numel():
            rdo["sh_mask_percent"] = 1.0 - kept_hard.detach().sum() / kept_hard.numel()
    sh_bit = {k: torch.logical_and(v, gs_bit) for k, v in sh_bit.items()}

    if bank.vq_active:
        rows = {"scale": gs_bit, "rot": gs_bit, "dc": gs_bit, **sh_bit}
        inputs = {
            "scale": scales[gs_bit],
            "rot": quats[gs_bit],
            "dc": sh0[gs_bit].reshape(-1, 3),
        }
        for k in SH_KEYS:
            a, b = SH_SLICES[k]
            inputs[k] = shn[sh_bit[k], a:b, :].reshape(-1, DIMS[k])
        update = bank.update_index() or not train
        rate_terms, vq_terms, out = [], [], {}
        for k in KEYS:
            x = inputs[k]
            cache = bank.cache(k, n, device)
            index = None if update else cache[rows[k]]
            x_hat, bits, index = bank.quantize(k, x, index)
            if train:
                cache[rows[k]] = index
            out[k] = (x_hat - x).detach() + x
            rate_terms.append(bits / float(cfg["lmbda"][k]))
            vq_terms.append(torch.linalg.vector_norm(x_hat - x, dim=-1).sum())
            rdo["bits"][k] = bits.detach()
        rdo["rate"] = sum(rate_terms) / n / VQ_DIM
        rdo["vq"] = sum(vq_terms) / n / VQ_DIM
        scales = scales.clone()
        scales[gs_bit] = out["scale"]
        quats = torch.nn.functional.normalize(quats, dim=-1).clone()
        quats[gs_bit] = torch.nn.functional.normalize(out["rot"], dim=-1)
        sh0 = sh0.clone()
        sh0[gs_bit] = out["dc"].reshape(-1, 1, 3)
        shn = shn.clone()
        for k in SH_KEYS:
            a, b = SH_SLICES[k]
            shn[sh_bit[k], a:b, :] = out[k].reshape(-1, b - a, 3)
    return scales, opacities, quats, sh0, shn, rdo


class RDORenderer:
    capabilities = {"render.rgb", "render.depth.expected_z", "gradient.screen.signed"}

    def __init__(self, spec):
        if spec.config:
            raise ValueError("rdo-renderer takes no options (the density owns the RDO config)")
        self.bank = None  # explicit bank (probes); otherwise bank.ACTIVE

    def active_bank(self):
        b = self.bank if self.bank is not None else bank_mod.ACTIVE["bank"]
        if b is None or b.baked or not b.any_active:
            return None
        return b

    def render(self, params, camera, sh_degree=3):
        b = self.active_bank()
        if b is None:
            return rasterize(params["means"], params["quats"], params["scales"].exp(),
                             torch.sigmoid(params["opacities"]), params["sh0"], params["shN"],
                             camera, sh_degree)
        scales, opacities, quats, sh0, shn, rdo = rdo_attributes(b, params,
                                                                 torch.is_grad_enabled())
        out = rasterize(params["means"], quats, scales, opacities, sh0, shn, camera, sh_degree)
        out["rdo"] = rdo
        return out


def rasterize(means, quats, scales, opacities, sh0, shn, camera, sh_degree):
    """splatstudio.backends.gsplat.Renderer.render on already-activated attributes."""
    render, alpha, info = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities,
        colors=torch.cat((sh0, shn), dim=1), viewmats=camera["viewmat"][None],
        Ks=camera["K"][None], width=camera["width"], height=camera["height"],
        sh_degree=sh_degree, packed=False, render_mode="RGB+ED",
    )
    # Runtime convention: RGB over a white background (official: black).
    return {"rgb": render[0, ..., :3] + (1 - alpha[0]), "depth": render[0, ..., 3:4],
            "alpha": alpha[0], "info": info}


def create_renderer(spec):
    return RDORenderer(spec)
