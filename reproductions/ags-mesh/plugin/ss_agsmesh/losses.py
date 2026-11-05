"""AGS-Mesh regularizers, Splatfacto variant (regularization_strategy.py:AGSMeshRegularization,
dn_model.py:get_loss_dict; github.com/maturk/dn-splatter @ 97588b4). ``ns = step - 1`` is the
Nerfstudio 0-based step (the runtime passes 1-based steps).

* ags-depth (C1, paper Eq. 1-3): EdgeAwareLogL1 (code; paper writes L1) x depth_lambda 0.2 on the
  raw sensor depth while ns < 7000, then on the DNC-filtered depth D_f = where(conf > 0, D, 0)
  with conf = 1 - mask/255 (offline DNC mask, tools/prepare_dnc.py). Valid mask D > 0.1.
* ags-normal (C2, paper Eq. 4-6): weight normal_lambda 0.1 only when ns > 7000. Terms (GL camera
  frame, values in [-1, 1], L1 = mean |a - b|):
    - surface term: rendered-depth normals vs Omnidata; ns < 15000 excluding the dilated
      Laplacian edges of the prior (find_edges), afterwards only where
      arccos(clip(<surf, prior>)) <= 0.1 rad (code; paper tau_N = 10 deg). Gradient-free with
      the official detached depth (renderer ``surface_grad`` false).
    - Gaussian-normal term: L1(rendered Gaussian normals, prior), unmasked.
  No TV term (not called by AGSMeshRegularization).
* ags-scale: min(exp(s)).mean() x 1 (DN-Splatter Eq. 6).
"""

import torch
import torch.nn.functional as F

DEPTH_DEFAULTS = {"depth_lambda": 0.2, "depth_mask_step": 7000, "depth_tolerance": 0.1,
                  "rgb_clamp_min": 10.0 / 255.0}
NORMAL_DEFAULTS = {"normal_lambda": 0.1, "normal_start": 7000, "normal_mask_step": 15000,
                   "anr_threshold": 0.1, "edge_threshold": 0.01, "edge_dilation": 1}
SCALE_DEFAULTS = {"weight": 1.0}
GL = (1.0, -1.0, -1.0)


def edge_aware_log_l1(pred, gt, rgb, mask):
    """losses.py:EdgeAwareLogL1 (implementation='scalar'). HWC tensors; mask [H,W,1] bool."""
    logl1 = torch.log(1 + torch.abs(pred - gt))
    grad_x = torch.mean(torch.abs(rgb[..., :, :-1, :] - rgb[..., :, 1:, :]), -1, keepdim=True)
    grad_y = torch.mean(torch.abs(rgb[..., :-1, :, :] - rgb[..., 1:, :, :]), -1, keepdim=True)
    loss_x = torch.exp(-grad_x) * logl1[..., :, :-1, :]
    loss_y = torch.exp(-grad_y) * logl1[..., :-1, :, :]
    loss_x = loss_x[mask[..., :, :-1, :]]
    loss_y = loss_y[mask[..., :-1, :, :]]
    return loss_x.mean() + loss_y.mean()


def find_edges(im, threshold=0.01, dilation_itr=1):
    """regularization_strategy.py:find_edges for a [3,H,W] map: per-channel Laplacian of
    1/(im + 1e-6) (zero padding) > threshold, dilated by a 3x3 ones kernel; returns bool [3,H,W]."""
    lap_k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=im.dtype, device=im.device)
    lap = F.conv2d((1.0 / (im + 1e-6))[:, None], lap_k[None, None], padding=1)[:, 0]
    edges = (lap > threshold).to(im.dtype)
    ones = torch.ones(1, 1, 3, 3, dtype=im.dtype, device=im.device)
    for _ in range(dilation_itr):
        edges = F.conv2d(edges[:, None], ones, padding=1)[:, 0]
    return edges > 0.0


def angular_error(a, b):
    """regularization_strategy.py:mean_angular_error on [3,H,W] maps -> [H,W] radians."""
    return torch.arccos(torch.clip(torch.sum(a * b, dim=0), -1.0, 1.0))


def _l1(a, b):
    if a.numel() == 0:  # official: mean of an empty tensor = NaN; never selected in practice
        return a.sum() * 0.0
    return torch.abs(a - b).mean()


def _prior(outputs, key):
    view = outputs.get("dn_view")
    if view is None or view.get(key) is None:
        raise ValueError(f"AGS-Mesh {key} loss: no sidecar prior for this training view (train "
                         "with --data at a prepare_mushroom.py + prepare_dnc.py scene)")
    return view[key]


def to_gl(n01_or_signed, encoded):
    """[H,W,3] OpenCV-frame map -> [3,H,W] GL-frame signed map."""
    n = 2 * n01_or_signed - 1 if encoded else n01_or_signed
    return (n * torch.tensor(GL, device=n.device, dtype=n.dtype)).permute(2, 0, 1)


def create_depth(spec):
    cfg = {**DEPTH_DEFAULTS, **spec.config}
    lam, t_d = float(cfg["depth_lambda"]), int(cfg["depth_mask_step"])
    tol, cmin = float(cfg["depth_tolerance"]), float(cfg["rgb_clamp_min"])

    def loss(params, outputs, target):
        ns = int(outputs["step"]) - 1
        gt = _prior(outputs, "depth")
        if ns >= t_d:  # Eq. 1 / 3: D_f
            gt = torch.where(_prior(outputs, "confidence") > 0, gt, torch.zeros_like(gt))
        return lam * edge_aware_log_l1(outputs["depth"], gt, target.clamp(min=cmin), gt > tol)

    return loss


def normal_terms(outputs, cfg):
    """Returns (surface_term, gaussian_term) before the lambda, plus the masks (for probes)."""
    gt = to_gl(_prior(outputs, "normal"), encoded=True)
    surf = outputs["surface_normal"].permute(2, 0, 1)
    pred = to_gl(outputs["normal"], encoded=True)
    ns = int(outputs["step"]) - 1
    if ns < int(cfg["normal_mask_step"]):
        keep = ~find_edges(gt, float(cfg["edge_threshold"]), int(cfg["edge_dilation"]))
        s_term = _l1(surf[keep], gt[keep])
    else:  # Eq. 5 (ANR)
        keep = angular_error(surf, gt) <= float(cfg["anr_threshold"])
        s_term = _l1(surf[:, keep], gt[:, keep])
    return s_term, _l1(pred, gt), keep


def create_normal(spec):
    cfg = {**NORMAL_DEFAULTS, **spec.config}
    lam, start = float(cfg["normal_lambda"]), int(cfg["normal_start"])

    def loss(params, outputs, target):
        ns = int(outputs["step"]) - 1
        if not ns > start:  # lambda_normal_l1 = 0 (skipped instead of multiplying by 0)
            return outputs["normal"].sum() * 0.0
        s_term, g_term, _ = normal_terms(outputs, cfg)
        return lam * (s_term + g_term)

    return loss


def create_scale(spec):
    cfg = {**SCALE_DEFAULTS, **spec.config}
    w = float(cfg["weight"])

    def loss(params, outputs, target):
        return w * torch.min(torch.exp(params["scales"]), dim=1, keepdim=True)[0].mean()

    return loss
