"""DN-Splatter regularizers (regularization_strategy.py:DNRegularization, losses.py), official
commit 97588b4. Weights follow the code, which differs from the paper (Eq. 10: 0.2 / 0.1 / 0.1):

* depth: ``depth_loss += depth_lambda * depth_loss`` (regularization_strategy.py:179) gives an
  effective weight 1 + depth_lambda = 1.2 on EdgeAwareLogL1 (Eq. 3);
* normal: L1 + TV with weight 1 each (normal_lambda is never applied);
* scale: min(exp(s)).mean() with weight 1 (Eq. 6).
"""

import torch

DEPTH_DEFAULTS = {"depth_lambda": 0.2, "official_double_count": True, "depth_tolerance": 0.1,
                  "rgb_clamp_min": 10.0 / 255.0}
NORMAL_DEFAULTS = {"l1_weight": 1.0, "tv_weight": 1.0}
SCALE_DEFAULTS = {"weight": 1.0}


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


def tv(pred):
    """losses.py:TVLoss on an HWC map."""
    h = pred[..., :, :-1, :] - pred[..., :, 1:, :]
    w = pred[..., :-1, :, :] - pred[..., 1:, :, :]
    return torch.mean(torch.abs(h)) + torch.mean(torch.abs(w))


def _prior(outputs, key):
    view = outputs.get("dn_view")
    if view is None:
        raise ValueError(f"DN-Splatter {key} loss: no sidecar prior for this training view "
                         "(train with --data pointing at a prepare_mushroom.py scene)")
    return view[key]


def create_depth(spec):
    cfg = {**DEPTH_DEFAULTS, **spec.config}
    lam = float(cfg["depth_lambda"])
    weight = 1.0 + lam if cfg["official_double_count"] else lam
    tol, cmin = float(cfg["depth_tolerance"]), float(cfg["rgb_clamp_min"])

    def loss(params, outputs, target):
        gt = _prior(outputs, "depth")
        mask = gt > tol
        return weight * edge_aware_log_l1(outputs["depth"], gt, target.clamp(min=cmin), mask)

    return loss


def create_normal(spec):
    cfg = {**NORMAL_DEFAULTS, **spec.config}
    wl, wt = float(cfg["l1_weight"]), float(cfg["tv_weight"])

    def loss(params, outputs, target):
        pred = outputs["normal"]
        gt = _prior(outputs, "normal")
        return wl * torch.abs(pred - gt).mean() + wt * tv(pred)

    return loss


def create_scale(spec):
    cfg = {**SCALE_DEFAULTS, **spec.config}
    w = float(cfg["weight"])

    def loss(params, outputs, target):
        return w * torch.min(torch.exp(params["scales"]), dim=1, keepdim=True)[0].mean()

    return loss


def resize_image(image, d):
    """splatfacto.py:resize_image: OpenCV-'area' downscale by an integer d (HWC -> H//d, W//d)."""
    image = image.to(torch.float32)
    weight = torch.full((1, 1, d, d), 1.0 / (d * d), dtype=torch.float32, device=image.device)
    out = torch.nn.functional.conv2d(image.permute(2, 0, 1)[:, None], weight, stride=d)
    return out.squeeze(1).permute(1, 2, 0)


def create_photometric(spec):
    """Base photometric loss (0.8 L1 + 0.2 D-SSIM) with Splatfacto's iteration-dependent target
    downscale (splatfacto.py:get_gt_img -> _downscale_if_required); the factor is the one the
    renderer used for this step (``outputs["downscale"]``, 1 = full resolution)."""
    from splatstudio.losses import create_photometric as base

    fn = base(spec)

    def loss(params, outputs, target):
        d = int(outputs.get("downscale", 1))
        if d > 1:
            target = resize_image(target, d)
        if target.shape != outputs["rgb"].shape:
            raise ValueError(f"target {tuple(target.shape)} != render {tuple(outputs['rgb'].shape)}")
        return fn(params, outputs, target)

    return loss
