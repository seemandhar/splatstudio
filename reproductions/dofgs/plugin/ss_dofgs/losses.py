"""DOF-GS objective (src/model_dofgs/training.py:186-231 of DOFGS_implementation @cfd635c).

step <= infocus_start (warm-up, supp. B.2):
    L_rec(I_defocus, I) = (1 - l) L1 + l (1 - SSIM)                               (Eq. 5)
step > infocus_start (CoC-guided detail enhancement, Sec. 3.3; code form, m = 1 - M*):
    m      = ILN(view, PE, [I_defocus, minmax(D)].detach(), minmax(M_coc).detach())   (Eq. 10)
    L_a    = L_rec((1-m) I_aif.detach() + m I_defocus, I)      # official first backward
    L_b    = L_rec((1-m) I_aif + m I_defocus.detach(), I)      # official second backward
    L_mk   = l_mask (1 - Pearson(m, minmax(log1p(M_coc))))                            (Eq. 11)
    L_reg  = -0.1 l_mask mean(m log(m + 1e-6))                                        (Eq. 14)
    total  = L_a + l_infocus L_b + L_mk + L_reg                                        (Eq. 15)
The two official backward passes accumulate gradients, so their sum is one backward here.
"""

from math import exp

import torch
import torch.nn.functional as F

OBJECTIVE_DEFAULTS = {"lambda_dssim": 0.2, "infocus_start_iter": 5000, "lambda_infocus": 1.0,
                      "detail_enhance": True}
# lambda_mask_loss: paper value / 7 of 9 official real configs; method() sets it per variant
MASK_DEFAULTS = {"lambda_mask_loss": 0.001, "reg_factor": 0.1, "use_mask_loss": True,
                 "infocus_start_iter": 5000}


def _window(size, sigma, channel, device, dtype):
    g = torch.tensor([exp(-((x - size // 2) ** 2) / float(2 * sigma**2)) for x in range(size)])
    g = (g / g.sum()).to(device=device, dtype=dtype)
    w2 = (g[:, None] @ g[None, :])[None, None]
    return w2.expand(channel, 1, size, size).contiguous()


def ssim_same(img1, img2, window_size=11):
    """utils/loss_utils.ssim (Inria): CHW, Gaussian 11x11 sigma 1.5, same padding, mean."""
    c = img1.shape[-3]
    w = _window(window_size, 1.5, c, img1.device, img1.dtype)
    x, y = img1[None], img2[None]
    pad = window_size // 2
    mu1 = F.conv2d(x, w, padding=pad, groups=c)
    mu2 = F.conv2d(y, w, padding=pad, groups=c)
    s11 = F.conv2d(x * x, w, padding=pad, groups=c) - mu1 * mu1
    s22 = F.conv2d(y * y, w, padding=pad, groups=c) - mu2 * mu2
    s12 = F.conv2d(x * y, w, padding=pad, groups=c) - mu1 * mu2
    C1, C2 = 0.01**2, 0.03**2
    return (((2 * mu1 * mu2 + C1) * (2 * s12 + C2))
            / ((mu1 * mu1 + mu2 * mu2 + C1) * (s11 + s22 + C2))).mean()


def l_rec(image, gt, lam=0.2):
    """CHW images."""
    return (1.0 - lam) * (image - gt).abs().mean() + lam * (1.0 - ssim_same(image, gt))


def pearson_correlation_coefficient(t1, t2, eps=1e-8):
    """training.py: unbiased variances (torch.var), biased covariance, eps floor."""
    m1, m2 = t1.mean(), t2.mean()
    v1, v2 = torch.var(t1), torch.var(t2)
    if v1 < eps:
        v1 = v1 + eps
    if v2 < eps:
        v2 = v2 + eps
    cov = torch.mean((t1 - m1) * (t2 - m2))
    return cov / (torch.sqrt(v1) * torch.sqrt(v2))


def pearson_mask_loss(img_src, img_target):
    scaled = img_target * (img_src.max() - img_src.min()) + img_src.min()
    return 1.0 - pearson_correlation_coefficient(scaled.reshape(-1), img_src.reshape(-1))


def minmax(t, eps=1e-5):
    return (t - t.min()) / (t.max() - t.min() + eps)


def _chw(t):
    return t.permute(2, 0, 1)


def infocus_mask(outputs):
    """Out-of-focus mask m [1,H,W] for the current training view (cached in outputs)."""
    dof = outputs["dof"]
    if "mask" not in dof:
        bank, view = dof["bank"], dof["view"]
        image, depth, coc = _chw(outputs["rgb"]), _chw(outputs["depth"]), _chw(outputs["coc"])
        inp = torch.cat([image, minmax(depth)], dim=0).detach()
        dof["mask"] = bank.mask_net(view, bank.pos_enc, inp, minmax(coc).detach())
    return dof["mask"]


def detail_active(outputs, start):
    dof = outputs.get("dof", {})
    return (dof.get("view") is not None and outputs["step"] > start
            and outputs.get("rgb_aif") is not None)


def create_objective(spec):
    unknown = set(spec.config) - set(OBJECTIVE_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown DOF-GS objective options: {sorted(unknown)}")
    c = {**OBJECTIVE_DEFAULTS, **spec.config}
    lam, start, lam_inf = float(c["lambda_dssim"]), int(c["infocus_start_iter"]), float(
        c["lambda_infocus"])
    enhance = bool(c["detail_enhance"])

    def loss(params, outputs, target):
        gt = _chw(target)
        image = _chw(outputs["rgb"])
        if not (enhance and detail_active(outputs, start)):
            return l_rec(image, gt, lam)
        aif = _chw(outputs["rgb_aif"])
        m = infocus_mask(outputs)
        blur1 = (1.0 - m) * aif.detach() + m * image
        blur2 = (1.0 - m) * aif + m * image.detach()
        return l_rec(blur1, gt, lam) + lam_inf * l_rec(blur2, gt, lam)

    return loss


def create_mask_losses(spec):
    unknown = set(spec.config) - set(MASK_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown DOF-GS mask-loss options: {sorted(unknown)}")
    c = {**MASK_DEFAULTS, **spec.config}
    lam, reg_factor, start = float(c["lambda_mask_loss"]), float(c["reg_factor"]), int(
        c["infocus_start_iter"])
    use = bool(c["use_mask_loss"])

    def loss(params, outputs, target):
        if not (use and detail_active(outputs, start)):
            return outputs["rgb"].sum() * 0.0  # inactive; attached zero, no ILN optimizer step
        m = infocus_mask(outputs)
        coc = _chw(outputs["coc"]).detach()
        log_coc = torch.log1p(coc)
        t = (log_coc - log_coc.min()) / (torch.abs(log_coc.max() - log_coc.min()) + 1e-5)
        mask_loss = lam * pearson_mask_loss(m.squeeze(), t.squeeze())
        reg = -1.0 * lam * reg_factor * (m * torch.log(m + 1e-6)).mean()
        return mask_loss + reg

    return loss
