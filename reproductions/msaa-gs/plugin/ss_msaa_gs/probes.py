"""Mechanism probes for MSAA-GS. Each raises when its mechanism is removed or wrong."""

import torch
from splatstudio.components import build

from .kernels import PIXEL_CENTRE, ROTATED_GRID_4X, rasterize_msaa
from .losses import aws_weights, gradient_difference
from .reference import dense_msaa
from .renderer import project


def _cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    return "cuda"


def synthetic_scene(n=250, seed=0, device="cuda", scale_range=(-3.2, -1.8)):
    """Gaussians in front of an identity camera (z in [2.5, 4.5]); small splats so that
    sub-pixel sample positions matter."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.stack([(torch.rand(n, generator=g) * 2 - 1) * 1.0,
                         (torch.rand(n, generator=g) * 2 - 1) * 0.8,
                         2.5 + torch.rand(n, generator=g) * 2.0], dim=-1)
    lo, hi = scale_range
    params = {
        "means": means,
        "scales": lo + torch.rand(n, 3, generator=g) * (hi - lo),
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.randn(n, generator=g) * 1.2 + 1.0,
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.8,
        "shN": torch.randn(n, 15, 3, generator=g) * 0.1,
    }
    return {k: v.to(device) for k, v in params.items()}


def camera(width, height, device="cuda"):
    f = 1.1 * width
    K = torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]], device=device)
    return {"viewmat": torch.eye(4, device=device), "K": K, "width": width, "height": height}


def _inputs(params, cam):
    with torch.no_grad():
        colors, meta = project(params, cam, 3)
    visible = (meta["radii"][0] > 0).all(-1)
    feats = torch.cat((colors[0], meta["depths"][0][..., None]), -1)
    leaves = {"means2d": meta["means2d"][0], "conics": meta["conics"][0], "colors": feats,
              "opacities": meta["opacities"][0]}
    leaves = {k: v.detach().clone().contiguous() for k, v in leaves.items()}
    return leaves, meta, visible


def _cuda_render(leaves, meta, offsets, grad=False):
    x = {k: v.clone().requires_grad_(grad) for k, v in leaves.items()}
    rgb, alpha = rasterize_msaa(
        x["means2d"], x["conics"], x["colors"], x["opacities"], meta["isect_offsets"][0],
        meta["flatten_ids"], meta["width"], meta["height"], offsets, meta["tile_size"])
    return x, rgb, alpha


def _dense(leaves, meta, visible, offsets, grad=False):
    x = {k: v[visible].double().clone().requires_grad_(grad) for k, v in leaves.items()}
    depths = meta["depths"][0][visible].double()
    rgb, alpha = dense_msaa(x["means2d"], x["conics"], x["colors"], x["opacities"], depths,
                            meta["width"], meta["height"], offsets)
    return x, rgb, alpha


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-12))


SIZES = ((64, 48), (61, 37))


def _offsets(method):
    renderer, _, _ = build(method)
    if len(renderer.offsets) != 4:
        raise AssertionError(f"renderer uses {len(renderer.offsets)} samples, paper uses 4")
    return renderer.offsets


# ----------------------------------------------------------------------------- C1 forward
def msaa_forward_vs_dense(method, device):
    dev = _cuda()
    offsets = _offsets(method)
    params = synthetic_scene(device=dev)
    out = {}
    for w, h in SIZES:
        leaves, meta, visible = _inputs(params, camera(w, h, dev))
        _, rgb, alpha = _cuda_render(leaves, meta, offsets)
        _, rgb_ref, alpha_ref = _dense(leaves, meta, visible, offsets)
        _, rgb_1x, _ = _dense(leaves, meta, visible, PIXEL_CENTRE)
        err_rgb = float((rgb.double() - rgb_ref).abs().max())
        err_a = float((alpha.double() - alpha_ref).abs().max())
        diff_1x = float((rgb_ref[..., :3] - rgb_1x[..., :3]).abs().max())
        out[f"{w}x{h}"] = {"max_err_rgbd": err_rgb, "max_err_alpha": err_a,
                           "msaa_vs_single_sample_maxdiff": diff_1x,
                           "mean_alpha": float(alpha.mean())}
        if err_rgb > 2e-3 * max(1.0, float(rgb_ref.abs().max())) or err_a > 2e-3:
            raise AssertionError(f"MSAA CUDA render != dense 4-sample reference: {out}")
        # removing MSAA would move the render by diff_1x: require it far above the kernel error
        if diff_1x < 5e-3 or err_rgb > 0.1 * diff_1x:
            raise AssertionError(f"MSAA render indistinguishable from single sample: {out}")
    return out


# ----------------------------------------------------------------------------- C1 backward
def msaa_backward_vs_dense(method, device):
    dev = _cuda()
    offsets = _offsets(method)
    params = synthetic_scene(device=dev)
    out = {}
    for w, h in SIZES:
        leaves, meta, visible = _inputs(params, camera(w, h, dev))
        g = torch.Generator(device="cpu").manual_seed(1)
        v_rgb = torch.randn(h, w, 4, generator=g).to(dev)
        v_rgb[..., 3] *= 0.05  # depth channel has larger values
        v_a = torch.randn(h, w, 1, generator=g).to(dev)
        x, rgb, alpha = _cuda_render(leaves, meta, offsets, grad=True)
        ((rgb * v_rgb).sum() + (alpha * v_a).sum()).backward()
        xr, rgb_r, alpha_r = _dense(leaves, meta, visible, offsets, grad=True)
        ((rgb_r * v_rgb.double()).sum() + (alpha_r * v_a.double()).sum()).backward()
        errs = {k: _rel(x[k].grad[visible], xr[k].grad) for k in x}
        out[f"{w}x{h}"] = errs
        if max(errs.values()) > 5e-3:
            raise AssertionError(f"CUDA MSAA backward != autograd of dense reference: {out}")
    return out


# ------------------------------------------------------------- C1 kernel adaptation fidelity
def single_sample_equals_gsplat(method, device):
    """With one sample at the pixel centre the kernel must reproduce gsplat's rasterizer
    (forward and backward): guards the adaptation of RasterizeToPixels3DGS{Fwd,Bwd}."""
    from gsplat import rasterize_to_pixels

    dev = _cuda()
    params = synthetic_scene(device=dev, seed=2)
    out = {}
    for w, h in SIZES:
        leaves, meta, _ = _inputs(params, camera(w, h, dev))
        g = torch.Generator(device="cpu").manual_seed(3)
        v_rgb = torch.randn(h, w, 4, generator=g).to(dev)
        v_a = torch.randn(h, w, 1, generator=g).to(dev)
        x, rgb, alpha = _cuda_render(leaves, meta, PIXEL_CENTRE, grad=True)
        ((rgb * v_rgb).sum() + (alpha * v_a).sum()).backward()
        y = {k: v.clone().requires_grad_(True) for k, v in leaves.items()}
        g_rgb, g_a = rasterize_to_pixels(
            y["means2d"][None], y["conics"][None], y["colors"][None], y["opacities"][None],
            w, h, meta["tile_size"], meta["isect_offsets"], meta["flatten_ids"])
        ((g_rgb[0] * v_rgb).sum() + (g_a[0] * v_a).sum()).backward()
        res = {"rgb": float((rgb - g_rgb[0]).abs().max()), "alpha": float((alpha - g_a[0]).abs().max())}
        res.update({f"grad_{k}": _rel(x[k].grad, y[k].grad) for k in x})
        out[f"{w}x{h}"] = res
        if res["rgb"] > 1e-4 or res["alpha"] > 1e-5 or max(
                v for k, v in res.items() if k.startswith("grad_")) > 1e-4:
            raise AssertionError(f"1-sample MSAA kernel != gsplat rasterize_to_pixels: {out}")
    _, rgb4, _ = _cuda_render(leaves, meta, ROTATED_GRID_4X)
    out["4x_vs_gsplat_maxdiff"] = float((rgb4 - g_rgb[0]).abs().max())
    if out["4x_vs_gsplat_maxdiff"] < 1e-2:
        raise AssertionError(f"4-sample render equals gsplat single-sample render: {out}")
    return out


# ----------------------------------------------------------------------------- C2 AWS
def _losses(method):
    _, _, losses = build(method)
    return dict(losses)


def _aws_norm(method):
    return bool(next(c for c in method.components if c.name == "aws-recon").config.get(
        "normalize_mean", False))


def aws_weighting(method, device):
    losses = _losses(method)
    norm = _aws_norm(method)
    if "photometric" in losses or "aws-recon" not in losses:
        raise AssertionError(f"aws-recon must replace photometric: {sorted(losses)}")
    from splatstudio.losses import ssim

    g = torch.Generator().manual_seed(0)
    target = torch.rand(40, 48, 3, generator=g)
    rgb = (target + 0.02 * torch.randn(40, 48, 3, generator=g)).requires_grad_(True)
    with torch.no_grad():
        rgb[5:12, 7:20] += 0.4  # an under-reconstructed region
    value = losses["aws-recon"](None, {"rgb": rgb, "step": 1}, target)
    # independent formula, Eq. 6-9 with alpha 0.5, eps 1e-6, lambda 0.8 / 0.2
    err = (rgb.detach() - target).abs()
    e = err.sum(-1)  # |.|_1 over RGB
    w = 0.5 + 0.5 * e / (e.max() + 1e-6 * 3)  # e/max e is scale-free; eps scaled with the sum
    if norm:  # iteration 1 (post-hoc): weights divided by their image mean
        w = w / w.mean()
    ref = 0.8 * (w[..., None] * err).mean() + 0.2 * (1 - ssim(rgb.detach(), target))
    plain = 0.8 * err.mean() + 0.2 * (1 - ssim(rgb.detach(), target))
    grad = torch.autograd.grad(value, rgb)[0]
    gw = aws_weights(rgb.detach(), target)
    out = {"loss": float(value), "reference": float(ref), "plain_l1_dssim": float(plain),
           "weight_min": float(gw.min()), "weight_max": float(gw.max()),
           "grad_ratio_hard_vs_easy": float(grad[5:12, 7:20].abs().mean()
                                            / grad[20:, 25:].abs().mean())}
    if abs(float(value) - float(ref)) > 1e-5:
        raise AssertionError(f"AWS loss != Eq. 6-9: {out}")
    if abs(float(value) - float(plain)) < 1e-3:
        raise AssertionError(f"AWS loss equals plain L1 + D-SSIM (weights removed): {out}")
    if not norm and not (0.5 - 1e-6 <= out["weight_min"] and out["weight_max"] < 1.0):
        raise AssertionError(f"weights outside [alpha, 1): {out}")
    if out["grad_ratio_hard_vs_easy"] < 2.0:
        raise AssertionError(f"high-error region not up-weighted: {out}")
    return out


# ----------------------------------------------------------------------------- C3 GDC
def gdc_gradient_difference(method, device):
    losses = _losses(method)
    if "gdc" not in losses:
        raise AssertionError("gdc loss missing")
    H, W = 20, 30
    # analytic case: pred is a horizontal ramp of slope a, gt of slope b, no vertical change
    a, b = 0.03, 0.01
    jj = torch.arange(W, dtype=torch.float64)[None, :, None].expand(H, W, 3)
    pred = (a * jj).clone().requires_grad_(True)
    gt = b * jj
    want = 0.1 * abs(a - b)  # mean|d_x| = |a-b|, mean|d_y| = 0, lambda_3 = 0.1
    v = losses["gdc"](None, {"rgb": pred, "step": 1}, gt)
    got = float(v)
    # constant offset must not be penalised (gradient-domain loss)
    off = float(losses["gdc"](None, {"rgb": gt + 0.3, "step": 1}, gt))
    # sharp edge vs blurred edge: nonzero gradient concentrated at the edge
    edge = torch.zeros(H, W, 3)
    edge[:, W // 2:] = 1.0
    blur = torch.nn.functional.avg_pool1d(edge.permute(0, 2, 1).reshape(-1, 1, W), 5, 1, 2,
                                          count_include_pad=False).reshape(H, 3, W).permute(0, 2, 1)
    blur = blur.clone().requires_grad_(True)
    ge = torch.autograd.grad(gradient_difference(blur, edge), blur)[0]
    near = float(ge[:, W // 2 - 3:W // 2 + 3].abs().sum())
    far = float(ge[:, :W // 2 - 4].abs().sum() + ge[:, W // 2 + 4:].abs().sum())
    out = {"ramp_loss": got, "ramp_expected": want, "offset_loss": off,
           "edge_grad_near": near, "edge_grad_far": far}
    if abs(got - want) > 1e-9 or off > 1e-9:
        raise AssertionError(f"GDC != Eq. 10-12 on analytic ramp: {out}")
    if near <= 0 or far > 1e-9:
        raise AssertionError(f"GDC gradient not localised at the edge: {out}")
    return out


def loss_weights(method, device):
    from splatstudio.losses import check_loss, ssim

    losses = _losses(method)
    g = torch.Generator().manual_seed(4)
    rgb = torch.rand(40, 48, 3, generator=g, requires_grad=True)
    target = torch.rand(40, 48, 3, generator=g)
    values = {n: fn(None, {"rgb": rgb, "step": 1}, target) for n, fn in losses.items()}
    for n, v in values.items():
        check_loss(v, n)
    total = float(sum(values.values()))
    with torch.no_grad():
        err = (rgb - target).abs()
        e = err.mean(-1, keepdim=True)
        w = 0.5 + 0.5 * e / (e.max() + 1e-6)
        if _aws_norm(method):
            w = w / w.mean()
        ref = (0.8 * (w * err).mean() + 0.2 * (1 - ssim(rgb, target))
               + 0.1 * gradient_difference(rgb, target))
    out = {"terms": {n: float(v) for n, v in values.items()}, "total": total,
           "reference": float(ref)}
    if sorted(losses) != ["aws-recon", "gdc"] or abs(total - float(ref)) > 1e-5:
        raise AssertionError(f"objective != 0.8 Lw + 0.2 D-SSIM + 0.1 Lgrad (Eq. 13): {out}")
    return out
