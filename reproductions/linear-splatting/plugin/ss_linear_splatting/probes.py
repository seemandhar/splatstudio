"""Mechanism probes for 3DLS. Each raises when its mechanism is removed or wrong."""

import math

import torch
from splatstudio.components import build

from .kernels import rasterize_linear
from .losses import PAPER_WEIGHTS, combined_loss
from .reference import dense_render
from .renderer import linear_radii, project

LAM = 2.5


def _cuda():
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is unavailable")
    return "cuda"


def synthetic_scene(n=250, seed=0, device="cuda", scale_range=(-2.8, -1.6)):
    """Gaussians in front of an identity camera (z in [2.5, 4.5])."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.stack([(torch.rand(n, generator=g) * 2 - 1) * 1.0,
                         (torch.rand(n, generator=g) * 2 - 1) * 0.8,
                         2.5 + torch.rand(n, generator=g) * 2.0], dim=-1)
    lo, hi = scale_range
    params = {
        "means": means,
        "scales": lo + torch.rand(n, 3, generator=g) * (hi - lo),
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.randn(n, generator=g) * 1.2,
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.8,
        "shN": torch.randn(n, 15, 3, generator=g) * 0.1,
    }
    return {k: v.to(device) for k, v in params.items()}


def camera(width, height, device="cuda"):
    f = 1.1 * width
    K = torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]], device=device)
    return {"viewmat": torch.eye(4, device=device), "K": K, "width": width, "height": height}


def _inputs(params, cam, lam=LAM):
    """Projected inputs (detached leaves) + tile lists + gsplat's own 3.33-sigma visibility."""
    from gsplat.cuda._wrapper import fully_fused_projection

    with torch.no_grad():
        colors, meta = project(params, cam, 3, lam)
        radii_gs = fully_fused_projection(
            params["means"], None, params["quats"], params["scales"].exp(), cam["viewmat"][None],
            cam["K"][None], cam["width"], cam["height"], eps2d=0.3, near_plane=0.01,
            far_plane=1e10, radius_clip=0.0, packed=False, calc_compensations=False)[0]
    visible = (radii_gs[0] > 0).all(-1)
    feats = torch.cat((colors[0], meta["depths"][0][..., None]), -1)
    leaves = {"means2d": meta["means2d"][0], "conics": meta["conics"][0], "colors": feats,
              "opacities": meta["opacities"][0]}
    leaves = {k: v.detach().clone().contiguous() for k, v in leaves.items()}
    return leaves, meta, visible


def _cuda_render(leaves, meta, lam=LAM, ags=False, grad=False):
    x = {k: v.clone().requires_grad_(grad) for k, v in leaves.items()}
    rgb, alpha = rasterize_linear(
        x["means2d"], x["conics"], x["colors"], x["opacities"], meta["isect_offsets"][0],
        meta["flatten_ids"], meta["width"], meta["height"], meta["tile_size"], lam, ags)
    return x, rgb, alpha


def _dense(leaves, meta, visible, lam=LAM, ags=False, kernel="linear", grad=False):
    x = {k: v[visible].double().clone().requires_grad_(grad) for k, v in leaves.items()}
    depths = meta["depths"][0][visible].double()
    rgb, alpha = dense_render(x["means2d"], x["conics"], x["colors"], x["opacities"], depths,
                              meta["width"], meta["height"], lam=lam, ags=ags, kernel=kernel)
    return x, rgb, alpha


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-12))


SIZES = ((64, 48), (61, 37))


# ----------------------------------------------------------------------------- C1 forward
def linear_forward_vs_dense(method, device):
    from gsplat import rasterize_to_pixels

    dev = _cuda()
    params = synthetic_scene(device=dev)
    out = {}
    for w, h in SIZES:
        cam = camera(w, h, dev)
        leaves, meta, visible = _inputs(params, cam)
        _, rgb, alpha = _cuda_render(leaves, meta)
        _, rgb_ref, alpha_ref = _dense(leaves, meta, visible)
        err_rgb = float((rgb.double() - rgb_ref).abs().max())
        err_a = float((alpha.double() - alpha_ref).abs().max())
        # the same inputs through gsplat's Gaussian rasterizer (tile lists from 3.33 sigma)
        from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

        with torch.no_grad():
            r = torch.ceil(3.33 * torch.stack(
                (leaves["conics"][:, 2], leaves["conics"][:, 0]), -1).div(
                (leaves["conics"][:, 0] * leaves["conics"][:, 2]
                 - leaves["conics"][:, 1] ** 2)[:, None]).sqrt()).int()
            r = torch.where(visible[:, None], r, torch.zeros_like(r))
            _, ids, flat = isect_tiles(leaves["means2d"][None], r[None], meta["depths"],
                                       16, meta["tile_width"], meta["tile_height"],
                                       packed=False, n_images=1)
            offs = isect_offset_encode(ids, 1, meta["tile_width"], meta["tile_height"])
            g_rgb, _ = rasterize_to_pixels(
                leaves["means2d"][None], leaves["conics"][None], leaves["colors"][None],
                leaves["opacities"][None], w, h, 16, offs.reshape(1, meta["tile_height"],
                                                                  meta["tile_width"]), flat)
            _, g_ref, _ = _dense(leaves, meta, visible, kernel="gaussian")
        err_gauss_ref = float((g_rgb[0, ..., :3].double() - g_ref[..., :3]).abs().max())
        diff_gauss = float((g_rgb[0, ..., :3].double() - rgb_ref[..., :3]).abs().max())
        out[f"{w}x{h}"] = {"max_err_rgbd": err_rgb, "max_err_alpha": err_a,
                           "reference_vs_gsplat_gaussian": err_gauss_ref,
                           "linear_vs_gaussian_maxdiff": diff_gauss,
                           "mean_alpha": float(alpha.mean())}
        if err_gauss_ref > 2e-3:
            raise AssertionError(f"dense reference disagrees with gsplat (gaussian): {out}")
        if err_rgb > 2e-3 * max(1.0, float(rgb_ref.abs().max())) or err_a > 2e-3:
            raise AssertionError(f"linear CUDA render != dense reference: {out}")
        if diff_gauss < 0.05:
            raise AssertionError(f"linear render indistinguishable from Gaussian: {out}")
    return out


# ----------------------------------------------------------------------------- C1 backward
def _grads(leaves, meta, visible, ags_cuda, ags_ref, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    H, W = meta["height"], meta["width"]
    v_rgb = torch.randn(H, W, 4, generator=g).to(leaves["means2d"].device)
    v_rgb[..., 3] *= 0.05  # depth channel has larger values
    v_a = torch.randn(H, W, 1, generator=g).to(leaves["means2d"].device)
    x, rgb, alpha = _cuda_render(leaves, meta, ags=ags_cuda, grad=True)
    ((rgb * v_rgb).sum() + (alpha * v_a).sum()).backward()
    xr, rgb_r, alpha_r = _dense(leaves, meta, visible, ags=ags_ref, grad=True)
    ((rgb_r * v_rgb.double()).sum() + (alpha_r * v_a.double()).sum()).backward()
    cuda = {k: v.grad[visible] for k, v in x.items()}
    ref = {k: v.grad for k, v in xr.items()}
    return cuda, ref


def linear_backward_vs_dense(method, device):
    dev = _cuda()
    params = synthetic_scene(device=dev)
    out = {}
    for w, h in SIZES:
        leaves, meta, visible = _inputs(params, camera(w, h, dev))
        cuda, ref = _grads(leaves, meta, visible, False, False)
        errs = {k: _rel(cuda[k], ref[k]) for k in cuda}
        out[f"{w}x{h}"] = errs
        if max(errs.values()) > 5e-3:
            raise AssertionError(f"CUDA backward != autograd of dense reference: {out}")
    return out


# ----------------------------------------------------------------------------- C3 AGS
def ags_gradient_scaling(method, device):
    dev = _cuda()
    renderer, _, _ = build(method)
    coef = renderer.ags  # omega = exp(-coef D^2); paper Eq. 13: coef = 1
    if coef <= 0:
        raise AssertionError("AGS is disabled in the configured renderer")
    params = synthetic_scene(device=dev)
    out = {}
    for w, h in SIZES:
        leaves, meta, visible = _inputs(params, camera(w, h, dev))
        cuda, ref = _grads(leaves, meta, visible, coef, coef)
        _, ref_plain = _grads(leaves, meta, visible, coef, 0.0)
        errs = {k: _rel(cuda[k], ref[k]) for k in cuda}
        # opacity/colour gradients do not pass through dk/dD: AGS must not change them
        unscaled = {k: _rel(ref[k], ref_plain[k]) for k in ("colors", "opacities")}
        changed = {k: _rel(ref_plain[k], ref[k]) for k in ("means2d", "conics")}
        out[f"{w}x{h}"] = {"ags_coef": coef, "err_vs_ags_reference": errs, "ags_effect_on_unscaled": unscaled,
                           "ags_effect_on_geometry": changed}
        if max(errs.values()) > 5e-3:
            raise AssertionError(f"CUDA AGS backward != scaled reference: {out}")
        if max(unscaled.values()) > 1e-9:
            raise AssertionError(f"AGS reference changed opacity/colour gradients: {out}")
        if min(changed.values()) < 0.05:
            raise AssertionError(f"AGS has no effect on geometry gradients: {out}")
    return out


# ----------------------------------------------------------------------------- C2 DA
def alignment_footprint(method, device):
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    dev = _cuda()
    renderer, _, _ = build(method)
    if abs(renderer.lam - LAM) > 1e-9:
        raise AssertionError(f"renderer lam {renderer.lam} != paper 2.5")
    out = {}
    W = H = 96
    s, o = 6.0, 0.99  # isotropic sigma in pixels, opacity
    for lam in (LAM, 1.0):
        m2 = torch.tensor([[48.0, 48.0]], device=dev)
        conic = torch.tensor([[1 / s**2, 0.0, 1 / s**2]], device=dev)
        op = torch.tensor([o], device=dev)
        col = torch.ones(1, 3, device=dev)
        depth = torch.ones(1, 1, device=dev)
        radii = linear_radii(m2[None], conic[None], op[None],
                             torch.ones(1, 1, 2, dtype=torch.int32, device=dev), W, H, lam)
        _, ids, flat = isect_tiles(m2[None], radii, depth, 16, 6, 6, packed=False, n_images=1)
        offs = isect_offset_encode(ids, 1, 6, 6).reshape(6, 6)
        _, alpha = rasterize_linear(m2, conic, col, op, offs, flat, W, H, 16, lam, True)
        ys, xs = torch.meshgrid(torch.arange(H, device=dev) + 0.5,
                                torch.arange(W, device=dev) + 0.5, indexing="ij")
        dist = ((xs - 48) ** 2 + (ys - 48) ** 2).sqrt()
        support = float(dist[alpha[..., 0] > 0].max())
        expected = lam * s * (1 - (1 / 255) / o)
        out[f"lam={lam}"] = {"support_px": support, "expected_px": expected,
                             "radius_box": radii[0, 0].tolist()}
        if not expected - 1.5 <= support <= expected + 1e-3:
            raise AssertionError(f"support radius is not lam*sigma: {out}")
    # Distribution Alignment claim: lam=2.5 matches the Gaussian's footprint (2D integral)
    area_linear = math.pi * LAM**2 / 3  # int max(0,1-r/lam) 2 pi r dr (unit sigma)
    area_gauss = 2 * math.pi
    out["footprint_ratio_linear_over_gaussian"] = area_linear / area_gauss
    if abs(area_linear / area_gauss - 1) > 0.1:
        raise AssertionError(f"lam does not align footprints: {out}")
    # tile boxes from lam never clip: covered by the dense comparison on a random scene
    params = synthetic_scene(device=dev, seed=3, scale_range=(-2.2, -1.2))
    leaves, meta, visible = _inputs(params, camera(64, 48, dev))
    _, _, alpha = _cuda_render(leaves, meta)
    _, _, alpha_ref = _dense(leaves, meta, visible)
    err = float((alpha.double() - alpha_ref).abs().max())
    out["large_splats_alpha_err"] = err
    if err > 2e-3:
        raise AssertionError(f"lam-derived bounding boxes clip the support: {out}")
    return out


# ----------------------------------------------------------------------------- recipe
def loss_weights(method, device):
    _, _, losses = build(method)
    g = torch.Generator().manual_seed(0)
    rgb = torch.rand(40, 48, 3, generator=g, requires_grad=True)
    target = torch.rand(40, 48, 3, generator=g)
    outputs = {"rgb": rgb, "step": 1}
    from splatstudio.losses import check_loss

    values = {name: fn(None, outputs, target) for name, fn in losses}
    for name, v in values.items():
        check_loss(v, name)
    grad = torch.autograd.grad(sum(values.values()), rgb)[0]
    if not torch.isfinite(grad).all() or not torch.count_nonzero(grad):
        raise AssertionError("loss gradient is zero or nonfinite")
    terms = {name: float(v) for name, v in values.items()}
    total = sum(terms.values())
    ref = float(combined_loss(rgb, target, PAPER_WEIGHTS))
    if abs(total - ref) > 1e-6 * max(1.0, abs(ref)):
        raise AssertionError(f"loss {total} != 0.6 L1 + 0.2 L2 + 0.2 (1-SSIM) = {ref}")
    for term in ("l1", "l2", "ssim"):
        dropped = dict(PAPER_WEIGHTS, **{term: 0.0})
        if abs(total - float(combined_loss(rgb, target, dropped))) < 1e-6:
            raise AssertionError(f"term {term} contributes nothing")
    if any("photometric" in name for name, _ in losses):
        raise AssertionError("base photometric loss must be replaced")
    return {"terms": terms, "total": total, "reference": ref}


def density_thresholds(method, device):
    _, density, _ = build(method)
    s = density.strategy
    got = {k: getattr(s, k) for k in ("grow_grad2d", "grow_scale3d", "prune_scale3d", "prune_opa",
                                      "refine_start_iter", "refine_stop_iter", "reset_every")}
    want = {"grow_grad2d": 0.0002, "grow_scale3d": 0.006, "prune_scale3d": 0.4,
            "prune_opa": 0.025, "refine_start_iter": 500, "refine_stop_iter": 15000,
            "reset_every": 3000}
    if got != want or not density.opacity_reset:
        raise AssertionError(f"density thresholds {got} != paper {want}")
    return got
