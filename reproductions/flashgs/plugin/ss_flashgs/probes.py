"""Executable checks for the FlashGS contributions (all CUDA).

The fixture is a synthetic scene with a mix of tiny, large, elongated, faint and opaque
Gaussians seen by a centred pinhole camera, so that single-tile and warp-cooperative
multi-tile paths, the opacity-aware radius and early termination are all exercised.
"""

import math
import statistics
import tempfile
from pathlib import Path

import numpy as np
import torch

from . import core


def _need_cuda(device):
    from splatstudio.verification import ProbeUnavailable

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise ProbeUnavailable("CUDA is unavailable")


def fixture(n=20000, width=640, height=480, seed=0, device="cuda"):
    g = torch.Generator().manual_seed(seed)
    means = torch.empty(n, 3)
    means[:, 0] = (torch.rand(n, generator=g) - 0.5) * 6
    means[:, 1] = (torch.rand(n, generator=g) - 0.5) * 4.5
    means[:, 2] = 2 + torch.rand(n, generator=g) * 8
    scales = torch.randn(n, 3, generator=g) * 0.6 - 3.2  # log; some large, some sub-pixel
    elong = torch.rand(n, generator=g) < 0.3
    scales[elong, 0] += 1.5
    quats = torch.randn(n, 4, generator=g)
    opac = torch.rand(n, generator=g)
    opac = torch.where(torch.rand(n, generator=g) < 0.5, opac * 0.3, opac)  # many faint ones
    params = {
        "means": means, "scales": scales, "quats": quats,
        "opacities": torch.logit(opac.clamp(0.004, 0.999)),
        "sh0": torch.randn(n, 1, 3, generator=g) * 0.8,
        "shN": torch.randn(n, 15, 3, generator=g) * 0.15,
    }
    params = {k: v.to(device) for k, v in params.items()}
    f = 0.9 * width
    camera = {
        "viewmat": torch.eye(4, device=device),
        "K": torch.tensor([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], device=device),
        "width": width, "height": height,
    }
    return params, camera


def _write_3dgs_ply(path, params):
    """3DGS point_cloud.ply layout read by upstream `ops.loadPly` (x,y,z,nx,ny,nz,f_dc*3,
    f_rest*45 channel-major, opacity, scale*3, rot*4)."""
    n = len(params["means"])
    sh0 = params["sh0"].cpu().reshape(n, 3)
    rest = params["shN"].cpu().transpose(1, 2).reshape(n, 45)  # [N,3,15] -> channel-major
    cols = [params["means"].cpu(), torch.zeros(n, 3), sh0, rest,
            params["opacities"].cpu().reshape(n, 1), params["scales"].cpu(), params["quats"].cpu()]
    data = torch.cat(cols, 1).float().numpy().astype("<f4")
    names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(45)] + ["opacity"]
             + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
    header += "".join(f"property float {p}\n" for p in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(np.ascontiguousarray(data).tobytes())


def parity(method, device):
    """FlashGS image == original 3DGS forward (and gsplat) on the fixture; our model conversion
    == upstream loadPly. Fails on a wrong camera convention, SH layout or covariance."""
    _need_cuda(device)
    params, camera = fixture(device=device)
    prep = core.prepare(params)
    with tempfile.TemporaryDirectory() as tmp:
        ply = Path(tmp) / "point_cloud.ply"
        _write_3dgs_ply(ply, params)
        fl = core.FlashGSRasterizer(device)
        _n, pos, shs, opa, cov = fl.ext.ops.loadPly(str(ply))
    conv = {
        "position": float((pos.to(device) - prep["position"]).abs().max()),
        "shs": float((shs.to(device) - prep["shs"]).abs().max()),
        "opacity": float((opa.to(device) - prep["opacity"]).abs().max()),
        "cov3d_rel": float((cov.to(device) - prep["cov3d"]).abs().max()
                           / prep["cov3d"].abs().max()),
    }
    bg = torch.tensor([1.0, 1.0, 1.0])
    with torch.no_grad():
        img = fl.forward(prep, camera, bg).float() / 255
        ref = core.Reference3DGS()
        c3 = ref.forward(ref.inputs(params), camera, bg.to(device)).clamp(0, 1)
        gs, _ = core.gsplat_forward(params, camera)
        gs = gs.clamp(0, 1)
        flipped = fl.forward(prep, {**camera, "viewmat": torch.diag(
            torch.tensor([-1.0, -1.0, 1.0, 1.0], device=device))}, bg).float() / 255
    out = {
        "loadply_max_abs_diff": conv,
        "psnr_flashgs_vs_3dgs": core.psnr(img, c3),
        "psnr_flashgs_vs_3dgs_trunc": core.psnr(img, core.quantize_trunc(c3).float() / 255),
        "psnr_flashgs_vs_gsplat": core.psnr(img, gs),
        "psnr_flashgs_rot180_vs_3dgs": core.psnr(flipped, c3),
        "pairs_flashgs": fl.last["num_rendered"],
        "pairs_3dgs": ref.last["num_rendered"],
    }
    if conv["position"] > 0 or conv["opacity"] > 1e-6 or conv["shs"] > 0 or conv["cov3d_rel"] > 1e-5:
        raise AssertionError(f"model conversion differs from upstream loadPly: {conv}")
    if out["psnr_flashgs_vs_3dgs"] < 40 or out["psnr_flashgs_vs_gsplat"] < 40:
        raise AssertionError(f"FlashGS image differs from the reference rasterizers: {out}")
    if out["psnr_flashgs_rot180_vs_3dgs"] > 30:
        raise AssertionError("parity check is insensitive to the camera pose")
    return out


def precise_intersection(method, device):
    """C1: every tile where some pixel gets alpha >= 1/255 (brute force from gsplat's own
    projection) is kept, and FlashGS emits strictly fewer pairs than both the 3DGS 3-sigma
    AABB and its own coarse rectangle."""
    _need_cuda(device)
    from gsplat import fully_fused_projection

    params, camera = fixture(n=3000, width=256, height=192, seed=1, device=device)
    prep = core.prepare(params)
    fl = core.FlashGSRasterizer(device)
    with torch.no_grad():
        fl.forward(prep, camera, torch.ones(3))
        tiles, gids = fl.pairs()
        ref = core.Reference3DGS()
        ref.forward(ref.inputs(params), camera, torch.ones(3, device=device))
        bound = int(core.bbox_tile_counts(prep, camera).sum())
        W, H, T = camera["width"], camera["height"], core.TILE
        gx = (W + T - 1) // T
        radii, m2d, _, conics, _ = fully_fused_projection(
            params["means"], None, params["quats"], params["scales"].exp(),
            camera["viewmat"][None], camera["K"][None], W, H, eps2d=0.3, near_plane=0.2)
        ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device),
                                indexing="ij")
        pix = torch.stack([xs, ys], -1).reshape(-1, 2).float() + 0.5
        tile_of_pix = (ys // T * gx + xs // T).reshape(-1)
        have = set(zip(tiles.tolist(), gids.tolist()))
        missing = 0
        needed = 0
        opac = params["opacities"].sigmoid()
        visible = torch.nonzero(radii[0].min(-1).values > 0).reshape(-1)
        for i in visible.tolist():
            d = pix - m2d[0, i]
            a, b, c = conics[0, i]
            q = a * d[:, 0] ** 2 + 2 * b * d[:, 0] * d[:, 1] + c * d[:, 1] ** 2
            alpha = opac[i] * torch.exp(-0.5 * q)
            hit = torch.unique(tile_of_pix[alpha >= 1.05 * core.ALPHA_PAIR_MIN]).tolist()
            needed += len(hit)
            missing += sum((t, i) not in have for t in hit)
    out = {"pairs_flashgs": len(gids), "pairs_coarse_rect": bound,
           "pairs_3dgs_aabb": ref.last["num_rendered"], "needed_pairs": needed,
           "missing_pairs": missing}
    if missing:
        raise AssertionError(f"FlashGS dropped tiles with visible contribution: {out}")
    if not (len(gids) < bound and len(gids) < ref.last["num_rendered"]):
        raise AssertionError(f"no reduction of key/value pairs: {out}")
    return out


def _timed(fn, repeats):
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def speed(method, device):
    """C2-C4 (scheduling / pipelined kernels): FlashGS forward is faster than the original
    3DGS forward on a 1080p fixture (median of repeated frames, same GPU; shared GPUs make
    the absolute value noisy, so only `faster` is asserted)."""
    _need_cuda(device)
    params, camera = fixture(n=300000, width=1920, height=1080, seed=2, device=device)
    prep = core.prepare(params)
    fl = core.FlashGSRasterizer(device)
    ref = core.Reference3DGS()
    inp = ref.inputs(params)
    bg = torch.ones(3)
    bgd = bg.to(device)
    with torch.no_grad():
        fl.reserve(prep, camera)
        fl.forward(prep, camera, bg, reserve=False)
        ref.forward(inp, camera, bgd)
        t_fl = _timed(lambda: fl.forward(prep, camera, bg, reserve=False), 15)
        t_3d = _timed(lambda: ref.forward(inp, camera, bgd), 15)
        t_gs = _timed(lambda: core.gsplat_forward(params, camera), 15)
    out = {"ms_flashgs": t_fl, "ms_3dgs": t_3d, "ms_gsplat": t_gs,
           "speedup_vs_3dgs": t_3d / t_fl, "speedup_vs_gsplat": t_gs / t_fl,
           "note": "shared-GPU timing, not comparable to the paper's RTX 3090"}
    if not t_fl < t_3d:
        raise AssertionError(f"FlashGS is not faster than the 3DGS forward: {out}")
    return out


def inference_switch(method, device):
    """The method's renderer uses FlashGS exactly when no gradient is needed."""
    _need_cuda(device)
    from splatstudio.runtime import build

    renderer, _, _ = build(method)
    params, camera = fixture(n=2000, width=160, height=128, seed=3, device=device)
    params = {k: torch.nn.Parameter(v) for k, v in params.items()}
    out = renderer.render(params, camera, 3)
    if "means2d" not in out["info"] or not out["rgb"].requires_grad:
        raise AssertionError("training render is not the differentiable gsplat path")
    with torch.no_grad():
        inf = renderer.render(params, camera, 3)
    if inf["info"].get("backend") != "flashgs":
        raise AssertionError("inference render did not use FlashGS")
    diff = core.psnr(inf["rgb"], out["rgb"].detach().clamp(0, 1))
    if not math.isfinite(diff) or diff < 35:
        raise AssertionError(f"training/inference images disagree: {diff:.2f} dB")
    return {"psnr_inference_vs_training_render": diff}
