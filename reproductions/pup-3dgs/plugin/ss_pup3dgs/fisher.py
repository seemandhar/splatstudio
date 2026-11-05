"""PUP 3D-GS per-Gaussian Fisher (mean + log-scale 6x6 block) via the official CUDA kernel.

`csrc/` vendors https://github.com/j-alex-hanson/rasterization_and_pup_fisher (commit eaef2fa):
the Inria 3DGS rasterizer whose backward pass (`renderFisherCUDA`, FisherRF-derived) returns,
instead of loss gradients, sum over pixels p and channels c of g g^T with
g = d I(p)_c / d(mean_i, log scale_i) (Inria licence, non-commercial research use; see
`csrc/LICENSE-3DGS.md`). This module converts SplatStudio cameras/params to that rasterizer's
conventions and reproduces `fisher_pool_xyz_scaling.py:pool_fisher_cuda` of the official code.
"""

import math
from pathlib import Path

import torch

_EXT = None
CSRC = Path(__file__).resolve().parent / "csrc"
ZNEAR, ZFAR = 0.01, 100.0  # 3DGS Camera defaults


def extension():
    """JIT-build (once, cached by torch) and return the vendored CUDA extension."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        sources = [
            CSRC / "cuda_rasterizer" / "rasterizer_impl.cu",
            CSRC / "cuda_rasterizer" / "forward.cu",
            CSRC / "cuda_rasterizer" / "backward.cu",
            CSRC / "rasterize_points.cu",
            CSRC / "ext.cpp",
        ]
        _EXT = load(
            name="ss_pup3dgs_fisher",
            sources=[str(s) for s in sources],
            extra_cuda_cflags=["-O3", f"-I{CSRC / 'third_party' / 'glm'}"],
            extra_cflags=["-O3", f"-I{CSRC / 'third_party' / 'glm'}"],
            verbose=False,
        )
    return _EXT


def low_res_camera(camera, resolution):
    """Official patch-wise Fisher: render at ceil(H/r) x ceil(W/r) with the same field of view."""
    width, height = int(camera["width"]), int(camera["height"])
    w, h = math.ceil(width / resolution), math.ceil(height / resolution)
    K = camera["K"].clone().float()
    K[0] *= w / width
    K[1] *= h / height
    return {"K": K, "viewmat": camera["viewmat"].float(), "width": w, "height": h}


def raster_matrices(camera):
    """3DGS rasterizer settings for an OpenCV pinhole camera (K, world-to-camera viewmat).

    3DGS maps ndc to pixels with ((ndc + 1) * S - 1) / 2 and samples integer pixel coords, while
    gsplat samples pixel centres at +0.5 with u = fx x/z + cx. Choosing P00 = 2 fx / W and
    P02 = 2 cx / W - 1 (same for y) makes both conventions render the same image, including an
    off-centre principal point (3DGS itself assumes cx = W/2, where this reduces to its matrix).
    """
    K, viewmat = camera["K"].float(), camera["viewmat"].float()
    W, H = int(camera["width"]), int(camera["height"])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    proj = torch.zeros(4, 4, device=K.device)
    proj[0, 0] = 2 * fx / W
    proj[0, 2] = 2 * cx / W - 1
    proj[1, 1] = 2 * fy / H
    proj[1, 2] = 2 * cy / H - 1
    proj[3, 2] = 1.0
    proj[2, 2] = ZFAR / (ZFAR - ZNEAR)
    proj[2, 3] = -(ZFAR * ZNEAR) / (ZFAR - ZNEAR)
    world_view = viewmat.T.contiguous()  # 3DGS stores transposed (row-vector) matrices
    full_proj = (world_view @ proj.T).contiguous()
    campos = torch.linalg.inv(viewmat)[:3, 3].contiguous()
    tanfovx = float(W / (2 * fx))
    tanfovy = float(H / (2 * fy))
    return world_view, full_proj, campos, tanfovx, tanfovy


def gaussian_inputs(params):
    """Activated per-Gaussian tensors in the 3DGS rasterizer layout."""
    means = params["means"].detach().float().contiguous()
    scales = params["scales"].detach().exp().float().contiguous()
    quats = torch.nn.functional.normalize(params["quats"].detach().float(), dim=-1).contiguous()
    opacities = params["opacities"].detach().sigmoid().float()[:, None].contiguous()
    sh = torch.cat((params["sh0"], params["shN"]), dim=1).detach().float().contiguous()
    return means, scales, quats, opacities, sh


@torch.no_grad()
def view_fisher(inputs, camera, background, sh_degree=3):
    """[P,21] upper-triangular (row-major) Fisher of one view: sum_p,c g g^T, dI/dpixel = 1."""
    ext = extension()
    means, scales, quats, opacities, sh = inputs
    world_view, full_proj, campos, tanfovx, tanfovy = raster_matrices(camera)
    empty = torch.empty(0, device=means.device)
    W, H = int(camera["width"]), int(camera["height"])
    num_rendered, _, radii, geom, binning, img = ext.rasterize_gaussians(
        background, means, empty, opacities, scales, quats, 1.0, empty, world_view, full_proj,
        tanfovx, tanfovy, H, W, sh, sh_degree, campos, False, False,
    )
    grad_out = torch.ones(3, H, W, device=means.device)
    outputs = ext.rasterize_gaussians_backward(
        background, means, radii, empty, scales, quats, 1.0, empty, world_view, full_proj,
        tanfovx, tanfovy, grad_out, sh, sh_degree, campos, geom, num_rendered, binning, img,
        False,
    )
    return outputs[-1]


# (row, col) of each of the 21 entries, in the kernel's order (fisher_pool_xyz_scaling.py).
TRIU = [(r, c) for r in range(6) for c in range(r, 6)]


def expand_sym(sym):
    """[P,21] -> [P,6,6] symmetric matrices."""
    rows = torch.tensor([r for r, _ in TRIU], device=sym.device)
    cols = torch.tensor([c for _, c in TRIU], device=sym.device)
    full = torch.zeros(sym.shape[0], 6, 6, dtype=sym.dtype, device=sym.device)
    full[:, rows, cols] = sym
    full[:, cols, rows] = sym
    return full


@torch.no_grad()
def scene_fisher(params, cameras, resolution=4, background=(1.0, 1.0, 1.0), sh_degree=3):
    """Sum of per-view Fishers over all training cameras, as [P,21] (float32 like the official)."""
    inputs = gaussian_inputs(params)
    device = inputs[0].device
    bg = torch.tensor(background, dtype=torch.float32, device=device)
    total = torch.zeros(inputs[0].shape[0], 21, device=device)
    for camera in cameras:
        total += view_fisher(inputs, low_res_camera(camera, resolution), bg, sh_degree)
    return total


@torch.no_grad()
def log_det_scores(sym, chunk=1 << 20):
    """Official score: torch.log(torch.linalg.svdvals(F)).sum(1) (= log|det F|, -inf if F = 0)."""
    out = torch.empty(sym.shape[0], device=sym.device)
    for start in range(0, sym.shape[0], chunk):
        full = expand_sym(sym[start : start + chunk])
        out[start : start + chunk] = torch.log(torch.linalg.svdvals(full)).sum(dim=1)
    return out


def prune_mask(scores, percent):
    """Official `prune_gaussians`: prune scores <= the int(percent*(N-1))-th smallest score."""
    sorted_scores, _ = torch.sort(scores, dim=0)
    threshold = sorted_scores[int(percent * (sorted_scores.shape[0] - 1))]
    return scores <= threshold
