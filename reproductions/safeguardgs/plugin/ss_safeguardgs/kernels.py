"""JIT-built CUDA top-k kernel (csrc/) and the per-view SafeguardGS keep mask.

gsplat 1.5.3 exposes per-view tile lists (`isect_offsets`, `flatten_ids`) and projected 2D
Gaussians in `rasterization`'s meta but no per-ray score ranking; `csrc/sg_topk.cu` re-traverses
the tile lists with gsplat's alpha numerics and the official SafeguardGS score.
"""

import functools
from pathlib import Path

import torch
from gsplat import rasterization, spherical_harmonics

from . import core

CSRC = Path(__file__).resolve().parent / "csrc"
SELECTIONS = ("tile", "pixel")


@functools.cache
def ext():
    """Build (first call, cached by torch in ~/.cache/torch_extensions) and import the module."""
    from torch.utils.cpp_extension import load

    return load(
        name="ss_safeguardgs_cuda",
        sources=[str(CSRC / "ext.cpp"), str(CSRC / "sg_topk.cu")],
        extra_cuda_cflags=["-O3"],
        extra_cflags=["-O3"],
        verbose=False,
    )


def gaussian_colors(params, camera, sh_degree):
    """3DGS computeColorFromSH: SH up to `sh_degree`, + 0.5, clamped >= 0 (gsplat's path)."""
    coeffs = torch.cat((params["sh0"], params["shN"]), dim=1)[:, : (sh_degree + 1) ** 2]
    campos = torch.linalg.inv(camera["viewmat"])[:3, 3]
    rgb = spherical_harmonics(sh_degree, params["means"] - campos[None], coeffs)
    return torch.clamp_min(rgb + 0.5, 0.0)


@torch.no_grad()
def project(params, camera, sh_degree):
    """gsplat projection + tile intersection of one camera (classic mode, as training)."""
    colors = gaussian_colors(params, camera, sh_degree)
    _, _, meta = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(),
        colors=colors[None],
        viewmats=camera["viewmat"][None],
        Ks=camera["K"][None],
        width=camera["width"],
        height=camera["height"],
        sh_degree=None,
        packed=False,
        render_mode="RGB",
        rasterize_mode="classic",
    )
    return meta, colors


@torch.no_grad()
def view_keep(meta, colors, image, func_id=core.SG_36, top_k=1, selection="tile",
              p_coef=1.0, c_coef=1.0):
    """Bool [N]: Gaussians that are among the top_k of some ray (selection 'pixel') or of some
    16x16 tile's slot keys (selection 'tile', official kernel) of this view.

    image: [H,W,3] ground truth in [0,1]; colors [N,3] the per-Gaussian colours rendered.
    """
    if selection not in SELECTIONS:
        raise ValueError(f"selection must be one of {SELECTIONS}")
    if meta["means2d"].dim() != 3 or meta["means2d"].shape[0] != 1:
        raise ValueError("SafeguardGS statistics expect one non-packed camera")
    n = meta["means2d"].shape[1]
    tile_size = int(meta["tile_size"])
    if selection == "tile" and tile_size * tile_size != 256:
        raise ValueError("official slot semantics need 16x16 tiles (BLOCK_SIZE 256)")
    offsets = meta["isect_offsets"][0].int().contiguous()
    flat = meta["flatten_ids"].int().contiguous()
    pad, slots, n_slots = core.padded_offsets(offsets.flatten(), flat.numel(), tile_size**2)
    out = ext().sg_topk(
        meta["means2d"][0].detach().float().contiguous(),
        meta["conics"][0].detach().float().contiguous(),
        meta["opacities"][0].detach().float().contiguous(),
        colors.detach().float().reshape(-1, 3).contiguous(),
        offsets,
        flat,
        int(meta["width"]),
        int(meta["height"]),
        tile_size,
        image.detach().float().reshape(-1, 3).contiguous(),
        int(func_id),
        float(p_coef),
        float(c_coef),
        int(top_k),
        selection == "tile",
        pad.contiguous(),
        n_slots,
    )
    if selection == "pixel":
        return out[0].bool()
    return core.tile_topk_keep(out[0], out[1], slots, top_k, n)
