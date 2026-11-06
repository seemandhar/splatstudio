"""JIT build + Python entry of the utilisation kernel (``csrc/ubp.cu``)."""

from pathlib import Path

CSRC = Path(__file__).resolve().parent / "csrc"
_EXT = None


def extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        _EXT = load(name="ss_spotlesssplats_ubp", sources=[str(CSRC / "ubp.cu")],
                    extra_cuda_cflags=["-O3"], extra_cflags=["-O3"], verbose=False)
    return _EXT


def utilization(info, colors, v_render):
    """Sum over pixels of the squared per-pixel screen-mean gradient, per Gaussian [N].

    ``info``: gsplat ``rasterization`` meta of a single-camera, non-packed render;
    ``colors``: [N,3] colours handed to the rasteriser; ``v_render``: dL/dI [H,W,3].
    """
    if info["means2d"].shape[0] != 1:
        raise ValueError("utilization expects a single camera")
    means2d = info["means2d"][0].detach().float().contiguous()
    conics = info["conics"][0].detach().float().contiguous()
    opac = info["opacities"][0].detach().float().contiguous()
    offsets = info["isect_offsets"][0].int().contiguous()
    ids = info["flatten_ids"].int().contiguous()
    return extension().ubp_sqrgrad(means2d, conics, colors.detach().float().contiguous(), opac,
                                   offsets, ids, v_render.detach().float().contiguous(),
                                   int(info["tile_size"]))
