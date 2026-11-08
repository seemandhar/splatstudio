"""JIT build of the plugin's own CUDA statistics kernel (csrc/stats.cu)."""

import functools
from pathlib import Path

CSRC = Path(__file__).resolve().parent / "csrc"


@functools.cache
def extension():
    from torch.utils.cpp_extension import load

    return load(name="ss_loss_aware_dc_stats", sources=[str(CSRC / "stats.cu")],
                extra_cuda_cflags=["-O3", "--use_fast_math"], verbose=False)


def surrogate_stats(means2d, conics, opacities, colors, isect_offsets, flatten_ids, c_total,
                    dl_di, omega, width, height, tile_size):
    """Per-Gaussian [N,24] statistics for one camera (see csrc/stats.cu header)."""
    f = lambda t: t.float().contiguous()  # noqa: E731
    return extension().surrogate_stats(
        f(means2d), f(conics), f(opacities), f(colors), isect_offsets.int().contiguous(),
        flatten_ids.int().contiguous(), f(c_total), f(dl_di), f(omega), int(width), int(height),
        int(tile_size))


def joint_split_accumulate(ids, means, logs, quats, stats, viewmat, K, width, height, drop_mixed,
                           S_acc):
    """S_acc[ids] += per-view joint splitting matrix (in place; S_acc is [N,6,6] float32)."""
    f = lambda t: t.float().contiguous()  # noqa: E731
    if not S_acc.is_contiguous() or S_acc.dtype != __import__("torch").float32:
        raise ValueError("S_acc must be a contiguous float32 tensor")
    extension().joint_split_accumulate(ids.long().contiguous(), f(means), f(logs), f(quats),
                                       f(stats), f(viewmat), f(K), int(width), int(height),
                                       bool(drop_mixed), S_acc)
