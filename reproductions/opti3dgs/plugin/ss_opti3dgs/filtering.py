"""Image frequency modulation (Opti3DGS Sec. 3.3-3.4).

Eq. 2: G[i,j] = sum_{u,v} H[u,v] F[i-u, j-v] with H the uniform (mean) kernel, stride 1 and
padding chosen so the image size is unchanged. The padding mode is not stated in the paper;
"reflect" equals OpenCV's cv2.blur default (BORDER_REFLECT_101), "zeros" is avg_pool2d with
count_include_pad, "replicate" repeats the edge.
"""

import bisect

import torch.nn.functional as F

PADDING_MODES = ("reflect", "replicate", "zeros")


def mean_blur(image, kernel_size, padding_mode="reflect"):
    """HWC image -> HWC image of the same size, KxK mean filter (Eq. 2). K=1 is the identity."""
    k = int(kernel_size)
    if k < 1 or k % 2 == 0:
        raise ValueError(f"kernel size must be a positive odd integer, got {kernel_size}")
    if padding_mode not in PADDING_MODES:
        raise ValueError(f"padding_mode must be one of {PADDING_MODES}")
    if k == 1:
        return image
    pad = k // 2
    if pad >= min(image.shape[0], image.shape[1]):
        raise ValueError(f"image {tuple(image.shape[:2])} too small for kernel {k}")
    x = image.permute(2, 0, 1)[None]
    if padding_mode == "zeros":
        y = F.avg_pool2d(x, k, stride=1, padding=pad, count_include_pad=True)
    else:
        y = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode=padding_mode), k, stride=1)
    return y[0].permute(1, 2, 0)


def check_schedule(kernel_sizes, boundaries):
    kernel_sizes = [int(k) for k in kernel_sizes]
    boundaries = [int(b) for b in boundaries]
    if len(kernel_sizes) != len(boundaries) or not kernel_sizes:
        raise ValueError("kernel_sizes and boundaries must be non-empty and equally long")
    if any(b2 <= b1 for b1, b2 in zip(boundaries, boundaries[1:])) or boundaries[0] < 1:
        raise ValueError("boundaries must be strictly increasing positive steps")
    if any(k < 1 or k % 2 == 0 for k in kernel_sizes):
        raise ValueError("kernel sizes must be positive odd integers")
    return kernel_sizes, boundaries


def kernel_at(step, kernel_sizes, boundaries):
    """Step decay (Fig. 3): level i covers steps (boundaries[i-1], boundaries[i]]; 1 afterwards.

    `step` is the runtime's 1-based step (= Inria 3DGS iteration number)."""
    i = bisect.bisect_left(boundaries, int(step))
    return kernel_sizes[i] if i < len(kernel_sizes) else 1
