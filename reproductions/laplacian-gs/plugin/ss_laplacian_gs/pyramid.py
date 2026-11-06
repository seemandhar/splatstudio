"""Laplacian pyramid operators of the official code (laplacian/laplacian_torch.py).

kernel = outer([1,4,6,4,1], [1,4,6,4,1]) / 256
Down(x)      = conv2d(reflect_pad(x, 2), kernel, stride 2)            (_pyr_down)
Up(x, size)  = bilinear(x, size, align_corners=False) [+ blur]         (_pyr_up, --blur_up)
blur(x)      = conv2d(reflect_pad(x, 2), kernel, stride 1)            (_gaussian_blur)
Eq. 4: I^{l+1} = Down(I^l), R^l = I^l - Up(I^{l+1});  Eq. 7: I^l = Up(I^{l+1}) + R^l.
All functions take HWC images (the native runtime convention).
"""

import torch
import torch.nn.functional as F


def kernel(device, dtype=torch.float32):
    k1 = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)
    k2 = torch.outer(k1, k1)
    return (k2 / k2.sum())[None, None]


def _nchw(x):
    return x.permute(2, 0, 1)[None]


def _hwc(x):
    return x[0].permute(1, 2, 0)


def down(x):
    """HWC -> HWC at ((H-1)//2+1, (W-1)//2+1)."""
    y = _nchw(x)
    c = y.shape[1]
    y = F.pad(y, (2, 2, 2, 2), mode="reflect")
    return _hwc(F.conv2d(y, kernel(x.device, x.dtype).expand(c, 1, 5, 5), stride=2, groups=c))


def blur(x):
    y = _nchw(x)
    c = y.shape[1]
    y = F.pad(y, (2, 2, 2, 2), mode="reflect")
    return _hwc(F.conv2d(y, kernel(x.device, x.dtype).expand(c, 1, 5, 5), groups=c))


def up(x, size, blur_up=True):
    """HWC -> HWC of `size` (h, w): bilinear, then the Gaussian blur when blur_up."""
    if tuple(x.shape[:2]) == tuple(size):
        # official get_upsampled: identical sizes are passed through untouched
        return x
    y = _hwc(F.interpolate(_nchw(x), size=tuple(size), mode="bilinear", align_corners=False))
    return blur(y) if blur_up else y


def down_size(h, w):
    return (h - 1) // 2 + 1, (w - 1) // 2 + 1


def level_sizes(height, width, levels):
    """[(h_0, w_0), ..., (h_{L-1}, w_{L-1})] of the Gaussian pyramid (level 0 = full)."""
    sizes = [(int(height), int(width))]
    for _ in range(levels - 1):
        sizes.append(down_size(*sizes[-1]))
    return sizes


def gaussian_pyramid(image, levels):
    pyr = [image]
    for _ in range(levels - 1):
        pyr.append(down(pyr[-1]))
    return pyr


def target_at_level(image, level):
    for _ in range(level):
        image = down(image)
    return image
