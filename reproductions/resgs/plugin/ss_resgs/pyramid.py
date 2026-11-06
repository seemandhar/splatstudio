"""Gaussian image pyramid matching OpenCV `cv2.pyrDown` (official utils/camera_utils.py:loadCam).

cv2.pyrDown: 5x5 binomial kernel [1, 4, 6, 4, 1] / 16 (separable), BORDER_REFLECT_101, then
keep even rows/columns; output size ((W + 1) // 2, (H + 1) // 2). PyTorch's "reflect" padding is
BORDER_REFLECT_101.
"""

import torch
import torch.nn.functional as F

_KERNEL = (1.0, 4.0, 6.0, 4.0, 1.0)


def pyr_size(height, width, level):
    for _ in range(int(level)):
        height, width = (height + 1) // 2, (width + 1) // 2
    return height, width


def pyr_down(image):
    """HWC float image -> HWC image of size ((H+1)//2, (W+1)//2)."""
    h, w, c = image.shape
    if min(h, w) < 3:
        raise ValueError("pyr_down needs images of at least 3 x 3 pixels")
    k = torch.tensor(_KERNEL, dtype=image.dtype, device=image.device) / 16.0
    x = image.permute(2, 0, 1)[None]  # 1,C,H,W
    x = F.pad(x, (2, 2, 2, 2), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, 5).expand(c, 1, 1, 5), groups=c)
    x = F.conv2d(x, k.view(1, 1, 5, 1).expand(c, 1, 5, 1), groups=c)
    x = x[:, :, ::2, ::2]
    return x[0].permute(1, 2, 0)


def pyramid_target(image, level):
    for _ in range(int(level)):
        image = pyr_down(image)
    return image


def scaled_camera(camera, level):
    """Same FoV at the pyramid size: pixel intrinsics scaled by (w'/W, h'/H)."""
    if level == 0:
        return camera
    H, W = int(camera["height"]), int(camera["width"])
    h, w = pyr_size(H, W, level)
    K = camera["K"].clone()
    K[0] *= w / W
    K[1] *= h / H
    return {**camera, "K": K, "width": w, "height": h}

