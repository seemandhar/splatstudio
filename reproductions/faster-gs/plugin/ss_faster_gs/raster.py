"""Python side of the vendored Faster-GS rasterizer.

`_Rasterize` reproduces `FasterGSCudaBackend/torch_bindings/rasterization.py` (main @ 090e8ad):
the kernels take the *raw* parameters (log-scales, unnormalised quaternions, opacity logits —
"fused activations") and the two SH buffers separately ("separate SH buffers"); the backward
optionally accumulates the 3DGS densification statistics into a persistent ``[2, N]`` buffer
(row 0: visibility count, row 1: sum of NDC-scaled ||dL/d mu_2D||).
"""

import torch

from .kernels import extension

NEAR, FAR = 0.2, 10000.0  # official dataset config (NEAR_PLANE / FAR_PLANE)


def settings(camera, background, sh_degree, near=NEAR, far=FAR):
    """RasterizerSettings for an OpenCV pinhole camera (viewmat = w2c, K in pixels).

    Pixel centres are at index + 0.5 in the Faster-GS kernels, the same convention as gsplat.
    """
    K = camera["K"].float()
    w2c = camera["viewmat"].float().contiguous()
    cam_position = torch.linalg.inv(w2c)[:3, 3].contiguous()
    return (
        w2c,
        cam_position,
        background.float().contiguous(),
        int((sh_degree + 1) ** 2),
        int(camera["width"]),
        int(camera["height"]),
        float(K[0, 0]),
        float(K[1, 1]),
        float(K[0, 2]),
        float(K[1, 2]),
        float(near),
        float(far),
        False,  # proper_antialiasing (Mip-Splatting 2D filter): off in the paper configuration
    )


class _Rasterize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, scales, rotations, opacities, sh0, shN, densification_info, s, stats):
        (image, primitive_buffers, tile_buffers, instance_buffers, bucket_buffers,
         n_instances, n_buckets, selector) = extension().forward(
            means, scales, rotations, opacities, sh0, shN, *s)
        stats["n_instances"] = int(n_instances)
        stats["n_buckets"] = int(n_buckets)
        ctx.s = s
        ctx.buffer_state = (n_instances, n_buckets, selector)
        ctx.save_for_backward(image, means, scales, rotations, opacities, shN,
                              primitive_buffers, tile_buffers, instance_buffers, bucket_buffers)
        ctx.densification_info = densification_info
        ctx.mark_non_differentiable(densification_info)
        return image

    @staticmethod
    def backward(ctx, grad_image):
        grads = extension().backward(ctx.densification_info, grad_image.contiguous(),
                                     *ctx.saved_tensors, *ctx.s, *ctx.buffer_state)
        return (*grads, None, None, None)


def rasterize(params, camera, sh_degree, background, densification_info=None, near=NEAR,
              far=FAR, stats=None):
    """Differentiable render; returns the CHW image (not clamped, like official training)."""
    means = params["means"]
    if densification_info is None:
        densification_info = torch.empty(0, device=means.device)
    stats = {} if stats is None else stats
    return _Rasterize.apply(
        means.contiguous(), params["scales"].contiguous(), params["quats"].contiguous(),
        params["opacities"][:, None].contiguous(), params["sh0"].contiguous(),
        params["shN"].contiguous(), densification_info,
        settings(camera, background, sh_degree, near, far), stats)
