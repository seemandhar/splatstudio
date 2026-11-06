"""Split-sum image-based lighting of GS-IR (official pbr/light.py + pbr/shade.py).

Environment light = learnable cube map ``base`` [6, 256, 256, 3]. ``build_mips``: 2x2 mips down
to 16^2; diffuse = cosine convolution of the 16^2 mip; specular level k (k < L-1) pre-filtered
with roughness 0.08 + k/(L-2) * 0.42, the last level with roughness 1.0 (energy cutoff 0.99).
``pbr_shading`` follows shade.py exactly (Eq. 15 of the paper as implemented):
  diffuse  = albedo * [cube_diffuse(n) * V + (1 - V) * irradiance]
  specular = cube_specular(reflect(v, n), mip(roughness)) * (F0 * A + B),
  F0 = (1 - m) 0.04 + m * albedo, (A, B) = BRDF_LUT(clamp(n.v, 1e-4, 1), roughness)
  rgb = clamp(diffuse + specular, 0, 1); pixels outside ``mask`` -> background (black).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import cubemap

LIGHT_MIN_RES = 16
MIN_ROUGHNESS = 0.08
MAX_ROUGHNESS = 0.5
_LUT = {}


def brdf_lut(device):
    """[1, 256, 256, 2] split-sum LUT shipped with GS-IR (pbr/brdf_256_256.bin, from nvdiffrec)."""
    key = str(device)
    if key not in _LUT:
        path = Path(__file__).resolve().parent / "data" / "brdf_256_256.bin"
        arr = np.fromfile(path, dtype=np.float32).reshape(1, 256, 256, 2)
        _LUT[key] = torch.from_numpy(arr).to(device)
    return _LUT[key]


def lut_lookup(lut, u, v):
    """nvdiffrast 2D 'linear' filter with 'clamp' boundary; u along width, v along height."""
    shape = u.shape
    grid = torch.stack((u * 2 - 1, v * 2 - 1), dim=-1).reshape(1, -1, 1, 2)
    out = F.grid_sample(lut.permute(0, 3, 1, 2), grid, mode="bilinear", padding_mode="border",
                        align_corners=False)  # [1, 2, P, 1]
    return out[0, :, :, 0].T.reshape(*shape, 2)


def init_base(res, generator=None, device="cuda"):
    """CubemapLight init: U(0, 1) * 0.5 + 0.25."""
    return torch.rand(6, res, res, 3, generator=generator, device=device) * 0.5 + 0.25


def build_mips(base, cutoff=0.99):
    """-> (specular levels list, diffuse cube) (CubemapLight.build_mips)."""
    levels = [base]
    while levels[-1].shape[1] > LIGHT_MIN_RES:
        levels.append(cubemap.mip_down(levels[-1]))
    diffuse = cubemap.diffuse_convolve(levels[-1])
    spec = []
    for i in range(len(levels) - 1):
        r = (i / (len(levels) - 2)) * (MAX_ROUGHNESS - MIN_ROUGHNESS) + MIN_ROUGHNESS
        spec.append(cubemap.specular_filter(levels[i], r, cutoff))
    spec.append(cubemap.specular_filter(levels[-1], 1.0, cutoff))
    return spec, diffuse


def mip_level(roughness, n_levels):
    """CubemapLight.get_mip."""
    lo = ((roughness.clamp(MIN_ROUGHNESS, MAX_ROUGHNESS) - MIN_ROUGHNESS)
          / (MAX_ROUGHNESS - MIN_ROUGHNESS) * (n_levels - 2))
    hi = (roughness.clamp(MAX_ROUGHNESS, 1.0) - MAX_ROUGHNESS) / (1.0 - MAX_ROUGHNESS) + n_levels - 2
    return torch.where(roughness < MAX_ROUGHNESS, lo, hi)


def pbr_shading(spec, diffuse, lut, normals, view_dirs, albedo, roughness, mask,
                metallic=None, occlusion=None, irradiance=None, background=None):
    """All maps HWC. normals / view_dirs unit, view_dirs point from the surface to the camera."""
    if background is None:
        background = torch.zeros_like(normals)
    ref = 2.0 * (normals * view_dirs).sum(-1, keepdim=True).clamp(min=0.0) * normals - view_dirs
    diffuse_light = cubemap.sample(diffuse, normals, "linear")
    if occlusion is not None:
        diffuse_light = diffuse_light * occlusion + (1 - occlusion) * irradiance
    diffuse_rgb = diffuse_light * albedo
    nov = (normals * view_dirs).sum(-1, keepdim=True).clamp(min=1e-4, max=1.0)
    fg = lut_lookup(lut, nov[..., 0], roughness[..., 0])
    level = mip_level(roughness, len(spec))[..., 0]
    spec_light = cubemap.sample_mip(spec, ref, level)
    if metallic is None:
        f0 = torch.ones_like(albedo) * 0.04
    else:
        f0 = (1.0 - metallic) * 0.04 + albedo * metallic
    specular_rgb = spec_light * (f0 * fg[..., 0:1] + fg[..., 1:2])
    rgb = (diffuse_rgb + specular_rgb).clamp(0.0, 1.0)
    rgb = torch.where(mask, rgb, background)
    return {"rgb": rgb, "diffuse_rgb": diffuse_rgb, "specular_rgb": specular_rgb,
            "diffuse_light": diffuse_light}
