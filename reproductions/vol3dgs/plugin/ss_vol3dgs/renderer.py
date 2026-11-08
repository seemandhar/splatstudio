"""Vol3DGS renderer: volumetric (analytic ray-integral) alpha in a tile rasterizer.

Pipeline (one camera):
1. Tiling and depth sorting from gsplat 1.5.3 `fully_fused_projection` (EWA, eps2d 0.3) under
   no_grad, with the official bounding radius ceil(3 sqrt(lambda_max)) (slang `splat_radius`,
   max(0.1, mid^2 - det) inside the sqrt) and the official near plane z > 0.2. The splatted 2D
   Gaussian is used ONLY for tile assignment, never for alpha (as in the official vertex shader).
   Fidelity note: the boxes come from gsplat's EWA projection (eps2d 0.3, its own culling) rather
   than the slang vertex shader; same 3-sigma + 0.3 formula, but boundary Gaussians may differ, and
   alpha beyond the 3-sigma box (high density) is clipped exactly as in the official code.
2. Differentiable torch: camera-space means and inverse 3D covariances, density
   kappa = -log(1 - 0.99 theta) mean(1/s) / 2.506 (code alpha2density; paper Eq. 20), colour
   softplus(SH + 0.5, beta = 10) (official --softplus_rgb).
3. CUDA kernel (csrc/vol_raster.cu): alpha = min(0.99, 1 - exp(-sqrt(2 pi) kappa beta G(gamma d))).

`info["means_in"]` is the graph entry for the means, so the density controller can read the
world-space position gradient dL/dmu used by the official densification.
"""

import math

import torch
import torch.nn.functional as F
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    spherical_harmonics,
)

from .kernels import rasterize_vol
from .reference import PRECIS_INDEX, alpha2density, camera_gaussians

CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.world.means")
OPTIONS = {
    "tile_size": 16,
    "eps2d": 0.3,
    "near_plane": 0.2,
    "radius_sigma": 3.0,
    "softplus_beta": 10.0,
    "sh_every": 2000,
    "background": "white",
}


def tile_radii(conics, base_radii, means2d, width, height, sigma=3.0):
    """Official splat_radius: ceil(sigma sqrt(lambda_max(Sigma2D))), square box [...,N,2] int32.

    Sigma2D = conic^-1 (already includes the 0.3 dilation). Gaussians gsplat culled
    (near/far plane, singular covariance) stay culled; boxes outside the image are dropped."""
    with torch.no_grad():
        a, b, c = conics[..., 0], conics[..., 1], conics[..., 2]
        det_c = (a * c - b * b).clamp_min(1e-12)
        cxx, cyy, cxy = c / det_c, a / det_c, -b / det_c
        mid = 0.5 * (cxx + cyy)
        det = cxx * cyy - cxy * cxy
        lam = mid + torch.sqrt((mid * mid - det).clamp_min(0.1))
        r = torch.ceil(sigma * lam.clamp_min(0).sqrt())
        mx, my = means2d[..., 0], means2d[..., 1]
        ok = (base_radii > 0).all(dim=-1)
        ok &= ~((mx + r <= 0) | (mx - r >= width) | (my + r <= 0) | (my - r >= height))
        out = torch.stack((r, r), dim=-1).to(torch.int32)
        return torch.where(ok[..., None], out, torch.zeros_like(out))


class VolRenderer:
    def __init__(self, spec):
        cfg = {**OPTIONS, **spec.config}
        unknown = set(cfg) - set(OPTIONS)
        if unknown:
            raise ValueError(f"Unknown Vol3DGS renderer options: {sorted(unknown)}")
        self.tile_size = int(cfg["tile_size"])
        self.eps2d = float(cfg["eps2d"])
        self.near = float(cfg["near_plane"])
        self.sigma = float(cfg["radius_sigma"])
        self.beta = float(cfg["softplus_beta"])
        self.sh_every = int(cfg["sh_every"])
        if cfg["background"] not in ("white", "black"):
            raise ValueError("background must be 'white' or 'black'")
        self.background = 1.0 if cfg["background"] == "white" else 0.0
        self.capabilities = set(CAPABILITIES)
        # Set by the density controller after each training step (paper SH schedule).
        self.next_step = None

    def set_next_step(self, step):
        self.next_step = int(step)

    def degree(self, sh_degree):
        # train_volr.py: `if iteration % 2000 == 0: oneupSHdegree()` -> min(step // 2000, 3).
        # The runtime's schedule (+1 every 1000) is never lower, so min() keeps eval unchanged.
        if torch.is_grad_enabled() and self.next_step is not None:
            return min(sh_degree, self.next_step // self.sh_every)
        return sh_degree

    def render(self, params, camera, sh_degree=3):
        means = params["means"]
        means_in = means.view_as(means)  # graph entry for the world-space position gradient
        viewmat, K = camera["viewmat"], camera["K"]
        W, H = int(camera["width"]), int(camera["height"])
        ts = self.tile_size
        with torch.no_grad():
            base_radii, means2d, depths, conics, _ = fully_fused_projection(
                means.detach(), None, params["quats"].detach(), params["scales"].detach().exp(),
                viewmat[None], K[None], W, H, eps2d=self.eps2d, near_plane=self.near,
                far_plane=1e10, radius_clip=0.0, packed=False, calc_compensations=False)
            radii = tile_radii(conics, base_radii, means2d, W, H, self.sigma)
            tw, th = math.ceil(W / ts), math.ceil(H / ts)
            _, isect_ids, flatten_ids = isect_tiles(
                means2d, radii, depths, ts, tw, th, packed=False, n_images=1)
            offsets = isect_offset_encode(isect_ids, 1, tw, th).reshape(th, tw)
        visible = (radii[0] > 0).all(dim=-1)
        mean_c, precis = camera_gaussians(means_in, params["quats"], params["scales"], viewmat)
        precis6 = precis[:, PRECIS_INDEX[0], PRECIS_INDEX[1]]
        theta = torch.sigmoid(params["opacities"])
        kappa = alpha2density(theta, params["scales"].exp())
        shs = torch.cat((params["sh0"], params["shN"]), dim=1)
        campos = torch.linalg.inv(viewmat)[:3, 3]
        degree = self.degree(sh_degree)
        colors = spherical_harmonics(degree, means_in - campos, shs, masks=visible)
        colors = F.softplus(colors + 0.5, beta=self.beta)
        feats = torch.cat((colors, mean_c[:, 2:3]), dim=-1)
        render, alpha = rasterize_vol(mean_c, precis6, kappa, feats, K, offsets, flatten_ids,
                                      W, H, ts)
        depth = render[..., 3:4] / alpha.clamp(min=1e-10)
        info = {
            "means_in": means_in,
            "visible": visible,
            "radii": radii[0],
            "means2d": means2d[0],
            "depths": depths[0],
            "sh_degree": degree,
            "set_next_step": self.set_next_step,
            "width": W,
            "height": H,
            "n_isects": int(flatten_ids.numel()),
        }
        return {"rgb": render[..., :3] + self.background * (1 - alpha), "depth": depth,
                "alpha": alpha, "info": info}


def create_renderer(spec):
    return VolRenderer(spec)
