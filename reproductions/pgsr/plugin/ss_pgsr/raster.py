"""Official PGSR plane rasterizer, JIT-built from the vendored ``csrc/``.

``csrc/`` is ``submodules/diff-plane-rasterization`` of github.com/zju3dv/PGSR at commit
de24f1a, unmodified (Inria/MPII Gaussian-Splatting licence + ZJU 3DV non-commercial research
licence, see ``csrc/LICENSE-*.md``). This module re-implements the thin Python binding
(``diff_plane_rasterization/__init__.py``) and ``gaussian_renderer.render`` /
``render_normal`` / ``GaussianModel.get_normal`` of the same commit.

Kernel conventions (forward.cu): integer pixel coordinates, principal point (W/2, H/2),
alpha = min(0.99, o G), stop at T < 1e-4, ``out_observe[i]`` = number of pixels where Gaussian
i was blended while T > 0.5, ``all_map`` channels 0-2 camera-frame normal, 3 alpha (1), 4 plane
distance; plane depth = D / -(N . ((x-cx)/fx, (y-cy)/fy, 1) + 1e-8).
``means2D.grad`` = 3DGS signed screen gradient, ``means2D_abs.grad`` = AbsGS per-pixel absolute
sums (backward.cu:602).
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from gsplat.utils import normalized_quat_to_rotmat

CSRC = Path(__file__).resolve().parent / "csrc"
ZNEAR, ZFAR = 0.01, 100.0  # official scene/cameras.py
_EXT = None


def extension():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        glm = CSRC / "third_party" / "glm"
        sources = [CSRC / "cuda_rasterizer" / f for f in
                   ("rasterizer_impl.cu", "forward.cu", "backward.cu")]
        sources += [CSRC / "rasterize_points.cu", CSRC / "ext.cpp"]
        _EXT = load(name="ss_pgsr_plane", sources=[str(s) for s in sources],
                    extra_cuda_cflags=["-O3", f"-I{glm}"], extra_cflags=["-O3", f"-I{glm}"],
                    verbose=False)
    return _EXT


def check_centered(camera):
    """The official kernel places the principal point at (W/2, H/2) (FoV camera model)."""
    K = camera["K"]
    W, H = int(camera["width"]), int(camera["height"])
    if abs(float(K[0, 2]) - W / 2) > 0.51 or abs(float(K[1, 2]) - H / 2) > 0.51:
        raise ValueError("PGSR rasterizer assumes a centred principal point (cx = W/2, cy = H/2)")


def intrinsics(camera, scale=1.0):
    """Official Camera.get_k(scale): fx/scale, fy/scale, Cx = W/2 / scale."""
    K = camera["K"].float()
    W, H = int(camera["width"]), int(camera["height"])
    out = torch.zeros(3, 3, device=K.device)
    out[0, 0], out[1, 1] = K[0, 0] / scale, K[1, 1] / scale
    out[0, 2], out[1, 2], out[2, 2] = 0.5 * W / scale, 0.5 * H / scale, 1.0
    return out


def raster_settings(camera, background, sh_degree, render_geo):
    """Inria conventions for an OpenCV pinhole camera (K, w2c viewmat)."""
    check_centered(camera)
    K, viewmat = camera["K"].float(), camera["viewmat"].float()
    W, H = int(camera["width"]), int(camera["height"])
    fx, fy = K[0, 0], K[1, 1]
    tanx, tany = float(W / (2 * fx)), float(H / (2 * fy))
    proj = torch.zeros(4, 4, device=K.device)  # getProjectionMatrix (symmetric frustum)
    proj[0, 0] = 1.0 / tanx
    proj[1, 1] = 1.0 / tany
    proj[3, 2] = 1.0
    proj[2, 2] = ZFAR / (ZFAR - ZNEAR)
    proj[2, 3] = -(ZFAR * ZNEAR) / (ZFAR - ZNEAR)
    world_view = viewmat.T.contiguous()
    return {
        "bg": background, "viewmatrix": world_view,
        "projmatrix": (world_view @ proj.T).contiguous(),
        "campos": torch.linalg.inv(viewmat)[:3, 3].contiguous(),
        "tanfovx": tanx, "tanfovy": tany, "height": H, "width": W,
        "sh_degree": int(sh_degree), "render_geo": bool(render_geo),
    }


class _Rasterize(torch.autograd.Function):
    """diff_plane_rasterization._RasterizeGaussians with identical argument order."""

    @staticmethod
    def forward(ctx, means3D, means2D, means2D_abs, sh, opacities, scales, rotations, all_map,
                s):
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], means3D, empty, opacities, scales, rotations, 1.0, empty, all_map,
                s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"], s["height"],
                s["width"], sh, s["sh_degree"], s["campos"], False, s["render_geo"], False)
        (num_rendered, color, radii, observe, out_all_map, plane_depth, geom, binning,
         img) = extension().rasterize_gaussians(*args)
        ctx.s = s
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(out_all_map, all_map, means3D, scales, rotations, radii, sh, geom,
                              binning, img)
        ctx.mark_non_differentiable(radii, observe)
        return color, radii, observe, out_all_map, plane_depth

    @staticmethod
    def backward(ctx, g_color, _g_radii, _g_observe, g_all_map, g_plane_depth):
        s = ctx.s
        (out_all_map, all_map, means3D, scales, rotations, radii, sh, geom, binning,
         img) = ctx.saved_tensors
        empty = torch.empty(0, device=means3D.device)
        args = (s["bg"], out_all_map, means3D, radii, empty, all_map, scales, rotations, 1.0,
                empty, s["viewmatrix"], s["projmatrix"], s["tanfovx"], s["tanfovy"],
                g_color.contiguous(), g_all_map.contiguous(), g_plane_depth.contiguous(), sh,
                s["sh_degree"], s["campos"], geom, ctx.num_rendered, binning, img,
                s["render_geo"], False)
        (g_means2D, g_means2D_abs, _g_colors, g_opac, g_means3D, _g_cov, g_sh, g_scales, g_rot,
         g_all) = extension().rasterize_gaussians_backward(*args)
        if not s["render_geo"]:
            g_all = None
        return g_means3D, g_means2D, g_means2D_abs, g_sh, g_opac, g_scales, g_rot, g_all, None


def activated(params):
    scales = params["scales"].exp()
    opac = params["opacities"].sigmoid()
    rots = F.normalize(params["quats"], dim=-1)  # wxyz, official get_rotation
    sh = torch.cat((params["sh0"], params["shN"]), dim=1)
    return scales, opac, rots, sh


def gaussian_normals(means, scales, rots, campos):
    """GaussianModel.get_normal: axis of the smallest scale, flipped to face the camera."""
    R = normalized_quat_to_rotmat(rots)  # columns are the local axes (pytorch3d convention)
    idx = scales.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
    normal = R.gather(2, idx).squeeze(2)
    flip = (normal * (campos[None] - means)).sum(-1) < 0.0
    return torch.where(flip[:, None], -normal, normal)


def plane_inputs(means, scales, rots, camera):
    """``all_map`` [N,5]: camera-frame normal, 1, plane distance |n_c . p_c| (render())."""
    viewmat = camera["viewmat"].float()
    campos = torch.linalg.inv(viewmat)[:3, 3]
    n_world = gaussian_normals(means, scales, rots, campos)
    n_cam = n_world @ viewmat[:3, :3].T
    p_cam = means @ viewmat[:3, :3].T + viewmat[:3, 3]
    dist = (n_cam * p_cam).sum(-1).abs()
    return torch.cat([n_cam, torch.ones_like(dist)[:, None], dist[:, None]], dim=-1)


def camera_rays(camera, scale=1.0):
    """Camera.get_rays: ((x - Cx)/Fx, (y - Cy)/Fy, 1) on the integer pixel grid. [H,W,3]."""
    K = intrinsics(camera, scale)
    W, H = int(int(camera["width"]) / scale), int(int(camera["height"]) / scale)
    dev = K.device
    iy, ix = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev),
                            indexing="ij")
    return torch.stack([(ix - K[0, 2]) / K[0, 0], (iy - K[1, 2]) / K[1, 1],
                        torch.ones_like(ix, dtype=torch.float32)], -1).float()


def normal_from_depth(camera, depth):
    """utils/graphics_utils.normal_from_depth_image (+ depth_pcd2normal, no offset).

    depth [H,W] -> camera-frame normals [H,W,3], zero on the 1-pixel border."""
    xyz = camera_rays(camera)[: depth.shape[0], : depth.shape[1]] * depth[..., None]
    bottom, top = xyz[2:, 1:-1], xyz[:-2, 1:-1]
    right, left = xyz[1:-1, 2:], xyz[1:-1, :-2]
    n = F.normalize(torch.cross(right - left, top - bottom, dim=-1), p=2, dim=-1)
    return F.pad(n.permute(2, 0, 1), (1, 1, 1, 1), mode="constant").permute(1, 2, 0)


def render(params, camera, sh_degree, background, render_geo=True):
    """gaussian_renderer.render. Returns HWC maps (normal is camera-frame, unnormalised)."""
    means = params["means"]
    grad = torch.is_grad_enabled()
    means2D = torch.zeros_like(means, requires_grad=grad) + 0
    means2D_abs = torch.zeros_like(means, requires_grad=grad) + 0
    if grad:
        means2D.retain_grad()
        means2D_abs.retain_grad()
    scales, opac, rots, sh = activated(params)
    settings = raster_settings(camera, background, sh_degree, render_geo)
    if render_geo:
        all_map = plane_inputs(means, scales, rots, camera).contiguous()
    else:
        all_map = torch.empty(0, device=means.device)
    color, radii, observe, out_all, plane_depth = _Rasterize.apply(
        means.contiguous(), means2D, means2D_abs, sh.contiguous(), opac[:, None].contiguous(),
        scales.contiguous(), rots.contiguous(), all_map, settings)
    out = {
        "rgb": color.permute(1, 2, 0),
        "info": {"means2d": means2D, "means2d_abs": means2D_abs, "radii": radii,
                 "observe": observe},
        "camera": camera,
        "geometry": bool(render_geo),
    }
    if render_geo:
        alpha = out_all[3:4].permute(1, 2, 0)
        depth = plane_depth[0]
        out.update({
            "normal": out_all[0:3].permute(1, 2, 0),
            "alpha": alpha,
            "distance": out_all[4:5].permute(1, 2, 0),
            "depth": depth[..., None],
            # render_normal(...) * rendered_alpha.detach()
            "depth_normal": normal_from_depth(camera, depth) * alpha.detach(),
        })
    return out
