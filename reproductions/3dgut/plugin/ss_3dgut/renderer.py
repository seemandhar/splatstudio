"""3DGUT renderer (arXiv 2412.12507 §4.1-4.2) on gsplat 1.5.3's UT / eval3d CUDA kernels.

gsplat's `rasterization(with_ut=True, with_eval3d=True)` hard-wires gsplat's default
`UnscentedTransformParameters` (alpha=0.1, require_all_sigma_points_valid=True). The paper and
the official 3dgrut config (`configs/render/3dgut.yaml`) use alpha=1, beta=2, kappa=0 and accept a
projection when any sigma point is valid, so this module reproduces the thin Python glue of
`gsplat.rendering.rasterization` (non-packed, single camera) and passes the paper's parameters.

Pipeline:
  1. UT projection (Eq. 5-9): 7 sigma points through the exact camera model -> 2D mean/cov; used
     only for tile assignment / bounding (not differentiable, as in the paper).
  2. SH colour per Gaussian from the camera centre (official per-particle features).
  3. eval3d rasterisation (Eq. 10): alpha = sigma * exp(-0.5 |o_g x d_g|^2) at the max-response
     point of each pixel's ray, gradients flow to means/quats/scales/opacity/colour directly.
"""

import math

import torch
from gsplat.cuda._wrapper import (
    FThetaCameraDistortionParameters,
    RollingShutterType,
    UnscentedTransformParameters,
    fully_fused_projection_with_ut,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels_eval3d,
    spherical_harmonics,
)

CAPABILITIES = frozenset(
    {"render.rgb", "render.depth.expected_z", "gradient.world.means", "camera.distorted.ut"}
)

# Paper §4.4 / official configs/render/3dgut.yaml.
DEFAULTS = {
    "ut_alpha": 1.0,
    "ut_beta": 2.0,
    "ut_kappa": 0.0,
    "ut_in_image_margin_factor": 0.1,
    "ut_require_all_sigma_points_valid": False,
    "eps2d": 0.3,
    "near_plane": 0.01,
    "far_plane": 1e10,
    "tile_size": 16,
    "background": [1.0, 1.0, 1.0],  # runtime convention (official code: black)
}

CAMERA_MODELS = ("pinhole", "fisheye", "ftheta")
SHUTTERS = {
    "global": RollingShutterType.GLOBAL,
    "top_to_bottom": RollingShutterType.ROLLING_TOP_TO_BOTTOM,
    "left_to_right": RollingShutterType.ROLLING_LEFT_TO_RIGHT,
    "bottom_to_top": RollingShutterType.ROLLING_BOTTOM_TO_TOP,
    "right_to_left": RollingShutterType.ROLLING_RIGHT_TO_LEFT,
}


def ut_parameters(config):
    return UnscentedTransformParameters(
        alpha=float(config["ut_alpha"]),
        beta=float(config["ut_beta"]),
        kappa=float(config["ut_kappa"]),
        in_image_margin_factor=float(config["ut_in_image_margin_factor"]),
        require_all_sigma_points_valid=bool(config["ut_require_all_sigma_points_valid"]),
    )


def camera_kwargs(camera):
    """Optional distortion / rolling-shutter keys of a camera dict -> gsplat keyword arguments.

    Keys: camera_model ('pinhole' | 'fisheye' | 'ftheta'), radial_coeffs ([6] OpenCV k1..k6 or
    [4] fisheye k1..k4), tangential_coeffs ([2] p1,p2), thin_prism_coeffs ([4] s1..s4),
    ftheta_coeffs (FThetaCameraDistortionParameters), rolling_shutter (see SHUTTERS) with
    viewmat_rs (end-of-exposure world-to-camera). Absent keys -> perfect pinhole, global shutter.
    """
    model = camera.get("camera_model", "pinhole")
    if model not in CAMERA_MODELS:
        raise ValueError(f"Unsupported camera_model {model!r}; expected one of {CAMERA_MODELS}")
    kwargs = {"camera_model": model}
    for key in ("radial_coeffs", "tangential_coeffs", "thin_prism_coeffs"):
        value = camera.get(key)
        if value is not None:
            kwargs[key] = torch.as_tensor(value, dtype=torch.float32, device=camera["K"].device)[
                None
            ]
    if model == "fisheye" and "radial_coeffs" in kwargs and kwargs["radial_coeffs"].shape[-1] != 4:
        raise ValueError("fisheye radial_coeffs must have 4 entries (k1..k4)")
    if model == "ftheta":
        coeffs = camera.get("ftheta_coeffs")
        if not isinstance(coeffs, FThetaCameraDistortionParameters):
            raise ValueError("ftheta cameras need FThetaCameraDistortionParameters")
        kwargs["ftheta_coeffs"] = coeffs
    shutter = camera.get("rolling_shutter", "global")
    if shutter not in SHUTTERS:
        raise ValueError(f"Unknown rolling_shutter {shutter!r}")
    kwargs["rolling_shutter"] = SHUTTERS[shutter]
    if shutter != "global":
        kwargs["viewmats_rs"] = camera["viewmat_rs"][None]
    return kwargs


def render_ut(
    means, quats, scales, opacities, sh, camera, sh_degree, config, ut_params, background=None
):
    """Single-camera 3DGUT render. Returns (rgb+depth [H,W,4], alpha [H,W,1], info)."""
    width, height = int(camera["width"]), int(camera["height"])
    viewmats, Ks = camera["viewmat"][None], camera["K"][None]
    cam = camera_kwargs(camera)
    radii, means2d, depths, conics, _ = fully_fused_projection_with_ut(
        means, quats, scales, opacities, viewmats, Ks, width, height,
        eps2d=config["eps2d"], near_plane=config["near_plane"], far_plane=config["far_plane"],
        calc_compensations=False, ut_params=ut_params, **cam,
    )  # radii [1,N,2]; means2d [1,N,2]; depths [1,N] (camera z at mid-shutter pose)
    campos = torch.linalg.inv(viewmats)[:, :3, 3]  # [1,3]
    if "viewmats_rs" in cam:
        campos = 0.5 * (campos + torch.linalg.inv(cam["viewmats_rs"])[:, :3, 3])
    dirs = means[None] - campos[:, None]
    visible = (radii > 0).all(dim=-1)
    colors = spherical_harmonics(sh_degree, dirs, sh[None].expand(1, *sh.shape), masks=visible)
    colors = torch.clamp_min(colors + 0.5, 0.0)
    colors = torch.cat((colors, depths[..., None]), dim=-1)  # RGB + camera-z channel
    tile = int(config["tile_size"])
    tw, th = math.ceil(width / tile), math.ceil(height / tile)
    _, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, tile, tw, th)
    offsets = isect_offset_encode(isect_ids, 1, tw, th).reshape(1, th, tw)
    bg = None
    if background is not None:
        bg = torch.cat((background, background.new_zeros(1)))[None]
    rendered, alphas = rasterize_to_pixels_eval3d(
        means, quats, scales, colors, opacities[None], viewmats, Ks, width, height, tile,
        offsets, flatten_ids, backgrounds=bg, ut_params=ut_params, **cam,
    )
    info = {
        "radii": radii, "means2d": means2d, "depths": depths, "conics": conics,
        "campos": campos[0], "width": width, "height": height, "n_cameras": 1,
    }
    return rendered[0], alphas[0], info


class Renderer:
    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown 3DGUT renderer options: {sorted(unknown)}")
        self.config = {**DEFAULTS, **spec.config}
        self.ut_params = ut_parameters(self.config)
        self.capabilities = set(CAPABILITIES)

    def render(self, params, camera, sh_degree=3):
        device = params["means"].device
        background = torch.tensor(self.config["background"], dtype=torch.float32, device=device)
        rendered, alpha, info = render_ut(
            params["means"], params["quats"], params["scales"].exp(),
            params["opacities"].sigmoid(), torch.cat((params["sh0"], params["shN"]), dim=1),
            camera, sh_degree, self.config, self.ut_params, background=background,
        )
        # Expected depth (gsplat "ED"): accumulated z / accumulated alpha.
        depth = rendered[..., 3:4] / alpha.clamp_min(1e-10)
        return {"rgb": rendered[..., :3], "depth": depth, "alpha": alpha, "info": info}


def create_renderer(spec):
    return Renderer(spec)
