"""gsplat renderer with Mini-Splatting2's visibility culling and half-resolution warm-up.

Official mini-splatting2: `render_imp(..., culling=gaussians._culling[:, view.uid])` skips culled
Gaussians in `preprocessCUDA` (radii = 0, no tiles, no gradient), and
`getTrainCameras_warn_up` trains on 0.5x-resolution cameras while iteration <= 3000.

When the density registered a `LinkTable` for these params (training) and the frame is a
training view, the renderer
- rasterizes only the Gaussians not culled for that view (`outputs["subset_index"]` maps the
  rasterized Gaussians back to the full set; None = no culling), and
- during the warm-up renders at int(W/2) x int(H/2) with the same FoV and provides the matching
  target (`outputs["loss_target"]`, bicubic antialiased, 8-bit like the official PIL images).
Every render forwards the frame's training-resolution target (`outputs["target_image"]`) and its
training-view index, which the density uses for depth-point colours. Without a table (evaluation,
probes) this is the stock gsplat renderer.
"""

import torch
import torch.nn.functional as F
from splatstudio.backends.gsplat import Renderer

from . import core, link

TARGET_CAPABILITY = "training.target_image"
CULLING_CAPABILITY = "render.visibility_culling"
FIELDS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def downsample_target(image, width, height):
    """HWC float target -> int(W/2) x int(H/2), bicubic with antialiasing, quantized to 8 bit."""
    x = image.permute(2, 0, 1)[None]
    y = F.interpolate(x, size=(height, width), mode="bicubic", antialias=True, align_corners=False)
    return (y[0].permute(1, 2, 0).clamp(0, 1) * 255).round() / 255


def half_res_camera(camera):
    w, h = int(camera["width"]), int(camera["height"])
    w2, h2 = core.half_res_size(w, h)
    cam = dict(camera)
    cam["width"], cam["height"] = w2, h2
    cam["K"] = core.scale_intrinsics(camera["K"], w, h, w2, h2)
    if camera.get("image") is not None:
        cam["image"] = downsample_target(camera["image"], w2, h2)
    return cam


class MSv2Renderer(Renderer):
    def __init__(self, spec):
        super().__init__(spec)
        self.capabilities = set(self.capabilities) | {TARGET_CAPABILITY, CULLING_CAPABILITY}

    def render(self, params, camera, sh_degree=3):
        table = link.lookup(params)
        k = None if table is None else table.view_index(camera["viewmat"])
        subset = None
        if k is not None:
            if table.half_res:
                camera = half_res_camera(camera)
            if table.cull is not None:
                if table.cull.shape[0] != len(params["means"]):
                    raise RuntimeError("Culling table is not aligned with the Gaussians")
                col = table.cull[:, k]
                if bool(col.any()):
                    subset = (~col).nonzero().squeeze(1)
        p = params if subset is None else {f: params[f][subset] for f in FIELDS}
        outputs = super().render(p, camera, sh_degree)
        outputs["subset_index"] = subset
        outputs["view_index"] = k
        outputs["target_image"] = camera.get("image")
        outputs["viewmat"] = camera["viewmat"]
        if k is not None and table.half_res:
            outputs["loss_target"] = camera["image"]
        return outputs


def create_renderer(spec):
    return MSv2Renderer(spec)


def full_index(outputs, n, device):
    """Indices into the full Gaussian set of the rasterized Gaussians."""
    idx = outputs.get("subset_index")
    return torch.arange(n, device=device) if idx is None else idx
