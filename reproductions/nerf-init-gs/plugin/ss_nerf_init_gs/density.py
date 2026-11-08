"""3DGS adaptive density control (unchanged) with the paper's point-cloud loading.

The NeRF point cloud "is loaded in exactly the same way as COLMAP point clouds" (Sec. 4.3), i.e.
the official 3DGS initialisation: per-point log sqrt(mean 3-NN squared distance) scales,
identity rotations, opacity 0.1, and 3DGS's camera-extent scene scale (1.1 x max camera distance
from the camera centroid) for the means LR and the grow/prune scale thresholds.

The runtime's `--recipe 3dgs` computes the 3-NN scales with a 4096 x N cdist (16 GB transient at
N = 500k), so the exact scales are precomputed by `ss-nerfinit-prep` (`nerfinit/init.npz`, keyed
by the SHA-256 of points.npz) and applied here; with `--recipe splatstudio` the camera extent is
restored in `set_training_cameras` (no-op under `--recipe 3dgs`, whose scene scale already is the
camera extent). Without a `--data` argument (ss-verify probes on synthetic scenes) the runtime
initialisation is kept and `state["init_scales"] = "runtime"` records it.
"""

import torch
from splatstudio.backends.gsplat import Density

from . import sidecar


class NerfInitDensity(Density):
    def __init__(self, spec):
        super().__init__(spec)
        self.table_override = None  # probes: object with .init() -> (sha, log_scales)

    @torch.no_grad()
    def initialize(self, params, optimizers, scene_scale=1.0):
        tab = self.table_override if self.table_override is not None else sidecar.table()
        source = "runtime"
        if tab is not None:
            sha, log_scales = tab.init()
            if sidecar.points_sha256(params["means"]) != sha:
                raise ValueError("Initial means differ from the nerfinit points.npz (was the point "
                                 "cloud subsampled? pass --num-gaussians >= its size)")
            params["scales"].copy_(log_scales.to(params["scales"])[:, None].expand(-1, 3))
            params["quats"].zero_()
            params["quats"][:, 0] = 1
            params["opacities"].fill_(float(torch.logit(torch.tensor(0.1))))
            source = "nerfinit/init.npz"
        self.optimizers = optimizers
        self.runtime_scene_scale = float(scene_scale)
        super().initialize(params, optimizers, scene_scale)
        self.state["init_scales"] = source
        self.state["runtime_scene_scale"] = self.runtime_scene_scale

    def set_training_cameras(self, cameras):
        centres = torch.stack([torch.linalg.inv(c["viewmat"])[:3, 3] for c in cameras])
        extent = max(1.1 * float(torch.linalg.vector_norm(centres - centres.mean(0), dim=1).max()),
                     0.01)
        group = self.optimizers["means"].param_groups[0]
        group["lr"] = group["lr"] / self.runtime_scene_scale * extent
        self.state["scene_scale"] = extent
        self.state["camera_extent"] = extent


def create_density(spec):
    return NerfInitDensity(spec)
