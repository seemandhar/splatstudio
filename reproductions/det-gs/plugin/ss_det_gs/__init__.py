"""DET-GS (Huang, Xu, Perry 2025, arXiv 2508.04099) as a native SplatStudio method.

3DGS + hierarchical monocular-depth supervision (C1), edge-aware depth smoothness (C2) and an
RGB-guided edge-preserving TV loss (C3). No official code exists; unspecified numerics follow
DNGaussian's official code (see README.md / reproductions/det-gs/paper.md).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .losses import DEPTH_DEFAULTS, EDGE_DEFAULTS, TV_DEFAULTS


def method():
    base = make_method("det-gs")
    comps = []
    for c in base.components:
        if c.kind == "renderer":
            c = replace(c, name="det-gs-renderer", factory="ss_det_gs.renderer:create_renderer")
        comps.append(c)
    comps += [
        ComponentSpec("det-gs-depth", "loss", "ss_det_gs.losses:create_depth",
                      ("state.gaussian3d", "camera.pinhole"), (), dict(DEPTH_DEFAULTS)),
        ComponentSpec("det-gs-edge", "loss", "ss_det_gs.losses:create_edge",
                      ("render.depth.expected_z",), (), dict(EDGE_DEFAULTS)),
        ComponentSpec("det-gs-tv", "loss", "ss_det_gs.losses:create_tv", ("render.rgb",), (),
                      dict(TV_DEFAULTS)),
    ]
    probes = (
        ProbeSpec("depth-formula", "ss_det_gs.probes:depth_formula",
                  "Eq. 5-7 patch/image normalised error-tolerant L2 vs an independent loop; affine invariance"),
        ProbeSpec("depth-schedule", "ss_det_gs.probes:depth_schedule",
                  "Hard-depth loss fires only every 5th step and its gradient reaches only the centres",
                  True),
        ProbeSpec("edge-formula", "ss_det_gs.probes:edge_formula",
                  "Eq. 10-11 masked cross-neighbourhood smoothness vs a per-pixel loop; edges excluded"),
        ProbeSpec("tv-formula", "ss_det_gs.probes:tv_formula",
                  "Eq. 13-15 RGB-guided hinge TV vs a per-pixel loop; GT edges and sub-margin gradients free"),
    )
    contributions = (
        ContributionSpec("hierarchical-depth",
                         "Hierarchical geometric depth supervision with Depth Anything V2 priors (C1)",
                         ("depth-formula", "depth-schedule")),
        ContributionSpec("edge-aware-depth", "Canny-masked edge-aware depth regularization (C2)",
                         ("edge-formula",)),
        ContributionSpec("rgb-guided-tv", "RGB-guided edge-preserving total variation (C3)",
                         ("tv-formula",)),
    )
    return replace(
        base,
        description="DET-GS native port (no official code; unspecified numerics from DNGaussian); "
        "paper fidelity unverified",
        components=tuple(comps),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=("https://arxiv.org/abs/2508.04099", "https://arxiv.org/abs/2403.06912"),
    )
