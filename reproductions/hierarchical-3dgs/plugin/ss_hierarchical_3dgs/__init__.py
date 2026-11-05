"""A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets
(Kerbl, Meuleman et al., SIGGRAPH 2024, arXiv 2406.12080) — native SplatStudio port of the
single-chunk path of github.com/graphdeco-inria/hierarchical-3d-gaussians:
train_single.py (30k) -> GaussianHierarchyCreator -> train_post.py (15k) -> render_hierarchy.py,
with the official hierarchy rasterizer and gaussian-hierarchy C++/CUDA vendored unmodified.

`--steps 30000` gives the unoptimized hierarchy ("Ours (tau)"), `--steps 45000` the optimized
one ("Ours opt (tau)"); evaluate granularities with `python -m ss_hierarchical_3dgs.eval_taus`.
Large-scene machinery (chunking, scaffold/skybox, consolidation, streaming), depth and exposure
regularization are not part of this port (see README / report).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from . import hier
from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2406.12080",
             "https://github.com/graphdeco-inria/hierarchical-3d-gaussians")


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_hierarchical_3dgs.probes:{fn}", desc, cuda)


def method(name="hierarchical-3dgs", density_overrides=None):
    base = make_method(name)
    photometric = base.components[2]  # 0.8 L1 + 0.2 D-SSIM, unchanged (train_single/post)
    comps = (
        ComponentSpec("hierarchy-renderer", "renderer",
                      "ss_hierarchical_3dgs.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("hierarchy-density", "density",
                      "ss_hierarchical_3dgs.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (),
                      {**DENSITY_DEFAULTS, **(density_overrides or {})}),
        photometric,
    )
    probes = (
        _probe("h3dgs-hierarchy-build", "hierarchy_build",
               "Vendored creator: every interior node equals the Eq. (merge) w'=o*S weighted "
               "merge of its leaf descendants (mean, covariance, falloff*S, rel < 5 %), "
               "consistent tree/boxes, leaves preserved", True),
        _probe("h3dgs-cut-selection", "cut_selection",
               "expand_to_size returns a valid cut with eps(n) < tau <= eps(parent); "
               "interpolation t and K match an independent evaluation; cut shrinks with tau",
               True),
        _probe("h3dgs-alpha-interpolation", "alpha_interpolation",
               "Kernel alpha' = 1-(1-alpha_p)^(1/K): K copies at t=0 render exactly as the parent "
               "(K = 2, 3), and differ without the correction", True),
        _probe("h3dgs-hierarchy-optimization", "hierarchy_optimization",
               "After the build: log-uniform random granularity in [0.005, 0.1), gradients "
               "through the cut interpolation, anchors frozen, interior nodes optimized", True),
        _probe("h3dgs-leaf-training", "leaf_training",
               "train_single.py: max screen-gradient statistic, grad*radius*sigma^0.2 >= 0.015 "
               "selection, split /1.6, opacity-only prune, reset, big-Gaussian shrink, "
               "sparse Adam, position lr 2e-5 -> 2e-7"),
    )
    contribs = (
        ContributionSpec("hierarchy-generation",
                         "BVH + KL-minimizing merge with w'=o*S, falloff, orientation matching, "
                         "compaction (Sec. 3.1, 4.2) via the official creator",
                         ("h3dgs-hierarchy-build",)),
        ContributionSpec("cut-selection-and-switching",
                         "Granularity cut and smooth parent-child interpolation incl. alpha' "
                         "(Sec. 3.2, Eq. t_interp, alphap)",
                         ("h3dgs-cut-selection", "h3dgs-alpha-interpolation")),
        ContributionSpec("hierarchy-optimization",
                         "Optimize interior nodes at random granularities, full resolution, "
                         "leaves frozen (Sec. 4.1)", ("h3dgs-hierarchy-optimization",)),
        ContributionSpec("chunk-training",
                         "Per-chunk 3DGS changes: max-gradient densification and the official "
                         "train_single schedule (Sec. 5.2; depth/exposure not ported)",
                         ("h3dgs-leaf-training",)),
    )
    # Non-learned storage of the tree (hier.py). The runtime requires lr > 0; the density
    # component replaces its optimizer with lr 0 and the renderer never reads it differentiably.
    fields = (FieldSpec(hier.FIELD, (hier.WIDTH,), 1e-30, 0.0),)
    return replace(
        base,
        description="Hierarchical 3DGS (SIGGRAPH 2024) single-chunk native port with the "
                    "official hierarchy rasterizer and creator; paper fidelity unverified.",
        components=comps,
        probes=base.probes + probes,
        contributions=base.contributions + contribs,
        fields=fields,
        citations=CITATIONS,
    )


def quick():
    """Pipeline smoke variant (not a paper configuration): hierarchy built after 1500 steps."""
    return method("hierarchical-3dgs-quick",
                  {"leaves_steps": 1500, "densify_until": 1200, "position_lr_max_steps": 1500})
