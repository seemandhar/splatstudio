"""EDGS: Eliminating Densification for Efficient Convergence of 3DGS (Kotovenko, Grebenkova,
Ommer; arXiv 2504.13204, CVPR 2026). Native SplatStudio port of https://github.com/CompVis/EDGS;
paper fidelity unverified.

Two stages: `ss-edgs-init` (RoMa correspondences -> triangulated `points.npz` + `edgs_init.npz`
in a dataset copy, training views only), then training with `edgs` / `edgs-nosh` on that copy
(`--recipe 3dgs --num-gaussians 10000000`).

- `edgs`: paper-full (Sec. 3.5 pseudo-inverse SH initialization).
- `edgs-nosh`: the released code (SH rest = 0; paper ablation "w/o SH init").
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2504.13204", "https://github.com/CompVis/EDGS")


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_edgs.probes:{fn}", description, cuda)


def _make(name, sh_init, description):
    base = make_method(name)
    renderer, _, photometric = base.components
    density = ComponentSpec(
        "edgs-density",
        "density",
        "ss_edgs.density:create_density",
        (),
        (),
        {**DEFAULTS, "sh_init": sh_init},
    )
    probes = (
        _probe("edgs-roma-matching", "roma_matching",
               "RoMa (pinned commit) loads from ~/models/roma; a 40 px horizontal shift yields "
               "flow dx=-40, dy=0 in the (x, y) NDC convention used for triangulation", True),
        _probe("edgs-triangulation", "triangulation",
               "Sec. 3.3 DLT least squares recovers synthetic points from two views (independent "
               "pinhole projection); tau_proj=0.01 L1-NDC filter rejects inconsistent matches"),
        _probe("edgs-sampling", "sampling",
               "Sec. 3.4 / official: c^i = max_j c^ij, weights 1 above tau_corr=0.05 (raw below), "
               "multinomial share; best neighbour = argmin max(eps_i, eps_j)"),
        _probe("edgs-sh-fit", "sh_fit",
               "Sec. 3.5 composed init (DC = reference colour, constrained ridge residual bands): "
               "degree-3 gsplat CUDA SH reproduces the reference colour exactly and fits the "
               "neighbour residual; |H| <= 2|R|; double-counting composition fails", True),
        _probe("edgs-sh-composed", "sh_composed",
               "Prepared garden init: per-point SH colour toward its observing training cameras "
               "matches the projected image pixels (ref MAE <= 0.01, neighbour <= 0.1); the "
               "double-counting composition fails", True),
        _probe("edgs-no-densify", "no_densify",
               "Official no_densify trainer: sidecar scales/opacity/SH applied, opacity LR 0.025, "
               "means LR = lr(max(step, 8000)), prune opacity<0.005 and opacity x0.99 / 10 steps "
               "only before 15000, never adds Gaussians"),
    )
    contributions = (
        ContributionSpec("dense-matching-prior",
                         "Dense RoMa correspondences from each reference view to its nearest "
                         "training views (Sec. 3.2; preprocessing ss-edgs-init)",
                         ("edgs-roma-matching",)),
        ContributionSpec("splat-triangulation", "One splat per triangulated match (Sec. 3.3)",
                         ("edgs-triangulation",)),
        ContributionSpec("sampling-distribution",
                         "Confidence- and reprojection-consistent sampling (Sec. 3.4)",
                         ("edgs-sampling", "edgs-triangulation")),
        ContributionSpec("densification-free-optimization",
                         "3DGS optimization without densification (official no_densify trainer)",
                         ("edgs-no-densify",)),
    )
    if sh_init:
        contributions += (
            ContributionSpec("sh-initialization",
                             "Pseudo-inverse SH from multi-view colours (Sec. 3.5, supp.)",
                             ("edgs-sh-fit", "edgs-sh-composed", "edgs-no-densify")),
        )
    # The built-in renderer-gradient probe initializes the density on a synthetic scene (no
    # edgs_init.npz) and checks screen-gradient statistics EDGS density never consumes.
    base_probes = tuple(p for p in base.probes if p.name != "renderer-gradient")
    base_contribs = tuple(c for c in base.contributions if c.name != "rasterizer-statistics")
    return replace(
        base,
        description=description,
        components=(renderer, density, photometric),
        probes=base_probes + probes,
        contributions=base_contribs + contributions,
        citations=CITATIONS,
    )


def method():
    return _make("edgs", True,
                 "EDGS (arXiv 2504.13204) native port: RoMa correspondence init (ss-edgs-init) + "
                 "pinv SH init + densification-free 3DGS training; paper fidelity unverified.")


def nosh():
    return _make("edgs-nosh", False,
                 "EDGS (arXiv 2504.13204) native port as released (no SH init): RoMa "
                 "correspondence init + densification-free training; paper fidelity unverified.")
