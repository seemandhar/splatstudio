"""Octree-GS (Ren et al., TPAMI 2025; arXiv 2403.17898) native SplatStudio port, Our-Scaffold-GS variant; paper
fidelity unverified.

The native Gaussian population is Octree-GS's *anchor* population (Scaffold-GS backbone) with per-anchor octree
level and LOD bias fields. The global learned state (Scaffold MLPs) and the octree scalars live in a bank owned by
the density component (the renderer reads it) and reach ss-eval packed into the `ogs_blob` anchor field. The
anchor / bank / blob design is adapted from the HAC and ContextGS native ports.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .renderer import BLOB, BLOB_WIDTH

CITATIONS = (
    "https://arxiv.org/abs/2403.17898",
    "https://github.com/city-super/Octree-GS",
)

FIELDS = (
    FieldSpec("ogs_feat", (DEFAULTS["feat_dim"],), learning_rate=0.0075, initial_value=0.0),
    FieldSpec("ogs_offset", (DEFAULTS["n_offsets"], 3), learning_rate=0.01, initial_value=0.0),
    FieldSpec("ogs_scaling", (6,), learning_rate=0.007, initial_value=0.0),
    # octree level L_j and LOD bias Delta L_j: set by the density, no gradient
    FieldSpec("ogs_level", (1,), learning_rate=1e-12, initial_value=0.0),
    FieldSpec("ogs_extra_level", (1,), learning_rate=1e-12, initial_value=0.0),
    # MLP weights + octree scalars, re-packed after every step (no gradient, never optimised)
    FieldSpec(BLOB, (BLOB_WIDTH,), learning_rate=1e-12, initial_value=0.0),
)


def _probe(name, description, cuda=True):
    return ProbeSpec(name, f"ss_octree_gs.probes:{name.replace('-', '_')}", description, cuda)


def method(name="octree-gs"):
    base = make_method(name)
    _, _, photometric = base.components
    dims = {"feat_dim": DEFAULTS["feat_dim"], "n_offsets": DEFAULTS["n_offsets"], "opacity_cap": 0.999}
    renderer = ComponentSpec(
        "ogs-renderer", "renderer", "ss_octree_gs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        ("render.rgb", "render.depth.expected_z", "gradient.screen.signed"), dims)
    density = ComponentSpec("ogs-density", "density", "ss_octree_gs.density:create_density",
                            ("gradient.screen.signed",), (), {})
    scaling = ComponentSpec("ogs-scaling-reg", "loss", "ss_octree_gs.losses:create_scaling_reg",
                            ("render.rgb",), (), {"weight": 0.01})
    keep = tuple(p for p in base.probes if p.name != "renderer-gradient")
    probes = (
        _probe("octree-init", "K = round(log2(d_max/d_min)) + 1 and standard distance from an independent "
               "two-stage quantile computation; per-level voxel centres equal a brute-force voxelization; the "
               "adaptive visible threshold equals the mean view frequency and weed_out keeps exactly the anchors "
               "above it; init_level = floor(K/2); vendored simple-knn equals exact 3-NN on a fixture"),
        _probe("lod-selection", "Anchor mask equals an independent per-anchor loop of Eq. 3 (code 'round' "
               "variant, half-voxel shifted distance); raising Delta L selects finer anchors; the progressive "
               "limit applies only to training renders", False),
        _probe("progressive-training", "Coarse intervals are geometric with ratio 1/1.5 and sum to 10000; "
               "a training render at step 1 uses no level above init_level, after 10000 all K levels"),
        _probe("neural-gaussians", "Anchor -> neural Gaussian decode equals a per-anchor loop reference "
               "(input [f, view dir] without distance, offsets*scaling[:3], scaling[3:]*sigmoid, opacity>0) and "
               "is view dependent"),
        _probe("visibility-filter", "Anchor prefilter equals the Scaffold fork's visible_filter (z > 0.2 and "
               "non-empty tile rect) on a constructed case that a z-only filter fails"),
        _probe("anchor-growing", "A mid-gradient offset grows a same-level anchor at its voxel with the parent "
               "feature, a high-gradient offset grows a level+1 anchor at the finer voxel with zero feature, Delta L "
               "+= 0.01 per exceeded level threshold, never-visible candidates are weeded out, the low-opacity "
               "anchor is pruned, fields and Adam moments stay aligned"),
        _probe("blob-eval", "A fresh renderer (no bank, checkpoint round trip, as in ss-eval) reproduces the "
               "in-process no-grad render exactly; the blob restores the trained MLPs and octree scalars; an "
               "empty blob is refused"),
    )
    contributions = (
        ContributionSpec(
            "scaffold-anchors",
            "Backbone (Our-Scaffold-GS): anchors with 32-D features, 10 offsets and 6-D scaling decoded per view "
            "by opacity/cov/color MLPs; Scaffold statistics, opacity pruning, volume regulariser",
            ("neural-gaussians", "visibility-filter", "state-transitions", "photometric-gradient"),
        ),
        ContributionSpec(
            "lod-anchors",
            "C1 (Sec. 3.1.1-3.1.2): K octree levels from the observed distance range; anchors voxelized per "
            "level at delta/2^L; view-frequency weed-out at initialisation",
            ("octree-init",),
        ),
        ContributionSpec(
            "lod-selection",
            "C2 (Sec. 3.1.3, Eq. 3): per-view cumulative LOD selection L_j <= round(log2(d_max/d) + Delta L)",
            ("lod-selection", "blob-eval"),
        ),
        ContributionSpec(
            "lod-anchor-control",
            "C3 (Sec. 3.2): next-level growing with tau_g 2^(beta L) thresholds, LOD-bias increase, opacity "
            "pruning and view-frequency filtering of new anchors",
            ("anchor-growing",),
        ),
        ContributionSpec(
            "progressive-training",
            "C4 (Sec. 3.3): coarse-to-fine level activation from floor(K/2) with geometric intervals (omega 1.5, "
            "10k iterations)",
            ("progressive-training",),
        ),
    )
    return replace(
        base,
        description=("Octree-GS native port (Our-Scaffold-GS: octree LOD anchors, view-dependent LOD selection, "
                     "LOD-aware growing, progressive training); paper fidelity unverified"),
        components=(renderer, density, photometric, scaling),
        fields=FIELDS,
        probes=keep + probes,
        contributions=base.contributions[:2] + contributions,
        citations=CITATIONS,
    )
