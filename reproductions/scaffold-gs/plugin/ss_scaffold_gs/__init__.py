"""Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering (Lu et al., CVPR 2024, arXiv 2312.00109)
native SplatStudio port; paper fidelity unverified.

Backbone extracted from the HGS native plugin (reproductions/hard-gs/plugin/ss_hard_gs, its `scaffold-gs-hgsbase`
control), which follows the official code github.com/city-super/Scaffold-GS @59c833b line by line.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .renderer import BLOB, BLOB_WIDTH

CITATIONS = (
    "https://arxiv.org/abs/2312.00109",
    "https://github.com/city-super/Scaffold-GS",
)

FIELDS = (
    FieldSpec("sgs_feat", (DEFAULTS["feat_dim"],), learning_rate=0.0075, initial_value=0.0),
    FieldSpec("sgs_offset", (DEFAULTS["n_offsets"], 3), learning_rate=0.01, initial_value=0.0),
    FieldSpec("sgs_scaling", (6,), learning_rate=0.007, initial_value=0.0),
    FieldSpec(BLOB, (BLOB_WIDTH,), learning_rate=1e-12, initial_value=0.0),
)

VOXEL = {"mipnerf360": 0.001, "tandt": 0.01, "db": 0.005}  # train_{mip360,tnt,db}.sh


def _probe(name, description, cuda=True):
    return ProbeSpec(name, f"ss_scaffold_gs.probes:{name.replace('-', '_')}", description, cuda)


def _build(name, voxel_size):
    base = make_method("gsplat")
    _, _, photometric = base.components
    dims = {"feat_dim": DEFAULTS["feat_dim"], "n_offsets": DEFAULTS["n_offsets"]}
    renderer = ComponentSpec(
        "scaffold-renderer", "renderer", "ss_scaffold_gs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        ("render.rgb", "render.depth.expected_z", "gradient.screen.signed"), {**dims, "opacity_cap": 0.999})
    density = ComponentSpec(
        "scaffold-density", "density", "ss_scaffold_gs.density:create_density",
        ("gradient.screen.signed",), (), {"voxel_size": voxel_size})
    scaling = ComponentSpec("scaffold-scaling-reg", "loss", "ss_scaffold_gs.losses:create_scaling_reg",
                            ("render.rgb",), (), {"weight": 0.01})
    keep = tuple(p for p in base.probes if p.name != "renderer-gradient")
    probes = (
        _probe("anchor-init", "Anchors are the de-duplicated voxel centres round(P/eps)*eps of the SfM cloud (Eq. 2) "
               "with log scaling = log sqrt(mean 3-NN sq. distance) (simple-knn CUDA vs brute force)"),
        _probe("neural-gaussians", "Anchor -> neural Gaussian decode equals a per-anchor loop reference "
               "([feat, view dir] MLP input, offsets*scaling[:3], scaling[3:]*sigmoid, tanh opacity > 0) and is "
               "view dependent", False),
        _probe("visibility-filter", "Anchor prefilter equals the Scaffold fork's visible_filter on constructed "
               "cases (behind camera, off-screen, off-screen centre with an on-screen footprint, near plane)", False),
        _probe("opacity-selection", "Only neural Gaussians with opacity > 0 are rasterized and the opacity cap is "
               "straight-through (capped forward value, unit gradient)"),
        _probe("anchor-density", "Growing puts an anchor in the voxel of a high-average-gradient offset (Eq. 7), "
               "not for a low-gradient one; pruning removes the low-opacity anchor; fields, Adam moments and "
               "statistics stay aligned and every field a trainable leaf"),
        _probe("blob-eval", "A fresh renderer without the live MLP bank (ss-eval) reproduces the training render "
               "exactly from the checkpointed sgs_blob; an empty blob refuses to render"),
    )
    contributions = (
        ContributionSpec("anchor-init", "C1 (§3.2.1, Eq. 2): sparse voxel anchors from SfM points with 32-D "
                         "feature, 6-D scaling and k=10 learnable offsets", ("anchor-init",)),
        ContributionSpec("neural-gaussians", "C2 (§3.2.2, Eq. 4-6): per-view on-the-fly decoding of k neural "
                         "Gaussians per visible anchor by opacity/cov/color MLPs from [feature, view direction]; "
                         "frustum prefilter and opacity selection", ("neural-gaussians", "visibility-filter",
                                                                     "opacity-selection", "blob-eval")),
        ContributionSpec("anchor-refinement", "C3 (§3.3, Eq. 7): 3-level voxel anchor growing from average "
                         "neural-Gaussian gradients with random elimination, accumulated-opacity pruning",
                         ("anchor-density",)),
    )
    return replace(
        base,
        name=name,
        description=(f"Scaffold-GS native port (voxel {voxel_size}, official released config: no feature bank, "
                     "no distance input, no appearance embedding); paper fidelity unverified"),
        components=(renderer, density, photometric, scaling),
        fields=FIELDS,
        probes=keep + probes,
        contributions=base.contributions[:2] + contributions,
        citations=CITATIONS,
    )


def method():
    return _build("scaffold-gs", VOXEL["mipnerf360"])


def method_tandt():
    return _build("scaffold-gs-tandt", VOXEL["tandt"])


def method_db():
    return _build("scaffold-gs-db", VOXEL["db"])
