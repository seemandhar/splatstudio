"""HGS: Hard Gaussian Splatting (Xu et al., CVPR 2025, arXiv 2412.04826) native SplatStudio
port on a Scaffold-GS backbone; paper fidelity unverified. No official HGS code exists
(github.com/GhiXu/HGS is 404); the backbone follows the official Scaffold-GS code.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .renderer import BLOB, BLOB_WIDTH

CITATIONS = (
    "https://arxiv.org/abs/2412.04826",
    "https://github.com/city-super/Scaffold-GS",
)

FIELDS = (
    FieldSpec("hgs_feat", (DEFAULTS["feat_dim"],), learning_rate=0.0075, initial_value=0.0),
    FieldSpec("hgs_offset", (DEFAULTS["n_offsets"], 3), learning_rate=0.01, initial_value=0.0),
    FieldSpec("hgs_scaling", (6,), learning_rate=0.007, initial_value=0.0),
    FieldSpec(BLOB, (BLOB_WIDTH,), learning_rate=1e-12, initial_value=0.0),
)

VOXEL = {"mipnerf360": 0.001, "tandt": 0.01, "db": 0.005}  # train_{mip360,tnt,db}.sh


def _probe(name, description, cuda=True):
    return ProbeSpec(name, f"ss_hard_gs.probes:{name.replace('-', '_')}", description, cuda)


def _build(name, voxel_size, pghgs=True, rehgs=True):
    base = make_method("gsplat")
    _, _, photometric = base.components
    dims = {"feat_dim": DEFAULTS["feat_dim"], "n_offsets": DEFAULTS["n_offsets"]}
    renderer = ComponentSpec(
        "hgs-renderer", "renderer", "ss_hard_gs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        ("render.rgb", "render.depth.expected_z", "gradient.screen.signed"), dict(dims))
    density = ComponentSpec(
        "hgs-density", "density", "ss_hard_gs.density:create_density",
        ("gradient.screen.signed",), (),
        {"voxel_size": voxel_size,
         "pghgs": {**DEFAULTS["pghgs"], "enabled": pghgs},
         "rehgs": {**DEFAULTS["rehgs"], "enabled": rehgs}})
    scaling = ComponentSpec("hgs-scaling-reg", "loss", "ss_hard_gs.losses:create_scaling_reg",
                            ("render.rgb",), (), {"weight": 0.01})
    keep = tuple(p for p in base.probes if p.name != "renderer-gradient")
    probes = (
        _probe("neural-gaussians", "Anchor -> neural Gaussian decode equals a per-anchor loop "
               "reference (Scaffold-GS: [feat, view dir] MLP input, offsets*scaling[:3], "
               "scaling[3:]*sigmoid, tanh opacity > 0) and is view dependent", False),
        _probe("visibility-filter", "Anchor prefilter equals the Scaffold fork's visible_filter "
               "on constructed cases (behind camera, off-screen, off-screen centre with an "
               "on-screen footprint, inside the near plane)", False),
        _probe("anchor-density", "Scaffold growing puts an anchor in the voxel of a high-average-"
               "gradient offset, prunes the low-opacity anchor, keeps fields, Adam moments and "
               "statistics aligned and every field a trainable leaf"),
        _probe("pghgs-criterion", "Running top-k equals a sort of all per-view gradient norms "
               "(Eq. 5); an offset with average < tau but 3rd-largest >= lambda tau grows only "
               "with PGHGS (Eq. 6); 2 large views (k=3) do not; stats reset per interval"),
        _probe("rehgs-criterion", "CUDA |S_i| equals a brute-force argmax_i w_i reference; "
               "Eq. 7 + Eq. 8 selection on constructed values; growth only when detected in >= 2 "
               "views and REHGS enabled; a real render with the tapped target yields detections"),
        _probe("blob-eval", "A fresh renderer without the live MLP bank (ss-eval) reproduces "
               "the training render exactly from the checkpointed hgs_blob; an empty blob "
               "refuses to render"),
    )
    contributions = (
        ContributionSpec(
            "scaffold-backbone",
            "Base method (§3, Scaffold-GS): anchors with 32-D features, 10 offsets and 6-D "
            "scaling decoded per view by opacity/cov/color MLPs; 3-level anchor growing from "
            "average neural-Gaussian gradients (Eq. 3) and opacity pruning",
            ("neural-gaussians", "visibility-filter", "anchor-density", "blob-eval",
             "state-transitions"),
        ),
    )
    if pghgs:
        contributions += (ContributionSpec(
            "pghgs",
            "C1 (§5.1, Eq. 5-6): positional-gradient driven hard Gaussians: k-th largest "
            "per-view view-space gradient (k=3) >= lambda tau (lambda=1) also triggers growth",
            ("pghgs-criterion",)),)
    if rehgs:
        contributions += (ContributionSpec(
            "rehgs",
            "C2 (§5.2, Eq. 7-8): rendering-error guided hard Gaussians: max-contribution pixel "
            "set |S_i| > 2e-4 |P| and SSIM(I, I^) < 0.7 at the projected centre in >= 2 views "
            "triggers growth",
            ("rehgs-criterion",)),)
    parts = [p for p, on in (("PGHGS", pghgs), ("REHGS", rehgs)) if on]
    return replace(
        base,
        name=name,
        description=(
            f"HGS native port (Scaffold-GS backbone + {' + '.join(parts) or 'no HGS'}, voxel "
            f"{voxel_size}); hard-Gaussian integration into Scaffold growing is a reproduction "
            "choice (no official code); paper fidelity unverified"
        ),
        components=(renderer, density, photometric, scaling),
        fields=FIELDS,
        probes=keep + probes,
        contributions=base.contributions[:2] + contributions,
        citations=CITATIONS,
    )


def method():
    return _build("hard-gs", VOXEL["mipnerf360"])


def method_tandt():
    return _build("hard-gs-tandt", VOXEL["tandt"])


def method_db():
    return _build("hard-gs-db", VOXEL["db"])


def method_scaffold():
    """Ablation / baseline: the same backbone without PGHGS and REHGS (Scaffold-GS)."""
    return _build("scaffold-gs-hgsbase", VOXEL["mipnerf360"], pghgs=False, rehgs=False)
