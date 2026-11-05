"""HAC (Chen et al., ECCV 2024) native SplatStudio port; paper fidelity unverified.

The native Gaussian population is HAC's *anchor* population (Scaffold-GS backbone); the
global learned state (Scaffold MLPs, binary hash grid, context MLP) lives in a module owned by
the density component (renderer reads it), and reaches ss-eval packed into the `hac_blob`
anchor field at the end of training (see renderer.py).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .renderer import BLOB, BLOB_WIDTH

CITATIONS = (
    "https://arxiv.org/abs/2403.14530",
    "https://github.com/YihangChen-ee/HAC",
)

LAMBDA_E = 0.004  # run_shell_mip360.py default (low-rate row of Table 1)

FIELDS = (
    FieldSpec("hac_feat", (DEFAULTS["feat_dim"],), learning_rate=0.0075, initial_value=0.0),
    FieldSpec("hac_offset", (DEFAULTS["n_offsets"], 3), learning_rate=0.01, initial_value=0.0),
    FieldSpec("hac_scaling", (6,), learning_rate=0.007, initial_value=0.0),
    FieldSpec("hac_mask", (DEFAULTS["n_offsets"], 1), learning_rate=0.01, initial_value=1.0),
    # decoder weights packed at the final step (no gradient, never optimised)
    FieldSpec(BLOB, (BLOB_WIDTH,), learning_rate=1e-12, initial_value=0.0),
)


def _probe(name, description, cuda=True):
    return ProbeSpec(name, f"ss_hac.probes:{name.replace('-', '_')}", description, cuda)


def method(lmbda=LAMBDA_E, name="hac"):
    base = make_method(name)
    _, _, photometric = base.components
    dims = {"feat_dim": DEFAULTS["feat_dim"], "n_offsets": DEFAULTS["n_offsets"]}
    renderer = ComponentSpec(
        "hac-renderer", "renderer", "ss_hac.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        ("render.rgb", "render.depth.expected_z", "gradient.screen.signed"), dict(dims))
    density = ComponentSpec("hac-density", "density", "ss_hac.density:create_density",
                            ("gradient.screen.signed",), (), {})
    scaling = ComponentSpec("hac-scaling-reg", "loss", "ss_hac.losses:create_scaling_reg",
                            ("render.rgb",), (), {"weight": 0.01})
    rate = ComponentSpec("hac-rate", "loss", "ss_hac.losses:create_rate", ("render.rgb",), (),
                         {"lmbda": lmbda, "mask_weight": 5e-4, **dims})
    keep = tuple(p for p in base.probes if p.name != "renderer-gradient")
    probes = (
        _probe("hash-grid", "CUDA binary hash grid (3D + three 2D planes) equals an "
               "independent torch multiresolution hash/dense-index trilinear reference, values "
               "and embedding gradients; embeddings are sign(theta) in {-1,+1}"),
        _probe("entropy-model", "Gaussian bit estimate equals -log2(Phi(x+Q/2)-Phi(x-Q/2)) "
               "computed with erf; Q = Q0 (1 + tanh r) in (0, 2 Q0); STE rounding to Q", False),
        _probe("codec-roundtrip", "HAC CUDA arithmetic coder: Gaussian and Bernoulli streams "
               "decode exactly and their sizes are within 2% of the entropy estimate"),
        _probe("neural-gaussians", "Anchor -> neural Gaussian decode equals a per-anchor loop "
               "reference (offsets*scaling[:3], scaling[3:]*sigmoid, masks, opacity>0) and is "
               "view dependent"),
        _probe("visibility-filter", "Anchor prefilter equals the Scaffold fork's visible_filter "
               "(z > 0.2 and non-empty tile rect of the anchor's own EWA footprint) on a "
               "constructed case that a z-only filter fails; renderer uses it"),
        _probe("anchor-density", "Growing adds a new anchor in the voxel of a high-gradient "
               "offset (not in an occupied voxel), pruning removes low-opacity anchors, fields "
               "and Adam moments stay aligned"),
        _probe("training-phases", "Phases at exact boundaries 3000/3001/10000/10001 (noise, "
               "bound update, rate), ~5% rate sample, rate-loss value lambda_e(b + hash/denom) "
               "+ lambda_m mean sigmoid(mask), gradient reach, get_expon_lr_func schedules"),
        _probe("bake-eval", "Encode -> decode -> bake on a fixture: a fresh renderer (no bank, "
               "checkpoint round trip, as in ss-eval) reproduces the baked render exactly; an "
               "un-baked state refuses to render"),
    )
    contributions = (
        ContributionSpec(
            "scaffold-anchors",
            "Backbone (§3.1, Scaffold-GS): anchors with 50-D features, 10 offsets and 6-D "
            "scaling decoded per view by opacity/cov/color MLPs; anchor growing and pruning",
            ("neural-gaussians", "visibility-filter", "anchor-density", "state-transitions"),
        ),
        ContributionSpec(
            "hash-grid-context",
            "C1 (§3.2, Eq. 2): binary hash grid (+ 2D planes) queried at the quantized anchor "
            "position, MLP_c -> per-attribute Gaussian (mu, sigma); L_hash from the +1 ratio",
            ("hash-grid", "entropy-model", "training-phases"),
        ),
        ContributionSpec(
            "adaptive-quantization-entropy",
            "C2 (§3.3, Eq. 3-7): adaptive Q = Q0 (1 + tanh r), uniform-noise training, "
            "Gaussian entropy rate loss lambda_e (+ offset-mask rate lambda_m)",
            ("entropy-model", "training-phases", "photometric-gradient"),
        ),
        ContributionSpec(
            "bitstream-and-decoded-eval",
            "C3 (§3.4): arithmetic-coded bitstream (anchors 16 bit, feat/scaling/offsets, "
            "hash, masks, fp32 MLPs) and evaluation of the decoded model",
            ("codec-roundtrip", "bake-eval"),
        ),
    )
    return replace(
        base,
        description=(
            f"HAC native port (Scaffold-GS anchors + hash-grid context entropy model, "
            f"lambda_e={lmbda}, arithmetic-coded size, decoded-model evaluation); paper "
            "fidelity unverified"
        ),
        components=(renderer, density, photometric, scaling, rate),
        fields=FIELDS,
        probes=keep + probes,
        contributions=base.contributions[:2] + contributions,
        citations=CITATIONS,
    )
