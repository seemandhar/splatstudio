"""ContextGS (Wang et al., NeurIPS 2024) native SplatStudio port; paper fidelity unverified.

The native Gaussian population is ContextGS's *anchor* population (Scaffold-GS backbone). The global learned state
(Scaffold MLPs, factorized hyperprior, per-level context MLPs) lives in a module owned by the density component
(the renderer reads it) and reaches ss-eval packed into the `cgs_blob` anchor field at the end of training.
Infrastructure shared with HAC is copied from the HAC native port (reproductions/hac/plugin), with attribution.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .renderer import BLOB, BLOB_WIDTH

CITATIONS = (
    "https://arxiv.org/abs/2405.20721",
    "https://github.com/wyf0912/ContextGS",
)

LAMBDA_E = 0.004  # low-rate rows of Table 1 / App. tables (lambda = 0.004)

HYPER_DIM = DEFAULTS["feat_dim"] // DEFAULTS["hyper_divisor"]

FIELDS = (
    FieldSpec("cgs_feat", (DEFAULTS["feat_dim"],), learning_rate=0.0075, initial_value=0.0),
    FieldSpec("cgs_offset", (DEFAULTS["n_offsets"], 3), learning_rate=0.01, initial_value=0.0),
    FieldSpec("cgs_scaling", (6,), learning_rate=0.007, initial_value=0.0),
    FieldSpec("cgs_mask", (DEFAULTS["n_offsets"], 1), learning_rate=0.01, initial_value=1.0),
    FieldSpec("cgs_hyper", (HYPER_DIM,), learning_rate=0.0075, initial_value=0.0),
    # render-MLP weights packed at the final step (no gradient, never optimised)
    FieldSpec(BLOB, (BLOB_WIDTH,), learning_rate=1e-12, initial_value=0.0),
)


def _probe(name, description, cuda=True):
    return ProbeSpec(name, f"ss_contextgs.probes:{name.replace('-', '_')}", description, cuda)


def method(lmbda=LAMBDA_E, name="contextgs"):
    base = make_method(name)
    _, _, photometric = base.components
    dims = {"feat_dim": DEFAULTS["feat_dim"], "n_offsets": DEFAULTS["n_offsets"]}
    renderer = ComponentSpec(
        "cgs-renderer", "renderer", "ss_contextgs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        ("render.rgb", "render.depth.expected_z", "gradient.screen.signed"), dict(dims))
    density = ComponentSpec("cgs-density", "density", "ss_contextgs.density:create_density",
                            ("gradient.screen.signed",), (), {})
    scaling = ComponentSpec("cgs-scaling-reg", "loss", "ss_contextgs.losses:create_scaling_reg",
                            ("render.rgb",), (), {"weight": 0.01})
    rate = ComponentSpec("cgs-rate", "loss", "ss_contextgs.losses:create_rate", ("render.rgb",), (),
                         {"lmbda": lmbda, "mask_weight": 5e-4})
    keep = tuple(p for p in base.probes if p.name != "renderer-gradient")
    probes = (
        _probe("entropy-model", "Gaussian bit estimate equals -log2(Phi(x+Q/2)-Phi(x-Q/2)) with erf; "
               "Q = Q0 (1 + tanh r) in (0, 2 Q0); STE rounding to Q with the +-15000 Q clamp", False),
        _probe("factorized-prior", "Hyperprior density equals an independent re-derivation of compressai's "
               "EntropyBottleneck (filters 3,3,3,3, init_scale 10): parameter count, initial values, "
               "cumulative-logit likelihood, noise / dequantize modes, 1e-9 likelihood bound", False),
        _probe("level-partition", "Levels are disjoint and cover all anchors, each coarse representative is "
               "the minimum index of its voxel (brute force), level ratios hit tau=0.2 with the "
               "binary-searched scales, masked anchors move to the origin before level 1", False),
        _probe("context-model", "Autoregressive order: level-k predictions do not change when finer-level "
               "attributes change and do change with coarser ones; context rows equal an independent "
               "gather of the official order; removing the hyperprior changes the predictions"),
        _probe("codec-roundtrip", "CUDA arithmetic coder: Gaussian, Bernoulli and factorized-prior streams "
               "decode exactly and their sizes are within 2% of the entropy estimate"),
        _probe("neural-gaussians", "Anchor -> neural Gaussian decode equals a per-anchor loop reference "
               "(offsets*scaling[:3], scaling[3:]*sigmoid, masks, opacity>0) and is view dependent"),
        _probe("visibility-filter", "Anchor prefilter equals the Scaffold fork's visible_filter (z > 0.2 and "
               "non-empty tile rect of the anchor's own EWA footprint) on a constructed case that a z-only "
               "filter fails; renderer uses it"),
        _probe("anchor-density", "Growing adds a new anchor in the voxel of a high-gradient offset with "
               "scatter-max feature and hyper latent, pruning removes low-opacity anchors, fields and Adam "
               "moments stay aligned"),
        _probe("training-phases", "Phases at exact boundaries 3000/3001/10000/10001 (noise, bound update, "
               "context rate), ~15% rate sample, rate-loss value lambda_e*bpp + lambda_m*mean sigmoid(mask), "
               "gradient reaches feat, hyper, mlp_grid and the prior, get_expon_lr_func schedules"),
        _probe("bake-eval", "Encode -> decode -> bake on a fixture: lossless, a fresh renderer (no bank, "
               "checkpoint round trip, as in ss-eval) reproduces the baked render exactly; an un-baked state "
               "refuses to render"),
    )
    contributions = (
        ContributionSpec(
            "scaffold-anchors",
            "Backbone (§3, Scaffold-GS + HAC-style quantization): anchors with 50-D features, 10 offsets and "
            "6-D scaling decoded per view by opacity/cov/color MLPs; masks; anchor growing and pruning",
            ("neural-gaussians", "visibility-filter", "anchor-density", "state-transitions"),
        ),
        ContributionSpec(
            "anchor-partitioning",
            "C1 (§4.1, Eq. 4-6): 3 disjoint anchor levels from voxelization at binary-searched coarser "
            "voxel sizes (tau = 0.2), minimum-index representatives, anchor forwarding",
            ("level-partition",),
        ),
        ContributionSpec(
            "anchor-level-context",
            "C2 (§4.2, Eq. 7-8): per-level MLPs predict (mu, sigma, Q) of f, l, O from the already coded "
            "coarser-level anchor; coding order level 2 -> 1 -> 0",
            ("context-model", "entropy-model", "training-phases"),
        ),
        ContributionSpec(
            "hyperprior",
            "C3 (§4.2, Eq. 9): per-anchor 12-D latent z coded with a factorized density, z-hat appended to "
            "every level's prior",
            ("factorized-prior", "context-model", "training-phases"),
        ),
        ContributionSpec(
            "rate-objective-bitstream",
            "C4 (§4.3, Eq. 10-11, App. A): lambda_e * bit_per_param + lambda_m L_m; arithmetic-coded "
            "bitstream (anchors 16 bit, hyper, feat/scaling/offsets per level, masks, fp32 MLPs) and "
            "evaluation of the decoded model",
            ("codec-roundtrip", "bake-eval", "training-phases", "photometric-gradient"),
        ),
    )
    return replace(
        base,
        description=(
            f"ContextGS native port (Scaffold-GS anchors + anchor-level context + hyperprior entropy model, "
            f"lambda_e={lmbda}, arithmetic-coded size, decoded-model evaluation); paper fidelity unverified"
        ),
        components=(renderer, density, photometric, scaling, rate),
        fields=FIELDS,
        probes=keep + probes,
        contributions=base.contributions[:2] + contributions,
        citations=CITATIONS,
    )
