"""Reduced-3DGS (Papantonakis et al., I3D 2024) native SplatStudio port; paper fidelity unverified."""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS

CITATIONS = (
    "https://repo-sam.inria.fr/fungraph/reduced_3dgs/",
    "https://doi.org/10.1145/3651282",
    "https://github.com/graphdeco-inria/reduced-3dgs",
)

# Per-Gaussian SH band count (official `_degrees`). It never receives a gradient (the renderer
# detaches it), so Adam never updates it; the runtime only requires a positive learning rate.
DEGREE = FieldSpec("sh_degree", (), learning_rate=1e-8, initial_value=3.0)


def _probe(name, description, cuda):
    return ProbeSpec(name, f"ss_reduced_3dgs.probes:{name.replace('-', '_')}", description, cuda)


def method():
    base = make_method("reduced-3dgs")
    renderer, baseline_density, photometric = base.components
    renderer = replace(renderer, name="reduced-renderer",
                       factory="ss_reduced_3dgs.renderer:create_renderer")
    density = ComponentSpec(
        "reduced-density",
        "density",
        "ss_reduced_3dgs.density:create_density",
        baseline_density.requires,
        (),
        {**DEFAULTS, "base": dict(baseline_density.config)},
    )
    opacity = ComponentSpec("opacity-l1", "loss", "ss_reduced_3dgs.losses:create_opacity_l1",
                            ("render.rgb",), (), {"weight": 0.001})
    sparsity = ComponentSpec("sh-sparsity", "loss", "ss_reduced_3dgs.losses:create_sh_sparsity",
                             ("render.rgb",), (), {"weight": 0.1, "degree_interval": 1000})
    probes = (
        _probe("opacity-l1-gradient", "Opacity L1: lambda/V sigmoid' on visible Gaussians", False),
        _probe("knn-exact", "CUDA simple-knn boxKnn2 equals brute-force 30-NN", True),
        _probe("redundancy-score", "Pixel footprint, sphere/ellipsoid score, min propagation and "
               "mercy mask vs an O(N^2) reference of the official code", True),
        _probe("sh-sparsity-gradient", "SH sparsity gradient lambda/(45V) sign(f_rest) on visible "
               "Gaussians' active bands (official CUDA backward)", False),
        _probe("transmittance-stats", "CUDA sum_T / blended-pixel counts vs per-pixel torch "
               "compositing on gsplat's projection", True),
        _probe("sh-band-culling", "Synthetic diffuse / band-1 / band-3 / band-3-logit<0 groups get "
               "0/1/3/0 bands with the official logit-opacity quirk, 0/1/3/3 without", True),
        _probe("shcull-variance-alias", "Official aliased-mean variance sum T (c - mean_new)^2 vs "
               "textbook West update, closed forms", False),
        _probe("codec-roundtrip", "K-means (official tie/empty rules), official PLY encode -> "
               "decode, float16 positions, memory_results size", True),
        _probe("schedule", "Mercy / dead-prune / SH-cull / encode steps equal official train.py",
               False),
    )
    contributions = (
        ContributionSpec(
            "redundancy-pruning",
            "C1 (§4.1): every 1000 its up to 27k, resolution-aware redundancy score (pixel "
            "footprint sphere vs 30-NN ellipsoids, min propagation), prune low-opacity half of "
            "candidates above max(mu+sigma, 3) and the 3% lowest opacity (<=0.05); opacity L1 "
            "0.001; dead-point pruning after 15k",
            ("knn-exact", "redundancy-score", "opacity-l1-gradient", "schedule"),
        ),
        ContributionSpec(
            "adaptive-sh-bands",
            "C2 (§4.2): at 15k, transmittance-weighted colour std < 0.04 -> 0 bands, colour "
            "distance to truncated SH < 6*sqrt(3)/255 -> 1 or 2 bands; SH sparsity 0.1; official "
            "statistics quirks (logit opacity, aliased variance) on by default",
            ("transmittance-stats", "sh-band-culling", "shcull-variance-alias",
             "sh-sparsity-gradient", "schedule"),
        ),
        ContributionSpec(
            "codebook-half-float-codec",
            "C3 (§4.3): 256-entry 1-D K-means codebooks for opacity, scale, rotation re/im, DC "
            "and each SH-rest coefficient, float16 positions and codebook entries; the final "
            "parameters are encoded to the official PLY layout and decoded, so evaluation "
            "renders the decoded model; encoded MB per generate_results.py",
            ("codec-roundtrip", "schedule"),
        ),
    )
    return replace(
        base,
        description=(
            "Reduced-3DGS native port (redundancy pruning + adaptive SH bands + codebook/half "
            "codec, decoded-model evaluation); paper fidelity unverified"
        ),
        components=(renderer, density, photometric, opacity, sparsity),
        fields=(DEGREE,),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


SMOKE_DENSITY = {"total_steps": 2000, "densify_until": 1000, "cull_sh_step": 1500,
                 "mercy_interval": 200, "mercy_finetune": 300, "save_steps": [700],
                 "base": {"absgrad": False, "grow_grad2d": 0.0002, "refine_stop_iter": 1000}}


def smoke():
    """Diagnostic 2000-step schedule with every event (not a benchmark configuration)."""
    base = method()
    comps = tuple(replace(c, config={**c.config, **SMOKE_DENSITY}) if c.kind == "density" else c
                  for c in base.components)
    return replace(base, name="reduced-3dgs-smoke", components=comps,
                   description="Reduced-3DGS compressed 2000-step smoke schedule (diagnostic only)")
