"""EAGLES (Girish et al., ECCV 2024) native SplatStudio port; paper fidelity unverified."""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2312.04564",
    "https://github.com/Sharath-girish/efficientgaussian",
)

# Per-Gaussian latents q_hat (continuous approximations of the integer latents). Their initial
# values and learning rates are set by eagles-density.initialize (decoder inversion, official
# lr / ||decoder scale||); the declared values only satisfy the runtime contract.
FIELDS = (
    FieldSpec("q_shN", (16,), learning_rate=0.037, initial_value=0.0),
    FieldSpec("q_rot", (8,), learning_rate=0.018, initial_value=0.0),
    FieldSpec("q_opa", (1,), learning_rate=0.05, initial_value=0.0),
)


def _probe(name, description, cuda=False):
    return ProbeSpec(name, f"ss_eagles.probes:{name.replace('-', '_')}", description, cuda)


def method():
    base = make_method("eagles")
    renderer, baseline_density, photometric = base.components
    renderer = replace(renderer, name="eagles-renderer",
                       factory="ss_eagles.renderer:create_renderer")
    density = ComponentSpec("eagles-density", "density", "ss_eagles.density:create_density",
                            baseline_density.requires, (), dict(DEFAULTS))
    # keeps the name "photometric" (the base photometric-gradient probe looks it up by name)
    loss = replace(photometric, factory="ss_eagles.losses:create_photometric")
    probes = (
        _probe("decoder-ste", "Decoders equal round(q)S+b / (round(q)F)*s+b with the DCT-II F; "
               "straight-through gradient; rendering uses the decoded attributes"),
        _probe("latent-init-lr", "Latent init D^-1 (identity quaternion, logit 0.1, zero SH), "
               "latent LR = lr/||S||_F (rotation, SH) / lr (opacity), decoder LR 1e-4"),
        _probe("entropy-size", "size() bits vs independent Shannon sum; range-coder roundtrip "
               "exact and within 1% of the entropy"),
        _probe("progressive-schedule", "Cosine 0.3->1 render scale over 70% of training, "
               "render size floor(sH), target F.interpolate with the same factor"),
        _probe("influence-kernel", "CUDA sum_p alpha*T vs per-pixel torch compositing on "
               "gsplat's projection", True),
        _probe("influence-prune", "Prunes exactly W <= 15% quantile and resets W; clones and "
               "split children start at W = 0", True),
        _probe("opacity-reset", "Reset writes D^-1(logit(min(sigmoid(D(q)),0.01))) into the "
               "opacity latent and zeroes its Adam moments"),
        _probe("codec-bake", "Encode -> decode -> bake: a bank-free render of the baked base "
               "fields equals the training-time decoded render", True),
        _probe("schedule", "Refine / opacity-reset / influence-prune / encode steps equal the "
               "official efficient-3dgs.yaml training loop"),
    )
    contributions = (
        ContributionSpec(
            "quantized-latent-attributes",
            "C1 (§3.1, supp. §1): rotation, opacity and SH-rest stored as integer latents "
            "(8/1/16 dims) decoded by learned linear decoders with straight-through rounding; "
            "entropy-coded size (official size()) and decoded-model evaluation",
            ("decoder-ste", "latent-init-lr", "entropy-size", "opacity-reset", "codec-bake"),
        ),
        ContributionSpec(
            "progressive-training",
            "C2 (§3.2): render and target downsampled by a cosine 0.3 -> 1.0 scale over the "
            "first 70% of iterations",
            ("progressive-schedule",),
        ),
        ContributionSpec(
            "influence-pruning",
            "C3 (§3.3, §4.1): densification every 125 steps until 18000, and every 5100 steps "
            "until 27000 prune the 15% Gaussians with the lowest accumulated sum alpha*T",
            ("influence-kernel", "influence-prune", "schedule"),
        ),
    )
    return replace(
        base,
        description=(
            "EAGLES native port (quantized latents + learned decoders + entropy-coded size, "
            "progressive training, influence pruning; decoded-model evaluation); paper "
            "fidelity unverified"
        ),
        components=(renderer, density, loss),
        fields=FIELDS,
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )
