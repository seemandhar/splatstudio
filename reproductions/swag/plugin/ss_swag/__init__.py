"""SWAG (Dahmani et al., arXiv 2403.10427) -- native SplatStudio port written from the paper only
(no official code released; every unspecified choice is an assumption in reproductions/swag/plan.md).

Per-image embeddings, the hash grid and the MLP live in ``ss_swag.bank.ACTIVE`` (owned by
``swag-density``) and are exported to ``density_state["bank"]``. The paper's NeRF-W evaluation is
``python -m ss_swag.evaluate --run RUN``; plain ``ss-eval`` has no bank and renders plain 3DGS.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2403.10427", "https://dahmanihiba.github.io/SWAG/")


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_swag.probes:{fn}", desc, cuda)


def method(name="swag", appearance=True, transient=True, label="full"):
    base = make_method(name)
    photometric = next(c for c in base.components if c.name == "photometric")
    comps = (
        ComponentSpec("swag-renderer", "renderer", "ss_swag.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      {**RENDERER_DEFAULTS, "appearance": appearance, "transient": transient}),
        ComponentSpec("swag-density", "density", "ss_swag.density:create_density",
                      DENSITY_REQUIRES, (), dict(DENSITY_DEFAULTS)),
        photometric,
    )
    probes = [
        _probe("hash-grid", "hash_grid",
               "HashGrid equals an independent per-point loop (dense and hashed levels, "
               "trilinear weights) and is not constant in x"),
        _probe("hash-cuda", "hash_cuda",
               "JIT CUDA hash-encoding kernel equals the PyTorch implementation (output, table "
               "and centre gradients)", True),
        _probe("concrete-opacity", "concrete_opacity",
               "Eq. 9/11 vs independent formula at fixed U; eval U=0.5 closed form; training "
               "samples near-binary at T=0.1"),
        _probe("transient-variance", "transient_variance_probe",
               "Var_I[delta_alpha_tilde] is ~0 for static Gaussians and >0 for image-varying ones"),
        _probe("appearance-mlp", "appearance_mlp",
               "Renderer colours/opacities equal an independent F_theta(c, emb(x), l_I) "
               "computation; l_I changes the render", True),
        _probe("bank-training", "bank_training",
               "A training step updates embeddings, hash table and MLP; means LR follows the 3DGS "
               "30k schedule; densification schedule 2000/30000", True),
        _probe("nerfw-protocol", "nerfw_protocol",
               "Left-half embedding optimisation lowers its loss; right-half metrics ignore the "
               "left half", True),
    ]
    if not transient:
        probes = [p for p in probes if p.name not in ("transient-variance",)]
    contribs = [
        ContributionSpec("appearance-conditioned-colour",
                         "MLP on SH colour + hash-encoded centre + image embedding (Eq. 7)",
                         ("hash-grid", "hash-cuda", "appearance-mlp", "bank-training")),
        ContributionSpec("transient-opacity",
                         "Binary-Concrete image-dependent opacity variation (Eq. 8-11)",
                         ("concrete-opacity", "appearance-mlp")),
        ContributionSpec("transient-decomposition",
                         "Transient Gaussians = nonzero Var over training images",
                         ("transient-variance",)),
        ContributionSpec("nerfw-test-time-appearance",
                         "NeRF-W left-half embedding optimisation (§5.3)", ("nerfw-protocol",)),
    ]
    if not transient:
        contribs = [c for c in contribs if c.name not in ("transient-opacity",
                                                          "transient-decomposition")]
    base_probes = tuple(p for p in base.probes if p.name in ("state-transitions",
                                                             "photometric-gradient"))
    base_contribs = tuple(c for c in base.contributions if c.name in ("state-lifecycle",
                                                                      "photometric-objective"))
    return replace(
        base, name=name,
        description=f"SWAG native port from the paper only [{name}: {label}; appearance "
                    f"{'on' if appearance else 'off'}, transient {'on' if transient else 'off'}]"
                    "; no official code, unspecified choices are assumptions; evaluate with "
                    "ss_swag.evaluate; paper fidelity unverified.",
        components=comps, probes=base_probes + tuple(probes),
        contributions=base_contribs + tuple(contribs), citations=CITATIONS)


def method_a():
    """SWAG-A (Tab. 3): transient opacity variation removed."""
    return method("swag-a", appearance=True, transient=False, label="SWAG-A ablation")


def method_t():
    """SWAG-T (Tab. 3): image-appearance colour dependency removed."""
    return method("swag-t", appearance=False, transient=True, label="SWAG-T ablation")
