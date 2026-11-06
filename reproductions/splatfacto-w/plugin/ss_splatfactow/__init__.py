"""Splatfacto-W (Xu, Kerr, Kanazawa, arXiv 2407.12306) -- native SplatStudio port written fresh
against the paper and the official Nerfstudio code (github.com/KevinXu02/splatfacto-w, Apache-2.0).

Per-image embeddings and the appearance / background MLPs live in ``ss_splatfactow.bank.ACTIVE``
(owned by ``sfw-density``) and are exported to ``density_state["bank"]``. The paper's evaluation
(right half of the NeRF-W test images) is ``python -m ss_splatfactow.evaluate --run RUN``; train
on ``~/datasets/phototourism/sfw/<scene>`` (code protocol, see tools/prepare_sfw.py).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .bank import BANK_DEFAULTS
from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .losses import ALPHA_DEFAULTS, PHOTOMETRIC_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS
from .schedule import LR_TABLE

CITATIONS = ("https://arxiv.org/abs/2407.12306", "https://github.com/KevinXu02/splatfacto-w")


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_splatfactow.probes:{fn}", desc, cuda)


def method(name="splatfacto-w", bg_model=True, robust_mask=True, alpha_loss=True, label="full"):
    if alpha_loss and not bg_model:
        raise ValueError("the alpha loss needs the background model")
    base = make_method(name)
    comps = [
        ComponentSpec("sfw-renderer", "renderer", "ss_splatfactow.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      {**RENDERER_DEFAULTS, "bg_model": bg_model}),
        ComponentSpec("sfw-density", "density", "ss_splatfactow.density:create_density",
                      DENSITY_REQUIRES, (), {**DENSITY_DEFAULTS, "bank_bg_model": bg_model}),
        ComponentSpec("sfw-photometric", "loss", "ss_splatfactow.losses:create_photometric",
                      ("render.rgb",), (), {**PHOTOMETRIC_DEFAULTS, "robust_mask": robust_mask}),
    ]
    if alpha_loss:
        comps.append(ComponentSpec("sfw-alpha", "loss", "ss_splatfactow.losses:create_alpha_loss",
                                   ("render.rgb",), (), dict(ALPHA_DEFAULTS)))
        comps.append(ComponentSpec("sfw-alpha-selected-frac", "loss",
                                   "ss_splatfactow.losses:create_alpha_diagnostic",
                                   ("render.rgb",), (), {}))
    fields = (FieldSpec("appearance_features", (int(BANK_DEFAULTS["appearance_features_dim"]),),
                        LR_TABLE["appearance_features"][0], 0.0),)
    probes = [
        _probe("appearance-colour", "appearance_colour",
               "Rendered Gaussian colours equal an independent MLP(l_j, f_i) + hand-written real SH "
               "(deg <= 3) + 0.5 clamp; l_j changes them; the MLP has no view-direction input", True),
        _probe("splatfacto-schedule", "splatfacto_schedule",
               "Resolution, SH, opacity-reset, densify-pause, cull and LR schedules equal "
               "independent formulas of the official code"),
        _probe("splatfacto-refine", "splatfacto_refine",
               "Refinement on a fixture splits/duplicates/culls exactly the expected Gaussians and "
               "keeps the optimizer state aligned"),
        _probe("ns-normalisation", "ns_normalisation",
               "Nerfstudio orient/centre/scale equals an independent numpy implementation"),
        _probe("masked-view-padding", "masked_view_padding",
               "Left-half crop + zero padding gives the same loss/robust mask as the code's full "
               "image x left-half mask"),
        _probe("nerfw-right-half", "nerfw_right_half",
               "Evaluation metrics depend only on columns >= W//2; test views reuse the embedding "
               "trained on their left half", True),
    ]
    contribs = [
        ContributionSpec("latent-appearance",
                         "Per-Gaussian features + per-image embeddings -> MLP -> SH colours (§4.1)",
                         ("appearance-colour",)),
        ContributionSpec("nerfw-right-half-protocol",
                         "Test images trained on their left half, scored on the right half "
                         "(official code protocol)", ("masked-view-padding", "nerfw-right-half")),
        ContributionSpec("splatfacto-base",
                         "Splatfacto densification, init, resolution and LR schedules",
                         ("splatfacto-schedule", "splatfacto-refine", "ns-normalisation")),
    ]
    if robust_mask:
        probes.append(_probe("robust-mask", "robust_mask",
                             "Robust mask equals an independent implementation; transient block "
                             "masked, upper 40% never masked, percentage interpolated"))
        contribs.append(ContributionSpec("robust-mask", "Transient handling with robust mask (§4.2)",
                                         ("robust-mask",)))
    if bg_model:
        probes.append(_probe("background-model", "background_model",
                             "SH background at infinity: camera-translation invariant, composite "
                             "C + (1-alpha) bg, degree-0 value C0*b0, depends on l_j", True))
        contribs.append(ContributionSpec("background-model",
                                         "SH background predicted from l_j (§4.3)",
                                         ("background-model",)))
    if alpha_loss:
        probes.append(_probe("alpha-loss", "alpha_loss",
                             "L_alpha = 0.15 mean alpha over independently selected background "
                             "pixels (box 3 > 0.6); gradient reaches alpha"))
        contribs.append(ContributionSpec("alpha-loss", "Alpha loss on background pixels (§4.3)",
                                         ("alpha-loss",)))
    base_probes = tuple(p for p in base.probes if p.name == "state-transitions")
    base_contribs = tuple(c for c in base.contributions if c.name == "state-lifecycle")
    return replace(
        base, name=name,
        description=f"Splatfacto-W native port [{name}: {label}] from the paper and the official "
                    "Nerfstudio code; evaluate with ss_splatfactow.evaluate (NeRF-W right half); "
                    "paper fidelity unverified.",
        components=tuple(comps), fields=fields, probes=base_probes + tuple(probes),
        contributions=base_contribs + tuple(contribs), citations=CITATIONS)


def method_a():
    """Splatfacto-W-A (Tab. 1): appearance model only."""
    return method("splatfacto-w-a", bg_model=False, robust_mask=False, alpha_loss=False,
                  label="Splatfacto-W-A: appearance only")


def method_t():
    """Splatfacto-W-T (Tab. 1): appearance model + robust mask."""
    return method("splatfacto-w-t", bg_model=False, robust_mask=True, alpha_loss=False,
                  label="Splatfacto-W-T: appearance + robust mask")
