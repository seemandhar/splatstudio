"""WildGaussians (Kulhanek, Peng, Kukelova, Pollefeys, Sattler; NeurIPS 2024, arXiv 2407.08447)
-- native SplatStudio port of the official code (github.com/jkulhanek/wild-gaussians, MIT).

Per-image appearance embeddings, the appearance MLP and the DINO uncertainty predictor live in
``ss_wildgaussians.bank.ACTIVE`` (owned by ``wg-density``) and are exported to the checkpoint's
``density_state["bank"]``. The paper's evaluation (zero embedding for NeRF On-the-go, NeRF-W
test-time embedding optimisation for Photo Tourism, NerfBaselines metrics) is
``python -m ss_wildgaussians.evaluate --run RUN``; ``ss-eval`` sees no bank and renders the
raw (untoned) colours.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .losses import COLOR_DEFAULTS, UNC_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2407.08447", "https://github.com/jkulhanek/wild-gaussians")

# configs/nerfonthego.yml on top of config.py defaults (30k schedule)
ONTHEGO = {"density": {}, "unc": {"warmup_start": 2000, "warmup_iters": 0, "protected_iters": 500}}
# configs/phototourism.yml
PHOTOTOURISM = {
    "density": {"iterations": 200000, "densify_from_iter": 4000, "densify_until_iter": 100000,
                "densification_interval": 400, "opacity_reset_interval": 15000,
                "position_lr_init": 1.6e-5, "position_lr_final": 1.6e-7, "scaling_lr": 0.0005},
    "unc": {"warmup_start": 35000, "warmup_iters": 5000, "protected_iters": 1500,
            "densify_until_iter": 100000, "opacity_reset_interval": 15000},
}


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_wildgaussians.probes:{fn}", desc, cuda)


def method(name="wildgaussians", appearance=True, schedule=None, label="On-the-go, paper"):
    schedule = schedule or ONTHEGO
    base = make_method(name)
    dens = {**DENSITY_DEFAULTS, **schedule["density"], "bank_appearance_enabled": appearance}
    unc = {**UNC_DEFAULTS, **schedule["unc"]}
    comps = (
        ComponentSpec("wg-renderer", "renderer", "ss_wildgaussians.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      {**RENDERER_DEFAULTS, "appearance": appearance}),
        ComponentSpec("wg-density", "density", "ss_wildgaussians.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (), dens),
        ComponentSpec("wg-uncertainty", "loss", "ss_wildgaussians.losses:create_uncertainty",
                      ("render.rgb",), (), unc),
        ComponentSpec("wg-color", "loss", "ss_wildgaussians.losses:create_color",
                      ("render.rgb", "render.rgb_raw"), (), dict(COLOR_DEFAULTS)),
    )
    fields = [FieldSpec("filter_3d", (), learning_rate=1e-8, initial_value=0.0)]
    if appearance:
        fields.insert(0, FieldSpec("wg_embedding", (24,), learning_rate=0.005,
                                   initial_value=0.0))
    probes = (
        _probe("appearance-toning", "appearance_toning",
               "Toned SH = independent affine formula (0.01 * MLP; offset/C0 on DC, mul on all 16 "
               "coefficients); Fourier embeddings = paper sin/cos(pi p 2^m); a different image "
               "embedding changes the toned colours but not the raw ones"),
        _probe("split-objective", "split_objective",
               "wg-color = 0.8 mean(M|toned-gt|) + 0.2 mean(M(1-SSIM(raw))); SSIM gradient only "
               "reaches the raw image, L1 only the toned image; masked pixels get no gradient"),
        _probe("dino-backbone", "dino_backbone",
               "Own DINOv2 ViT-S/14-reg block-5 normed patch features equal timm's "
               "vit_small_patch14_reg4_dinov2 with the same official checkpoint"),
        _probe("uncertainty-loss", "uncertainty_loss",
               "L_unc = mean(clip(2-2cos,0,1) * down(1/(2 sigma^2))) + 0.5 mean(log sigma) vs an "
               "independent computation; mask = (1/(2 sigma^2) > 1); gradient reaches the head "
               "only; protected window and warm-up schedule"),
        _probe("sky-init", "sky_init",
               "Fibonacci sphere at 10 x q97 radius, official visibility rule; sky/SfM opacity, "
               "colour and sh0 = RGB initialisation"),
        _probe("densify-official", "densify_official",
               "Union of grad >= 2e-4 and abs-grad >= quantile(1 - ratio); clones start with zero "
               "statistics while originals keep theirs (no reset); world-size prune gated"),
        _probe("filter-3d", "filter_3d",
               "3D filter = official formula (W/2 principal point, max focal); filter-aware reset "
               "clamps the filtered opacity to 0.01"),
        _probe("render-two-pass", "render_two_pass",
               "Raw and toned gsplat passes equal independent renders; both passes' screen "
               "gradients enter the density statistics", True),
        _probe("nerfw-protocol", "nerfw_protocol",
               "Test-time embedding optimisation on the left half lowers its loss and moves the "
               "toned render towards the target; right-half metrics ignore the left half", True),
    )
    contribs = (
        ContributionSpec("appearance-modeling", "Per-image/per-Gaussian embeddings + affine "
                         "appearance MLP (Eq. 4-5, Fourier init)",
                         ("appearance-toning", "render-two-pass")),
        ContributionSpec("split-objective", "D-SSIM on the raw, L1 on the toned render (Eq. 6/11)",
                         ("split-objective",)),
        ContributionSpec("test-time-appearance", "Test-time embedding optimisation, NeRF-W "
                         "protocol", ("nerfw-protocol",)),
        ContributionSpec("dino-uncertainty", "DINO-cosine uncertainty predictor (Eq. 8-9)",
                         ("dino-backbone", "uncertainty-loss")),
        ContributionSpec("uncertainty-mask", "Binary uncertainty mask on the colour loss with "
                         "warm-up / protected windows (Eq. 10-11)",
                         ("uncertainty-loss", "split-objective")),
        ContributionSpec("sky-init", "Fibonacci sky-sphere initialisation", ("sky-init",)),
        ContributionSpec("mip-absgrad-base", "Mip-Splatting filters + GOF abs-grad "
                         "densification with official statistics", ("densify-official",
                                                                      "filter-3d")),
    )
    if not appearance:  # released On-the-go config: no appearance state at all
        probes = tuple(p for p in probes if p.name not in ("appearance-toning", "nerfw-protocol"))
        contribs = tuple(c for c in contribs
                         if c.name not in ("appearance-modeling", "test-time-appearance"))
    base_probes = tuple(p for p in base.probes if p.name == "state-transitions")
    base_contribs = tuple(c for c in base.contributions if c.name == "state-lifecycle")
    return replace(
        base, name=name,
        description=f"WildGaussians native port [{name}: {label}, appearance "
                    f"{'on' if appearance else 'off'}] (appearance MLP + embeddings, DINOv2 "
                    "uncertainty mask, Mip-Splatting + abs-grad densification, sky init); "
                    "evaluate with ss_wildgaussians.evaluate; paper fidelity unverified.",
        components=comps, fields=tuple(fields), probes=base_probes + probes,
        contributions=base_contribs + contribs, citations=CITATIONS,
    )


def method_noapp():
    """Released configs/nerfonthego.yml: appearance_enabled false."""
    return method("wildgaussians-noapp", appearance=False, label="On-the-go, released config")


def method_pt():
    """configs/phototourism.yml (200k steps)."""
    return method("wildgaussians-pt", appearance=True, schedule=PHOTOTOURISM,
                  label="Photo Tourism")
