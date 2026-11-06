"""BAGS (Peng et al., ECCV 2024, arXiv 2403.04926) -- native SplatStudio port of the official
code (github.com/snldmt/BAGS @ed04144, built on Mip-Splatting).

Blur Proposal Network state lives in ``ss_bags.bank.ACTIVE`` during training (exported into
``density_state["bank"]`` of the checkpoint). Test / validation renders are always the sharp
image C at full resolution: the BPN only shapes the training loss.
Official Deblur-NeRF metrics (8-bit PNG, PSNR, skimage SSIM, LPIPS-Alex):
``python -m ss_bags.evaluate --run RUN``.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .losses import DEPTH_TV_DEFAULTS, MASK_DEFAULTS, OBJECTIVE_DEFAULTS, RGB_TV_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS
from .variants import GENERIC, OFFICIAL, SMOKE, scene_config

CITATIONS = (
    "https://arxiv.org/abs/2403.04926",
    "https://github.com/snldmt/BAGS",
    "https://nwang43jhu.github.io/BAGS/",
)
LOSS_REQUIRES = ("render.rgb", "render.multiscale", "render.blur_view")


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_bags.probes:{fn}", desc, cuda)


def _build(name, v, label):
    up = list(v.get("upsample_steps", RENDERER_DEFAULTS["upsample_steps"]))
    density_cfg = {**DENSITY_DEFAULTS, "init_dgt": v["init_dgt"], "total_steps": v["iterations"],
                   "ms_steps": v["ms_steps"], "use_another_mlp": v["use_another_mlp"],
                   "bpn_lr": v["bpn_lr"], "min_opacity": v.get("min_opacity", 0.1),
                   "upsample_steps": up,
                   "bpn_start": v.get("bpn_start", DENSITY_DEFAULTS["bpn_start"])}
    base = make_method("gsplat")
    comps = (
        ComponentSpec("bags-raster", "renderer", "ss_bags.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      {**RENDERER_DEFAULTS, "upsample_steps": up}),
        ComponentSpec("bags-density", "density", "ss_bags.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (), density_cfg),
        ComponentSpec("bags-objective", "loss", "ss_bags.losses:create_objective",
                      LOSS_REQUIRES, (), dict(OBJECTIVE_DEFAULTS)),
        ComponentSpec("bags-mask", "loss", "ss_bags.losses:create_mask_loss",
                      LOSS_REQUIRES, (), dict(MASK_DEFAULTS)),
        ComponentSpec("bags-depth-tv", "loss", "ss_bags.losses:create_depth_tv",
                      LOSS_REQUIRES + ("render.depth.ray_distance",), (),
                      {**DEPTH_TV_DEFAULTS, "use_depth_loss": v["use_depth_loss"],
                       "depth_loss_alpha": v["depth_loss_alpha"]}),
        ComponentSpec("bags-rgb-tv", "loss", "ss_bags.losses:create_rgb_tv",
                      LOSS_REQUIRES, (),
                      {**RGB_TV_DEFAULTS, "use_rgbtv_loss": v["use_rgbtv_loss"]}),
    )
    probes = (
        _probe("spatial-conv-reference", "spatial_conv_reference",
               "Unfold spatially varying KxK convolution (K=5,9,17) equals an independent "
               "float64 shift-and-add loop in value and in autograd gradients w.r.t. image and "
               "kernels; impulse/offset orientation, delta kernel = identity, gradcheck"),
        _probe("bpn-architecture", "bpn_architecture",
               "BPN layer shapes (80->64 base, heads 25/81/289, no bias), softmax kernels sum "
               "to 1, sigmoid mask, 2-D PE equals an independent formula, input detached, "
               "mask blend C_out = m C~ + (1-m) C"),
        _probe("objective-terms", "objective_terms",
               "Objective uses C_out when the BPN is active and C otherwise (it <= 250 of each "
               "phase); mask / depth-TV / rgb-TV values equal independent numpy; target resized "
               "to the render size; removing the BPN changes the loss"),
        _probe("multiscale-schedule", "multiscale_schedule",
               "Training render 150x100/K5 before 3000, 300x200/K9 before 6000, 600x400/K17 "
               "after (scaled intrinsics vs explicit-K render); test views sharp full-res; "
               "Euclidean-distance depth equals an independent render", True),
        _probe("phase-schedule", "phase_schedule",
               "Official loop clocks: xyz LR restart at ms_steps, init_dgt -> 2e-4, densify / "
               "reset / filter step sets, BPN off for it <= 250 in both phases, final-step "
               "skip, single-scale-net switch"),
        _probe("density-recipe", "density_recipe",
               "Clone/split/prune(min_opacity) vs independent formula, NDC gradient scaling, "
               "filter-aware reset, 3D filter from the current-scale cameras", True),
        _probe("official-config", "official_config",
               "Method config equals the official scripts/<scene>.sh values"),
    )
    contribs = (
        ContributionSpec("blur-proposal-network",
                         "Per-view, per-pixel dense KxK kernels + residual quality mask from "
                         "view embedding, positional encoding and RGB-D CNN features applied to "
                         "the rendered image during training (Sec. 3.1, Eq. 1-3)",
                         ("spatial-conv-reference", "bpn-architecture", "objective-terms")),
        ContributionSpec("coarse-to-fine",
                         "Multi-scale BPN (K 5/9/17 heads) with 1/4 -> 1/2 -> full resolution "
                         "training and the official phase restart (Sec. 3.2, Eq. 4-5, code)",
                         ("multiscale-schedule", "phase-schedule")),
        ContributionSpec("official-scene-config",
                         f"Mip-Splatting base + official densification; config: {label}",
                         ("density-recipe", "official-config")),
    )
    base_probes = tuple(p for p in base.probes if p.name == "state-transitions")
    base_contribs = tuple(c for c in base.contributions if c.name == "state-lifecycle")
    return replace(
        base, name=name,
        description=f"BAGS native port [{label}] (Mip-Splatting gsplat render, multi-scale Blur "
                    "Proposal Network on training views, coarse-to-fine schedule, official "
                    "densification); test renders sharp; paper fidelity unverified.",
        components=comps, probes=base_probes + probes, contributions=base_contribs + contribs,
        fields=(FieldSpec("filter_3d", (), learning_rate=1e-8, initial_value=0.0),),
        citations=CITATIONS,
    )


def method():
    """Official argument defaults (60k iterations, ms_steps 6000, init_dgt 6e-4, min_opacity
    0.005); use the per-scene methods ``bags-<scene>`` for the paper tables."""
    return _build("bags", GENERIC, "official arguments defaults")


def method_smoke():
    return _build("bags-smoke", SMOKE, "compressed smoke-test schedule (not a paper config)")


def _scene_method(scene):
    def factory():
        v = scene_config(scene)
        return _build(f"bags-{scene}", v, f"official scripts/{'blur' if v['dataset'] == 'motion' else 'defocus'}{scene}.sh")

    factory.__name__ = f"method_{scene}"
    return factory


for _s in OFFICIAL:
    globals()[f"method_{_s}"] = _scene_method(_s)


def _nobpn_method(scene):
    """Control: the official scene config with the BPN never activated (bpn_start =
    total_steps) = Mip-Splatting under the BAGS coarse-to-fine schedule and densification;
    the paper's comparison row (Mip-Splatting) and the BPN ablation."""

    def factory():
        v = {**scene_config(scene)}
        v["bpn_start"] = v["iterations"]
        return _build(f"bags-{scene}-nobpn", v, f"{scene} script with the BPN disabled "
                      "(control, not a paper method)")

    factory.__name__ = f"method_{scene}_nobpn"
    return factory


NOBPN_SCENES = ("ball", "caps")
for _s in NOBPN_SCENES:
    globals()[f"method_{_s}_nobpn"] = _nobpn_method(_s)
