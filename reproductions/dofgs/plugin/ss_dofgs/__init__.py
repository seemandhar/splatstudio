"""DOF-GS (Wang, Chakravarthula, Chen; CVPR 2025, arXiv 2405.17351) -- native SplatStudio port
of the official code (github.com/rongduo/DOFGS_implementation @cfd635c) with its DOF CUDA
rasterizer vendored unmodified (ss_dofgs/csrc).

Per-view focal distance / aperture and the ILN live in ``ss_dofgs.bank.ACTIVE`` during training
and are exported into the checkpoint's ``density_state["bank"]``. Refocusing / DOF rendering of
any camera: ``ss_dofgs.renderer.render_dof(params, camera, sh_degree, bg, F, K)``.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .losses import MASK_DEFAULTS, OBJECTIVE_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS
from .variants import REG_FACTOR, SCENE_VARIANT, VARIANTS

CITATIONS = (
    "https://arxiv.org/abs/2405.17351",
    "https://github.com/rongduo/DOFGS_implementation",
    "https://yujiewang.info/projects/dof_gs/",
)
LOSS_REQUIRES = ("render.rgb", "render.rgb.all_in_focus", "render.depth.ray_distance",
                 "render.coc", "render.dof_view")


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_dofgs.probes:{fn}", desc, cuda)


def method(name="dofgs"):
    """Official real-scene config groups (variants.py): ``dofgs`` = cake/daisy (band_limit,
    lambda_mk 0.002), ``dofgs-bl-m001`` = cisco/sausage/seal/tools (band_limit, 0.001),
    ``dofgs-nobl`` = caps/cupcake/cups (no 3D filter, plain reset, 0.001)."""
    band_limit, lam = VARIANTS[name]
    base = make_method("dofgs")
    comps = (
        ComponentSpec("dof-raster", "renderer", "ss_dofgs.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      {**RENDERER_DEFAULTS, "band_limit": band_limit}),
        ComponentSpec("dofgs-density", "density", "ss_dofgs.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (),
                      {**DENSITY_DEFAULTS, "band_limit": band_limit}),
        ComponentSpec("dofgs-objective", "loss", "ss_dofgs.losses:create_objective",
                      LOSS_REQUIRES, (), dict(OBJECTIVE_DEFAULTS)),
        ComponentSpec("dofgs-mask", "loss", "ss_dofgs.losses:create_mask_losses",
                      LOSS_REQUIRES, (),
                      {**MASK_DEFAULTS, "lambda_mask_loss": lam, "reg_factor": REG_FACTOR}),
    )
    probes = (
        _probe("dof-kernel-reference", "dof_kernel_reference",
               "Official DOF kernel RGB / ray-distance depth / CoC map equal an independent "
               "float64 dense EWA renderer with Sigma'+aI, a=R^2/(2 ln 4), "
               "R=Q/2|1/z-1/f| (Q>0); removing the CoC term fails", True),
        _probe("pinhole-limit", "pinhole_limit",
               "Q=0 render equals the reference without CoC (standard 3DGS+Mip2D), Q=1e-3 is "
               "continuous with it, and Gaussians on the focal plane z=f stay sharp", True),
        _probe("analytic-blur", "analytic_blur",
               "Single Gaussian: rendered image second moment equals sigma'^2 + 0.1 + "
               "R^2/(2 ln 4) for three apertures", True),
        _probe("fk-gradients", "fk_gradients",
               "Kernel dL/dF, dL/dK and dL/dmeans equal autograd of the float64 reference; "
               "CoC->z gradient disabled (official coc2z_coeff=0)", True),
        _probe("camera-init", "camera_init",
               "f_m/Q_m initialisation (median diopter, tau=15 with 5/95% quantiles) equals an "
               "independent numpy computation; Adam schedule: accumulate <100, update to 15000, "
               "frozen after"),
        _probe("iln-mask-losses", "iln_mask_losses",
               "ILN architecture/PE shapes; Pearson mask loss and entropy reg equal numpy; "
               "composite objective routes AiF/defocus gradients as the two official backwards"),
        _probe("objective-schedule", "objective_schedule",
               "Objective is L_rec(defocused) up to step 5000 and the detail-enhancement "
               "composite afterwards; dropping the AiF branch changes the loss"),
        _probe("density-recipe", "density_recipe",
               "Clone/split/prune(0.005)/extra points (2M IDW)/compactness vs independent "
               "formula/filter reset/xyz LR (max_steps 80000)", True),
        _probe("official-config", "official_config",
               "Variant config (band_limit in renderer and density, lambda_mk, lambda_reg = "
               "0.1 lambda_mk) equals the official config_real_<scene>.yaml of every scene "
               "mapped to it"),
        _probe("band-limit-switch", "band_limit_switch",
               "band_limit=false renders identical to the unfiltered kernel call and resets "
               "opacity to min(sigmoid(o), 0.01) without the filter; band_limit=true differs "
               "(3D filter in render, filter-aware reset)", True),
    )
    contribs = (
        ContributionSpec("dof-rasterizer", "Finite-aperture thin-lens camera with "
                         "differentiable circle-of-confusion convolution in the official CUDA "
                         "rasterizer (Sec. 3.1.2, 3.2, supp. A)",
                         ("dof-kernel-reference", "pinhole-limit", "analytic-blur")),
        ContributionSpec("per-view-camera", "Per-view learnable focal distance and aperture with "
                         "heuristic initialisation (Eq. 7, supp. B.1)",
                         ("fk-gradients", "camera-init")),
        ContributionSpec("coc-iln", "Rendered CoC/depth maps and In-Focus Localization Network "
                         "with L_mk and L_reg (Sec. 3.3)", ("iln-mask-losses",)),
        ContributionSpec("detail-enhancement", "All-in-Focus / defocused composite supervision "
                         "after a 5k warm-up (Sec. 3.3, supp. B.2)", ("objective-schedule",)),
        ContributionSpec("densification-extras", "Extra uniform points at 2k, code-only "
                         "compactness densification, Mip-Splatting 3D filter (supp. B.2, code)",
                         ("density-recipe", "band-limit-switch")),
        ContributionSpec("official-scene-config", "Per-scene official configuration "
                         f"(this variant: band_limit={band_limit}, lambda_mk={lam}; scenes "
                         f"{', '.join(s for s, v in SCENE_VARIANT.items() if v == name)})",
                         ("official-config",)),
    )
    base_probes = tuple(p for p in base.probes if p.name == "state-transitions")
    base_contribs = tuple(c for c in base.contributions if c.name == "state-lifecycle")
    return replace(
        base, name=name,
        description=f"DOF-GS native port [{name}: band_limit={band_limit}, "
                    f"lambda_mk={lam}] (official DOF CUDA rasterizer, per-view focus/aperture, "
                    "ILN detail enhancement, official densification, Mip 3D filter iff "
                    "band_limit); evaluated All-in-Focus; paper fidelity unverified.",
        components=comps, probes=base_probes + probes, contributions=base_contribs + contribs,
        fields=(FieldSpec("filter_3d", (), learning_rate=1e-8, initial_value=0.0),),
        citations=CITATIONS,
    )


def method_bl_m001():
    return method("dofgs-bl-m001")


def method_nobl():
    return method("dofgs-nobl")
