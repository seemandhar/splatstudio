"""CRiM-GS (Lee et al., arXiv 2407.03923v2) -- native SplatStudio port written from the paper.

No official CRiM-GS code exists; omitted numerics follow the authors' successor CoMoGaussian
(github.com/Jho-Yonsei/CoMoGaussian @cd6d536, MIT). Training renders N = 9 poses along a
per-image continuous camera trajectory (latent neural ODEs for a rigid screw motion and an
adaptive distortion-aware near-SE(3) correction) and blends them with CNN pixel-wise weights and
a scalar mask into the blurry training image; test views are rendered sharp at the input pose.
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .density import REQUIRES as DENSITY_REQUIRES
from .losses import DEFAULTS as LOSS_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2407.03923",
    "https://github.com/Jho-Yonsei/CoMoGaussian",
    "https://arxiv.org/abs/2311.16493",
)


def _probe(name, fn, desc, cuda=False):
    return ProbeSpec(name, f"ss_crim_gs.probes:{fn}", desc, cuda)


def method():
    base = make_method("crim-gs")
    photometric = replace(base.components[2], config={"ssim_weight": 0.3})  # lambda_c = 0.3
    comps = (
        ComponentSpec("crimgs-renderer", "renderer", "ss_crim_gs.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), tuple(sorted(CAPABILITIES)),
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("crimgs-density", "density", "ss_crim_gs.density:create_density",
                      tuple(sorted(DENSITY_REQUIRES)), (), dict(DENSITY_DEFAULTS)),
        photometric,
        ComponentSpec("crimgs-regularizers", "loss", "ss_crim_gs.losses:create_regularizers",
                      ("render.motion_blur",), (), dict(LOSS_DEFAULTS)),
    )
    probes = (
        _probe("dopri5", "dopri5",
               "Own Dormand-Prince 5(4) solver matches matrix_exp on a linear ODE (rel. err "
               "< 1e-4 at rtol 1e-4), fixed-step tableau converges at 5th order, adaptive steps "
               "are taken; a single Euler step fails the same test"),
        _probe("screw-se3", "screw_se3",
               "Eq. 9/10 rotation and G(theta) equal torch.matrix_exp of the twist [S]theta; "
               "decoded axis has unit norm and theta is independent of ||w||"),
        _probe("kernel-poses", "kernel_poses",
               "Trajectory viewmats equal inv(c2w T_r T_a) recomposed independently from the "
               "decoded latents; ODE latent at tau_0 equals E(embed(i)); initial trajectory is "
               "within 5e-3 of the input pose and R_a within 1e-3 of I", True),
        _probe("blur-synthesis", "blur_synthesis",
               "Renderer training output equals an independent combination of independent "
               "gsplat renders: sharp at step <= 1000, mean of N + sharp at 2000, "
               "(1-M) sharp + M sum P I at 4000; the kernel receives gradients; eval = sharp",
               True),
        _probe("objective", "objective",
               "Regulariser = 1e-4 mean_s ||R_a^T R_a - I||_F + 1e-3 mean(M) (independent "
               "formula); photometric uses lambda_c = 0.3"),
        _probe("kernel-schedule", "kernel_schedule",
               "Kernel Adam lr 1e-4 decays linearly by (1e-4 - 1e-6)/40000 per step after "
               "step 1000; no kernel update at or before step 1000", True),
        _probe("density-schedule", "density_schedule",
               "No opacity reset (CoMo recipe) at steps 3000/6000; 3D filter recomputed on the "
               "official steps; no densification at or after step 20000", True),
    )
    contribs = (
        ContributionSpec("continuous-rigid-motion",
                         "Latent neural ODE (dopri5) + screw-axis decoder with independent unit "
                         "axis and angle, SE(3) exponential (Sec. 4.2, Eq. 5-10)",
                         ("dopri5", "screw-se3", "kernel-poses")),
        ContributionSpec("adaptive-distortion-transform",
                         "Second latent ODE decoding R_a = A + I and t_a, composed as "
                         "T = T0 T_r T_a, with orthogonality loss (Sec. 4.3)",
                         ("kernel-poses", "objective")),
        ContributionSpec("pixel-weights-and-mask",
                         "CNN softmax pixel-wise weights over the N renders and BAGS scalar "
                         "mask blending with the sharp render (Sec. 4.4)", ("blur-synthesis",)),
        ContributionSpec("objective-and-schedule",
                         "(1-0.3) L1 + 0.3 D-SSIM + 1e-4 L_o + 1e-3 L_M; trajectory from 1k, "
                         "pixel weights from 3k; kernel lr schedule (App. A, CoMo code)",
                         ("objective", "kernel-schedule", "blur-synthesis")),
        ContributionSpec("mip-splatting-base",
                         "Mip-Splatting 3D filter + 2D Mip filter, CoMo densification without "
                         "opacity reset", ("density-schedule",)),
    )
    base_probes = tuple(p for p in base.probes if p.name in ("state-transitions",
                                                              "photometric-gradient"))
    base_contribs = tuple(c for c in base.contributions if c.name in ("state-lifecycle",
                                                                       "photometric-objective"))
    return replace(
        base,
        description="CRiM-GS native port (latent neural-ODE rigid + adaptive camera trajectory, "
                    "N=9 renders blended by CNN pixel weights and mask, Mip-Splatting base); "
                    "written from the paper with CoMoGaussian code defaults; paper fidelity "
                    "unverified.",
        components=comps, probes=base_probes + probes, contributions=base_contribs + contribs,
        fields=(FieldSpec("filter_3d", (), learning_rate=1e-8, initial_value=0.0),),
        citations=CITATIONS,
    )
