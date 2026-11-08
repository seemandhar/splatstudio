"""Denoising-GS: Gaussian Splatting with Spatial-aware Denoising (arXiv 2605.14880).

Native SplatStudio port written from the paper's LaTeX (no official code; the Appendix with
implementation details is not in the arXiv source). Paper fidelity unverified.

Methods:
  denoising-gs        3DGS-MCMC backbone + momentum drift (C1) + spatial gradient denoising (C2)
                      + Fisher-uncertainty pruning (C3) + kNN-sparsity refinement (C4).
  denoising-gs-mcmc   control: the same MCMC backbone/budget with C1-C4 disabled.
  denoising-gs-quick  smoke-test variant: C3 at step 1500, C4 at 1600..2000 (total_steps 2000).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .losses import DEFAULTS as REG_DEFAULTS

CITATIONS = ("https://arxiv.org/abs/2605.14880", "https://arxiv.org/abs/2404.09591",
             "https://arxiv.org/abs/2311.17874", "https://arxiv.org/abs/2406.10219")


def _probe(name, fn, description, cuda=False):
    return ProbeSpec(name, f"ss_denoising_gs.probes:{fn}", description, cuda)


PROBES = (
    _probe("momentum-drift", "momentum_drift",
           "Mean displacement over 3 steps equals -alpha*lr*m with m an independent lagged EMA "
           "(beta1) of the hooked mean gradients; alpha=0 ablation differs"),
    _probe("spatial-denoise", "spatial_denoise",
           "Delta-mu equals an independent per-Gaussian numpy loop (quaternion -> R, dominant "
           "axes, |dS_j| sign(dM^G_j)); analytic R=I aligned / misaligned cases; density applies "
           "beta2 R Delta-mu"),
    _probe("fisher-prune", "fisher_prune",
           "u_i equals sum of singular values of an explicit 6x6 sum_v g g^T from per-view "
           "autograd of the L2 loss; prune removes exactly the 10 % lowest u with aligned "
           "optimizer state; opacity ranking ablation differs", True),
    _probe("scr-densify", "scr_densify",
           "d_i equals brute-force mean squared 3-NN distance; the sparsest 0.05 % are split; "
           "source/copy opacity and scale follow hand-written Eqs. p6/p7 with N=2", True),
    _probe("split-finite", "split_finite",
           "Regression: SCR split of primitives with opacity down to 1e-9, then 4 full training "
           "steps (render, Adam, C1/C2) keep all fields finite; N=2 closed form equals gsplat "
           "compute_relocation where well conditioned", True),
    _probe("schedule", "schedule_probe",
           "Prune at total-5000, SCR five times every 500 after it, MCMC refine window "
           "500-25000; cap from the paper's MCMC row for the scene"),
)

CONTRIBUTIONS = (
    ContributionSpec("momentum-exploration",
                     "C1 momentum-biased stochastic exploration (Sec. 4.2, Eqs. m2/msgld)",
                     ("momentum-drift",)),
    ContributionSpec("spatial-gradient-denoising",
                     "C2 local-frame mean/scale gradient denoising term (Sec. 4.3, Eqs. m3-m5)",
                     ("spatial-denoise",)),
    ContributionSpec("uncertainty-denoising",
                     "C3 Fisher-information uncertainty pruning of 10 % (Sec. 4.4, Eqs. f2-f3)",
                     ("fisher-prune", "schedule")),
    ContributionSpec("spatial-coherence-refinement",
                     "C4 kNN-sparsity MCMC split of the sparsest 0.05 %, 5 times (Sec. 4.5)",
                     ("scr-densify", "split-finite", "schedule")),
)


def _build(name, density_cfg, description, contributions=CONTRIBUTIONS, probes=PROBES):
    base = make_method(name)
    density = ComponentSpec("denoising-density", "density",
                            "ss_denoising_gs.density:create_density", (), (),
                            {**DEFAULTS, **density_cfg})
    reg = ComponentSpec("mcmc-reg", "loss", "ss_denoising_gs.losses:create_reg",
                        ("render.rgb",), (), dict(REG_DEFAULTS))
    return replace(
        base,
        description=description,
        components=(base.components[0], density, base.components[2], reg),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


def method():
    return _build("denoising-gs", {},
                  "Denoising-GS (arXiv 2605.14880) native port on a gsplat 3DGS-MCMC backbone; "
                  "paper fidelity unverified")


def mcmc_control():
    off = {"momentum": False, "spatial_denoise": False, "fisher_prune": False, "scr": False}
    return _build("denoising-gs-mcmc", off,
                  "3DGS-MCMC control for Denoising-GS: same backbone, per-scene budget, reg and "
                  "init, with all four Denoising-GS contributions disabled",
                  contributions=(), probes=(PROBES[5],))


def quick():
    cfg = {"total_steps": 2000, "prune_offset": 500, "densify_every": 100}
    return _build("denoising-gs-quick", cfg,
                  "Denoising-GS with the prune/SCR schedule compressed into 2000 steps "
                  "(smoke test only)")
