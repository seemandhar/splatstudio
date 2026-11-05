"""SparseGS (Xiong et al., arXiv 2312.00206) as a native SplatStudio method.

Paper -> components:
  C1 softmax / mode depth rendering   renderer  sparsegs-renderer (custom CUDA depth pass)
  C2 local + global Pearson loss      loss      sparsegs-pearson
  C3 SDS (Stable Diffusion 2.1-base)  loss      sparsegs-sds
  C4 depth-warping loss               loss      sparsegs-warp (targets built by the density)
  C5 dip-test floater pruning         density   sparsegs-density (Inria density + pruning)
Defaults = the paper's 12-view Mip-NeRF 360 setting (10k iterations); see README.md.
"""

from dataclasses import replace

from splatstudio.methods import make_method
from splatstudio.specs import ComponentSpec, ContributionSpec, ProbeSpec

TOTAL_STEPS = 10000
# Floater pruning step: the paper prunes once after training completes (4_experiments.tex:26,
# 3_methods.tex:157), i.e. at the last step. The official code has no default step.
PRUNE_STEP = TOTAL_STEPS
# TUNED variant only: step 6667 was chosen from a test-split 2k-step smoke (review-1 major); it is
# not supported by the paper or the official README recipe and is labelled "tuned".
TUNED_PRUNE_STEP = round(TOTAL_STEPS * 2 / 3)


def method():
    base = make_method("sparsegs")
    photometric = next(c for c in base.components if c.kind == "loss")
    renderer = ComponentSpec(
        "sparsegs-renderer", "renderer", "ss_sparsegs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        ("render.rgb", "gradient.screen.signed", "render.depth.softmax", "render.depth.alpha",
         "render.depth.mode"),
        {"beta": 5.0, "near_plane": 0.2, "background": 0.0})
    density = ComponentSpec(
        "sparsegs-density", "density", "ss_sparsegs.density:create_density",
        ("gradient.screen.signed",), (),
        {"total_steps": TOTAL_STEPS, "prune_step": PRUNE_STEP, "prune_perc": 0.97,
         "prune_exp": 7.5, "min_opacity": 0.05, "warp_start": 4999, "warp_angles": [-2.5, 2.5],
         "mode_range": "official", "beta": 5.0})
    pearson = ComponentSpec(
        "sparsegs-pearson", "loss", "ss_sparsegs.losses:create_pearson",
        ("render.depth.softmax",), (),
        {"global_weight": 0.15, "local_weight": 0.15, "patch": 64, "p_corr": 0.5})
    warp = ComponentSpec(
        "sparsegs-warp", "loss", "ss_sparsegs.losses:create_warp", ("render.rgb",), (),
        {"weight": 0.05, "start": 4999, "prob": 0.4, "angles": [-2.5, 2.5]})
    sds = ComponentSpec(
        "sparsegs-sds", "loss", "ss_sparsegs.losses:create_sds", ("render.rgb",), (),
        {"weight": 5e-4, "total_steps": TOTAL_STEPS, "start_frac": 2.0 / 3.0, "freq": 0.1,
         "step_ratio": 0.99, "guidance_scale": 100.0})
    probes = (
        ProbeSpec("softmax-depth", "ss_sparsegs.probes:softmax_depth",
                  "CUDA softmax/alpha/mode depth and gradients vs an independent per-pixel "
                  "torch reference; beta->0 limit", True),
        ProbeSpec("pearson", "ss_sparsegs.probes:pearson",
                  "Global/local Pearson loss vs torch.corrcoef, invariances, patch count"),
        ProbeSpec("warp", "ss_sparsegs.probes:warp",
                  "Forward depth warp (identity, known shift), Rodrigues warp cameras, masked "
                  "loss, Pearson skipped on warp iterations"),
        ProbeSpec("sds", "ss_sparsegs.probes:sds",
                  "SDS surrogate gradient w(eps_hat - eps), fixed timestep, elliptic views look "
                  "at the centre"),
        ProbeSpec("floater-prune", "ss_sparsegs.probes:floater_prune",
                  "Dip-test pruning removes an opaque floater in front of a wall; official "
                  "index formula", True),
    )
    contributions = (
        ContributionSpec("softmax-depth", "C1: mode-selection and softmax-scaling depth "
                         "rendering (Eq. 1-5, supp. A)", ("softmax-depth",)),
        ContributionSpec("pearson-depth", "C2: patch-based and global Pearson correlation to "
                         "the monocular depth (Eq. 6)", ("pearson",)),
        ContributionSpec("sds", "C3: SDS loss with Stable Diffusion 2.1 on elliptic-cylinder "
                         "views (Eq. 7-8)", ("sds",)),
        ContributionSpec("depth-warp", "C4: depth-warping loss on rotated pseudo views "
                         "(Eq. 9-10, supp. B)", ("warp",)),
        ContributionSpec("floater-pruning", "C5: dip-test floater pruning (supp. C, Alg. 1)",
                         ("floater-prune",)),
    )
    return replace(
        base,
        description="SparseGS native port (gsplat RGB + official softmax-depth CUDA pass); "
                    "paper fidelity unverified.",
        components=(renderer, density, photometric, pearson, warp, sds),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=("https://arxiv.org/abs/2312.00206", "https://github.com/ForMyCat/SparseGS"),
    )


def method_3dgs():
    """Protocol-matched 3DGS control (paper Table 1/3 "3DGS" row = the SparseGS code with every
    component disabled): same renderer, Inria density (min_opacity 0.05), photometric loss
    only, no warp targets, no floater pruning."""
    full = method()
    comps = []
    for c in full.components:
        if c.name in ("sparsegs-pearson", "sparsegs-warp", "sparsegs-sds"):
            continue
        if c.name == "sparsegs-density":
            c = replace(c, config={**c.config, "warp": False, "prune": False})
        comps.append(c)
    keep = {"state-transitions", "photometric-gradient", "renderer-gradient", "softmax-depth"}
    return replace(
        full, name="sparsegs-3dgs",
        description="SparseGS-protocol 3DGS control (no depth / UVR / pruning); not a paper method.",
        components=tuple(comps),
        probes=tuple(p for p in full.probes if p.name in keep),
        contributions=tuple(c for c in full.contributions
                            if set(c.probes) <= keep and c.probes))


def method_paperprune():
    """`sparsegs` with the floater pruning range the paper describes (first contributor .. mode)
    instead of the official forward.cu indices (thread-rank offset bug)."""
    full = method()
    comps = tuple(replace(c, config={**c.config, "mode_range": "paper"})
                  if c.name == "sparsegs-density" else c for c in full.components)
    return replace(full, name="sparsegs-paperprune", components=comps,
                   description=full.description + " Paper pruning range.")


def method_tuned():
    """TUNED: `sparsegs` with floater pruning at step 6667 instead of after training. The step was
    picked on a test-split smoke run; report it only as a labelled tuned result."""
    full = method()
    comps = tuple(replace(c, config={**c.config, "prune_step": TUNED_PRUNE_STEP})
                  if c.name == "sparsegs-density" else c for c in full.components)
    return replace(full, name="sparsegs-tuned6667", components=comps,
                   description=full.description + " TUNED: prune at 6667 (test-split smoke).")
