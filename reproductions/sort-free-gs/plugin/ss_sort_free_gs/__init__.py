"""Sort-free Gaussian Splatting via Weighted Sum Rendering (Hou et al., ICLR 2025,
arXiv 2410.18931). Native SplatStudio port; paper fidelity unverified.

No official code. The unofficial re-implementation github.com/LiYukeee/sort-free-gs (linked by the
first author) was read for unstated learning rates and initial values. The WSR CUDA rasterizer
(csrc/wsr_raster.cu) is adapted from gsplat 1.5.3 RasterizeToPixels3DGS{Fwd,Bwd}.cu (Apache-2.0).

Methods
- `sort-free-gs`: LC-WSR (Eq. 7 + Eq. 10) + view-dependent opacity (Eq. 11), 3DGS loss and
  densification without opacity pruning (the variant of the paper's per-scene tables).
- `sort-free-gs-dir`: ablation DIR-WSR (Eq. 8; depth-independent weight).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .renderer import CAPABILITIES

CITATIONS = (
    "https://arxiv.org/abs/2410.18931",
    "https://github.com/LiYukeee/sort-free-gs",
    "https://github.com/nerfstudio-project/gsplat/tree/v1.5.3",
)

FIELDS = (
    # H_i of Eq. 11 (u = 0.1 + Y(dir; H)); lrs follow the unofficial code (opacity_lr, /20)
    FieldSpec("opa_sh0", (1,), 0.05, 0.0),
    FieldSpec("opa_shN", (15,), 0.0025, 0.0),
    # global sigma (Eq. 10, init 10, Sec. 3.4) and w_B (Eq. 7, init 0.02 unofficial), stored as
    # identical per-Gaussian rows; their lrs are set by the density component
    FieldSpec("wsr_sigma", (), 0.1, 10.0),
    FieldSpec("wsr_wbg", (), 0.01, 0.02),
)

# 3DGS densification unchanged (gsplat defaults = 3DGS), opacity pruning removed (prune_opa 0),
# no opacity reset; WSR-specific learning rates (unofficial code arguments/__init__.py)
DENSITY = {"grow_grad2d": 0.0002, "prune_opa": 0.0, "opacity_reset": False, "vi_lr": 0.01,
           "sigma_lr_init": 0.1, "sigma_lr_final": 0.001, "sigma_lr_max_steps": 30000,
           "wbg_lr": 0.01, "log_every": 1000}


def _probe(name, fn, description, cuda=True):
    return ProbeSpec(name, f"ss_sort_free_gs.probes:{fn}", description, cuda)


PROBES = (
    _probe("wsr-forward-vs-dense", "wsr_forward_vs_dense",
           "CUDA weighted sums equal a dense float64 reference, are invariant to Gaussian order, "
           "and differ from sorted alpha compositing on an occluding fixture"),
    _probe("wsr-backward-vs-dense", "wsr_backward_vs_dense",
           "CUDA gradients (means2d, conics, features, u, w) equal autograd of the dense "
           "reference"),
    _probe("lc-weight", "lc_weight",
           "Per-Gaussian weight equals max(0, 1 - z/sigma) v (zero beyond sigma); sigma and w_B "
           "receive gradients (DIR: weight independent of sigma)"),
    _probe("view-dependent-opacity", "view_dependent_opacity",
           "u equals 0.1 + SH(view dir; H) (checked against gsplat SH), changes with the "
           "viewpoint and is not clamped to [0,1]"),
    _probe("recipe", "recipe",
           "No opacity prune/reset, v_i lr 0.01, sigma lr 0.1->0.001 x scale, global rows and "
           "Adam moments synchronised after cloning; global param equals a scalar Adam across N changes", False),
)


def _make(name, weight, description):
    base = make_method(name)
    renderer = ComponentSpec(
        "wsr-renderer", "renderer", "ss_sort_free_gs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"), CAPABILITIES,
        {"weight": weight, "u_offset": 0.1, "cull_nonpositive": True, "den_eps": 1e-4,
         "clamp_output": True, "tile_size": 16, "eps2d": 0.3,
         "background": 0.0})
    density = replace(base.components[1], name="wsr-density",
                      factory="ss_sort_free_gs.density:create_density", config=dict(DENSITY))
    contributions = (
        ContributionSpec("weighted-sum-rendering",
                         "Order-independent normalised weighted sum with learnable background "
                         "weight replaces sorted alpha compositing; custom CUDA (Sec. 3.2, Eq. 7)",
                         ("wsr-forward-vs-dense", "wsr-backward-vs-dense")),
        ContributionSpec("depth-weight",
                         ("LC-WSR w = max(0, 1 - d/sigma) v_i with learnable sigma, v_i (Eq. 10)"
                          if weight == "lc" else "DIR-WSR w = 1 (Eq. 8, ablation)"),
                         ("lc-weight",)),
        ContributionSpec("view-dependent-opacity",
                         "Unclamped opacity u_i from per-Gaussian SH of the view direction "
                         "(Sec. 3.3, Eq. 11)", ("view-dependent-opacity",)),
        ContributionSpec("training-recipe",
                         "3DGS loss and densification, opacity pruning removed (Sec. 3.4, App.)",
                         ("recipe",)),
    )
    return replace(base, description=description, components=(renderer, density)
                   + base.components[2:], fields=FIELDS, probes=base.probes + PROBES,
                   contributions=base.contributions + contributions, citations=CITATIONS)


def sort_free_gs():
    return _make("sort-free-gs", "lc",
                 "Sort-free GS via Weighted Sum Rendering (arXiv 2410.18931), LC-WSR + "
                 "view-dependent opacity; native port, paper fidelity unverified.")


def sort_free_gs_dir():
    return _make("sort-free-gs-dir", "dir",
                 "Sort-free GS ablation DIR-WSR (arXiv 2410.18931 Table 4); native port, paper "
                 "fidelity unverified.")


method = sort_free_gs
