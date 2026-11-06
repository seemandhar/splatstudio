"""GS-IR: 3D Gaussian Splatting for Inverse Rendering (Liang, Zhang, Feng, Shan, Jia; CVPR 2024,
arXiv 2311.16473) — native SplatStudio port written against the paper and the official code
(github.com/lzhnb/GS-IR @ e5a030b).

One ``ss-train`` run executes the three stages: initial stage (3DGS + depth-derived normal
supervision of per-Gaussian normals), occlusion baking at ``pbr_start`` and the decomposition
stage (albedo / roughness / metallic, cube-map light, irradiance volumes, split-sum PBR).
Rendered RGB after ``pbr_start`` (and therefore ``ss-eval``) is the PBR image.

``gs-ir``        Mip-NeRF 360 recipe of the official README: 30000 + 10000 steps (--steps 40000).
``gs-ir-smoke``  identical pipeline, decomposition from step 1500 (smoke tests <= 2000 steps).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS as DENSITY_DEFAULTS
from .losses import COLOR_DEFAULTS, LIGHT_DEFAULTS, MATERIAL_DEFAULTS, NORMAL_DEFAULTS
from .renderer import CAPABILITIES
from .renderer import DEFAULTS as RENDERER_DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2311.16473",
    "https://github.com/lzhnb/GS-IR",
)

# official create_from_pcd: normal (0,0,1) (set by the density), albedo/roughness/metallic raw 1;
# all four use opacity_lr 0.05 (gaussian_model.py training_setup)
FIELDS = (
    FieldSpec("gsir_normal", (3,), learning_rate=0.05, initial_value=0.0),
    FieldSpec("gsir_albedo", (3,), learning_rate=0.05, initial_value=1.0),
    FieldSpec("gsir_roughness", (1,), learning_rate=0.05, initial_value=1.0),
    FieldSpec("gsir_metallic", (1,), learning_rate=0.05, initial_value=1.0),
)


def _probe(name, fn, desc, cuda=True):
    return ProbeSpec(name, f"ss_gsir.probes:{fn}", desc, cuda)


def _loss(name, fn, requires, defaults):
    return ComponentSpec(name, "loss", f"ss_gsir.losses:{fn}", requires, (), dict(defaults))


def _build(name, density_overrides, description):
    base = make_method("gsplat")
    dcfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DENSITY_DEFAULTS.items()}
    dcfg.update(density_overrides)
    comps = (
        ComponentSpec("gsir-renderer", "renderer", "ss_gsir.renderer:create_renderer",
                      ("state.gaussian3d", "camera.pinhole"), CAPABILITIES,
                      dict(RENDERER_DEFAULTS)),
        ComponentSpec("gsir-density", "density", "ss_gsir.density:create_density",
                      ("gradient.screen.signed",), (), dcfg),
        _loss("photometric", "create_color", ("render.rgb",), COLOR_DEFAULTS),
        _loss("gsir-normal", "create_normal",
              ("render.normal.world", "render.normal_from_depth.world"), NORMAL_DEFAULTS),
        _loss("gsir-material", "create_material", ("render.pbr.gbuffer",), MATERIAL_DEFAULTS),
        _loss("gsir-light", "create_light", ("render.pbr.gbuffer",), LIGHT_DEFAULTS),
    )
    probes = (
        _probe("depth-linear-interp", "depth_linear_interp",
               "Semi-transparent 2-Gaussian fixture: rendered depth = sum(w d)/sum(w) (Eq. 7), "
               "inside [d_min, d_max], != volumetric accumulation sum(w d)"),
        _probe("normal-from-depth", "normal_from_depth_probe",
               "Analytic tilted plane: depth-derived world normal = plane normal; L_n-p/TV "
               "gradients reach gsir_normal and never means/scales/opacities"),
        _probe("occlusion-baking", "occlusion_baking",
               "Opaque wall 0.25 from a probe: baked SH visibility ~0 toward the wall, ~1 away; "
               "SH projection of constant visibility = 4 pi Y00; masked trilinear vs loops; "
               "dilation fills the grid"),
        _probe("cube-ops", "cube_ops",
               "Texel lookup round trip and face-rotation consistency, edge continuity, "
               "diffuse / GGX pre-filter of a constant map = constant, filter gradients"),
        _probe("pbr-shading", "pbr_shading_probe",
               "Constant environment closed form: albedo*E*V + (1-V)*irr*albedo + E*(F0 A + B); "
               "gradients reach albedo / roughness / metallic / cube base"),
        _probe("indirect-volumes", "indirect_volumes",
               "Irradiance trilinear SH query vs independent loops; gradient reaches the "
               "coefficients; V=0 shades with irradiance only"),
        _probe("stage-switch", "stage_switch",
               "Tiny scene: density bakes at pbr_start, renderer switches to PBR, light Adam "
               "steps, geometry/SH get no gradient in stage 2, bank exported in state"),
    )
    contribs = (
        ContributionSpec("linear-interpolated-depth", "Eq. 7 depth = sum(w d) / sum(w)",
                         ("depth-linear-interp",)),
        ContributionSpec("depth-derived-normal-supervision",
                         "Gaussian normals tied to depth-gradient normals (Eq. 8-9, supp. Eq. 2), "
                         "gradient-isolated from geometry", ("normal-from-depth",)),
        ContributionSpec("baked-occlusion-volumes",
                         "6-face depth cube probes -> binary visibility -> SH (Eq. 10-11), masked "
                         "trilinear recovery (supp. Eq. 6-7)", ("occlusion-baking",)),
        ContributionSpec("indirect-illumination-volumes",
                         "Learnable SH irradiance volumes in occluded directions (Eq. 13)",
                         ("indirect-volumes",)),
        ContributionSpec("split-sum-pbr-decomposition",
                         "Albedo/roughness/metallic + cube-map light, split-sum IBL (Eq. 14-16)",
                         ("cube-ops", "pbr-shading")),
        ContributionSpec("staged-training", "Initial -> baking -> decomposition in one run",
                         ("stage-switch",)),
    )
    base_probes = tuple(p for p in base.probes if p.name in ("state-transitions",
                                                              "photometric-gradient",
                                                              "renderer-gradient"))
    base_contribs = tuple(c for c in base.contributions)
    return replace(base, name=name, description=description, components=comps,
                   probes=base_probes + probes, contributions=base_contribs + contribs,
                   fields=FIELDS, citations=CITATIONS)


def method():
    return _build("gs-ir", {},
                  "GS-IR native port (gsplat + own CUDA/PyTorch PBR, baking, volumes); "
                  "Mip-NeRF 360 recipe 30k + 10k steps; paper fidelity unverified.")


def method_smoke():
    return _build("gs-ir-smoke", {"pbr_start": 1500, "stage1_steps": 1500},
                  "GS-IR pipeline with the decomposition stage from step 1500 (smoke test "
                  "only); paper fidelity unverified.")
