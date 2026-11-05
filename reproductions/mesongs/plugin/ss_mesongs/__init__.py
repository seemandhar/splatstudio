"""MesonGS (Xie et al., ECCV 2024) native SplatStudio port; paper fidelity unverified.

Per-scene methods (the official code selects hyper-parameters by scene name):
`mesongs-<scene>` = released `scripts/mesongs_block.sh` (`--hyper_config config3`, 8000 fine-tune
iterations), `mesongs-c1-<scene>` = `universal` config (repo csv "c1"), `mesongs-noft-<scene>` =
config3 without fine-tune (paper "Our" row), `mesongs-smoke` (diagnostic 1500 + 500 steps).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS

CITATIONS = (
    "https://arxiv.org/abs/2409.09756",
    "https://github.com/ShuzhaoXie/MesonGS",
)

# official mesongs.py: config3 / universal_config (prune, cb, depth, n_block), cal_imp beta_list,
# lr_scale_list (x4 in training())
SCENES = ("bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen",
          "bonsai", "train", "truck", "drjohnson", "playroom")
PRUNE_C3 = {"bicycle": 0.4, "bonsai": 0.34, "counter": 0.22, "garden": 0.28, "kitchen": 0.24,
            "room": 0.32, "stump": 0.3, "flowers": 0.5, "treehill": 0.5, "drjohnson": 0.41,
            "playroom": 0.2, "train": 0.22, "truck": 0.4}
PRUNE_C1 = {"bicycle": 0.3, "bonsai": 0.24, "counter": 0.12, "garden": 0.18, "kitchen": 0.14,
            "room": 0.22, "stump": 0.2, "flowers": 0.2, "treehill": 0.2, "drjohnson": 0.41,
            "playroom": 0.0, "train": 0.12, "truck": 0.38}
CODEBOOK = {s: 2048 for s in SCENES} | {"kitchen": 8192, "train": 4096, "truck": 4096}
DEPTH = {s: 20 for s in SCENES} | {"bonsai": 19, "counter": 19, "kitchen": 19, "room": 19}
BETA = {s: 0.1 for s in SCENES} | {"bicycle": 0.03, "stump": 0.01}
LR_SCALE = {"bicycle": 0.15, "bonsai": 0.1, "counter": 0.1, "garden": 0.1, "kitchen": 0.1,
            "room": 0.1, "stump": 0.01, "flowers": 0.05, "treehill": 0.05, "drjohnson": 0.1,
            "playroom": 0.1, "train": 0.05, "truck": 0.2}
N_BLOCK = {"c3": 57, "c1": 66}
FT_ITERATIONS = 8000  # scripts/mesongs_block.sh


def scene_config(scene, config="c3"):
    return {
        "scene": scene,
        "prune_percent": (PRUNE_C3 if config == "c3" else PRUNE_C1)[scene],
        "codebook_size": CODEBOOK[scene],
        "depth": DEPTH[scene],
        "n_block": N_BLOCK[config],
        "beta": BETA[scene],
        "lr_scale": LR_SCALE[scene] * 4,
    }


def _probe(name, description, cuda=True):
    return ProbeSpec(name, f"ss_mesongs.probes:{name.replace('-', '_')}", description, cuda)


def _make(name, overrides, description):
    base = make_method(name)
    renderer, baseline_density, photometric = base.components
    renderer = replace(renderer, name="mesongs-renderer",
                       factory="ss_mesongs.renderer:create_renderer")
    config = {**DEFAULTS, "base": dict(baseline_density.config), **overrides}
    density = ComponentSpec("mesongs-density", "density", "ss_mesongs.density:create_density",
                            baseline_density.requires, (), config)
    probes = (
        _probe("importance", "I_d = sum alpha T per Gaussian vs an independent per-pixel "
               "compositor; volume term (V / V[0.9N desc])^beta; official prune quantile"),
        _probe("octree", "Voxel keys, mean merge and lower-corner decode = line-by-line port of "
               "the official octreecodes / decode_oct", False),
        _probe("raht", "Fast RAHT tables = verbatim port; inverse(forward(x)) = x; orthonormal; "
               "custom adjoint backward = autograd of the functional loop; DC = sum / sqrt(N)"),
        _probe("block-quant", "VanillaQuan running range, scale/zero point, STE, split_length",
               False),
        _probe("euler", "R(euler(q)) = R(q) with the official Eq. 5 matrix and argument order; "
               "matrix_to_quaternion recovers R", False),
        _probe("codec-roundtrip", "Decoded bitstream renders = bank render; size = real zip "
               "bytes; official member layout"),
        _probe("pipeline", "Tiny end-to-end run: prune count, merge, fine-tune changes "
               "attributes and codebook but not positions, last step frozen, encode -> decode"),
        _probe("schedule", "Compress at base, codebook steps, freeze, encode at total, "
               "run-length guard", False),
    )
    contributions = (
        ContributionSpec("universal-pruning", "C1 (§3.3 Eq. 3): I_g = I_d I_i, I_d = sum over "
                         "training pixels of alpha T, I_i = (V / V_90%)^beta; prune the tau "
                         "lowest", ("importance", "pipeline")),
        ContributionSpec("octree-geometry", "C2 (§3.4): depth-d voxel keys, mean merge of "
                         "duplicates, positions = voxel corners, keys stored", ("octree",
                                                                               "pipeline")),
        ContributionSpec("euler-replacement", "C3 (§3.5 Eq. 4/5): quaternions replaced by Euler "
                         "angles, covariance from the Euler rotation", ("euler",
                                                                       "codec-roundtrip")),
        ContributionSpec("raht", "C4 (§3.5 Eq. 6): RAHT of [opacity, Euler, SH DC], DC kept in "
                         "float, AC coded", ("raht", "codec-roundtrip")),
        ContributionSpec("block-quantization", "C5 (§3.5 Eq. 7/8): 8-bit per-block min-max "
                         "quantisation of AC coefficients and log scales (no RAHT on scales)",
                         ("block-quant", "codec-roundtrip")),
        ContributionSpec("sh-vq", "C6 (§3.5): batched k-means VQ of the 45 SH-rest values "
                         "(codebook 2048, 1000 x 262144, decay 0.8)", ("pipeline",)),
        ContributionSpec("entropy-encoding", "C7 (§3.5): official npz members, DEFLATE, zip; "
                         "size = zip bytes; evaluation renders the decoded bitstream",
                         ("codec-roundtrip", "pipeline")),
        ContributionSpec("finetune", "C8 (§3.5): positions fixed, encode/decode simulated in "
                         "the forward with STE, official fine-tune LRs, 8000 iterations",
                         ("pipeline", "schedule")),
    )
    return replace(
        base,
        description=description,
        components=(renderer, density, photometric),
        probes=base.probes + probes,
        contributions=base.contributions + contributions,
        citations=CITATIONS,
    )


def build(scene, config="c3", finetune=True):
    over = scene_config(scene, config)
    if finetune:
        over["total_steps"] = DEFAULTS["base_steps"] + FT_ITERATIONS
        label = f"mesongs-{scene}" if config == "c3" else f"mesongs-{config}-{scene}"
        what = f"{FT_ITERATIONS} fine-tune iterations"
    else:
        over["total_steps"] = DEFAULTS["base_steps"]
        label = f"mesongs-noft-{scene}"
        what = "no fine-tune"
    desc = (f"MesonGS native port, {scene} {config} config (single run: 3DGS 30k + prune, octree, "
            f"Euler, RAHT, block quant, SH VQ, {what}; decoded-bitstream evaluation); "
            "paper fidelity unverified")
    return _make(label, over, desc)


def _factory(scene, config="c3", finetune=True):
    def method():
        return build(scene, config, finetune)
    return method


for _s in SCENES:
    globals()[f"method_{_s}"] = _factory(_s)
for _s in ("garden", "bicycle", "counter"):
    globals()[f"method_c1_{_s}"] = _factory(_s, "c1")
    globals()[f"method_noft_{_s}"] = _factory(_s, "c3", False)


def smoke():
    """Diagnostic 2000-step schedule (1500 base + 500 fine-tune), garden table; not a benchmark."""
    over = {**scene_config("garden"), "base_steps": 1500, "total_steps": 2000, "vq_steps": 100}
    return _make("mesongs-smoke", over, "MesonGS 2000-step smoke schedule (diagnostic only)")
