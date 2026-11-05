"""Compact 3D Gaussian Representation (Lee et al., CVPR 2024) native SplatStudio port; paper
fidelity unverified.

Global learned state (hash grid, MLP, R-VQ codebooks) lives in `model.CompactBank`, owned by the
density component and read by the renderer through `model.ACTIVE`; the decoded model reaches
ss-eval through the base fields plus `c3dgs_feature` / `c3dgs_blob` (see renderer.py, plan.md).
"""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method

from .density import DEFAULTS
from .renderer import BLOB, BLOB_WIDTH, FEATURE, MASK

CITATIONS = (
    "https://arxiv.org/abs/2311.13681",
    "https://github.com/maincold2/Compact-3DGS",
)

FIELDS = (
    FieldSpec(MASK, (), learning_rate=0.01, initial_value=1.0),  # official _mask, mask_lr 0.01
    # eval-only state written by the final bake; never receives a gradient
    FieldSpec(FEATURE, (32,), learning_rate=1e-12, initial_value=0.0),
    FieldSpec(BLOB, (BLOB_WIDTH,), learning_rate=1e-12, initial_value=0.0),
)


def _probe(name, description, cuda=True):
    return ProbeSpec(name, f"ss_compact_3dgs.probes:{name.replace('-', '_')}", description, cuda)


def method(comp=True, name="compact-3dgs"):
    base = make_method(name)
    _, _, photometric = base.components
    renderer = ComponentSpec(
        "compact-renderer", "renderer", "ss_compact_3dgs.renderer:create_renderer",
        ("state.gaussian3d", "camera.pinhole"),
        ("render.rgb", "render.depth.expected_z", "gradient.screen.signed"),
        {"mask_threshold": DEFAULTS["mask_threshold"]})
    density = ComponentSpec("compact-density", "density", "ss_compact_3dgs.density:create_density",
                            ("gradient.screen.signed",), (), {**DEFAULTS, "comp": bool(comp)})
    mask_loss = ComponentSpec("mask-loss", "loss", "ss_compact_3dgs.losses:create_mask_loss",
                              ("render.rgb",), (), {"weight": 5e-4})
    keep = tuple(p for p in base.probes if p.name != "renderer-gradient")
    probes = (
        _probe("hash-grid", "CUDA hash grid equals an independent int64 torch implementation of "
               "tcnn's grid_scale/resolution, dense index, coherent prime hash and trilinear "
               "weights (values and parameter gradients); level table = 6,299,960 entries x 2"),
        _probe("neural-color", "Renderer colour equals max(C0*MLP(cat[hash(contract(p)), "
               "SH9(2d-1)])+0.5, 0) computed independently; contraction = paper Eq. 8; tcnn SH "
               "table vs closed-form real SH; colour is view dependent"),
        _probe("volume-mask", "Masked Gaussians (sigmoid(m)<=0.01) do not contribute, STE mask "
               "gradient = (dL/ds_hat . s + dL/do_hat o) sigmoid'(m), mask loss = lambda mean "
               "sigmoid(m) and its gradient, mask pruning at refine steps / every 1000 after 15k"),
        _probe("rvq", "R-VQ vs an independent loop: K-means (1 iter) init, nearest code, residual "
               "cascade, in-place Adam(1e-4) codebook step on mse, re-quantise, L x straight-"
               "through gradient; renderer uses R-VQ only for steps > 29000", False),
        _probe("huffman-codec", "dahuffman-port code lengths give the optimal Huffman cost (vs an "
               "independent textbook Huffman incl. EOF); canonical encode/decode lossless; "
               "post_quant matches the official formula (min/max before pruning)"),
        _probe("net-schedule", "Closed-form net lr equals torch ChainedScheduler(LinearLR, "
               "MultiStepLR) replay; mask-prune / R-VQ / encode steps equal the official loop",
               False),
        _probe("bake-eval", "Encode -> bytes -> decode -> bake on a fixture: a fresh bank-free "
               "renderer after a checkpoint round trip reproduces the baked render exactly; sizes "
               "equal the official final_prune formulas; an un-baked state refuses to render"),
        _probe("train-render-gradient", "Live-bank training render yields finite non-zero "
               "screen-space, hash-grid, MLP, mask, scale and opacity gradients"),
    )
    contributions = (
        ContributionSpec(
            "volume-mask",
            "C1 (§3.1, Eq. 1-3): learnable binary (STE) mask on scale and opacity, lambda_m 5e-4 "
            "mean-sigmoid loss, mask pruning during and after densification",
            ("volume-mask", "net-schedule", "state-transitions"),
        ),
        ContributionSpec(
            "neural-view-dependent-color",
            "C2 (§3.3, Eq. 7-8): tcnn hash grid (16 levels x 2, 2^19, 16..4096) at the contracted "
            "position + SH(deg 3) direction encoding -> 2x64 MLP -> SH0 colour; net Adam with "
            "warm-up and step decay",
            ("hash-grid", "neural-color", "net-schedule", "train-render-gradient"),
        ),
        ContributionSpec(
            "geometry-codebook",
            "C3 (§3.2, Eq. 4-6): 6-stage x 64-code residual VQ of scale and rotation in the last "
            "1000 iterations (K-means init, in-place codebook learning, straight-through)",
            ("rvq", "net-schedule"),
        ),
        ContributionSpec(
            "post-processing-storage",
            "C4 (§4.1, Table tab:size): fp16 positions, R-VQ indices + codebooks, 8-bit opacity "
            "and pruned 8-bit hash, Huffman coding, fp16 MLP; encoded bytes are decoded and the "
            "decoded model is evaluated; paper-formula and actual sizes",
            ("huffman-codec", "bake-eval", "photometric-gradient"),
        ),
    )
    variant = "Ours+PP" if comp else "Ours (no PP)"
    return replace(
        base,
        description=(
            f"Compact-3DGS native port ({variant}: volume mask + hash-grid neural colour + R-VQ "
            "geometry codebooks, encoded/decoded-model evaluation); paper fidelity unverified"
        ),
        components=(renderer, density, photometric, mask_loss),
        fields=FIELDS,
        probes=keep + probes,
        contributions=base.contributions[:2] + contributions,
        citations=CITATIONS,
    )


def method_no_pp():
    return method(comp=False, name="compact-3dgs-nopp")
