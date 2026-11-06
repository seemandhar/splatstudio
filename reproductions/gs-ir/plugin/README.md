# splatstudio-method-gs-ir

Native SplatStudio port of **GS-IR: 3D Gaussian Splatting for Inverse Rendering** (Liang et al.,
CVPR 2024, [arXiv 2311.16473](https://arxiv.org/abs/2311.16473)), written against the paper
and the official code ([lzhnb/GS-IR](https://github.com/lzhnb/GS-IR) @ e5a030b). Paper fidelity
unverified until benchmarked.

## Methods
| name | schedule |
|---|---|
| `gs-ir` | Mip-NeRF 360 recipe of the README: initial stage 30000 steps, baking at 30000, decomposition 10000 steps → `--steps 40000` |
| `gs-ir-smoke` | same pipeline with the switch at 1500 (≤2000-step smoke tests) |

`--steps` must exceed `pbr_start` for the decomposition stage to run; the evaluated RGB after
the switch is the PBR image (the paper's NVS metric).

## Mechanism → code
| paper | code |
|---|---|
| Eq. 7 linear-interpolated depth | gsplat `RGB+ED` (Σwd/Σw) — `renderer.render_initial` |
| Eq. 8–9, supp. Eq. 2 normal supervision | `geometry.normal_from_depth` (port of `depthmapToNormalCUDA`, 3×3 median), `losses.create_normal` (L1 on valid pixels ×1 + edge-aware TV ×5); normals rendered with detached geometry (official backward isolation) |
| Eq. 10–11 occlusion baking | `baking.bake_occlusion`: 256³ grid over ±16, probes at the 8 corners of every Gaussian's cell, 6×256² depth faces (gsplat, near 0.2, crop ±1.5), threshold 0.4, 16 SH coefficients, dilation |
| supp. Eq. 6–7 recovery | `volumes.masked_trilinear` + CUDA `sh_reconstruction` (256 cosine samples, curand jitter) |
| Eq. 13 indirect | `volumes.query_irradiance` (64³ × 9 SH, 1 channel, AABB ±1.5, learnable) |
| Eq. 14–16 split-sum PBR | `light.build_mips` / `light.pbr_shading`, `cubemap.*` (cube sampling, mips, diffuse convolution, CUDA GGX pre-filter), BRDF LUT `data/brdf_256_256.bin` (from the GS-IR repo) |
| Eq. 16 / supp. Eq. 3–5 | `losses`: L1(PBR), TV_mat ×1 (+ code-only roughness/metallic prior ×0.001), TV_light ×0.01 |
| stages / global state | `density.GSIRDensity` owns `bank.GSIRBank` (stage, cube light, irradiance, baked volumes, light Adam lr 0.05) and exports it to `density_state["gsir"]` |

## Evaluation
`ss-eval --run RUN` works directly: the renderer restores the bank from `RUN/checkpoint.pt`
(`--run` of the ss-eval command line, or env `GSIR_RUN`) and checks a fingerprint of the
Gaussians. Eval-time quirks of `render.py` are reproduced (roughness += final transmittance,
normal padding, no roughness remap).

## Custom CUDA
`ss_gsir/csrc/gsir_kernels.cu` (JIT-built on first use, ~1 min): GGX cube pre-filter
(bounds/forward/backward, same weights as nvdiffrec) and the GS-IR SH occlusion reconstruction.
nvdiffrast is not used (its installation into the shared env was refused); the needed texture
ops are re-implemented in PyTorch.

## Fidelity gaps
See `../plan.md` and `../report.md`: gsplat rasterizer conventions (alpha clamp 0.999, near
plane 0.01 for training renders, white stage-1 background), cube corner filtering approximation,
runtime metrics (whole-image PSNR, LPIPS on [-1,1]), deterministic dilation, sampling with
replacement.

## Commands
```bash
pip install -e .
ss-inspect gs-ir && ss-verify gs-ir --device cuda
ss-train gs-ir --data ~/datasets/mipnerf360/native/garden --output runs/x --steps 40000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
ss-eval --run runs/x --split test
python -m pytest -q tests
```
