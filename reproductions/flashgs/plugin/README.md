# splatstudio-method-flashgs

**FlashGS: Efficient 3D Gaussian Splatting for Large-scale and High-resolution Rendering**
(arXiv [2408.07967](https://arxiv.org/abs/2408.07967)), official code
[InternLandMark/FlashGS](https://github.com/InternLandMark/FlashGS) @ `cdfc4e40` (MIT).

## What FlashGS is — and why this is a render-only plugin
FlashGS is an inference rasterizer: opacity-aware exact ellipse/tile intersection (§4.1),
warp-cooperative size-aware scheduling fused with key generation (§5.1), software-pipelined
rendering (§5.2) and low-level optimisations (§5.3). Its quality claim is *unchanged output*
(same PSNR as 3DGS), its result is *speed*. The public release has **no backward pass**, so
it cannot train; the paper trains its models with unmodified 3DGS.

* `flashgs` method: trains exactly like the `gsplat` baseline (gsplat renderer whenever
  gradients are enabled, unchanged density and loss) and renders with the FlashGS kernels
  under `torch.no_grad()` — i.e. in the final validation and in `ss-eval`. Inference
  outputs contain RGB only (FlashGS produces no depth/alpha).
* `apply_flashgs.py`: the paper protocol — take any finished 3DGS/gsplat run (e.g. the shared
  `runs/baseline3dgs-tandt-30000/gsplat/tandt-truck`), render the test views with gsplat, the
  original 3DGS rasterizer and FlashGS; write SplatStudio metrics, image-to-image parity PSNR,
  key/value pair counts, and per-view same-GPU timings at native / 1920 / 3840 widths.

## CUDA (vendored unmodified, JIT-built on first use, cached in `~/.cache/ss_flashgs`)
* `ss_flashgs/csrc/` — FlashGS `csrc/` byte-identical, MIT (`csrc/LICENSE-FlashGS.md`),
  upstream flags `-O1 -Xptxas=-O1`. ~20 s build.
* `ss_flashgs/ref3dgs_csrc/` — graphdeco `diff-gaussian-rasterization@59f5f77e`, Inria
  Gaussian-Splatting licence (research/evaluation use only), used forward-only as the paper's
  baseline. ~80 s build.

Wrapper-side guards (kernels untouched): key/value buffers are sized per frame from a torch
replica of FlashGS's coarse rectangle (upstream writes unchecked into a fixed 2²⁷ buffer); the
colour buffer has one spare tile row because the empty-tile branch writes the full tile
unchecked. Known upstream issue not patched: for W % 16 ≠ 0, an *empty* right-edge tile writes
background into the first pixels of the next row — counted per view as `edge_wrap_tiles`.

## Config (`flashgs-renderer`)
| key | default | meaning |
|---|---|---|
| `inference_backend` | `flashgs` | `gsplat` disables FlashGS (ablation) |
| `tile` | 16 | upstream example uses `render_16x16`; only 16 is wired |

Kernel constants (fixed in upstream code): τ = 1/256 (`ln2*8`; paper says 1/255), near 0.2,
opacity cull 1/255, 2D dilation 0.3, zNear/zFar 0.01/100, SH degree 3 (higher bands zeroed
by the driver when evaluating a lower degree), output uint8 truncation, centred principal point
required (true for all our datasets; otherwise a `ValueError`).

## Fidelity gaps
Shared ~100 %-busy RTX A6000 vs paper RTX 3090 (ratios only); test views instead of the paper's
frame sets; 1080p/4K by rescaled intrinsics (T&T → 1920×1071); gsplat-trained models (white
background); FlashGS gets precomputed cov3D (its API) while 3DGS/gsplat compute it per frame;
memory-reduction claim not measured; MatrixCity/Rubble not available. See `../plan.md`.

## Commands
```bash
cd reproductions/flashgs/plugin && ../../../tools/ssenv pip install -e .
tools/ssenv ss-inspect flashgs
CUDA_VISIBLE_DEVICES=<g> tools/ssenv ss-verify flashgs --device cuda
tools/ssenv python -m pytest -q reproductions/flashgs/plugin/tests
CUDA_VISIBLE_DEVICES=<g> tools/ssenv python reproductions/flashgs/plugin/apply_flashgs.py \
    --run runs/baseline3dgs-tandt-30000/gsplat/tandt-truck --out reproductions/flashgs/results/truck.json
```
