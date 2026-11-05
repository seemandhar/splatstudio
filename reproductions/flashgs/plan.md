# FlashGS — native SplatStudio plan

- **Method name**: `flashgs`  · **module**: `ss_flashgs` · **workspace**: `reproductions/flashgs/`
- **Feasibility verdict**: the official release (`InternLandMark/FlashGS@cdfc4e40`) is
  **forward-only** — no backward kernels, no autograd wrapper (pybind exposes `loadPly`,
  `preprocess`, `sort_gaussian`, `get_sort_buffer_size`, `render_{16x16,32x16,32x32}`; output is
  `uchar3`). A trainable FlashGS method is therefore impossible without writing a new backward,
  which the paper never proposes (its models are trained by unmodified 3DGS, §6.1.2).
  The faithful reproduction is a **render-only renderer at evaluation time** on models trained
  by the standard pipeline, verifying (a) image/PSNR parity with the reference rasterizer and
  (b) render speedup on the same GPU. `feasible_natively: true` in that scoped sense.

## How it fits the plugin model
| piece | kind | what |
|---|---|---|
| `flashgs-renderer` | renderer (`ss_flashgs.renderer:create_renderer`) | grad enabled → gsplat renderer (unchanged baseline, provides `render.rgb`, `render.depth.expected_z`, `gradient.screen.signed`); `torch.no_grad()` (SplatStudio `measure`: final validation, `ss-eval`) → FlashGS kernels, uint8 RGB over white |
| `adaptive-density` | density | unchanged gsplat `DefaultStrategy` + 3DGS opacity reset |
| `photometric` | loss | unchanged L1 + 0.2 D-SSIM |
| `apply_flashgs.py` | evaluation tool (like `fcgs/plugin/apply_fcgs.py`) | paper protocol: load a finished baseline run (`runs/baseline3dgs-{tandt,db}-30000/gsplat/<scene>`), render test views with gsplat / original 3DGS / FlashGS, SplatStudio metrics + parity PSNR + same-GPU speed at native, 1920 ("1080p"), 3840 ("4k") |

`ss-train flashgs` is therefore bit-identical in training to `gsplat`; it exists so `ss-eval`
and `ss-verify` exercise the FlashGS path. The benchmark does **not** retrain: it applies the
tool to the shared baseline checkpoints.

No new fields. No schedules.

## Kernels (custom CUDA, vendored unmodified, JIT via `tools/ssenv`)
- `ss_flashgs/csrc/` = upstream `csrc/` byte-identical (MIT, `LICENSE-FlashGS.md` added);
  built with upstream setup.py flags (`-O1`, `-Xptxas=-O1`).
- `ss_flashgs/ref3dgs_csrc/` = graphdeco `diff-gaussian-rasterization@59f5f77e` unmodified
  (Gaussian-Splatting licence, research/evaluation use) — the paper's baseline, forward only.
- Driver (`core.py`) replicates upstream `example.py`'s `Rasterizer.forward`; conversions:
  cov3D = R S Sᵀ Rᵀ upper triangle (checked against upstream `loadPly`), SH [N,16,3] flat 48,
  camera = centre + camera→world rotation (3DGS `cameras.json` convention), zNear/zFar 0.01/100.
  Wrapper-side safety only (kernel untouched): key/value capacity sized from a torch replica of
  the coarse rectangle before each frame (upstream writes unchecked into a fixed 2²⁷ buffer);
  colour buffer padded by one tile row (empty-tile branch writes without bounds check).

## Probes
| probe | contribution | fails when |
|---|---|---|
| `flashgs-parity` | C1, C2/C3 (output unchanged) | FlashGS image < 40 dB vs original 3DGS or gsplat on a synthetic fixture; our model conversion ≠ upstream `loadPly`; pose-insensitive check |
| `precise-intersection` | C1 | a tile with brute-force alpha ≥ 1/255 (gsplat projection) is missing from FlashGS pairs, or pairs not fewer than 3DGS 3σ-AABB and the coarse rect |
| `render-speed` | C2–C4 | FlashGS 1080p forward not faster than original 3DGS forward (same GPU) |
| `inference-switch` | integration | training render not differentiable gsplat, or no-grad render not FlashGS |
| base `state-transitions`, `photometric-gradient`, `renderer-gradient` | baseline training path | — |

## Benchmark choice
Paper reports per-scene speed (Tab. 1) for Truck, Train, Playroom, DrJohnson at 1080p/4K and only
a single-frame PSNR-equality claim. Chosen: `tandt` truck+train and `db` drjohnson+playroom
(all 4 reported scenes we have), existing 30K baseline checkpoints, all test views
(llffhold8), 10 timed repeats per view. Targets: PSNR(FlashGS) within ±0.05 dB of the
reference rasterizer (paper: "does not change the quality"); speedup ratio vs original 3DGS
reported next to the paper's 3090 ratios (not expected to match exactly on a busy A6000).

## Fidelity gaps
1. **GPU**: shared, ~100 % utilised RTX A6000 vs idle RTX 3090/A100 → absolute ms not
   comparable; ratios measured back-to-back are noisy. Cannot be closed here.
2. **Frames**: paper times "hundreds or thousands of frames per scene" (unspecified
   trajectory); we time the llffhold8 test views. Avg/Max/Min speedups depend on frame set.
3. **1080p/4K cameras**: produced by rescaling test-camera intrinsics to width 1920/3840 (aspect
   kept: T&T 1920×1071, not 1080). Paper's exact procedure unstated.
4. **Model**: our gsplat-trained baselines (white background, SplatStudio recipe), not
   models trained by the original 3DGS code; Gaussian counts differ from theirs.
5. **Timing scope**: forward only, inputs resident on GPU, FlashGS buffers pre-reserved (as in
   upstream example), cov3D precomputed for FlashGS (upstream `loadPly` API) while 3DGS/gsplat
   compute covariance per frame (their default APIs). 3DGS timing includes its per-call torch
   buffer allocation (as in the original).
6. **Output format**: FlashGS writes truncated uint8; references are float. Reported both raw
   and with references truncated the same way.
7. **Upstream edge bug**: empty right-edge tiles with W % 16 ≠ 0 write bg into the first pixels
   of the next row; counted per view (`edge_wrap_tiles`), not patched.
8. **Memory reduction claim (49 %)** not measured (FlashGS preallocates; not comparable).
9. MatrixCity / Rubble scenes unavailable (large-scale datasets not on disk).
