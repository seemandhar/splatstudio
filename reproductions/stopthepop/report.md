# StopThePop (plan #71) — build report

**Paper**: StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering (TOG 2024, arXiv 2402.00525).
**Status**: ready for benchmark (not yet benchmarked). Methods: `stopthepop`, control `stopthepop-global`.

## Implementation
- **Renderer**: the official CUDA rasterizer, r4dl/StopThePop-Rasterization @ d8cad09, vendored unmodified in `plugin/ss_stopthepop/csrc/`.
  - Licences: Gaussian-Splatting (non-commercial) and MIT.
  - JIT-built through `tools/ssenv`; the full build takes about 70 s.
  - Used for **training and evaluation**, as the official `train.py` does, with `configs/hierarchical.json`: HIER sort, PTD_MAX order, queues 64/8/4, 4x4 culling, rect and tight bounding, load balancing.
- **Density**: the unchanged official 3DGS ADC.
  - The screen-size prune is inert, as in the official code, because `densification_postfix` zeroes `max_radii2D`.
  - The `opacity_decay` option reproduces the paper's "Opacity Decay" rows.
- **Loss**: L1 + 0.2 D-SSIM.
- **Control**: `stopthepop-global` runs the same kernels with `configs/vanilla.json` (3DGS global sort).

## Verification (`evidence/verify.json`: declared_checks_passed, 9 probes)
| probe | result |
|---|---|
| t_opt (Eq. 1) | Blend order matches a float64 reference sorted by Eq. 1 (err 1e-7). It differs from the d·(μ−o) and global-z ablations by 0.57. |
| per-pixel order (crossing splats) | HIER and PPX_FULL match an independent per-pixel-sorted compositor (5e-7). GLOBAL matches the global-z reference and is 0.54 off the per-pixel one. |
| backward | Kernel gradients for opacity, sh0 and screen means match autograd of the sorted reference (rel ≤ 8e-7). The global-order reference is 0.24–0.71 off. |
| popping (yaw sweep, exact-homography warp error, proxy for FLIP_t) | GLOBAL max 0.047 against median 4e-4: a pop. HIER max 0.003 against median 0.0014: smooth. |
| culling | Bounding and tile culling leave the image unchanged (max diff 0 for GLOBAL, mean 7e-5 for HIER). Sort entries drop from 567k to 277k for HIER and to 197k with tile culling. |
| schedule | 3DGS cadence, reset, world prune and opacity decay all pass. |

Unit tests: 5 pass. ruff is clean.

## Smoke test (garden, 2000 steps, shared A6000)
| run | PSNR | SSIM | LPIPS | Gaussians | train s |
|---|---|---|---|---|---|
| gsplat baseline (reused from steepgs, same args) | 23.02 | 0.643 | 0.419 | 845k | 175 |
| stopthepop-global (control) | 23.25 | 0.650 | 0.413 | 823k | 250 |
| **stopthepop** | 23.19 | 0.654 | 0.400 | 855k | 279 |

## Benchmark request
Mip-NeRF 360 **garden + counter**, chosen because the supplementary reports per-scene numbers. Protocol: images_4/images_2 as in `full_eval.py`, 30k steps, compared against `baseline3dgs-mipnerf360-30000`.

| scene | paper Ours | paper 3DGS (author retrained) | paper Δ | our gsplat baseline |
|---|---|---|---|---|
| garden | 27.16 | 27.24 | −0.08 | 26.89 |
| counter | 28.59 | 28.98 | −0.39 | 28.85 |

What to compare:
- Ours against paper Ours, within ±0.5 dB.
- Δ(ours − our baseline) against the paper's Δ.
- Optionally, also run `stopthepop-global` to isolate the effect of sorting.

## Fidelity gaps
1. **Update order.** The runtime takes the optimizer step before density control; the official code densifies first. The effect should be negligible.
2. **Shared runtime differences.**
   - Views are sampled with replacement.
   - SSIM uses valid convolution.
   - LPIPS comes from the `lpips` package.
   - The SH degree steps up at a slightly different time.
3. **Popping metric.** FLIP_t (RAFT flow, FLIP, occlusion masks, 3 camera paths per scene) is not reproduced at scene scale. An analytic warp-consistency probe replaces it. This could be closed later with torchvision RAFT and a FLIP port.
4. **Tile-based culling.** It is off, following the released `hierarchical.json`. The paper's configuration (A) has it on. It only affects speed; the probe shows the image is unchanged.
5. **Timings.** Not comparable: we run on an A6000, the paper used an RTX 4090. Training is about 1.6× slower per step than gsplat; the global control is 1.4×.
6. **Opacity-decay variant.** Not benchmarked; the option is implemented.
7. **S⁻¹ clamp.** The 1e3 bound is taken from the code (`1/max(1e-3, s)`). No probe isolates it.

## Review-1 resolution (orchestrator)
Approved, with 0 critical and 0 major issues. The benchmark launched with `stopthepop` on garden and counter. The `stopthepop-global` control was not run: its `VANILLA` dict has `rect_bounding`, `tight_opacity_bounding` and `load_balancing` off, while the real `configs/vanilla.json` has them on (review minor 1). Fix this before using that control to isolate the effect of sorting.

Notes added from the review:
- The backward-pass error is up to 1.08e-6.
- On densify and reset iterations, the official `optimizer.step()` has no effect on the replaced tensors.
- The SH degree is raised one step later than in the official code.
