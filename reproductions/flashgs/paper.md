# FlashGS — implementation spec

**Paper**: FlashGS: Efficient 3D Gaussian Splatting for Large-scale and High-resolution Rendering
(Feng, Liao, et al.), arXiv [2408.07967](https://arxiv.org/abs/2408.07967) (CVPR 2025).
**Official code**: https://github.com/InternLandMark/FlashGS @ `cdfc4e4002318423eda356eed02df8e01fa32cb6`,
MIT licence. `source/code/` (read, never executed from there).

## 1. Method summary
- FlashGS is a **render-only (inference) CUDA rasterizer** for already-trained 3DGS models.
  The public release has **no backward pass** (grep of `csrc/`: no gradient kernels; the pybind
  module exposes only `loadPly`, `preprocess`, `sort_gaussian`, `get_sort_buffer_size`,
  `render_16x16/32x16/32x32`). Training is unchanged 3DGS (paper §6.1.2: "We train 30K
  iterations on each dataset with 3DGS to obtain the Gaussian model for rendering").
- Output equals the 3DGS forward up to numerically benign differences; claim is *speed at equal
  quality* ("FlashGS does not change the quality, keeping 31.52 PSNR with 8.57x speedup", §6.5).

Contributions:
- **C1 Opacity-aware precise intersection** (§4.1, Eq. 2–3, Alg. 2): effective radius
  `r = sqrt(2 ln(α0/τ) λ)`, τ=1/255; tight axis-aligned bounding rect from the 2D covariance
  diagonal; then exact ellipse–tile test (centre-in-tile, or ellipse chord on an edge line
  overlapping that edge). Fewer key/value pairs → cheaper sort + render.
- **C2 Algebraic simplification + adaptive size-aware scheduling** (§5.1): division/sqrt-free
  segment test (Eq. 7); single-tile Gaussians handled per thread, multi-tile Gaussians handled
  cooperatively by the whole warp (ballot/shuffle); fused preprocess + duplicateWithKeys.
- **C3 Low-latency pipelined rendering** (§5.2): software-pipelined two-stage prefetch of
  Gaussian index + features (lane-distributed `__ldg` + `__shfl_sync`), 2×(BLOCK/8×BLOCK/4)
  pixels per thread, warp-level early termination, `ex2.approx` alpha.
- **C4 Common optimisations** (§5.3): static buffer allocation, inline PTX (`ex2/lg2/rsqrt.approx`).

## 2. Equations (verbatim)
- Eq. 1 (3DGS): `r = 3\sqrt{\lambda}`
- Eq. 2: `\alpha = \alpha_0 \times \text{exp}(-\frac{r^2}{2\lambda})`
- Eq. 3: `r = \sqrt{2ln(\frac{\alpha_0}{\tau})\lambda}`, τ = 1/255.
  Code (`preprocess.cu`): `power = ln2*8 + ln2*log2(opacity)` = ln(256·α0) (τ = 1/256 in code),
  `width = int(1.414214*sqrt(cov.x*power)+1)`, `height = int(1.414214*sqrt(cov.z*power)+1)`
  (axis-aligned half extents from Σ'₀₀, Σ'₁₁, not the eigenvalue).
- Eq. 5–7 (segment test): `a_i \leq b_j \&\& a_j \geq b_i`;
  `(e_1 \leq 0 || e_1^2 \leq DELTA) \&\& (e_2 \geq 0 || e_2^2 \leq DELTA)`, `e_1=2Aa_i+B`, `e_2=2Ab_i+B`.
  Code: `segment_intersect_ellipse`, `block_intersect_ellipse` (nearest vertical edge, then
  nearest horizontal edge), quadratic in the edge coordinate with `c = conic·d² − 2·power`.
- Rendering (code): `alpha = 2^(log2 α0 + q)`, q = −½·log2e·dᵀΣ'⁻¹d; **no 0.99 alpha clamp and
  no per-pixel 1/255 skip** (both are in 3DGS); stop when every pixel of the thread has T<1e-4.

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| τ (alpha threshold) | 1/255 paper, 1/256 code (`ln2*8`) | §4.1 / `preprocess.cu` |
| near cull | z ≤ 0.2 | `preprocess.cu` (= 3DGS) |
| opacity cull | 255·α < 1 | `preprocess.cu` |
| 2D low-pass | +0.3 on Σ' diagonal | `computeCov2D` (= 3DGS) |
| tile | 16×16 (also 32×16, 32×32 kernels) | `example.py` uses 16×16 |
| zNear / zFar | 0.01 / 100 | `example.py` |
| SH degree | fixed 3 (48 floats) | `shs_deg3_t` |
| output | uchar3, `uint8(clamp(x*255))` (truncation) | `render.cu:encode` |
| background | argument (example uses black) | `example.py` |

## 4. Schedules
None (no training).

## 5. Builds on
3DGS (Kerbl et al. 2023) forward rasterizer; model trained by unchanged 3DGS for 30K iterations.

## 6. Protocol
- Tab. 1 (RTX 3090; A100 avg row): Truck, Train (T&T), Playroom, DrJohnson (DB) at 1080p and 4K,
  MatrixCity 1080p/4K, Rubble 4608×3456. Avg/Max frame time FlashGS vs 3DGS and Avg/Max/Min
  speedup, over "hundreds or thousands of frames per scene" (which cameras is unstated).
- Quality: PSNR of FlashGS = 3DGS (one MatrixCity frame: 31.52 dB both). No per-scene PSNR table.
- Timing: `example.py` times `Rasterizer.forward` (preprocess + host read of num_rendered +
  sort + render) with model attributes preloaded (cov3D precomputed at load).

## 7. Ambiguities
- How 1080p/4K cameras were produced from T&T/DB (native ~980×545 / ~1300×850): assumed
  intrinsics rescaled to width 1920 / 3840 with unchanged aspect.
- τ = 1/255 (paper) vs 1/256 (code); code used.
- 3DGS timing method (rasterizer forward only vs full render) unstated; we time the
  rasterizer forward of each backend with inputs already on the GPU.
- Principal point must be the image centre (projection from width/height/focal only) — holds
  for all our T&T/DB/Mip-NeRF 360 cameras (cx = W/2, cy = H/2).
