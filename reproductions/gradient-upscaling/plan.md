# Plan — gradient-aware upscaling (plan id 80)

Method family `gradient-upscaling-*`, module `ss_gradient_upscaling`, pip package
`splatstudio-method-gradient-upscaling` in `plugin/`.

The paper is primarily a **training-time renderer** ("Spline (Ours)": render at W/s with
analytic image derivatives, spline-upscale to W, loss at W) plus an **inference-only** use on
3DGS models trained at W/s ("Spline (3DGS)"). Both are implemented: a renderer component for
training and `plugin/apply_upscaling.py` for evaluation of any finished checkpoint.

No official CUDA exists (repo is README-only; brush fork = WGSL viewer, Apache-2.0, forward
only). We therefore write the CUDA kernels ourselves: forward follows gsplat 1.5.3's
`RasterizeToPixels3DGSFwd.cu` structure/numerics (Apache-2.0) plus the derivative recurrences of
the paper / brush `rasterize.wgsl`; backward is the reverse-mode of the derivative jets with the
inversion trick (supp. "Backward Pass"). JIT build via `torch.utils.cpp_extension.load` under
`tools/ssenv`.

## Methods (entry points)
| method | renderer | density | paper row |
|---|---|---|---|
| `gradient-upscaling-2x` / `-4x` | jet rasterizer @ W//s + spline (analytic derivs) | Taming growth → 1 M | Spline (Ours) |
| `bicubic-upscaling-2x` / `-4x` | gsplat @ W//s + `F.interpolate(bicubic)` | Taming → 1 M | Bicubic |
| `gradient-upscaling-3dgs` | stock gsplat (trained with `--downscale s`) | Taming → 1 M | 3DGS (eval at full W via `apply_upscaling.py`) |

Loss: unchanged `photometric` (L1 + 0.2 D-SSIM) at full target resolution.
Density: `ss_taming_3dgs.density` (installed Taming port, plan #30) with
`mode=final_count, budget=1_000_000, densification_interval=100, high_opacity=off,
sh_update_every=1, lr={}` — only the growth/score densification, the paper's stated use.
Budget is a config key (5 M for Mip-NeRF360 full-res, not benchmarked).

## Components
| component | kind | requires | provides | config |
|---|---|---|---|---|
| `upscale-renderer` | renderer | state.gaussian3d, camera.pinhole | render.rgb, gradient.screen.signed, training.target_image | factor, upscaler ∈ {spline, bicubic}, tile_size 16 |
| `taming-density` | density | gradient.screen.signed, training.target_image | – | above |
| `photometric` | loss | render.rgb | – | – |

Renderer steps: low-res camera (K rows scaled by w/W, h/H; w=W//s like brush/runtime) →
gsplat `fully_fused_projection` (eps2d 0.3, classic) → SH colours (`spherical_harmonics`,
+0.5, clamp≥0) → `isect_tiles`/`isect_offset_encode` → `JetRasterize` (CUDA autograd fn) →
jets J=(I, I_x, I_y, I_xy) [4,h,w,3] incl. white background (`bg·T` jet) →
`hermite_upscale` (separable tensor-product Hermite = paper's A=C⁻¹FC⁻ᵀ) → rgb [H,W,3].
`info` = gsplat-style meta at low res (means2d with grad, radii, width=w, height=h,
n_cameras=1) so densification statistics work. Output pixel X ↦ low-res coordinate
(X+0.5)·w/W − 0.5 (consistent with the scaled intrinsics; = PyTorch align_corners=False).

CUDA (csrc/jet_raster.cu):
- fwd: per pixel, front-to-back; α=min(0.999, o·e^{−σ}), skip σ<0 or α<1/255, stop when
  T(1−α)≤1e-4 (gsplat). α jet: α_x=−α σ_x, α_y=−α σ_y, α_xy=α(σ_xσ_y−b) with σ_x=a dx+b dy,
  σ_y=b dx+c dy, d = pixel − mean; zero derivatives where clamped. w=α⊗T, C+=c·w,
  T←T⊗(1−α). Store out jets, T_final jet, last_ids. Optional sub-pixel offset (probe only).
- bwd: per pixel back-to-front over the same lists; T_i = T_{i+1}⊘(1−α_i) (inversion trick in
  jet algebra); M_k = c_k T_i − (buf_k + bg_k T_N)⊘(1−α_i); dL/dα-jet = adjoint of M⊗;
  dL/dc = ⟨G, w⟩; chain to o, conic, mean2d; warp reduce + atomics (gsplat style).

## Probes
| probe | checks | fails if |
|---|---|---|
| `jet-value-equals-gsplat` | value channel = gsplat `rasterization` RGB (white bg) | rasterizer numerics differ |
| `jet-derivs-vs-finite-diff` | I_x, I_y, I_xy vs central differences of the kernel's own image under sub-pixel shifts | derivatives wrong/removed |
| `jet-vs-torch-autograd` | fwd jets and bwd grads (means2d, conics, colours, opacities) vs an independent dense float64 torch reference using autograd double-backward for spatial derivatives | backward/forward wrong |
| `spline-equals-paper-matrix` | separable Hermite = A=C⁻¹F(Cᵀ)⁻¹, p=[1 x x² x³]A[1 y y² y³]ᵀ | upscaler formula wrong |
| `spline-equals-catmull-rom` | with central-difference f_x,f_y,f_xy the spline equals Keys a=−0.5 bicubic (interior) | gradient input ignored / wrong scaling |
| `analytic-beats-fd` | on a synthetic smooth scene (GT = native full-res render), spline(analytic) PSNR > spline(FD) and > torch bicubic at 4× | analytic gradients removed |
| `renderer-end-to-end` | full renderer at s=4 trains: finite grads, means2d grad, output H×W | integration broken |

## Fidelity gaps
1. No official code: CUDA derived from the paper; numerics follow gsplat (0.999 clamp,
   opacity-aware radii) not Inria (0.99). Expected effect small; cannot be closed until release.
2. Datasets: Mip-NeRF360 in the paper is full-res 5187×3361 with 5 M Gaussians; we have only
   images_4/2 conversions and the cost is prohibitive → benchmark T&T + DB only (paper per-scene
   numbers available). DB drjohnson native is 1332×876 (paper header says 1264×832 = playroom).
3. White background (runtime) vs black (Inria); derivative includes bg·∂T (brush ignores bg).
4. Upscaler border: edge-cell cubic extrapolation (brush: mirrored fraction + edge clamp).
5. Taming: only densification is used; interval 100 assumed; Taming score views rendered at
   full GT resolution.
6. Control row "3DGS": runtime `--downscale s` (LANCZOS GT downsampling; Inria uses PIL
   default resize) and Taming 1 M; evaluated at full W by `apply_upscaling.py`.
7. Speed: our spline upscaler is PyTorch ops (paper: SLANG.D kernel); timings are same-GPU
   ratios on shared A6000s, not L4 ms. Upscaler backward via autograd (paper: atomics kernel).
8. View sampling with replacement, SSIM valid conv, LPIPS-VGG package — shared with baseline.
