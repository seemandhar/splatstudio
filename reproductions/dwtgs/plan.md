# DWTGS plan — method `dwtgs`, module `ss_dwtgs`

Built on the in-repo DropGaussian port (`reproductions/dropgaussian`, matched 20.36 vs paper 20.76;
DWTGS's own DropGaussian re-run is 20.39). Its renderer and photometric loss are reused **by
factory reference** (`ss_dropgaussian.*`, hashed into the run's source manifest); the density is a
thin subclass that additionally stores the training cameras and initial points.

## Components
| name | kind | factory | config (paper value) |
|---|---|---|---|
| dropgaussian-renderer | renderer | ss_dropgaussian.renderer:create_renderer | gamma 0.2, total 10000, black bg, clamp, ST alpha cap |
| dropgaussian-density | density | ss_dwtgs.density:create_density | DropGaussian defaults (tau 5e-4, ...) |
| photometric | loss | ss_dropgaussian.losses:create_photometric | L1 + 0.2 D-SSIM |
| dwtgs-lf | loss | ss_dwtgs.losses:create_lf | weight λ_LL 0.5, levels N 2, haar_norm average (Eq. 5) |
| dwtgs-hf | loss | ss_dwtgs.losses:create_hf | weight 1, interval 10, stop 5000, start 1, haar_norm average, novel_views interp (slerp/lerp of a random training pair, t~U(0,1)), seed 0 (Eq. 6); option fsgs_llff = FSGS poses |

Variant `dwtgs-lf` (paper Table III row "+LF 2-level", 20.70) = no `dwtgs-hf`.
Fields: none new.

## Implementation notes
- `wavelet.haar_dwt(x[C,H,W])` → LL, LH, HL, HH, averaging Haar (review 1, C1): LL=(a+b+c+d)/4,
  LH=(a+b-c-d)/4, HL=(a-b+c-d)/4, HH=(a-b-c+d)/4 (orthonormal /2 = labelled option) on 2×2 blocks
  [a b; c d]; odd H/W zero-padded at the end (pytorch_wavelets mode='zero').
- LF: Σ_{n=1..N} λ mean|LL_n(render) − LL_n(gt)|, render = training (dropout) render HWC→CHW.
- HF: on active steps build a novel camera by spherical interpolation between a random pair of
  training views (Fig. 2 note 3; `interp.py`), render it with a private
  `ss_dropgaussian` Renderer with gamma 0 (official `render(is_train=False)` default), at the
  runtime's current SH degree min((step-1)//1000, 3); loss = mean|HH_1|. Gradient flows into
  params; densification statistics use only the training render (as FSGS/official).
  Inactive steps return `0 * rgb.sum()`.

## Probes
| probe | checks | fails if |
|---|---|---|
| haar-dwt | configured norm = paper averaging; subbands vs an independent loop (/4), LL = avg-pool, a = LL+LH+HL+HH reconstruction, zero-pad on odd sizes | wrong filter/orientation, orthonormal normalisation |
| lf-loss | equals independent Σλ L1(LL); invariant to a level-1 HH checkerboard; levels/weight = paper | LF loss removed / supervises HF |
| hf-schedule | active only at step%10==0, ≤5000; zero & graph-attached otherwise | schedule wrong |
| hf-novel-view (CUDA) | loss renders a pseudo camera ≠ training cams, value = mean|HH_1| of an independent render, nonzero grad to means/sh0 | HF on training view / no gradient |
| pseudo-poses | novel_views = interp; each view's centre on a training-pair segment and rotation on the geodesic at the same t; t spread over (0,1) | non-interpolated source (e.g. FSGS poses) |
+ reused DropGaussian probes for renderer/photometric/density.

## Fidelity gaps
1. Haar normalisation: averaging chosen from §IV-B/Figs. 2–3 (review 1); orthonormal (3× effective LF weight) would differ; closable only with official code. Review 2k horns smokes: orthonormal 18.04 dB/152k G, averaging 18.44–18.64/86k, DropGaussian 18.88/59k.
2. Novel views: paper (Fig. 2 note 3) says spherical interpolation between training views; pair choice, t distribution and intrinsics are unstated → random distinct pair, t~U(0,1), slerp R + lerp centre/K (our reading).
3. Novel view rendered without dropout (official render default); paper silent.
4. HF weight 1 (Eq. 6 has none).
5. Inherited DropGaussian gaps (view sampling with replacement, Adam-before-densify, gsplat rasterizer/ST cap, lpips package, 1 seed).
6. Mip-NeRF 360 12-view and Blender not benchmarked (LLFF is the cheapest complete average).
7. LF gradient is in the densification statistic (same backward as the main loss, as in official 3DGS-style code) → more Gaussians (≈1.45× at 2k with averaging); paper reports no counts, report #G.
8. Novel-view renderer is a private gamma-0 DropGaussian renderer instance (its own step counter; harmless, rate 0).
