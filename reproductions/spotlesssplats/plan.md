# SpotLessSplats — native SplatStudio plan

Method names: **`spotlesssplats`** (headline SLS-mlp w/ UBP, paper Fig. 3–5 / Fig. 8 last row),
`spotlesssplats-noubp` (Fig. 8 "SLS-mlp w/o UBP"), `spotlesssplats-agg` (Fig. 8 "SLS-agg").
Module `ss_spotlesssplats`. Port target = the official gsplat trainer
(`source/code/examples/spotless_trainer.py`), whose numerics override the paper (paper.md §7).

## Components
| name | kind | factory | requires | provides | config (paper/code value) |
|---|---|---|---|---|---|
| sls-renderer | renderer | `ss_spotlesssplats.renderer:create_renderer` | state.gaussian3d, camera.pinhole | render.rgb, gradient.screen.signed, render.rgb_grad, render.sh_colors | `background: black` (official: no bg) |
| sls-density | density | `ss_spotlesssplats.density:create_density` | gradient.screen.signed (+ render.rgb_grad, render.sh_colors when ubp) | – | refine_start 500, refine_stop 15000, refine_every 100, grow_grad2d 2e-4, grow_scale3d 0.01, prune_opa 0.005, ubp true, ubp_thresh 1e-14, reset_sh_step 8002, reset_sh_value 0.001 |
| sls-robust | loss | `ss_spotlesssplats.losses:create_robust` | render.rgb | – | mask_mode mlp/agg/filter, robust_percentile 0.7, lower_bound 0.5, upper_bound 0.9, bins 10000, hist_decay 0.95, stats_stop 15000, schedule true, schedule_beta -3e-3, schedule_div 1.5, ssim_weight 0 |
| sls-mlp | loss | `ss_spotlesssplats.losses:create_mlp_loss` | render.rgb | – | lr 1e-3, hidden 16, pe_freqs 20, reg_weight 0.5 (no-gradient reg as in code) |

The base `photometric` loss (L1+0.2 D-SSIM) is **replaced** by `sls-robust` (code: masked L1,
ssim_lambda 0). The gsplat DefaultStrategy density is replaced by `sls-density`, a line-by-line
port of the official refine loop (dup/split/prune via `gsplat.strategy.ops`, which has the same
numerics as the official `refine_*`), plus UBP and the SH reset. No opacity reset, no scale or
screen-size prune (official `step > reset_every` guard never fires).

Shared run context (`context.py`): data root from `--data` in argv or `SLS_DATA`; training frames
of `transforms.json`; per-frame SD features `<root>/SD/<stem>.npy` [1280,50,50] (CPU, moved per
step) and, for agg, `<root>/SLS_clusters/<stem>.npy` (int16 [50,50] labels); lookup by the native
viewmat (as SparseGS/DNGaussian). Step convention: runtime step k (1-based) ↔ official t = k-1
for every schedule (α, histogram stop, refine, SH reset).

## Fields
None (no per-Gaussian learned fields). Density state: grad2d [N], count [N], sqrgrad [N].
MLP (22 k params) lives in the `sls-mlp`/`sls-robust` shared context with its own Adam.

## Implementation notes (equation → code)
1. Renderer: SH colours evaluated explicitly with `gsplat.spherical_harmonics(deg, means - campos,
   coeffs)` → `+0.5`, `clamp_min(0)` (identical to gsplat's internal path), passed to
   `rasterization(sh_degree=None)`, RGB mode, no background. A tensor hook on the rendered RGB stores
   dL/dÎ in `outputs["rgb_grad"]` (UBP input).
2. sls-robust (Eq. 1, 2, 7–9; §4.2.2): `Î = clamp(rgb,0,1)`, `e = mean_c|Î-I|`;
   thresholds from the discounted histogram (`hist = 0.95 hist + histogram(e, 10000, [0,1])`,
   threshold = `linspace(0,1,10001)[first k: cumsum ≥ q·sum]`) updated after the loss while t<15000;
   `robust_mask` = inlier OR (3×3 box mean of inliers > 0.5).
   mlp mode: MLP on [bilinear(SD feat) ‖ PE(20)] — the first Linear is applied at 50×50 and then
   bilinearly upsampled (exactly equal: bilinear interpolation is linear with weights summing
   to 1); hinge targets lower = mask(thr q=0.5), upper = mask(thr q=0.9).
   agg mode: cluster maps (nearest upsample) vote by inlier fraction > 0.5 (Eq. 3–4).
   Bernoulli: `M ~ B(clip(α + (1-α) M*, 0, 1))`, `α = exp(β ⌊(t+1)/1.5⌋)`. Loss = `mean(M·|Î-I|)`.
3. sls-mlp (Eq. 6–7): `mean(relu(p - M_upper) + relu(M_lower - p)) + 0.5·reg.detach()`; the MLP
   Adam step for step k is applied at the start of step k+1 (same order as official).
4. sls-density: t=k-1; if t<15000: accumulate grad2d (×W/2, ×H/2 normalised, radii>0), count and,
   with UBP, `sqrgrad += Σ_pixels (v_x² + v_y²)` from the JIT CUDA kernel `csrc/ubp.cu` (gsplat
   1.5.3 RasterizeToPixels3DGSBwd re-derived for means2d only, with per-pixel squaring before the
   warp sum, as in the official 1.1.1 fork). If t>500 and t%100==0: duplicate (grad≥2e-4 & small),
   split (grad≥2e-4 & large, original Gaussians only), prune (opacity<0.005 | sqrgrad<1e-14),
   new Gaussians get sqrgrad=1, reset stats. At t=8002 clamp shN to ≤0.001 and zero its Adam moments.

## Probes (each fails if the mechanism is removed)
| probe | measures |
|---|---|
| robust-mask | `robust_mask` vs an independent NumPy implementation; mask changes when an outlier blob is injected |
| quantile-histogram | threshold = discounted-histogram quantile vs NumPy reference; discount matters |
| mlp-equivalence | low-res-first-layer MLP == naive upsample-then-MLP; hinge gradient pushes p into [lower, upper] |
| warmup-schedule | α at t=0, 1000, 3000 vs closed form; Bernoulli mean ≈ α+(1-α)M |
| agg-vote | cluster vote (Eq. 3–4) vs a loop reference |
| ubp-kernel (CUDA) | kernel `Σ_p (∂(g_p·Î_p)/∂x)²` vs brute-force per-pixel autograd through gsplat |
| density-ubp (CUDA) | unutilised Gaussian pruned, utilised kept, new Gaussians get sqrgrad=1, SH reset clamps & zeros moments |
| state-transitions | built-in alignment probe |

## Data (new benchmark datasets)
`tools/prepare_spotless.py` (workspace) converts the authors' preprocessed scenes (Borealis
doi:10.5683/SP3/WOFXFT, downloaded to `/bigdata/sejain/datasets/spotless/raw`) to
`~/datasets/spotless/native/<scene>`: images_8 hard links; explicit splits train = `clutter`,
test = `extra`, validation = hard-linked copies of test (never used for selection); world
normalised exactly like the official Parser (`similarity_from_cameras` + `align_principle_axes`,
all COLMAP cameras); points.npz = normalised SfM points; `SD/` = symlink to the precomputed SD 2.1
features; `SLS_clusters/` for agg (sklearn 1.x ward clustering, official parameters).
Registered in `tools/datasets.json` as `robustnerf` (android, statue, yoda, crab2) and `onthego`
(mountain, fountain, corner, patio, spot, patio-high); only android and patio are downloaded.

Benchmark (per-scene reported → 2 scenes): `robustnerf/android` (indoor, robot capture) and
`onthego/patio` (outdoor casual), 30k steps, `--recipe 3dgs --num-gaussians 10000000`.
A `gsplat` baseline on the same scenes is needed (no shared baseline exists for these datasets).

## Fidelity gaps
1. **GLO appearance (C8)** not implemented — the official code has none; paper numbers include it
   (README: gsplat port without GLO is within ~0.2 dB on android/patio). Could be closed with a
   per-image embedding renderer (runtime lacks per-image state; doable inside the renderer by viewmat).
2. Per-image utilisation (Eq. 10) uses the official loss-gradient formulation (threshold 1e-14), not
   the paper's mask·∂Î/∂x with 1e-8.
3. View sampling: runtime samples with replacement; official shuffles without replacement per epoch.
4. Optimiser step happens before density in the runtime (official: refine before step, so the
   refine step's update is skipped). One step in 100; negligible.
5. scene_scale: runtime uses training cameras (clutter) only; official uses all COLMAP cameras.
6. Initial quaternions identity (runtime) vs random (official); seed/RNG streams differ.
7. Metrics: runtime LPIPS-VGG (paper) — official code AlexNet; SSIM valid-conv vs torchmetrics.
8. SD features are the authors' released `.npy` files (exact); our extractor (tools/extract_sd_features.py)
   is provided for other scenes and checked for parity on a few images only.
9. MLP Lipschitz reg contributes no gradient (as in the code; paper intends a gradient).
10. Paper reports means of 5 seeds; we run one.
