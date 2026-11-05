# DOF-GS native plan

Method `dofgs`, module `ss_dofgs`, package `splatstudio-method-dofgs` in `plugin/`.

## Architecture
The official DOF CUDA rasterizer (`submodules/diff-gaussian-rasterization-dof`, Inria/MPII
licence) is vendored **unmodified** in `ss_dofgs/csrc/` and JIT-built through
`torch.utils.cpp_extension.load` (run via `tools/ssenv`). Per-view learned state (f_m, Q_m
reparameterised as Fs/Ks, plus the ILN conv net) lives in a process-local `DofBank`
(nn.Module + own Adam), created by the density component from `set_training_cameras`, shared
with renderer/loss through `ss_dofgs.bank.ACTIVE` (HAC/EAGLES pattern) and stepped by the
density component in `after_backward`. The bank's tensors are also exported into
`density.state` (saved in checkpoint.pt) for refocusing; evaluation never needs them because
test views are rendered All-in-Focus (Q = 0), exactly like the official evaluation.

| component | kind | factory | requires / provides | config (paper/code values) |
|---|---|---|---|---|
| `dof-raster` | renderer | `ss_dofgs.renderer:create_renderer` | req state.gaussian3d, camera.pinhole; provides render.rgb, render.rgb.all_in_focus, render.depth.ray_distance, render.coc, render.dof_view, gradient.screen.signed | background black, kernel_size 0.1, infocus_start 5000, band_limit (per variant) |
| `dofgs-density` | density | `ss_dofgs.density:create_density` | req gradient.screen.signed | 3DGS clone/split thr 2e-4, percent_dense 0.03, 500–25000/100, prune opac<0.005, reset /3000 (filter-aware if band_limit, else plain min(σ(o),0.01)), compactness /1000 K=3 shrink 1.5, extra points @2000 M=30000 factor 0.25, Mip 3D filter (band_limit only), xyz lr 5e-4→5e-6 (max 80000), scales lr 0.01, F/K lr 0.05/0.01 start 100 stop 15000, mask lr 5e-4, coc_max 15 |
| `dofgs-objective` | loss | `ss_dofgs.losses:create_objective` | req render.rgb, render.rgb.all_in_focus, render.depth.ray_distance, render.coc, render.dof_view | λ_dssim 0.2, infocus_start 5000, λ_infocus 1, detail_enhance true |
| `dofgs-mask` | loss | `ss_dofgs.losses:create_mask_losses` | same | λ_mask per variant (0.002 or 0.001), reg factor 0.1 (λ_reg = 0.1 λ_mask), use_mask_loss true |

Method variants (official per-scene `config_real_<scene>.yaml`; only `band_limit` and
`lambda_mask_loss` differ between the 9 real scenes, `ss_dofgs/variants.py`):

| method | band_limit | λ_mask | scenes |
|---|---|---|---|
| `dofgs` | true | 0.002 | cake, daisy |
| `dofgs-bl-m001` | true | 0.001 | cisco, sausage, seal, tools |
| `dofgs-nobl` | false | 0.001 | caps, cupcake, cups |

The base `photometric` loss is **replaced** by `dofgs-objective` (paper deactivates L_rec after
warm-up; before 5k the objective is exactly L_rec = 0.8 L1 + 0.2 (1-SSIM_same-pad)).

Renderer logic: if the camera's viewmat matches a training view of the live bank → defocused
render with (f_m, Q_m) as `rgb` (+ depth, coc, info for densification); if additionally
bank.step+1 > infocus_start and grad is enabled → AiF render (F=[1], K=[0]) as `rgb_aif`.
Any other camera (test/val, or no bank in ss-eval) → AiF render as `rgb`.

## Fields
| field | shape | lr | notes |
|---|---|---|---|
| filter_3d | () | 1e-8 (never receives grad) | Mip-Splatting 3D filter std, recomputed from training cameras |

## Probes
| probe | contribution | measures / fails if removed |
|---|---|---|
| dof-kernel-reference (cuda) | C1 | official kernel RGB, depth, CoC map vs independent float64 dense renderer with Σ'+aI (a=R²/2ln4), for Q=0 and Q>0; fails if CoC term missing or wrong |
| pinhole-limit (cuda) | C1 | Q=0 render == reference without CoC; Q→1e-4 continuous; focus plane z=f unblurred |
| analytic-blur (cuda) | C1 | single Gaussian: measured image variance == σ'²+k+R²/(2ln4) across 3 apertures |
| fk-gradients (cuda) | C2 | kernel dL/dF, dL/dK == autograd of float64 reference; dL/dz from CoC disabled |
| camera-init | C2 | F/K init equals independent numpy of supp. B.1 / setup_defocus_params; bank schedule: step 100 start, freeze 15000 |
| iln-mask-losses | C3 | ILN shapes (5→48→16, 64→20→1), Pearson/entropy losses vs numpy; gradient routing of the two composites |
| objective-schedule | C4 | ≤5000 equals L_rec(defocused); >5000 composite; removing the AiF branch changes it |
| density-recipe | C5 | extra points (60000 candidates, IDW attributes), compactness formula vs independent impl., prune 0.005, filter reset, xyz-lr(80000) |
| official-config | C6 | variant band_limit (renderer and density), λ_mask, λ_reg and all shared keys equal the official yaml of each mapped scene (pytest also checks the transcription against the yaml files) |
| band-limit-switch (cuda) | C5 | band_limit=false renders identical to the unfiltered kernel call, does no filter recompute and resets to min(σ(o),0.01); band_limit=true differs |
| state-transitions | base | gsplat ops alignment |

## Fidelity gaps
1. View sampling with replacement (runtime) vs shuffled stack — minor, can't close without runtime.
2. Runtime optimizer step happens before densification. Officially densify/prune/compactness/
   reset replace every Gaussian nn.Parameter (grad None) before optimizer.step, so on each of
   those steps (~245 of 40k) the official run skips the Adam step for ALL Gaussians, not only
   the new ones; the runtime steps them — negligible.
3. Last-iteration optimizer skip not replicated — negligible.
4. SSIM metric: SplatStudio valid-conv SSIM vs skimage SSIM in paper (PSNR comparable;
   SSIM not directly comparable). LPIPS not in the paper.
5. Paper checkpoints lost; official val scripts pick per-scene iterations (val_real.sh: cake
   15k, caps 15k, tools 40k; test-set selection). The protocol is the final 40k state; a 15k
   number may be reported only as a labelled diagnostic.
10. SH degree is raised one step later than official (runtime step 1001 vs official 1000) —
   negligible.
6. No 3DGS baseline in the paper; compare to gsplat baseline on the same scenes (must be run)
   and paper's Mip-Splatting row.
7. Random seeds/CUDA atomics: non-deterministic as official.
8. torch_cluster (knn for extra points and compactness) installed to
   `~/models/dofgs-pylib` (sys.path fallback) — same library as official.
9. Datasets: only the real defocus set is downloaded; the authors' COLMAP synthetic set folder
   failed to list (Drive 500) — synthetic not benchmarked.
