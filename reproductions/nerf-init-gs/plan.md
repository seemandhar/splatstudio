# Plan — nerf-init-gs (plan id 106, arXiv 2404.12547)

Method names: `nerf-init-gs` ("NerfAcc @ 5k + Depth Loss", Table 4 best row) and
`nerf-init-gs-nodepth` ("NerfAcc @ 5k"). Module `ss_nerf_init_gs`.

## Pipeline
1. `ss-nerfinit-prep --source ~/datasets/mipnerf360/native/<scene> --output ~/datasets/nerfinit-gs/mipnerf360/<scene>`
   (iNGP 5k → 500k weight samples → 3-NN scales → depth of all training pixels → hard-linked copy;
   same transforms.json + images ⇒ same protocol_id as the native scene, so the shared
   `baseline3dgs-mipnerf360-30000` is comparable). `--init random-box` gives the 50³ variant.
2. `ss-train nerf-init-gs --data <copy> --recipe splatstudio --num-gaussians 10000000 --steps 30000`.

## Reuse
iNGP copied from the ConeGS plugin (`reproductions/conegs/plugin/ss_conegs/{ingp,hashgrid}.py`):
JIT CUDA tcnn-layout hash grid, nerfacc proposal sampler, pdf loss, training loop. Changed to the
nerfacc Mip-NeRF 360 example defaults (far 1e3, wd 0, 5000 its); ConeGS's median-depth query
removed; added Eq. 4 sampler and Eq. 6 depth.

## Components
| C | kind | factory | requires → provides | config |
|---|---|---|---|---|
| C2 init | preprocessing + density | `preprocess:main`, `density:create_density` (wraps splatstudio gsplat `Density` = 3DGS ADC) | gradient.screen.signed | base DefaultStrategy config |
| C3 depth | renderer | `renderer:create_renderer` (built-in gsplat renderer + `depth_accum` = ED·α, `nerf_depth` sidecar lookup) | → render.rgb, render.depth.expected_z, render.depth.accumulated_z, nerfinit.nerf_depth, gradient.screen.signed | — |
| C3 depth | loss | `losses:create_depth_loss` | render.depth.accumulated_z, nerfinit.nerf_depth | λ_init 0.9, decay 0.9, decay_steps 100 |
| — | loss | built-in photometric (L1 + 0.2 D-SSIM) | render.rgb | unchanged |
| C1 random box | preprocessing `--init random-box` | — | — | 50k, size 50 |

No new FieldSpecs.

Density: loads exact 3-NN log scales (`nerfinit/init.npz`, SHA-checked against the means),
opacity 0.1, identity quats; `set_training_cameras` sets gsplat `state["scene_scale"]` and the
means LR to the 3DGS camera extent (so `--recipe splatstudio` ≡ `--recipe 3dgs` without the
runtime's 16 GB 4096×N cdist at N = 500k).

## Probes
| probe | measures | fails if |
|---|---|---|
| nerfinit-hashgrid (cuda) | kernel vs torch reference | encoding wrong |
| nerfinit-proposal-sampler | nerfacc docstring cases, lindisp inverse, pdf loss | sampler wrong |
| nerfinit-surface-sampling | two-surface weights 0.3/0.7 → sample fractions, Eq. 6 depth, colour at sample | median/argmax/uniform placement |
| nerfinit-ingp-fit (cuda) | 3k-it fit on textured sphere: depth & samples on surface | NeRF path broken |
| nerfinit-init-loading | scales from init.npz, opacity 0.1, extent, LR, SHA mismatch raises | runtime init kept / wrong scale |
| nerfinit-depth-loss | λ(1,100,1000), L1 value/grad, raises without depth | schedule/loss wrong or silent no-op |
| nerfinit-renderer-depth (cuda) | depth_accum = gsplat "D"; nerf_depth only in training | normalised depth / leakage |

## Benchmark
Per-scene numbers exist (Table 4). Scenes: garden (outdoor, 27.37) + room (indoor, 31.63);
paper 3DGS re-run: 27.29 / 31.43 (gain +0.08 / +0.20). Compare against
`baseline3dgs-mipnerf360-30000`. Optional ablation `nerf-init-gs-nodepth` (27.21 / 32.11).

## Fidelity gaps
1. **GS depth d_i**: gsplat camera-z of the Gaussian centre instead of the ray-conditional mean
   (Supp.); NeRF depth converted from ray distance to camera z. Differs only within a Gaussian's
   extent; λ is ~0 after 3–5k its. Small effect. Closable with a custom rasterizer.
2. **iNGP precision**: fp32 PyTorch MLPs + own hash kernel vs tcnn fp16 FullyFusedMLP; NeRF quality
   at 5k may differ slightly → different init. Medium/small.
3. **NeRF training resolution**: images_2 for indoor scenes vs nerfacc factor 4. Small.
4. **Depth storage fp16** (rel. 1e-3). Negligible.
5. Runtime: white background (official 3DGS black), random-with-replacement view sampling, SSIM
   valid conv, LPIPS package — common to all SPLATIFY runs; compare gains vs baseline.
6. Ambiguities 1–3, 6–7 of paper.md (inverse CDF space, opaque-background samples, schedule
   scaling, continuous λ, unchanged densification params).
7. Timing: NeRF preprocessing time is recorded in `nerfinit/nerfinit.json`; must be added to
   training time.
8. Not reproduced: OMMO, TUM/Replica SLAM (C4), Table 1 random-init benchmark (variant exists).
