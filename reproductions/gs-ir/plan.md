# GS-IR native plan

Method names: `gs-ir` (Mip-NeRF 360 recipe, 30000 + 10000 = 40000 steps),
`gs-ir-smoke` (same pipeline, stage switch at 1500 for ≤2000-step smoke tests).
Module: `ss_gsir`. One `ss-train` run executes all three stages; the stage switch and the
baking happen inside the density component at `pbr_start`.

## Components
| component | kind | factory | requires | provides | config (paper/code value) |
|---|---|---|---|---|---|
| gsir-renderer | renderer | ss_gsir.renderer:create_renderer | state.gaussian3d, camera.pinhole | render.rgb, render.depth.expected_z, gradient.screen.signed, render.normal.world, render.normal_from_depth.world, render.pbr.gbuffer | near_plane 0.01 (runtime), roughness range [0.04,1], eval-mode quirks (inference T-fill, pad_normal) |
| gsir-density | density | ss_gsir.density:create_density | gradient.screen.signed | – | DefaultStrategy (3DGS schedule, grow_grad2d 2e-4) + opacity-reset fix, pbr_start 30000, stage1_steps 30000 (means-LR schedule), baking: occ_res 256, occ_bound 16, occ_threshold 0.4, occ_valid 1.5, occ_degree 4, cubemap_res 256, near 0.2; light: cube_res 256, irr_res 64, irr_degree 3, irr_bound 1.5, light_lr 0.05 |
| photometric | loss | ss_gsir.losses:create_color | render.rgb | – | stage 1: 0.8 L1 + 0.2 D-SSIM (runtime SSIM); stage 2: L1(PBR, gt) |
| gsir-normal | loss | ss_gsir.losses:create_normal | render.normal.world, render.normal_from_depth.world | – | weight 1.0, tv 5.0 (stage 1 only) |
| gsir-material | loss | ss_gsir.losses:create_material | render.pbr.gbuffer | – | brdf_tv 1.0, lamb 0.001 (stage 2 only) |
| gsir-light | loss | ss_gsir.losses:create_light | render.pbr.gbuffer | – | env_tv 0.01 on 512×1024 lat-long (stage 2 only) |

The base `photometric` loss is replaced (same name, so the runtime probe still applies) because
stage 2 uses pure L1 on the PBR image (official train.py:330).

## Fields (Gaussian-owned)
| field | shape | lr | init | notes |
|---|---|---|---|---|
| gsir_normal | (3,) | 0.05 | (0,0,1) set in density.initialize | normalised at use |
| gsir_albedo | (3,) | 0.05 | 1.0 (sigmoid → 0.731) | |
| gsir_roughness | (1,) | 0.05 | 1.0 | |
| gsir_metallic | (1,) | 0.05 | 1.0 | |
gsplat ops copy them on duplicate/split (official densification_postfix repeats them).

## Global state: component-owned bank (`ss_gsir.bank.GSIRBank`)
Owned by the density, published in `bank.ACTIVE`: stage flag, cube-map base [6,256,256,3]
(init U(0.25,0.75)), irradiance SH grid [64,64,64,9,1] (init 0, AABB ±1.5), baked occlusion
ids [256³] int32 + coefficients [n,16,1], Adam (lr 0.05) over cube+irradiance, cube clamp ≥0.
Exported through `density.state["gsir"]` (saved in checkpoint.pt) with a fingerprint
(N, Σ means). `ss-eval` builds a fresh renderer: it restores the bank from
`<--run>/checkpoint.pt` (argv of `ss-eval`/`splatstudio eval`, or env `GSIR_RUN`), verifies
the fingerprint and refuses to evaluate a stage-2 run without it.

## Stage logic
- Stage 1 (step ≤ pbr_start): pass A = gsplat RGB+ED (identical to the baseline renderer,
  white background); pass B = normals Σw·n with **detached geometry** (official backward sends
  no gradient from normal maps to α/means/cov). normal_from_depth = port of
  `depthmapToNormalCUDA` on the 3×3-median-filtered detached ED depth.
- After the optimizer step of `pbr_start`: density bakes occlusion volumes from the frozen
  Gaussians (gsplat batched 6-face renders, per-probe crop as opacity masks, near 0.2 = 3DGS
  frustum cull), allocates light state, stage = 2, strategy disabled.
- Stage 2: one gsplat pass with all geometry detached, 8 G-buffer channels (albedo, roughness,
  metallic, normal) + ED depth; PBR shading in torch (cube sampling, mips, split sum) + CUDA
  (specular pre-filter, SH occlusion reconstruction). Light Adam stepped in
  `density.after_backward`. Geometry / SH params get no gradient → Adam skips them.
- Means LR: density overrides the runtime's 0.01^(step/steps) with 0.01^(min(step,30000)/30000)
  (official position_lr_max_steps 30000).

## Custom CUDA (JIT, `ss_gsir/csrc/gsir_kernels.cu`, own implementation of the official math)
- GGX specular cube pre-filter: bounds kernel (tile culling), forward (Σ w·L, Σ w), backward
  (atomic scatter) — same weights as nvdiffrec `specular_cubemap` (cited, not vendored).
- SH occlusion reconstruction: 256 Hammersley cosine samples + curand jitter (seed 1234,
  pixel id), mean of clamp(SH, 0, 1) — port of GS-IR `SH_reconstruction_kernel`.
Everything else (cube sampling with seamless edges, diffuse convolution as a dense
1536×1536 operator, masked trilinear, irradiance trilinear, dilation, SH projection) in torch.
No nvdiffrast: its install into the shared env was refused; the needed ops are re-implemented.

## Probes
| probe | contribution | fails if |
|---|---|---|
| depth-linear-interp | C1 | depth is Σwd (vol. accum.) or peak; checks D = Σwd/Σw on a semi-transparent 2-Gaussian fixture |
| normal-from-depth | C2 | analytic tilted plane normal not recovered, or normal loss leaks gradient to geometry / misses gsir_normal |
| occlusion-baking | C3 | wall fixture: visibility toward the wall not ≈0 / opposite side not ≈1; masked trilinear ≠ independent formula; SH projection of constant ≠ analytic |
| cube-ops | C5 | lookup round trip, edge continuity, diffuse/specular convolution of a constant ≠ constant |
| pbr-shading | C5 | constant-environment closed form (diffuse + split-sum) mismatch; no gradient to materials/cube |
| indirect-volumes | C4 | trilinear irradiance ≠ independent formula; no gradient to coefficients; V=0 not using irradiance |
| stage-switch | all | at pbr_start the density does not bake / switch / step the light optimiser; geometry receives gradients in stage 2 |

## Fidelity gaps
1. gsplat rasterizer instead of the official fork: alpha clamp 0.999 vs 0.99, near plane 0.01
   vs 0.2 in training renders (baking uses 0.2), white vs black stage-1 background (target images
   are opaque) — same as our baseline; small effect.
2. Runtime samples views with replacement; optimizer steps on the final iteration.
3. Cube sampling / mip filtering re-implemented (no nvdiffrast): corner texels of the seamless
   cube filter are approximated by nearest-face padding. Expected effect < 0.05 dB.
4. Runtime metrics: PSNR on whole-image MSE (official: mean of per-channel PSNR), LPIPS on
   [-1,1] (official feeds [0,1]); SSIM valid conv.
5. Dilation of empty occlusion cells is deterministic (Jacobi) instead of the official racy
   in-place kernel; affects only cells without nearby Gaussians.
6. TensoIR-synthetic (normal/albedo/relighting metrics) not benchmarked (no data; Mip-NeRF 360
   per-scene NVS numbers are available).
