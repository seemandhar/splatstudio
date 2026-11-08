# ConeGS — review 1

**Verdict: not approved yet. 0 critical, 1 major (integration, not plugin code), 6 minor.**
The plugin follows the official code (baranowskibrt/conegs @ f86df2f) closely. I found no
fidelity bug in the iNGP, the init, the cone densification, the budget logic or the losses. The
one blocker: the benchmark cannot be launched until the dataset copy is registered in
`tools/datasets.json`.

Checked against: `source/code/train.py`, `scene/nerf_model.py`, `scene/gaussian_model.py`,
`utils/{ngp,nerf_utils,ray_utils}.py`, `scene/{__init__,cameras}.py`, configs; nerfacc 57ccfa1
(`cuda/csrc/pdf.cu`, prop_net); tiny-cuda-nn 3daa6e5 (`common_device.h`). Runtime:
`splatstudio/runtime.py`, `state.py`, `data.py`, `backends/gsplat.py`, gsplat 1.5.3 `strategy/ops.py`.

Checks I ran:
- pytest: 8 passed.
- Hash-grid layout computed: per-level resolutions are 16…4096 (radiance field), 16…128 and
  16…256 (proposals). Levels with res³ ≤ 2^19 are dense, the rest hashed. This matches tcnn
  `grid_scale`/`grid_resolution`.
- smoke-1 checkpoint state: 402,000 rays cast in 2000 steps (201/step, none invalid); 107,733
  pruned and 107,733 refilled; N stays at 1,000,000; camera extent 4.92.
- Protocol: the copy has the same protocol_id (53deee9d…) as `baseline3dgs-mipnerf360-30000`
  garden, and the same 24 test views in the same order. Images are hard links (same inode).
  cx = W/2 and cy = H/2 in transforms.json, so the official pinhole assumption holds.

## Issues

| sev | where | problem | evidence | fix |
|---|---|---|---|---|
| **major** | `tools/datasets.json`, `bench_request.json` | `mipnerf360-conegs` is not registered. `bench.py launch --dataset mipnerf360-conegs` fails in `scenes_for` (`DATASETS[dataset]` KeyError). | `bench.py:41-52`; keys listed in datasets.json have `mipnerf360-edgs` but no conegs entry | Add the `proposed_datasets_json_entries` block (native_root `~/datasets/conegs-init/mipnerf360`, scenes garden: images_4, room: images_2). Same pattern as `mipnerf360-edgs`. |
| minor | `bench_request.json` `extra` | `"--recipe splatstudio"` is passed through `--extra`. bench.py also adds its own `--recipe 3dgs` (the default). argparse takes the last value, so the run is correct, but jobs.json records `recipe: 3dgs`. | `bench.py:195-197` | Use the bench flag `--recipe splatstudio` and leave `--extra` empty. |
| minor | prerequisite | The room copy does not exist yet. The init job is running (pid 226948, at iNGP iteration ~10k). | `ls ~/datasets/conegs-init/mipnerf360/room` → missing | Wait for `room/conegs_init.json` before launching. Otherwise launch fails on the transforms.json check. |
| minor | gap analysis | The paper's 3DGS row is 3DGS capped at 1M primitives (garden 26.70, room 31.62). Our `baseline3dgs-mipnerf360-30000` is unbudgeted (garden 26.89 with 4.8M Gaussians, room 31.52 with 1.7M). A baseline-adjusted gap against it is not like for like. | `paper_results.json` notes | Report the absolute gap to paper ConeGS-1M as the primary number. Treat the unbudgeted-baseline delta as context only, or add a 1M-capped gsplat baseline. |
| minor | `density.py:156-158` (`error_map`) | The error map is computed on the white-composited render. The official code uses a black background (`train.py` `l1_full`). | runtime convention; Mip-NeRF 360 renders have alpha ≈ 1 almost everywhere | Already listed as gap 2. Effect should be negligible. |
| minor | `probes.py:hashgrid` | The kernel is compared with a reference that shares the same index logic, so the probe cannot catch a tcnn misunderstanding common to both. It does check the level layout (16/4096, 2^19). | I checked by hand against tcnn `grid_index` (MAX_BASES[3] = 0x659, dense iff res³ ≤ T), `coherent_prime_hash` (1, 2654435761, 805459861), `pos_fract` (x·scale + 0.5, floor) and per-level counts (next multiple of 8, min with 2^T): all match | Optional: add a hard-coded index from tcnn for one hashed and one dense vertex. |
| minor | `preprocess.py:56` | Exact 3-NN init scales versus approximate `distCUDA2` in the official code. iNGP runs in fp32 instead of tcnn fp16. | gaps 4 and 5 in plan.md | Already documented. Keep them in report.md. |

## Fidelity by item (all exact unless noted)

- **iNGP.** Radiance field: 16 levels, T = 2^19, resolution 16→4096 (per-level scale from
  max_res, as in `ngp.py`, not the yaml's 1.45), bias-free 64×1 base MLP → 1+15, SH4 + 15 → 64×2
  → sigmoid, `trunc_exp(x−1)`, selector, unisphere ℓ2 contraction. Proposal fields: 5 levels,
  T = 2^17, 16→128 and 16→256. Sampler: the nerfacc `importance_sampling` kernel line for line
  (u_step = Δcdf/n, per-ray bias, upper_bound on cdf[:-1], clamped midpoint edges), lindisp
  near 0.2 / far 1e6, prop [256, 96] + 48, opaque last sample. Training: `_pdf_loss`,
  `get_proposal_requires_grad_fn(5, 1000)`, prop scheduler stepped every iteration (as in
  `update_every_n_steps`), smooth-L1 ×1024 without unscaling (the weight-decay interplay is
  reproduced), Adam(1e-2, eps 1e-15, wd 1e-5), LinearLR + MultiStep (10k/15k/18k, ×0.33),
  4096 uniform rays over the stacked training images, 20k iterations (official effectively
  19,999; negligible).
- **Normalisation and rays.** `get_gs2nerf_transform` matches, using all poses. The official
  OpenGL flip of c2w and the flip of the ray directions cancel, so the plugin's
  R_align·c2w·K⁻¹[u+.5, v+.5, 1] is equivalent. Origin = s·(R_align t + translate).
- **Median depth and cone.** Inclusive transmittance < 0.5, using `argmax`; point at the sample
  midpoint; `alpha > 0` validity mask; colour = volume-rendered colour with zero view
  directions (SH4 of the zero vector, as tcnn gives for (0+1)/2); std = radii·t/‖d‖/scale ×
  1.0. This matches `select_for_densification` plus `pixels_to_rays`.
- **Init.** Rays are sampled until ≥ 1M valid, then a uniform 1M subset is taken (multinomial
  over unit weights in the official code; randperm here). Scales are 3-NN within the 1M set
  (exact; the official uses distCUDA2), not clamped. Opacity 0.1, identity rotation, DC
  clamped to [0,1], SH rest 0. No SfM points; the official code also never calls
  `create_from_pcd`.
- **Densification.** Pixels are drawn with multinomial sampling without replacement, weighted
  by mean_c|I−I*|, at each step 0 < s < 25000. Every 100 steps: prune opacity ≤ 0.005 (only
  s < 25000), then merge, truncated to the first budget−N stash rows in chronological order.
  `get_num_rays_to_cast` is exact (growth 0.02 from the code; the paper text says 0.2;
  N_added only with a budget; minimum 10; initial 201). Merges after 25000 match the official
  behaviour. Ordering matches the official `train.py`: render → backward → `optimizer.step` →
  cast → prune/merge (runtime.py:200-211).
- **Budget enforcement.** The density gets `budget: 1_000_000`. Init (1M) raises an error if it
  exceeds the budget, and each merge caps N at the budget. The runtime `--max-gaussians 12M`
  from bench.py never binds. The paper's budgeted mode is `optimization.max_points=1000000`;
  this is equivalent.
- **Opacity penalty.** 2e-4 × mean(logits), signed, matching `train.py:110-111`. The probe
  distinguishes it from the post-activation penalty.
- **SH.** All bands from step 1: the renderer ignores the runtime degree. This matches
  `start_sh_degree=3` (the `oneupSHdegree` calls in the official loop are no-ops).
- **LRs.** 3DGS rates. The means LR is rescaled to the 3DGS camera extent: `initial_lr` is
  read after `set_training_cameras`, so the runtime decay keeps the extent-scaled value.
  `0.01^(s/T)` equals the 3DGS exponential with no delay.
- **requires_grad pitfall.** `append_gaussians` builds each Parameter from `p.detach()` and
  copies `requires_grad`. gsplat `remove` preserves `requires_grad`. A trainable-leaf assertion
  runs after every merge. New rows get zero Adam moments. Correct.

## Probes
All of these would fail if their mechanism were removed: cone-placement (quadrant error, pixel
centres, depth, size_mult 2 fails), median-depth (analytic profile), budget (refill count,
oldest rows, Adam zero rows, ray formula), opacity-penalty (gradient λ/N, differs from the
post-activation penalty), renderer-sh, ingp-fit (whole pipeline on a sphere). The
proposal-sampler probe uses the nerfacc docstring cases. The hashgrid probe is weaker (see
minor issue above).

## Launch (after registering the dataset and once the room copy exists)
```
mkdir -p /bigdata/sejain/runs/conegs-mipnerf360-conegs && ln -s /bigdata/sejain/runs/conegs-mipnerf360-conegs runs/conegs-mipnerf360-conegs
python3 tools/bench.py launch --exp conegs-mipnerf360-conegs --methods conegs \
    --dataset mipnerf360-conegs --scenes garden room --steps 30000 --recipe splatstudio
```
bench.py itself adds `--num-gaussians 10000000 --max-gaussians 12000000`. Add the iNGP and
init seconds from each `conegs_init.json` (garden 492 s) to the training time.
