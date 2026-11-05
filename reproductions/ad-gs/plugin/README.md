# splatstudio-method-ad-gs

Native SplatStudio port of **AD-GS: Alternating Densification for Sparse-Input 3D Gaussian
Splatting** (Patle et al., SIGGRAPH Asia 2025, arXiv 2509.11003). Written against the paper and
the official code (https://github.com/gurutvapatle/AD-GS, built on CoR-GS/FSGS/3DGS).
Paper fidelity unverified.

## Mechanism → components
| paper | component | notes |
|---|---|---|
| two co-trained fields (CoR-GS) | `adgs-density` owns G2; `adgs-cotrain` renders/back-props it | G2 cloned from the initial G1 state; only G1 is evaluated/saved |
| alternating densification (§3.3, Table 7) | `adgs-density` | warm-up ≤1500: τ 5e-4 / ε 0.005; then `step % 200 < 100` high: τ 2e-4 / ε 0.005, else low: τ 5e-4 / ε 0.1; every 100 steps in (500, 10000); FSGS extra split; opacity reset to ≤0.05 at 3501/6501/9501 |
| L_ph (Eq. 5) | built-in `photometric` (G1), `losses.photometric` (G2) | λ_ssim 0.2 |
| edge-aware depth smoothness + range (Eq. 7) | `adgs-depth` (G1), inside `adgs-cotrain` (G2) | official code form: ω1 0.01, λ_r 0.001, GT-image edge weights, means over pixels; low phase only |
| pseudo-view consistency (Eq. 6) | `adgs-cotrain` | random LLFF pseudo poses (`generate_random_poses_llff` port), detached partner target, λ3 = 1; low phase only, step ≥ 500 |
| random training background, black test background | `adgs-renderer` | depth = Σ z α T (RGB+D) |

## Data
Few-view LLFF scenes are built by `reproductions/ad-gs/tools/prepare_llff_fewview.py`
(FSGS/CoR-GS split, PIL bicubic ×4 downscale, training-view-only COLMAP triangulation + MVS
`fused.ply` init, as the official `tools/colmap_llff.py`).

## Run
```bash
tools/ssenv ss-train ad-gs --data ~/datasets/llff/fewview3/horns --output runs/x \
  --steps 10000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test
```

## Fidelity gaps
See `reproductions/ad-gs/plan.md` / `report.md`: SH degree every 1000 (official 500) steps;
pseudo-pose near/far bounds from the init cloud instead of `poses_bounds.npy` (+6 % focus
distance on horns); paper's ω2 pseudo-view depth term not used (disabled in official code);
runtime SSIM (valid conv.); step-before-densify ordering; different RNG streams.

## Iteration 1
- Pseudo poses use the official LLFF `poses_bounds.npy` near/far (density `pseudo_bounds:
  official`, per-scene `scene_bounds` table matched by training-camera centres; values from
  `prep.json`). `pseudo_bounds: points` restores iteration 0.
- Renderer applies the official SH schedule min(t // 500, 3) during training (counts autograd
  renders) and degree 3 at evaluation (config `sh_interval`, `max_sh_degree`, `eval_sh_degree`).
- Both remaining gaps above are therefore closed; probes `pseudo-view-consistency` and
  `sh-schedule` fail on a mismatch.
