# AtomGS (plan #66) — build report

- Plugin `atomgs` (`reproductions/atomgs/plugin`, module `ss_atomgs`), port of official code
  RongLiu-Leo/AtomicGS @5b9ce904; official feature rasterizer vendored unmodified in
  `ss_atomgs/csrc` (Inria/MPII licence), JIT-built.
- Components: `atomgs-raster` (Inria kernel, mean/median depth, alpha, black bg),
  `atomgs-density` (atom scale P1 init + clamp, atomize ×0.99 every 100 it < 7k, split-first with
  warm-up threshold and max s > S, clone with the official index-shifted gradients, prune
  α<0.005 / >0.1·extent, opacity reset every 300 to 0.1, xyz LR over 7k),
  `atomgs-photometric` (0.9 L1 + 0.1(1−3-scale SSIM)), `edge-aware-normal` (λ 0.1, < 7k).
- Verify: declared_checks_passed (feature-raster kernel vs float64 compositor rel err < 1e-6;
  densify equals literal transcription of the official code).
- Smoke garden 2k steps: 23.73 / 0.757 / 0.245 (7.1M Gaussians, 11.3 GB, 536 s) vs gsplat
  baseline 23.07 / 0.643 / 0.418.
- Benchmark request: mipnerf360 garden (paper 27.44, its 3DGS 27.18) and room (31.30 / 31.22),
  30k, images_4/2; baseline3dgs-mipnerf360-30000 (26.89 / 31.52).

## Fidelity gaps
1. Optimizer step precedes density ops (official skips one Adam step per densify iteration).
2. View sampling with replacement; SH degree +1 step later.
3. Delta vs baseline includes gsplat/white-bg vs Inria-kernel/black-bg difference.
4. Resolution protocol not stated in paper; assumed 3DGS full_eval (images_4/2) from LPIPS.
5. No mesh / DTU chamfer evaluation.
6. Large populations (7.1M at 2k); bench extra raises the 12M guard to 20M.

## Review-1 resolution (orchestrator)
**Review outcome:** approved, with 0 critical and 0 major issues.

**Benchmark launched:** garden and room on `mipnerf360` (images_4/2), `--max-gaussians 15000000`, `--min-free-gb 16`.

**Notes recorded from the review:**
- **Densification schedule:** densification runs until step 15000 (official `densify_until_iter`). Only atomize, the threshold warm-up and the normal loss stop at 7000.
- **Resolution protocol:** inferred, because the repo has no eval script. The supplementary 3DGS LPIPS matches images_4/2.
- **Atom-scale initialisation:** uses exact KNN rather than the official approximate `distCUDA2`, so S0 may differ slightly.
- **Small expected gains:** the paper's gains on these two scenes are small (+0.26 on garden, +0.08 on room). Our baseline room (31.52) already exceeds the paper's AtomGS room (31.30), so absolute numbers matter here more than deltas.
