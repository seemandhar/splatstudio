# Deblurring 3D Gaussian Splatting — native SplatStudio plugin

Paper: Lee et al., ECCV 2024, https://arxiv.org/abs/2401.00834 · code:
https://github.com/benhenryL/Deblurring-3D-Gaussian-Splatting (ported, not executed).

Methods: `deblurring-3dgs-defocus` (configs/real_defocus.txt) and `deblurring-3dgs-motion`
(configs/real_camera_motion.txt). Train with `--recipe 3dgs --num-gaussians 10000000
--steps 20000` on `deblur3dgs-defocus` / `deblur3dgs-motion` scenes (the authors' re-released
data with re-run COLMAP, built by `../tools/prepare_official.py`).

Mechanism: a small MLP (bank.GTnet, in the density-owned `bank.BlurBank`, own Adam lr 1e-3,
exported to `density_state["bank"]`) maps detached (x, s, q) and the camera centre to
per-Gaussian factors in [1, 1.1] (and position offsets for motion). Training renders use the
transformed Gaussians (Σ built from the *unnormalised* q·δr, as the Inria rasterizer does);
motion averages 5 renders. ss-eval renders the original Gaussians. Extra KNN-coloured points
are added at step 2500 (all Gaussians re-initialised, fresh Adam), opacity pruning is
depth-scaled, no opacity reset. See `../plan.md` for the fidelity-gap list.
