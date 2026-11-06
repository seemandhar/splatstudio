# splatstudio-method-deblurgs

DeblurGS (Oh et al., ECCV 2024, arXiv 2404.11358). This is a native SplatStudio port of the
official code, github.com/taekkii/deblurgs @e0587475. Paper fidelity is unverified.

## Mechanism
- **Trajectories:** each training image gets a 9th-order Bezier trajectory in se(3). It uses
  pytorch3d exp/log maps, vendored verbatim in `ss_deblurgs/pytorch3d_se3.py`, BSD licence.
- **Sub-frame times:** 21 sorted times, with learnable inner times ν and fixed end points.
- **Blur synthesis:** the training view is rendered at all 21 poses in one batched gsplat call
  over a random background, and the renders are averaged in linear space.
- **Loss:**
  - L1 against GT^2.2;
  - a temporal smoothness term, mean |Δ sub-frame|, with λ going 1e-3 → 1e-5;
  - an opacity hinge term, 0.1.
- **Densification annealing:** the threshold θ goes 4e-4 → 2e-4 over 25k steps; densification
  runs every 200 steps until 75k.
- **Official activations:** opacity is the raw value clamped to [0, 1] (straight-through cap at
  0.999), max SH degree is 2, and the near plane is 0.2.
- **Trajectory state:** it lives in `ss_deblurgs.bank.ACTIVE`, owned by `deblurgs-density`, and
  is saved in `density_state["bank"]`.

## Evaluation
- **Sharp render:** it uses a white background and γ = x^(1/2.2). This is what `ss-eval` reports
  (unaligned).
- **Paper protocol (official test.py):** test-pose post-optimisation for 2000 epochs, then PSNR,
  SSIM (same-padded) and LPIPS-Alex. Run it with
  `python -m ss_deblurgs.evaluate --run RUN`, which writes `test_tpo.json`.

## Methods
| Method | ν learning rate |
|---|---|
| `deblurgs` | 3e-3 (paper value), on the code's schedule: 0 until 30k, then decay to 1e-7 at 150k |
| `deblurgs-nu0` | 0, the released-code default |

## Config sources
The defaults are the official `arguments/__init__.py` values. The paper↔code differences are
listed in `../paper.md` and `../report.md`.

## Commands
```
tools/ssenv ss-train deblurgs --data ~/datasets/deblurnerf/native-motion/ball --output RUN --steps 150000 --recipe 3dgs --num-gaussians 10000000
tools/ssenv ss-eval --run RUN --split test
tools/ssenv python -m ss_deblurgs.evaluate --run RUN
```
