# CRiM-GS review, round 1

**Verdict: approved for benchmarking.** No critical or major issues. Six minor issues, listed below.

## The red flag: densification nearly stops after step 1000

**This is not a bug. It matches the CoMoGaussian code exactly.**

The counts in `evidence/smoke-ball/training.jsonl`:

| steps | Gaussians added per 100 steps |
|---|---|
| 900 → 1000 | +11.7k |
| 1000 → 1100 | +65 |
| 1100 → 2000 | +65 to +120 |

The stall begins exactly when the trajectory starts (step > 1000).

**Why it happens.**
- The densification statistics come only from the sharp render: `renderer.py:58-61` passes the sharp `info` to `density.py:187,192`.
- Between steps 1001 and 3000, `rgb = mean(N renders + sharp)` (`renderer.py:77`). So the sharp render's `means2d` gradient is exactly 1/(N+1) = 1/10 of what it would be without blending.
- With a heavy-tailed gradient distribution, a 10× smaller gradient cuts the number of Gaussians above the 2e-4 threshold by about 100×. That matches the observed drop.

**What CoMo does** (`source/comogaussian/train.py`):
- `viewspace_point_tensor`, `visibility_filter` and `radii` are bound at l.141-142, from the sharp render only.
- The warped renders at l.153 overwrite `render_pkg`, but none of those three variables.
- `add_densification_stats(viewspace_point_tensor, …)` at l.200 therefore uses only the sharp gradient.
- The blend at l.164-166 is `torch.mean` over N + 1 images.
- `gaussian_renderer` creates a fresh `screenspace_points` with `retain_grad` for every call.

CoMo therefore has the same 1/10 scaling and the same stall. After step 3000 the sharp weight becomes (1 − M), with M ≈ 0.5 at init and pushed down by L_M, so growth should partly resume.

**The suspected bugs, checked one by one:**
- **Batched `[C,N,2]` normalisation:** not involved. The 9-camera call's `info` is discarded (`renderer.py:70`). The density sees a single-camera `info` (C = 1), and gsplat's `×W/2·n_cameras` scaling then equals Inria's NDC gradient.
- **Gradient retention:** `step_pre_backward` retains the gradient on the sharp `means2d`. Growth is small but non-zero, so the gradients do arrive.
- **absgrad:** off, which matches Mip-Splatting at the paper-era commit and CoMo.
- **Reference ports:** `deblurring-3dgs` feeds batched `info` (all renders) to the density, and BAGS accumulates from a single render. CRiM/CoMo is the single-render case.

**Recommendation.** Keep the current behaviour (faithful). In the 40k runs, log the Gaussian count at 3k, 5k, 10k and 20k, and state the stall in `report.md`. If ball or cozy2room comes in well short of the target and the count stays at ≈50k after 3k, try one ablation run that accumulates statistics from all N + 1 renders, and label it as a deviation.

## Other checks (all OK)

| item | result |
|---|---|
| **dopri5** (`ode.py`) | <ul><li>Tableau (A, B5, B4, FSAL) verified.</li><li>Error against exact solutions:<ul><li>y' = cos(t)·y over 0..8: 8.5e-7 relative error at rtol 1e-6;</li><li>backward integration (y' = y² from 4 to 0, and y' = −y): 1.4e-7 and 6.6e-7;</li><li>gradient through the solver against `matrix_exp`: 1e-9.</li></ul></li><li>Observed order 5.3–5.4.</li><li>Accepted steps: 28, against 29 for SciPy RK45 at the same tolerances.</li></ul> |
| **Time grid** | <ul><li>τ = 0..8, integrated forward from τ0: this is the paper's text ("from τ0 to τs").</li><li>CoMo instead integrates ±4 from the middle pose (`como_kernel.py:139-148`) and uses rk4 on the integer grid, so its rtol/atol are unused.</li><li>The gap is documented; see minor issue m1.</li></ul> |
| **SE(3)** | <ul><li>`kernel.py:382` computes `inv(c2w · T_r · T_a)`, which is Eq. 1 with T = c2w (right multiplication, camera frame).</li><li>Rodrigues (Eq. 9) and G (Eq. 10) in `se3.py` match CoMo l.314-332 term by term.</li><li>The axis is normalised with +1e-10, as CoMo does.</li></ul> |
| **CNN and mask** | <ul><li>Structure follows the CRiM appendix text: 3 conv layers, 64 channels, 5/3/3 kernels, ReLU + InstanceNorm.</li><li>Weight head: 1×1 conv, softmax over N. Mask head: 1×1 conv, averaged over N, then sigmoid.</li><li>The CNN sees the renders detached; blend is `(1−M)·sharp + M·ΣP·I` (`renderer.py:72-74`).</li><li>All of this matches CoMo l.158-160, which also uses the CNN for the weights.</li></ul> |
| **Schedule** | <ul><li>Trajectory starts at step > 1000; pixel weights and mask at step > 3000.</li><li>Kernel Adam steps only when step > 1000.</li><li>Kernel lr starts at 1e-4 and drops by (1e-4 − 1e-6)/40000 per step. This matches CoMo l.217-221 and `adjust_lr`.</li></ul> |
| **Loss** | 0.7·L1 + 0.3·D-SSIM + 1e-4·L_o + 1e-3·mean(M). Matches CoMo l.175. |
| **Mip-Splatting** | <ul><li>3D filter: coef 0.2, from training cameras only.</li><li>Filter recompute steps (`filters.py:333`, limits 500 < s < 20000, then 20000 < s < 39900) equal CoMo l.202-212.</li><li>2D filter: gsplat antialiased with eps2d 0.1. Background black.</li></ul> |
| **Densification** | <ul><li>gsplat `refine_stop` uses `step >= 20000`, CoMo `iteration < 20000`: these agree.</li><li>No reset and no size prune (CoMo passes `size_threshold=None`).</li><li>The opacity prune uses the raw sigmoid, as CoMo's `get_opacity` does.</li></ul> |
| **Dataset, synthetic** | <ul><li>cozy2room raw: 34 images at 600×400; `images` and `images_1` are byte-identical.</li><li>Laplacian variance:<ul><li>idx % 8 == 0 (000, 008, 016, 024, 032): 666–810, the sharp images;</li><li>all others: 43–386, blurry.</li></ul></li><li>Split: 29 train (blurry only), 5 test (sharp). Validation is copies of test.</li></ul> |
| **Dataset, real** | ball: 600×400; test = 000/007/014/021 from the release's hold file; 27 train. |
| **requires_grad pitfall** | <ul><li>Densification goes only through gsplat ops.</li><li>A leaf / requires_grad guard runs on every field after every step (`density.py:200-202`).</li><li>`filter_3d` is only ever read with `.detach()`.</li></ul> |

## Minor issues

- **m1. Latent growth over τ = 0..8** (`kernel.py:347-358`).
  - Measured at init: ‖z_r‖ grows 5 → 16 → 91 at τ = 0 / 4 / 8, and z_a reaches up to 119. Each ODE takes about 15 adaptive steps.
  - Cause: the default `nn.Linear` init in an autonomous ReLU-linear f, integrated for 8 time units, gives roughly e^{0.4·8} growth. CoMo integrates 4 units each way from a zero-initialised embedding (`como_kernel.py:165`), so its growth is about 5×.
  - Effect: the late poses are about 20× more sensitive than τ0. The ODE may stiffen during training (more steps, slower runs, more GPU syncs from `float()` in `ode.py:51`).
  - The paper's text permits this choice. Monitor `ode_accepted` per call in the kernel export. If it blows up, use a centred grid or a zero embedding init (as CoMo) and label it.
- **m2. f is autonomous** (`kernel.py:259`).
  - The paper writes f(z, τ; φ). CoMo adds a learned per-time embedding (`como_kernel.py:89-108`).
  - The gap is documented (plan gap 2). Keep it in `report.md`.
- **m3. The 1/(N+1) densification stall** (above) is faithful, but `report.md` must state it, with Gaussian counts from the full runs.
- **m4. L_o uses the Frobenius norm** averaged over the N poses (`losses.py:16-18`). CoMo uses mean |R Rᵀ − I|, which differs by about a 3–9× scale. With weight 1e-4 the effect is negligible. Already documented.
- **m5. Embedding initialisation.** `nn.Embedding` uses the default N(0,1) init; CoMo initialises it to zeros and concatenates `linear_Rt(Rt)`. CRiM's text describes only the embedding, so this is acceptable. The probe tolerance was relaxed to 5e-3 because of the N(0,1) init.
- **m6. Background bookkeeping.** `config.json` records `protocol.background: "white"` while the renderer uses black. This is harmless for RGB-only targets, but the report should note it.

## Probe assessment

The probes would fail if their mechanism were removed:
- **dopri5:** includes an Euler ablation.
- **screw-se3:** the G·v vs v·θ swap check.
- **kernel-poses:** checks that the composition order is observable.
- **blur-synthesis:** independent gsplat renders at steps 1000, 2000 and 4000.
- **density-schedule:** no reset and the exact filter steps.

**Gap:** no probe asserts where the densification statistics come from (sharp only). That is the decision the red flag questions. A small probe could compare the strategy's `grad2d` after one step against an independent sharp-only `means2d` gradient.
