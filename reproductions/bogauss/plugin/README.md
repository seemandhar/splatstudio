# splatstudio-method-bogauss

BOGausS: Better Optimized Gaussian Splatting (Pateux et al., arXiv 2504.01844), native
SplatStudio port written from the paper — **no official code exists**. Paper fidelity unverified.

Method `bogauss` = "Ours (Big)": final count = vanilla 3DGS count of the scene (Taming 3DGS big
budgets, looked up by initial SfM point count; set density config `budget` elsewhere).

## Mechanism → code
| paper | code |
|---|---|
| Precision Δ_i = α tr(Σ⁻¹)^-½, α=2 (§Precision) | `core.precision_delta` (σ_uv = 1 px, 1.3× frustum) |
| Unbiased sparse Adam, per-Gaussian t (Eq. adamNewState) | `optim.BOGausSAdam` |
| Δ-scaled position / linear size updates, minimal size Δ | `BOGausSAdam(mode="position"/"size")` |
| State inheritance α_t=1, α_g=0.2 | `ops.split` |
| Effective-opacity pruning τ=0.02, ≤1 % | `kernels.pixel_stats` (CUDA `csrc/bogauss_stats.cu`), `core.prune_selection` |
| Density-preserving split ±0.3σ, SplitAdaptation | `ops.split`, `core.split_adaptation_table` |
| P_SNR = Σ_c SE_ci / SE_c | `density._accumulate`, `core.snr_priority_increment` |
| Densify every 500 its, budget growth | `core.count_target` (Taming parabola), 1000..15000 |

## Choices for unspecified details (fidelity gaps)
- η of the Δ-scaled updates: `lr_means`/`lr_scales` = "auto" → median initial step equals
  3DGS's (1.6e-4·extent, 0.005 relative); the runtime's ×0.01 decay applies to means.
- σ_uv = 1 pixel; cameras = training views whose frustum contains the centre.
- S_i = contributing pixel count; statistics from the views rendered during training.
- SplitAdaptation = offline 1-D alpha-profile fit (children only change along the split axis).
- Growth curve and window 500..15000 from Taming 3DGS; pruning only in the window.
- Other learning rates, loss, SH schedule: 3DGS defaults. Exposure correction not ported.

## Commands
```bash
pip install -e .
ss-inspect bogauss && ss-verify bogauss --device cuda
ss-train bogauss --data ~/datasets/tandt_db/native/truck --output runs/x --recipe 3dgs --num-gaussians 10000000
```
