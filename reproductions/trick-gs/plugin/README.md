# splatstudio-method-trick-gs

Native SplatStudio port of **Trick-GS: A Balanced Bag of Tricks for Efficient Gaussian
Splatting** (Armagan et al., ICASSP 2025, https://arxiv.org/abs/2501.14534). No official code is
released; the port follows the paper and, where it is silent, the official code of the paper each
trick is adopted from. Paper fidelity unverified.

Method `trick-gs` = the headline "Trick-GS" rows (all eight tricks, Table 3 caption).

| trick | where | values (source) |
|---|---|---|
| GM volume mask (Compact-3DGS) | renderer STE on scale+opacity, `gaussian-mask` loss, density prune | ε_m = λ_m = 0.05, lr 0.5 (§4.2), init logit 1 (Compact-3DGS) |
| SG significance (LightGaussian) | `significance.py` + CUDA count kernel (from the LightGaussian port) | steps 5K/10K/15K/20K/22K/23K (Fig. 3), rate 0.6·0.7^k (§4.2), v_pow 0.1 |
| SHM SH masks (RDO-Gaussian) | renderer per-band STE, `sh-mask` loss, baked at the end | ε_sh 0.1, λ_sh 0.05, lr 0.05 (§4.2), init 0, start 3500 (Fig. 3) |
| BL progressive blur (EAGLES) | photometric target | 9×9, σ 2.4, every 100 its to 19.5K, scaled by DS factor |
| DS progressive resolution (Eq. 6) | renderer K scaling + target downsampling | 1/8 → 1 linearly to 19.5K |
| SC progressive scale (RAIN-GS) | renderer eps2d, density ABE split, 20 % lowest-error SfM init | s = max(HW/9πN, 0.3) every 1000 its, ≤10K |
| AT (Taming 3DGS) | `IntervalAdam` for shN (16 its), separable SSIM | |
| DE late densification | density | every 100 its in (20K, 20.5K] |
| FP16 storage | density at the last step | |

The 20 % lowest-error init needs per-point COLMAP errors, which the native `points.npz` does not
store: `make_sfm_errors.py` writes `../sfm_errors/<sha256(xyz)>.npy` (done for garden, bicycle,
counter, truck, train, drjohnson, playroom).

```bash
T=/home/sejain/splatify/tools/ssenv
$T pip install -e . && $T ss-inspect trick-gs && $T python -m pytest -q tests
$T ss-verify trick-gs --device cuda
$T ss-train trick-gs --data ~/datasets/tandt_db/native/truck --output runs/x --steps 30000 \
   --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```

Fidelity gaps: see `../plan.md` and `../paper.md` §7 (SG steps from Fig. 3, blur decay law, mask
inits, SH-mask start, DS linear vs "logarithmic", white background / random view sampling).
