# splatstudio-method-gaussianpop

GaussianPOP (Lee et al., arXiv 2602.06830, Fraunhofer HHI), post-training scenario, native
SplatStudio port. Paper fidelity unverified. No official code exists (project page "Coming Soon").

Mechanism: per-Gaussian removal error ΔSE_k = ||T_k α_k (c_k − b_{k+1})||² (Eq. 4),
b_{k+1} = (C_render − P_k)/(T_{k+1}+ε) (Alg. 1, ε=1e-9), summed over all pixels of all
training views by a JIT CUDA kernel (`csrc/pop_error.cu`, adapted from gsplat 1.5.3
RasterizeToPixels3DGSFwd.cu, Apache-2.0, via the AuGS plugin's contrib_stats.cu). Pruning:
8 re-quantification cycles at step 30000 (compound split, keep_c = N0(1−P)^{c/C}), then 5k
fine-tune with fresh Adam and the 3DGS final means LR.

Methods: `gaussianpop` (P=0.8), `gaussianpop-p07`, `gaussianpop-p09`; run with `--steps 35000`
(`--steps 30000` = post-prune without fine-tune). Config keys in `ss_gaussianpop/density.py`
`DEFAULTS` (paper values; sources in ../paper.md).

Fidelity gaps: gsplat base model (white bg); per-cycle split, N_max semantics, fine-tune LR and
background in C_render are unstated in the paper (choices in ../plan.md); on-training mode not
ported.

Smoke: `tools/ssenv python ../tools/smoke_accelerated.py <scene> <out>`.
