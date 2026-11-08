# splatstudio-method-efa-gs

Native SplatStudio port of **EFA-GS** — *Low-Frequency First: Eliminating Floating Artifacts in
3D Gaussian Splatting* (Wang et al., Cyberworlds 2025, [arXiv 2508.02493](https://arxiv.org/abs/2508.02493)),
following the official code <https://github.com/jcwang-gh/EFA-GS> (commit 57f330f,
`Mip-Splatting/` variant = the paper's default EFA-GS(Mip); Inria Gaussian-Splatting license,
non-commercial research use). Mip-Splatting filter functions are vendored from
`reproductions/mip-splatting` (`ss_efa_gs/mip_filters.py`).

Methods:
- `efa-gs`: Mip-Splatting (3D filter, 2D mip filter 0.1, filter-aware reset, GOF absolute-gradient
  densification) + LFCF every 2nd densification (c_max 1.5, c_min 1, depth & scale strategies,
  linear log-factor annealing to 15k, depth-dependent split probability, N = 1 splits).
- `efa-gs-base`: identical with LFCF off — the official EFA repo's Mip-Splatting base (control).

Config values and their code sources are in `ss_efa_gs/__init__.py`; the full spec and the paper
vs code discrepancies (c(i) labels, annealing direction) in `../paper.md`; fidelity gaps in
`../plan.md` (black background as official, configurable via renderer `background`; with-replacement view sampling, densify after optimizer step,
gsplat rasterizer statistics, RNG streams).

```bash
tools/ssenv pip install -e reproductions/efa-gs/plugin
tools/ssenv ss-verify efa-gs --device cuda
tools/ssenv ss-train efa-gs --data ~/datasets/mipnerf360/pilr/garden --output runs/x --recipe 3dgs \
  --num-gaussians 10000000 --max-gaussians 12000000
```
