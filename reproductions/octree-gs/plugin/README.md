# splatstudio-method-octree-gs

Native SplatStudio port of **Octree-GS** (Ren et al., *Octree-GS: Towards Consistent Real-time Rendering with
LOD-Structured 3D Gaussians*, TPAMI 2025, [arXiv 2403.17898](https://arxiv.org/abs/2403.17898)), the
**Our-Scaffold-GS** variant of the official code ([city-super/Octree-GS](https://github.com/city-super/Octree-GS)
@ `9db64fe`). Paper fidelity unverified until benchmarked.

## Mechanism → code
| contribution | code |
|---|---|
| Scaffold backbone (anchors, 32-D feature, 10 offsets, 6-D scaling, MLP decode, vol. reg.) | `renderer.neural_gaussians`, `model.scaffold_mlps`, `losses.py` |
| C1 octree LOD anchors (K from distance quantiles, per-level voxelization, weed-out) | `octree.set_level / octree_box / octree_sample / weed_out`, `density.build` |
| C2 view-dependent LOD selection with LOD bias ΔL (Eq. 3, code `round`) | `octree.anchor_mask`, `renderer.render` |
| C3 next-level growing, ΔL update, opacity pruning, view-frequency filter | `density.anchor_growing / adjust_anchor` |
| C4 progressive training (start ⌊K/2⌋, geometric intervals ω=1.5 over 10k) | `octree.coarse_intervals / coarse_index` |

Design (from the HAC / ContextGS native ports): anchors are the native population; per-anchor fields `ogs_feat`,
`ogs_offset`, `ogs_scaling`, `ogs_level`, `ogs_extra_level`; the MLPs + octree scalars live in a process-local bank
owned by the density (`model.ACTIVE`) and are packed into the per-anchor `ogs_blob` field after every step so that
`ss-eval` (fresh process, no component state) can render. Scaffold's `visible_filter` is re-implemented in torch.

Custom CUDA: `simple-knn` (`distCUDA2`, initial anchor scaling) vendored **unmodified** from the official repo
(`ss_octree_gs/csrc/simple_knn`, Inria Gaussian-Splatting licence, non-commercial research use), JIT-built with
`torch.utils.cpp_extension` through `tools/ssenv` (`-include cfloat` for FLT_MAX). The official repo's rasterizer
(Scaffold fork of `diff-gaussian-rasterization`) is **not** used; rendering is gsplat.

## Config (defaults = official mip360 script, `ogs-density` config)
`total_steps 40000`, `base_layer 10`, `visible_threshold -1` (adaptive), `fork 2`, `update_ratio 0.2`,
`extra_ratio 0.25`, `extra_up 0.01`, `dist_ratio 0.999`, `progressive True`, `coarse_iter 10000`,
`coarse_factor 1.5`, `start_stat 500`, `update_from 1500`, `update_interval 100`, `update_until 25000`,
`min_opacity 0.005`, `success_threshold 0.8`, `densify_grad_threshold 0.0002`, LRs of `arguments/__init__.py`.
T&T: `visible_threshold 0.9`; DB: `base_layer 12, visible_threshold 0.9` (not registered as presets).

## Fidelity gaps
gsplat rasterizer (+ 0.999 opacity straight-through cap) instead of the Inria/Scaffold fork; white background
(official black); random-with-replacement view sampling and valid-convolution SSIM; runtime Adam step before the
density update; MLP weights transported through `ogs_blob`; paper-vs-code choices follow the code (`round` LOD without
opacity blending, adaptive τ_v, next-level growing during progressive training — `~True` quirk, update_until 25000).

## Commands
```bash
T=/home/sejain/splatify/tools/ssenv
cd reproductions/octree-gs/plugin && $T pip install -e . && $T ss-inspect octree-gs && $T python -m pytest -q tests
$T ss-verify octree-gs --device cuda
python3 tools/bench.py launch --exp octree-gs-mipnerf360-1600 --methods octree-gs --dataset mipnerf360-1600 \
    --scenes garden counter --steps 40000
$T python reproductions/octree-gs/plugin/report_counts.py runs/octree-gs-mipnerf360-1600/octree-gs/mipnerf360-1600-*
```
Runs need `--recipe 3dgs`, `--num-gaussians` larger than the SfM cloud and `--steps 40000` (= `total_steps`).
