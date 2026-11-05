# splatstudio-method-gaussianspa

Native SplatStudio port of **GaussianSpa** (Zhang, Jia, Niu, Yin, CVPR 2025, arXiv 2411.06019),
written against the official code https://github.com/noodle-lab/GaussianSpa
(`train_opacity.py`, `train_imp_score.py`, `optimizing_spa.py`). Paper fidelity unverified.

## Mechanism
GaussianSpa runs Mini-Splatting up to 15k, then solves `min L s.t. ||a||_0 <= kappa` on the
activated opacities `a` with an ADMM-style "optimizing-sparsifying" loop:

| step | what | code |
|---|---|---|
| 15000 | importance sampling keeps `int(N(1-r1))`, p ∝ imp+1, reinit from centres | `density._simplify_sample` |
| 15200 | `z = prox(a)`, `u = 0` | `density._sparsify(update_u=False)` |
| (15200, 25200], every 50 | loss `+= 0.5·delta·‖a − z + u‖²` (Eq. 11) | `losses.create_penalty` |
| (15200, 25200], every 50 | `z = prox(a + u)` (Eq. 16), `u += a − z` (Eq. 17) | `density._sparsify` |
| 25200 | hard prune to `int((1-r2)N)` (opacity top-k or importance rank) | `density._hard_prune` |
| 25200–40000 | light tuning | — |

`prox` keeps the top-ranked entries: by `a + u` (opacity criterion) or by Mini-Splatting
importance (importance criterion, recomputed over all training views at every update). `z` and
`u` are Gaussian-owned fields `spa_z`, `spa_u` (never receive gradients), so population ops keep
them aligned and the checkpoint carries them.

## Config (paper/code values)
`delta` 5e-4 (`rho_lr`), interval 50, window 15200–25200, iterations 40000, `imp_metric`
indoor (both scripts), Mini-Splatting defaults otherwise (num_depth 3.5M, num_max 4.5M, blur
split 2e-4, depth reinit every 5000 before 15k, LR restart lr(it−10000), SH 0 until 15k).

Methods: `gaussianspa` (opacity, 0.5/0.8), `gaussianspa-imp` (importance, 0.5/0.7), and
`gaussianspa-<scene>` for the official README presets (13 scenes; e.g. bicycle importance
0.50/0.72, garden importance 0.60/0.50, counter opacity 0.60/0.50).

## Prior art
`ms_density.py`, `core.py`, `kernels.py`, `renderer.py`, `ms_probes.py`, `csrc/` are copied with
attribution from the SPLATIFY Mini-Splatting port (`reproductions/mini-splatting/plugin`), whose
kernel is adapted from official Mini-Splatting (Inria 3DGS licence, `csrc/LICENSE-3DGS.md`).
Edits: extension name `ss_gaussianspa_cuda`, a `_reinit_extra` hook for the ADMM fields, an
overridable option table, probes building scenes with the ADMM fields.

## Fidelity gaps
Inherited from the Mini-Splatting port (white background, random-with-replacement views,
optimizer-before-density order, gsplat alpha numerics, `lpips` package, valid-conv SSIM); z/u
update and hard prune use the opacity after the step's Adam update (official: before); per-scene
presets come from the README, not the paper.

## Commands
```bash
tools/ssenv pip install -e reproductions/gaussianspa/plugin
tools/ssenv ss-verify gaussianspa-garden --device cuda
tools/ssenv ss-train gaussianspa-garden --data ~/datasets/mipnerf360/w1600/garden --output <run> \
  --steps 40000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv python reproductions/gaussianspa/plugin/smoke_schedule.py gaussianspa-garden <data> <out>
```
