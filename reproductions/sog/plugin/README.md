# splatstudio-method-sog

Native SplatStudio port of **Compact 3D Scene Representation via Self-Organizing Gaussian Grids**
(Morgenstern, Barthel, Hilsmann, Eisert, ECCV 2024; arXiv 2312.13299). Paper fidelity unverified.
Official code: https://github.com/fraunhoferhhi/Self-Organizing-Gaussians (`5f2b360`); numerics
of the codec / sorting / neighbour loss are copied from it (Inria/MPII Gaussian-Splatting license,
`ss_sog/LICENSE-SOG-official.md`). Sorter: PLAS (Apache-2.0) as the installed package `plas`.

Methods: `sog` (paper configuration, encodes at step 30000) and `sog-smoke` (identical, encodes
at step 2000; diagnostic only).

## Mechanism → code
| contribution | code | values (source) |
|---|---|---|
| C1 square grid + PLAS sort | `sorting.py`, `density.py:_regrid` | keys xyz/DC/scale (weights 1, standardized, activated), shuffle, improvement_break 1e-4; at init and every 1000 steps after 500 (`train.py:65, 240-246`) |
| C2 smoothness loss | `losses.py` | λ 1, Huber, blur 5 / σ 3 (x circular, y reflect), opacity 1 / rotation 10 (= 0.09 / 0.91) |
| C3 log positions | `renderer.py`, `density.py:split_world`, `initialize` | `x = sign(y)(e^{|y|}-1)` (§3.3) |
| C4 quantize + JPEG XL, decoded eval | `codec.py`, `jxl_worker.py`, `density.py:_encode` | experiment `jxl_quant_sh`: xyz 14 bit lossless, DC lossy level 100, rest 5 bit, scale/rot/opacity 6 bit, ranges of `compression/codec.py` |
| C5 densification | `density.py:DEFAULTS["base"]` | every 1000 its in (500, 15000), grad 7e-5, prune opacity < 0.1, percent dense 0.1, no reset / size pruning |

At `total_steps` the density prunes to square, encodes, decodes and **replaces the parameters**,
so `validation.json`, `checkpoint.pt` and `ss-eval` measure the decoded model (pattern copied from
the Reduced-3DGS / EAGLES plugins). The encoded files are stored in the density state.

**Note:** `params["means"]` in the checkpoint are log-space (official `_xyz`); only `sog-renderer`
renders them correctly.

## Dependencies
- `plas`: `tools/ssenv pip install --no-deps "git+https://github.com/fraunhoferhhi/PLAS.git@782a3ae8adf5223ece31aa4ea97fc5e66931180a"`
  (kornia, tqdm already in the env; PLAS's other setup.py deps are only for its examples).
- JPEG XL: `install_codec_env.sh` → `reproductions/sog/codec_env` with `imagecodecs==2023.9.18`
  + `numpy==1.26.4` (official pin; 2024.x writes ~4× larger files, upstream issue #3). It cannot be
  imported next to the env's numpy 2.2.6, so `jxl_worker.py` runs in a subprocess with
  `PYTHONPATH=codec_env`. Override with config `codec_env` or env var `SOG_CODEC_ENV`.

## Commands
```bash
tools/ssenv ss-train sog --data ~/datasets/mipnerf360/native/garden --output runs/x --steps 30000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test
tools/ssenv python reproductions/sog/plugin/report_size.py runs/x   # size in MB (bytes / 1e6)
```
`--steps` must equal the density `total_steps` (checked at step 1), or no encode happens.

## Fidelity gaps
See `../plan.md`: runtime order (optimizer step before densify/sort, also on the official
skipped steps), white background, random view sampling, gsplat `>` threshold; isolated codec
subprocess; metadata files written without pandas (same format); PLAS RNG; no w/o-SH variant.
