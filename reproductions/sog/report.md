# Self-Organizing Gaussians (SOG) — reproduction report

**Paper:** *Compact 3D Scene Representation via Self-Organizing Gaussian Grids*, by Morgenstern, Barthel,
Hilsmann and Eisert (ECCV 2024, arXiv 2312.13299 v2).

**Official code:** fraunhoferhhi/Self-Organizing-Gaussians @ `5f2b360`. Read but never executed.

**Sorter:** PLAS @ `782a3ae`, used as an installed package.

**Method:** `sog`, plugin `plugin/` (`ss_sog`), frozen at `356909644029a2b9`.

**Outcome: matched.** On all 3 Mip-NeRF 360 scenes, the decoded-model PSNR is within ±0.5 dB of the
official per-scene results (−0.01 / −0.12 / +0.05 dB). The compressed sizes are +14.8 / −1.5 / +5.3 %
off the reference, and each gap follows our Gaussian count. The bytes per Gaussian are within 4 %.

## 1. Target numbers: why the repo CSV and not Table 1
- **What the paper reports.** arXiv v2 Table 1 gives dataset averages only. For Mip-NeRF 360 it lists
  "Ours" at **27.64** PSNR / 0.220 LPIPS / 40.3 MB.
- **Where the per-scene numbers come from.** They exist only in the official repo's
  `results/MipNeRF360/*.csv`. The authors produced these files on 2024-09-16: they decompressed the
  released ECCV models and rendered and measured them with the original 3DGS code
  (`eval/download_eval.sh`).
- **Why they differ from Table 1.** The README says this re-evaluation fixes the earlier W&B-based
  numbers: those omitted the quantization/compression loss, and they put Dr Johnson in T&T.
  - The CSV Mip-NeRF 360 average is **27.08** PSNR, 0.2304 LPIPS, 40.28 MB.
  - The sizes agree with Table 1; only the quality changed.
- **Target choice.** The CSV numbers are therefore the right target for evaluating the decoded model,
  and they are the only per-scene numbers available. The pass criterion in `state.json` (per-scene
  PSNR within ±0.5 dB of the CSV) was fixed before the benchmark.
- **Average check.** The v2 average (27.64) is the published claim. It cannot be checked
  per-scene, and our 3-scene subset is **not** a dataset average.
- **Size definition.** Size is the sum of all files in the compressed directory: 6 `.jxl` files plus
  `compression_info.csv` and `compression_config.yml` (`decompress.py:get_size_of_files_in_dir`).
  MB = bytes / 1e6.

## 2. Results (30k steps, decoded JPEG XL model, test split, llffhold 8)
Runs:
- SOG: `runs/sog-mipnerf360/sog/mipnerf360-<scene>/`
- Baseline: `runs/baseline3dgs-mipnerf360-30000/gsplat/`

Sizes come from `plugin/report_size.py`, which writes the files to
`<run>/compression/jxl_quant_sh/`, re-sums them and checks the sum against the recorded size.

| scene | PSNR ours / CSV (Δ) | SSIM ours / CSV | LPIPS ours / CSV | size MB ours / CSV (Δ) | #Gaussians ours / CSV | bytes/Gaussian ours / CSV |
|---|---|---|---|---|---|---|
| bicycle | 24.53 / 24.54 (**−0.01**) | 0.738 / 0.726 | 0.250 / 0.252 | 61.06 / 53.20 (+14.8 %) | 3.36M / 2.89M | 18.15 / 18.39 |
| garden | 26.91 / 27.03 (**−0.12**) | 0.848 / 0.840 | 0.140 / 0.135 | 82.26 / 83.49 (−1.5 %) | 4.45M / 4.34M | 18.50 / 19.22 |
| counter | 28.51 / 28.46 (**+0.05**) | 0.895 / 0.896 | 0.272 / 0.193 | 18.56 / 17.62 (+5.3 %) | 1.01M / 0.96M | 18.45 / 18.35 |

**Size breakdown (MB).** Higher-order SH (`_features_rest`) dominates:

| scene | xyz | DC | rest | scale | rot | opacity |
|---|---|---|---|---|---|---|
| bicycle | 13.34 | 3.09 | 34.53 | 4.78 | 4.23 | 1.09 |
| garden | 16.57 | 4.03 | 48.04 | 7.13 | 5.14 | 1.35 |
| counter | 3.67 | 0.98 | 10.94 | 1.44 | 1.22 | 0.30 |

**Relative to our gsplat baseline** (`baseline3dgs`, same protocol):

| scene | baseline PSNR | SOG − baseline | baseline Gaussians | SOG Gaussians / size | 3DGS-paper per-scene reference (CSV − 3DGS paper) |
|---|---|---|---|---|---|
| bicycle | 24.92 | −0.39 | 6.59M | 3.36M / 61 MB | −0.71 |
| garden | 26.89 | +0.02 | 4.76M | 4.45M / 82 MB | −0.38 |
| counter | 28.85 | −0.34 | 1.22M | 1.01M / 19 MB | −0.24 |

**Reading the baseline comparison.**
- The paper's own Table 1 shows 3DGS 27.55 vs Ours 27.64 (pre-fix). The CSV corresponds to roughly
  −0.47 dB on average versus that 3DGS row.
- Our deltas (−0.39 / +0.02 / −0.34) fall in the same range.
- The baseline's uncompressed size would be about 248 B per Gaussian (59 floats), so SOG is roughly
  13× smaller per Gaussian.

**Metric caveat.** The CSV numbers were rendered by the 3DGS code on a black background with
same-padding SSIM and the lpipsPyTorch VGG. Ours use a white background, valid-padding SSIM and the
`lpips` package. Absolute SSIM/LPIPS are therefore not directly comparable; counter's LPIPS (0.272 vs
0.193) is the clearest case. PSNR matched within the noise.

## 3. Mechanism evidence
**PLAS / sorting**
- 30 sorts per run: the initial one plus every 1000 steps from 1000 to 29000. The final grid sides
  were 1834 / 2109 / 1003.
- Pruning to a square removed 15.1k / 11.8k / 10.6k Gaussians in total over the run.
- The longest single sort took 263 / 323 / 56 s, and the sorts totalled 3941 / 3863 / 1171 s.
- These times were on shared, fully loaded A6000s. The paper reports under 10 s typical on a 4090,
  and an isolated test of a random 1500² grid took 345 s under the same contention.
- **Wall-clock cost:** training took 11.6k / 12.2k / 8.2k s, against 6.8k / 3.2k / 4.1k s for the
  baseline.
- The `grid-sort` probe shows the sort reduces the mean neighbour distance of the sort keys from 2.25
  to 0.20 on a 40² grid. It also confirms that params, Adam moments and densification state are
  permuted together.

**Codec**
- The final encode took 578 / 398 / 412 s (single-threaded JPEG XL on the CPU).
- The `codec-roundtrip` probe checks that the decoded lossless grids equal an independent
  quantize/normalize formula (error ≤ 1e-7).
- The same probe shows smooth grids compress more than 30 % smaller than shuffled ones.

**Smoke ablation** (garden, 2000 steps, test PSNR):

| configuration | PSNR |
|---|---|
| gsplat default | 22.99 |
| gsplat with only SOG's densification settings | 21.03 |
| SOG without the encode step | 21.12 |
| SOG decoded | 20.93 |

At 2k steps the gap to gsplat comes from the sparse densification schedule. The codec costs about
0.19 dB.

## 4. Codec environment (`codec_env`)
- The official environment pins `imagecodecs[all]==2023.9.18` with numpy < 2. Version 2024.9.22
  writes about 4× larger files (upstream issue #3).
- That wheel cannot be imported next to the splatstudio env's numpy 2.2.6, so it is installed with
  `numpy==1.26.4` into an isolated target directory: `reproductions/sog/codec_env`, built by
  `plugin/install_codec_env.sh`.
- `ss_sog/jxl_worker.py` imports only numpy and imagecodecs. It runs as a subprocess with
  `PYTHONPATH=codec_env` and calls `imagecodecs.imwrite(file, array, level=level)` /
  `imagecodecs.imread` exactly as `compression/jpeg_xl.py` does. The bundled library is libjxl 0.8.2.
- All quantization and normalization arithmetic runs in-process. The reviewer checked that float32
  scalar promotion gives the same results under numpy 1.26 and 2.x.
- PLAS was installed into the env with `pip --no-deps` from commit 782a3ae. kornia 0.8.2 was already
  present.

## 5. Contributions → components; probes
| contribution | component | probes |
|---|---|---|
| C1 square grid + PLAS sort (§3.1) | `sog-density` (`sorting.py`) | grid-sort, schedule |
| C2 smoothness Huber loss (§3.2) | `sog-neighbor` loss | neighbor-loss |
| C3 log positions (§3.3) | `sog-renderer`, `split_world`, `initialize` | log-xyz |
| C4 quantization + JPEG XL, decoded evaluation (§3.4) | `sog-density._encode`, `codec.py`, `jxl_worker.py` | codec-roundtrip, decoded-eval |
| C5 densification (supplementary) | `sog-density` base config | schedule |

**Checks:** `ss-inspect` reports no issues; 9 CPU unit tests pass; ruff is clean. `ss-verify --device
cuda` returns `declared_checks_passed` on 10 probes (the base `renderer-gradient` probe is replaced by
`sog-renderer-gradient`, which runs in the runtime's order).

**Review** (`reviews/review-1.json`): approved, with no critical or major issues. All 8 equation items
are marked exact. The minors:
1. Optimizer step order: the runtime steps Adam on densify/sort steps and at step 30000. This is a
   shared runtime gap.
2. The "vs 3DGS paper" deltas mix protocols; we compare against our own baseline instead.
3. The full 30k path had not been exercised at review time. The benchmark now covers it.
4. gsplat densifies on gradient `>` threshold where the official code uses `>=`.

## 6. Fidelity gaps (all kept)
1. **Runtime differences shared with the baseline:**
   - Adam steps before densify/sort, including the iterations the official code skips and step 30000.
   - White background (the official code uses black).
   - Random-with-replacement view sampling.
   - The SH degree is raised one step later.
   - The position learning rate has no delay.
   - gsplat rasterizer.
2. gsplat densifies on `>` threshold, the official code on `>=`.
3. The codec runs in a separate numpy-1.26 subprocess. It is the same codec version with the same
   calls.
4. `compression_info.csv` is written without pandas, in pandas' format. It and
   `compression_config.yml` may differ by a few bytes (together about 900 B).
5. PLAS randomness differs from the official runs; sorting is stochastic in the official code too.
6. The official intermediate logging compressions at 7k/10k/20k are not run.
7. There is no "w/o SH" variant, and SOG-specific metrics are only reported for 3 scenes, so there is
   no dataset average.
8. **Sizes follow our Gaussian count:** bicycle has +16 % Gaussians, which gives +15 % MB. Bytes per
   Gaussian are within 4 % of the reference.

## 7. Reproduce
```bash
T=/home/sejain/splatify/tools/ssenv
$T pip install --no-deps "git+https://github.com/fraunhoferhhi/PLAS.git@782a3ae8adf5223ece31aa4ea97fc5e66931180a"
reproductions/sog/plugin/install_codec_env.sh          # -> reproductions/sog/codec_env
cd reproductions/sog/plugin && $T pip install -e . && $T ss-verify sog --device cuda
$T ss-train sog --data ~/datasets/mipnerf360/native/garden --output runs/x --steps 30000 \
   --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000   # --steps must be 30000
$T ss-eval --run runs/x --split test
$T python reproductions/sog/plugin/report_size.py runs/x               # MB = bytes / 1e6
```
Benchmark: `python3 tools/bench.py` with `bench_request.json` (bicycle, garden, counter).
