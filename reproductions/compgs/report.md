# CompGS (Compact3D) — reproduction report

Paper: CompGS: Smaller and Faster Gaussian Splatting with Vector Quantization (Navaneet et al., ECCV 2024, arXiv 2311.18159).
Official code: github.com/UCDvision/compact3d (MIT, commit dccc07e), ported into `plugin/ss_compgs`.
Variant: **CompGS 32K with opacity regularisation** (supp. per-scene table). Benchmark: `runs/compgs-mipnerf360`, 30000 steps, decoded model evaluated (`ss-eval` on baked C[idx] params).

**Outcome: matched** (both scenes within ±0.5 dB). Not separately reviewed.

## Results (Mip-NeRF 360, test split, decoded model)
| scene | ours PSNR | paper PSNR | Δ vs paper | our gsplat baseline | ours − baseline | paper − its 3DGS | #Gaussians ours / paper | size ours (paper formula) / paper | train time |
|---|---|---|---|---|---|---|---|---|---|
| garden | 26.56 | 26.822 | −0.26 | 26.89 | −0.33 | −0.42 | 1.23M / 1.37M | 27.53 MB / 30 MB | 103 min |
| room | 31.32 | 31.131 | +0.18 | 31.52 | −0.21 | −0.22 | 349k / 327k | 9.12 MB / 9 MB | 76 min |

Size lines printed by `plugin/report_size.py <run>` (MB = 1e6 bytes):
| line | garden | room | meaning |
|---|---|---|---|
| `paper_size` | 27.53 | 9.12 | paper §3 formula: fp32 position + opacity, fp32 codebooks, ceil(log2 K)-bit indices, rotation index run-length coded (K counts). **Compare this one to the paper.** |
| `official_disk_size` | 34.32 | 10.61 | what the released code would write: ply (xyz, opacity) + indices at ceil(log2 N) bits (`save_kmeans` uses len(cls_ids)=N) + codebooks |
| `uncompressed_3dgs` | 290.5 | 82.39 | same Gaussian count stored as plain 3DGS (59 fp32 per Gaussian), for the compression ratio |

## Paper-vs-code choices used
| item | paper | released code (`run.sh`) | used |
|---|---|---|---|
| VQ start | after 20K | 15K | 20K (paper) |
| k-means iterations per assignment | 1 (text); 5 (Table perf_vs_time row that equals the headline 27.12) | 10 | 5 |
| SH codebook | 4096 | 512 | 4096 |
| scale / rotation codebooks | 32K each (CompGS 32K) | 4096 | 32768 each, separate |
| assignment frequency | every 100 until 25K, then frozen | same (it%100==1, freq 5000 after 25000) | same: assignments at 20001…24901, 25001 |
| opacity reg | λ=1e-7, 15K–20K, prune every 1000 | same, prune σ<0.005, no size criterion | same |

## Deviations / fidelity gaps
- Shared runtime differences: white background, random-with-replacement view sampling, gsplat rasteriser, SSIM/LPIPS implementations (see CLAUDE.md calibration).
- Centroid averaging uses `index_add_` instead of the official padded gather (same math, different float summation order).
- Opacity pruning runs after the optimizer step of that iteration, not before (one extra Adam step on pruned Gaussians).
- At step 30000 the runtime still steps means/opacity (the official code skips the last step); the decoded attributes are the centroids used for the 30000 render, like the official saved codebook.
- Size is computed analytically from the formula; no official file set (ply/bin/pth) is written.
- k-means init for tiny N (< K) samples with replacement (probes only; never hit on real scenes).
- Gaussian count: garden 10% fewer than the paper, room 7% more.

## Evidence
`evidence/verify.json` (8/8 probes), `evidence/test.json` (smoke: garden 2k steps 23.19 dB decoded vs gsplat 23.07), `bench_request.json`, `plan.md`, `paper.md`, `paper_results.json`.
