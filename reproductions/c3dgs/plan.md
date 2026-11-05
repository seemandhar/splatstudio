# c3dgs — native SplatStudio plan

Method `c3dgs`, module `ss_c3dgs`, package `splatstudio-method-c3dgs`. Paper spec: `paper.md`.

## Run shape (single run, precedents pup-3dgs / reduced-3dgs)
One `ss-train c3dgs --steps 35000` run:
* steps 1..30000: the unchanged SplatStudio 3DGS density (gsplat DefaultStrategy + opacity-reset
  fix, `make_method` density config) with the means LR of a 30000-step run
  (`schedule.pretrain_means_lr`, copied from pup-3dgs) — identical recipe to
  `baseline3dgs-mipnerf360-30000`, so that baseline is the pre-compression reference.
* after step 30000 (density `after_backward`): C1 sensitivity over all training cameras + zero
  sensitivity pruning, C2 colour and shape clustering; a `CompressedBank` (codebooks, indices,
  log eta, FakeQuantize modules, its own Adam) is registered in the process-local `bank.ACTIVE`.
* steps 30001..35000: C3 quantization-aware fine-tuning through the normal runtime loop (the
  renderer renders from the bank; photometric loss unchanged; no densification; means frozen
  as in the official code; runtime Adam states cleared = fresh optimizer).
* step 35000: C4 Morton sort + official `save_npz` layout written with `np.savez_compressed`
  to memory, size = len(bytes)/2**20 (= `os.path.getsize`/1024**2), decode the npz bytes into the
  base fields (means, log scales, quats, logit opacities, sh0, shN) so checkpoint,
  `validation.json` and `ss-eval` measure the decoded model. The npz is kept in the density
  state (uint8 tensor) and `report_size.py` prints/exports it.

## Components
| contribution | kind | factory | config (paper/code value) |
|---|---|---|---|
| base 3DGS | density (wrapped) | `splatstudio.backends.gsplat.Density` | make_method config |
| C1–C4 | density | `ss_c3dgs.density:create_density` | base_steps 30000, total_steps 35000, color/gaussian codebook 4096, beta_c 6e-7, beta_g 3e-6, prune_threshold 0, color steps 100 batch 2^18, gaussian steps 800 batch 2^20, decay 0.8, LRs (dc 0.0025, rest 0.000125, scaling 0.005, scaling_factor 0.005, rotation 0.001), freeze_means true, morton true, sensitivity_background black |
| C3 renderer | renderer | `ss_c3dgs.renderer:create_renderer` | gsplat base caps; renders bank attributes while a bank is active |
| loss | loss | `splatstudio.losses:create_photometric` | 0.8 L1 + 0.2 D-SSIM (= official finetune loss) |

`requires` of the density = base density requires (`gradient.screen.signed`); renderer
capabilities identical to the gsplat renderer.

## State
No FieldSpecs. Global/indexed state lives in the bank (runtime cannot express global learned
fields): `cb_dc [K',1,3]`, `cb_rest [K',15,3]`, `cb_scale [K'',3]` (linear, activation
normalize(relu)), `cb_rot [K'',4]`, per-Gaussian `log_eta [N,1]`, `color_idx [N]`,
`gauss_idx [N]`; opacity and means stay runtime fields. No density op runs after compression,
so alignment is trivial; pruning at 30000 uses `gsplat.strategy.ops.remove`.

## Numerics mapped to code
* `quant.py`: `torch.ao.quantization.FakeQuantize(dtype=torch.qint8)` per attribute (exact
  official module), `HalfSTE`, `morton_encode` (official bit tricks).
* `vq.py`: official `weighted_distance.cu` (JIT, csrc, licence kept) for nearest centroid;
  `vq_features` / `join_features` / EMA update / trace re-normalisation / uniform init copied
  from `compression/vq.py`; `extract_rot_scale`, `matrix_to_quaternion`, `to_full_cov` from
  `utils/splats.py`.
* `sensitivity.py`: per training camera, leaves `sh [N,16,3]` (fake-quantised) and normalised
  covariance `[N,6]`; colours = SH(dirs) + 0.5 **unclamped** (`gsplat.spherical_harmonics`),
  covariance = cov_n * eta^2, black background, E = sum(rgb), accumulate |grad| per image,
  divide by sum of pixels.
* `bank.py`: forward attributes as in the official indexed `render()`; quantizer call order per
  render = one observer update per attribute.
* `codec.py`: official `save_npz(sort_morton=True)` keys/dtypes/shapes; decode = official
  `load_npz` dequantisation + render activations (rotation normalised; stored scale used as is,
  which equals the fake-quantised value the paper's `compress.py` evaluation renders).

## Probes
| probe | fails if |
|---|---|
| `sensitivity-oracle` (CUDA) | |dE/dsh| differs from the independent W_g * Y_k(d) formula (per-Gaussian blending weights from a one-hot multi-channel render), clamping is used, abs is taken after the sum, or pixel normalisation is wrong; cov sensitivity != eta^2 x full-cov gradient |
| `vq-official` (CUDA) | CUDA nearest != brute force; EMA update != formula; weighted centroid ignores sensitivity; trace re-normalisation off; kept vectors not exact; eigendecomposition does not reconstruct cov |
| `qa-codebook` (CUDA) | values not on an 8-bit grid / positions not float16; STE gradient missing; codebook gradient != sum of per-Gaussian gradients; LR groups wrong |
| `codec-roundtrip` (CUDA) | decoded render != bank render; reported size != on-disk `np.savez_compressed` size; Morton order not sorted / does not shrink a coherent scene vs a random order |
| `pipeline` (CUDA) | end-to-end tiny scene: zero-sensitivity Gaussian not pruned, bank not active, fine-tune step does not change codebooks, means move, decode does not replace params |
| `schedule` (CPU) | events at base_steps / total_steps differ; run-length guard missing |

## Files
`plugin/pyproject.toml`, `ss_c3dgs/{__init__,schedule,quant,vq,sensitivity,bank,codec,renderer,density,probes}.py`,
`ss_c3dgs/csrc/{weighted_distance.cu,weighted_distance.h,ext.cpp,LICENSE-c3dgs.md}`,
`report_size.py`, `README.md`, `tests/test_c3dgs.py`. Smoke config entry point `c3dgs-smoke`
(base 1500 + 500 fine-tune; diagnostic only).

## Fidelity gaps
1. Base model: our gsplat 3DGS (white background, gsplat rasteriser) instead of the 3DGS
   authors' released models — compare deltas vs our baseline (paper: -0.20 / -0.43 / -0.22 dB on
   bicycle / garden / counter). Not closable without the released models (could be downloaded
   and converted later; out of scope for one run).
2. Fine-tune background white (runtime convention) vs black in the official mip360 runs; the
   sensitivity pass does use black as in the code.
3. Runtime takes an optimizer step for opacity at step 35000 (official last iteration has none);
   bank parameters skip it. Negligible.
4. Random view sampling with replacement (runtime) vs the official shuffled stack.
5. gsplat vs official CUDA rasteriser (sensitivity gradients via gsplat; checked by probe).
6. Final metrics: ss-eval on the decoded npz vs compress.py on the in-memory fake-quantised
   model (identical except rotation stored as Q(normalize(r)) vs rendered normalize(FQ(r))).
7. Hardware-rasteriser speed claim (WebGPU renderer) not reproduced.
8. Scatter sums via `index_add_` instead of torch_scatter (same math, float order differs).
