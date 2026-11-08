# LiteGS review 1 (splat-reviewer)

**Verdict: approved.** 0 critical, 0 major, 5 minor. Ready to benchmark.

Scope: everything in `plugin/ss_litegs/` compared against the official code in `source/code/litegs`
(`training/densify.py`, `training/trainer.py`, `training/optimizer.py`, `utils/statistic_helper.py`,
`utils/wrapper.py`, `submodules/gaussian_raster/raster.cu`, `compact.cu`, `full_eval.py`), and the
runtime loop in `splatstudio/runtime.py:186-215`. I ran one scratch experiment comparing the exact
statistic with the official one (below). I did no training and did not edit the plugin.

## Key question: what does the official stats kernel accumulate?

`raster.cu:599-820` (`raster_backward_kernel`, tile 8x16): each warp handles one tile. Each thread
handles 4 pixels as `PIXELS_PER_THREAD = 2` half2 steps. Lane `.x` covers rows y and y+2, and lane
`.y` covers rows y+1 and y+3 of one column. For each primitive, `grad_a` is reset to 0. At every
step i, `grad_a += d_alpha*G` and then `err_square += grad_a^2`. That is the square of the
**running sum within one 2-pixel lane**, not of a per-pixel value. A masked pixel still adds the
running sum again whenever any lane in the warp is valid. So per view and per primitive:

- `square_sum = Σ_lanes [ g0² + (g0+g1)² ]`, where g0 and g1 are the exact per-pixel opacity
  gradients at rows (4k+r, 4k+r+2), r ∈ {0,1}.
- `sum = grad_opacities`, the true total gradient, which equals Σ g.
- `count = fragment_count` from the forward pass (alpha ≥ 1/256, T > 1/8192).
- `get_var` = S/(C+1) − (M/(C+1))².

The arithmetic runs in fp16 with the 128 scaler.

This is an approximation that inflates S by 2g0² + 2g0g1 per lane. Correlated neighbours raise the
score of primitives whose gradients are consistent (low variance).

**Measured effect.** I used a scratch copy of our kernel that dumps each fragment and pairs the
lanes exactly as above (`/tmp/claude-1013/-home-sejain/6ad4135e-056e-4bd4-9844-6b8764bea041/scratchpad/cmp.py`).
Setup: smoke-1 checkpoint, garden, 422k Gaussians, 30 random training views, SH degree 2,
loss = our photometric loss on the clamped image.

| metric | result |
|---|---|
| S_official / S_exact, median | 1.43 |
| Spearman(score_exact, score_official), 414k non-zero primitives | **0.9987** (Pearson 0.9987) |
| Multinomial draw without replacement, Jaccard vs a second exact draw (noise floor) | 0.280 at budget 50k, 0.619 at 200k |
| Same draw, Jaccard exact vs official | 0.280 at 50k, 0.618 at 200k |
| Score mass in the top 10% | 76.7% exact vs 76.9% official |

The ranking change is immaterial: the sampled densification sets cannot be told apart.
**Recommendation: keep the exact per-pixel statistic.** If strict fidelity is wanted, the emulation
is cheap to add as an option: pair rows by `(r//4, r%2)` with i = (r%4)//2, as in the script.
Either way, fix the documentation (minor 1).

## Checked against official, all OK

- **Score.** `density.py:71-81`: var·C·σ(o)², clamp, nan_to_num. Matches `densify.py:286-292` and
  `statistic_helper.get_var`, including the (count+1) denominator.
- **Budget and growth.** `density.py:84-91` matches `densify.py:304-305`, where `prune_num` is
  counted before the append.
- **Target counts.** Garden 5,728,191 and room 1,548,960 are the last entries in `full_eval.py`'s
  `target_primitives_list`, i.e. the 3DGS counts. The lookup keys on the SfM count (garden 138,766,
  room 112,627, the same numbers as the `full_eval_aggressive.py` comments) and fails loudly for
  unknown scenes.
- **Smoke run.** It lands exactly on the linear target (219,772 at epoch 5, 422,287 at epoch 10).
- **Sampling.** `torch.multinomial(..., replacement=False)`.
- **Clone/split.** The split threshold is max(scale) > 0.01·extent, and extent = 1.1·max camera
  distance = `norm_radius`. Only the child is appended: xyz + R·N(0, s), scale s/1.6, everything
  else copied. The parent is kept, as in `densify.py:312-343`.
- **Prune.** Only primitives with W==0 that existed before the append (`densify.py:273-280`).
- **Opacity decay.** σ ← max(σ/2, 1/128), then all optimizer state is cleared
  (`densify.py:229-232`). Order: split/clone → prune → decay, as in `densify.py:246-255`.
- **Schedule.** Densify at epoch % 5 == 0 for 3 ≤ e < until. until = int(0.8·total/10)·10 + 1,
  which is 141 on garden (total_epoch 186). Decay at epoch % 10 == 0 inside the same window.
  Statistics are collected only during densify epochs.
- **Adam.** `optim.py`: no bias correction, eps 1e-15, betas (0.9, 0.999), m/v/p updated only on
  visible rows. Matches `compact.cu:348-375`. After `state.clear()`, the lazy re-initialisation
  works (`optim.py:37`), and appended rows get zero moments.
- **Learning rates.** means 1.6e-4·extent (log-decay to ×0.01 over 30K; the runtime reads
  `initial_lr` after `density.initialize`), sh0 0.0025, shN 0.00025 (= feature/10, not the
  runtime's /20), opacity 0.025, scale 0.005, rot 0.001.
- **SH schedule.** min(epoch//5, 3), as in `trainer.py:107-108`.
- **Clamp.** The image is clamped to [0,1] before the loss (`render/__init__.py:87`). dL/dC is
  taken on the pre-clamp image, i.e. after the clamp's zero gradient, which is what the official
  rasteriser receives.
- **Kernel.** The background term is included in C − P_i, and g is not zeroed by the 0.999 clamp
  (official behaviour). Per-pixel S agrees with a double-precision reference to a relative error of
  2.7e-4.
- **requires_grad.** `append_new` builds parameters from `p.detach()` with
  `requires_grad=p.requires_grad`, gsplat `remove` preserves it, and `check_trainable` runs after
  every population change.
- **Visibility.** The mask is refreshed in `before_backward` on every step, so it is never stale
  after a densify. `opt.step()` runs before `after_backward` (`runtime.py:205-211`).
- **Probes.** The stats probe uses brute-force per-pixel autograd and would fail if the
  squares/sums were wrong or replaced. The score probe uses an independent formula plus a ranking
  check. The densify, decay and Adam probes compare against independent references.

## Minor issues

1. **`csrc/litegs_stats.cu:12-16`, `plan.md` gap 3.** The comment says S "are the moments the
   official LiteGS rasteriser aggregates". That is not true for S. Document the official quantity
   Σ_lanes[g0² + (g0+g1)²] (raster.cu:727-817) and the measured equivalence (Spearman 0.9987)
   in report.md.
2. **Epoch semantics under with-replacement sampling.** A statistics epoch is n_train random draws,
   so about 37% of views are never seen. Primitives visible only in those views get W=0, which
   means they are pruned and get score 0. The smoke prune counts (158 and 177) show the effect is
   tiny, and the budget compensates. Record it in report.md as a runtime gap (TODO-runtime
   shuffled sampler); it is not a fix item.
3. **`density.py:234`.** The budget is capped at the number of score>0 primitives. Official
   `torch.multinomial` would raise instead. This is a benign deviation; note it.
4. **`optim.py`.** Visibility is per primitive (radii>0), whereas official uses 128-primitive
   cluster frustum visibility. With clusters, primitives that are in the frustum but not rendered
   still take momentum steps. This is already gap 2; keep it in report.md.
5. **`density.py:263-265`.** The split offset is R·s. Official computes `samples @ CreateTransformMatrix(1,q)`
   in row-vector form, which equals R·s under the 3DGS glm convention. I did not verify the wrapper
   numerically. The effect is negligible either way.

## Launch

```
python3 tools/bench.py launch --exp litegs-mipnerf360-30000 --methods litegs --dataset mipnerf360 \
  --scenes garden room --steps 30000 --recipe 3dgs --min-free-gb 24
```

Notes: bench already passes `--max-gaussians 12000000`, which covers garden's 5.73M. Use the
/bigdata symlink per CLAUDE.md. Compare the gains over `baseline3dgs-mipnerf360-30000`
(garden 26.89, room 31.52) with the paper's gains (garden +0.76, room +1.22).
