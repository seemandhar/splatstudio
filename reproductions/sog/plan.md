# SOG native plan

Method `sog`, module `ss_sog`, package `splatstudio-method-sog`; diagnostic `sog-smoke`
(identical except `total_steps = 2000`, so a 2000-step smoke run reaches the encode step).

## Architecture
Decoded-evaluation pattern copied from the Reduced-3DGS / EAGLES plugins (copied, not imported):
the density encodes at the final step (`total_steps`), decodes, and replaces the parameters, so
the runtime's validation, `checkpoint.pt` and `ss-eval` all measure the decoded model. The
encoded files (bytes) are kept in the density state; `report_size.py` writes them to
`<run>/compression/jxl_quant_sh/` and reports `Size [Bytes]` / 1e6 exactly as the official
`decompress.py:get_size_of_files_in_dir`.

Grid = parameter row order (as official). Between sort events N is always a perfect square,
because densification only happens on sort steps and every sort step first prunes to square.

**Log xyz (C3)**: `params["means"]` holds `sign(x)·log1p|x|` (official `_xyz`). The renderer maps
to world space before rasterization; the density converts the SfM init in `initialize()` and its
split samples children in world space and maps back (official `densify_and_split`). The
checkpoint therefore stores log-space means; the method's renderer is the only consumer.

**Codec runtime**: imagecodecs 2023.9.18 (pinned by the official repo; 2024.x gives 4× larger
files, issue #3) needs numpy < 2, but the splatstudio env has numpy 2.2.6 (other plugins/opencv
depend on it). It is installed into an isolated target dir `reproductions/sog/codec_env`
(`pip install --target ... imagecodecs==2023.9.18 numpy==1.26.4`, `plugin/install_codec_env.sh`)
and invoked as a subprocess (`ss_sog/jxl_worker.py`, only numpy+imagecodecs) for JXL
write/read. All quantization/normalization numerics run in-process (copied from
`compression_exp.py`). PLAS (Apache-2.0, pinned 782a3ae) is installed into the env with `--no-deps`.

## Components
| contribution | component | factory | config (paper/code value) |
|---|---|---|---|
| C3 log xyz | renderer `sog-renderer` (gsplat Renderer subclass) | `ss_sog.renderer:create_renderer` | none |
| C1 square prune + PLAS sort; C5 densification; C4 codec; C3 split | density `sog-density` (wraps splatstudio gsplat `Density`; strategy subclass with world-space split) | `ss_sog.density:create_density` | base: refine_every 1000, refine_start 500, refine_stop 15000, grow_grad2d 7e-5, prune_opa 0.1, grow_scale3d 0.1, reset_every 10^9 (no reset, no size pruning), opacity_reset false; sort: weights xyz/dc/scale 1, normalize, activated, shuffle, improvement_break 1e-4, sort_every 1000, sort_after 500; total_steps 30000; codec `jxl_quant_sh` table |
| C2 smoothness | loss `sog-neighbor` | `ss_sog.losses:create_neighbor_loss` | lambda 1.0, kernel 5, sigma 3, huber, weights opacities 1 / quats 10 |
| photometric | unchanged built-in | | L1 + 0.2 D-SSIM |

Renderer capabilities = gsplat's; density requires `gradient.screen.signed`.
No new FieldSpecs.

## after_backward order (official iteration order, runtime steps the optimizer first)
1. base gsplat DefaultStrategy post-backward (stats; at 1000·k, k=1..14: clone/split/prune < 0.1);
2. if step > 500 and step % 1000 == 0 and step < total_steps: prune to square + PLAS sort
   (params, Adam moments and strategy state permuted together);
3. step == total_steps: prune to square → quantize/normalize → JXL encode (subprocess) → decode →
   replace params (moments zeroed), record size.

## Probes
| probe | checks | fails if |
|---|---|---|
| log-xyz (CUDA) | log/inverse round trip; renderer(log means) == gsplat(world means); split children = world parent + R·S·n mapped back (seeded); init converts SfM means | activation removed / split in log space |
| neighbor-loss | value & gradient vs independent conv (explicit circular/reflect pad, Huber formula, detached target), weights 1/11, 10/11; constant grid → 0 | blur/border/detach/weights wrong |
| grid-sort (CUDA) | prune keeps top-S² opacity in order; PLAS permutation applied consistently to params/moments/state; sorted neighbour distance ≪ shuffled | sort or permutation alignment removed |
| codec-roundtrip | worker uses imagecodecs 2023.9.18; decoded lossless attributes == independent quantize/normalize formula; DC error bounded; size == Σ files incl. csv/yml; sorted grid compresses smaller than shuffled | quantization/normalization/codec wrong |
| schedule | sort/densify/encode steps equal official train.py; no reset; densify config values | schedule wrong |
| decoded-eval (CUDA) | after final step params == decoded arrays; fresh renderer render == decoded render; state holds files whose sizes sum to the report | decode/replace skipped |
| sog-renderer-gradient (CUDA) | renderer gradient probe in runtime order (initialize before render) | no screen gradient |

## Fidelity gaps
1. Runtime (shared with baseline): optimizer step runs every step and before densify/sort
   (official skips the step on sort/densify iterations and at 30000), random-with-replacement
   view sampling, white background (official black), SH degree bump one step later, means LR
   exponential without delay, gsplat rasterizer.
2. gsplat densification uses `grad > thr` (official `>=`); identical otherwise.
3. Codec subprocess env (isolated numpy 1.26) instead of the official single env; same codec
   version, same numerics.
4. Metadata files (`compression_info.csv` written without pandas in pandas' format,
   `compression_config.yml` via yaml.dump) may differ by a few bytes from the official ones.
5. The official intermediate compressions at 7k/10k/20k are logging-only; not run.
6. PLAS RNG differs (global torch RNG state); sorting is stochastic in the official code too.
7. "w/o SH" variant not provided (paper main row is with SH).
