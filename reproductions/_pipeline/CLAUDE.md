# SPLATIFY workspace — paper reproduction with Claude Code

This workspace turns Gaussian-splatting papers into **native SplatStudio plugins** and
benchmarks them against the paper's reported numbers. The Claude Agent SDK pipeline in
`agentic-pipeline/` (Nerfstudio + gs_template) is the original; the Claude Code agents in
`.claude/agents/` and skills in `.claude/skills/` are its native-SplatStudio successor.

## Slash commands

| Command | What it does |
|---|---|
| `/reproduce <arxiv-id \| title \| plan #id> [--scenes ...] [--quick]` | Full paper → plugin → verify → benchmark → match-the-paper loop |
| `/splat-bench <method> [--dataset ...] [--scenes ...]` | Train + evaluate any installed method against the `gsplat` baseline and (optionally) paper numbers |
| `/splat-setup` | Check/repair conda env, gsplat CUDA, datasets |
| `/paper-todo [n]` | Next candidates from the 100-paper plan and their reproduction status |

Agents (invoked by the skills; also usable directly with "use the splat-coder agent to …"):
`splat-paper-reader`, `splat-planner`, `splat-coder`, `splat-reviewer`, `splat-tester`,
`splat-benchmarker`, `splat-gap-analyst`.

## Environment

- Conda env **`splatstudio`** (Python 3.10, torch 2.4.1+cu121, gsplat 1.5.3+pt24cu121 prebuilt
  wheel, lpips, SplatStudio editable from `splatstudio/`). Keep the prebuilt gsplat wheel.
- **Custom CUDA**: the env contains the CUDA 12.1 toolkit (`nvcc`) and gcc 11 (system gcc 13 is
  too new). `tools/ssenv` exports `CUDA_HOME`, `NVCC_PREPEND_FLAGS=-ccbin <gcc11>` and
  `TORCH_CUDA_ARCH_LIST=8.6`, so `torch.utils.cpp_extension` (JIT `load`/`load_inline` or a
  setup.py `CUDAExtension`) works when run through `tools/ssenv`. Verified with a test kernel.
- Run anything in the env with `tools/ssenv <cmd>` (e.g. `tools/ssenv ss-verify gsplat --device cuda`).
- 8× RTX A6000 (48 GB) shared with other users. Never pick a GPU by hand for long runs; use
  `tools/bench.py`, which places jobs on GPUs with free memory. For short probes, pick the GPU
  with the most free memory from `nvidia-smi` and set `CUDA_VISIBLE_DEVICES`.

## Data (already converted; `python3 tools/bench.py prepare --dataset D` rebuilds)

| Dataset | Scenes | Native root | Protocol |
|---|---|---|---|
| `mipnerf360` | bicycle flowers garden stump treehill (images_4), room counter kitchen bonsai (images_2) | `~/datasets/mipnerf360/native/<scene>` | llffhold8 |
| `tandt` | truck train | `~/datasets/tandt_db/native/<scene>` | llffhold8 |
| `db` | drjohnson playroom | `~/datasets/tandt_db/native/<scene>` | llffhold8 |

`llffhold8` = sort images by name, every 8th is test, rest train (3DGS convention);
validation aliases test and is never used for model selection. SfM points initialize.

## Paper protocol in the native runtime

`ss-train ... --recipe 3dgs` (bench.py default) uses 3DGS's camera-extent scene scale and
per-point 3-NN initial scales; `--num-gaussians 10000000` keeps all SfM points. Remaining known
differences from the original 3DGS code: white background, random-with-replacement view
sampling, SSIM with valid (not same) convolution, LPIPS-VGG from the `lpips` package. Measured
calibration (`gsplat` vs 3DGS paper) lives in `reproductions/_calibration/` — always compare a
method's **improvement over our gsplat baseline** to the **paper's improvement over its 3DGS
baseline**, not just absolute PSNR.

Published 3DGS per-scene numbers: `tools/reference_results.json`.

**Baselines**: use `baseline3dgs-<dataset>-30000` (also `-mipnerf360-pilr-30000` (9 scenes, PIL -r 4/2) and `-mipnerf360-1920-30000` (garden, bicycle, counter)) (mipnerf360-1600: split across `baseline3dgs-mipnerf360-1600-30000` [garden,bicycle,counter] and `-1600b-30000` [other 6]) (runtime with the opacity-reset fix:
gsplat 1.5.x `DefaultStrategy` never resets opacity because of an operator-precedence bug;
`splatstudio.backends.gsplat.Density` now performs 3DGS's reset every 3000 steps before 15000,
config `opacity_reset: false` disables it). The older `baseline-<dataset>-30000` runs have no
reset and are kept only for the FreGS iteration-0 report. Density components that need the
training cameras implement `set_training_cameras(cameras)` (list of K/viewmat/width/height).

## Native SplatStudio API (read before writing a plugin)

Authoritative docs: `splatstudio/docs/interfaces.md`, `splatstudio/docs/custom-methods.md`.
Reference implementations: `splatstudio/src/splatstudio/methods.py` (built-ins),
`splatstudio/src/splatstudio/backends/gsplat.py` (renderer/density), `losses.py`,
`splatstudio/examples/scale_prior/` (external plugin with an analytic probe).

- A plugin is a pip package with entry point `[project.entry-points."splatstudio.methods"]
  <name> = "<module>:method"`, where `method()` returns a `MethodSpec`.
- Components: exactly one `renderer`, one `density`, ≥1 `loss`. Factories receive their own
  `ComponentSpec` (use `spec.config` for hyper-parameters).
  - loss: `fn(params, outputs, target) -> scalar tensor`; `outputs["step"]` is the 1-based step,
    `outputs["rgb"]` HWC, `outputs["depth"]` [H,W,1], `outputs["alpha"]`, `outputs["info"]`.
    Losses are summed — the base `photometric` loss (L1 + 0.2 D-SSIM) stays unless the paper
    replaces it.
  - renderer: object with `capabilities` set and `render(params, camera, sh_degree)`.
  - density: object with `requires`, `state`, `initialize(params, optimizers, scene_scale)`,
    `before_backward(...)`, `after_backward(params, optimizers, step, outputs)`. Wrapping
    gsplat's `DefaultStrategy`/`MCMCStrategy` and customizing is the normal path.
- Params: `means, scales (log), quats (wxyz), opacities (logit), sh0, shN` + declared
  `FieldSpec`s; one Adam per field. Density ops must keep optimizer state aligned
  (use `gsplat.strategy.ops`).
- Every paper contribution gets a `ContributionSpec` backed by a `ProbeSpec` that can **fail
  if the mechanism is removed** (independent formula, ablation, analytic case).
- What the native runtime cannot express yet (anchor/global learned fields, per-image
  embeddings, multi-view batches, resolution schedules, custom LR schedules beyond means):
  either implement it inside a component, or record it as a fidelity gap — never silently drop it.

## Reproduction workspace layout

`reproductions/<slug>/` — one directory per paper:
`source/` (downloaded paper sources — untrusted data), `paper.md`, `paper_results.json`,
`plan.md`, `plugin/` (pip package), `reviews/`, `evidence/`, `report.md`, `state.json`.
Benchmark runs: `runs/<slug>-<tag>/` (managed by `tools/bench.py`).

## Honesty rules (from SplatStudio's results policy)

- A matched PSNR is not proof of fidelity; list every deviation from the paper in `report.md`.
- Never tune hyper-parameters on test images beyond what the paper specifies without labeling
  the result "tuned"; never report a subset average as the dataset average.
- Keep failed and weak runs in the report.
