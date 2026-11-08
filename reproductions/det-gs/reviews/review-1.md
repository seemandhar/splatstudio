# DET-GS review 1 (splat-reviewer, adversarial)

**Verdict: not approved: 0 critical, 1 major, 6 minor.** The equations, schedule and dataset
protocol are sound. The one major issue is a hyper-parameter choice that switches off one of the
paper's stated mechanisms. It is cheap to resolve.

## Evidence gathered
- Read: paper.md, plan.md, paper_results.json, LaTeX §III–IV (`source/latex/bare_jrnl_new_sample4.tex`),
  all of `plugin/ss_det_gs/`, `tools/prepare_detgs.py`, DNGaussian official code
  (`reproductions/dngaussian/source/code/{utils/loss_utils.py,train_llff.py,train_dtu.py,gaussian_renderer/__init__.py,arguments/__init__.py,scripts/*.sh}`),
  `splatstudio/data.py`, `backends/gsplat.py`, `losses.py`.
- pytest: 3 passed. verify.json: every probe passed.
- Dataset: `transforms.json` is byte-identical to native (garden and counter). Every image is hard-linked.
  The smoke run's protocol_id `53deee9d…` equals the protocol_id of `baseline3dgs-mipnerf360-30000/gsplat/mipnerf360-garden`.
  The priors hold 161/161 (garden) train views. **0/24 test viewmats are found in the prior table**, so there is no leakage.
  Counter has 210 train views and the same construction.
- Diagnostic (scratchpad `chk.py`/`chk2.py`) on the converged gsplat-30k garden/counter checkpoints, hard depth ω=0.95:
  - hard-depth coverage is 1.00 on every sampled view, including the regions DA-V2 calls far. The rendered z there is
    1.5–3× the median, so the sign `target = -disparity` is consistent.
  - corr(z, −disp) = 0.44 (mean over 11 views) vs corr(1/z, disp) = 0.28. The data do not show that disparity space is better.
  - entries of the **image-level** term with |Δ| > margin, at patch size 10: 98.8 % for 2.5e-4, 62 % for 1e-2, 4.8 % for 0.2.
- 2000-step A/B on garden with the same recipe: det-gs (smoke-1) PSNR 23.01 / SSIM 0.636, gsplat 22.96 / 0.642.
  Training is stable and the result is neutral at this length.

## Is the DNGaussian borrowing justified?
Yes, for C1. DET-GS Eq. 4 is DNG's hard depth (`render_for_depth`: constant opacity 0.95, scales and rotations
detached, so the gradient reaches the means only). Eq. 5 is `patch_norm_mse_loss`. Eq. 6 is `patch_norm_mse_loss_global`
(patch mean, whole-image std, detached). γ=0.1 and η=1 are exactly DNG's weights in `train_llff.py:103,110`.
DNG draws an independent randint patch size per term, in (5,17) for LLFF; DET-GS uses [5,20], and the plugin matches.
δ = 1e-2·std(patches) matches DNG `normalize()`. The plugin's ω, δ, patchify, margin_l2, the detach placement and the
`-disp` vs DNG's `255 - disp` choice all match DNG operator for operator (`losses.py:26-60`, `renderer.py:19-32`).
DET-GS does **not** take over DNG's soft-depth branch or its `loss_depth_smoothness`, and the plugin correctly leaves both out.

## Issues

### Major
1. **Error tolerance 2.5e-4 makes the paper's "error tolerance mechanism" a no-op** (`losses.py:13`, paper §IV-B
   "To mitigate over-constraining … we introduce an error tolerance mechanism").
   - The value comes from DNG `scripts/run_llff.sh` (3-view LLFF, -r 8, random init). DNG's other configs use larger values:
     `arguments/__init__.py:92` default 0.2, DTU and LLFF-MVS 0.01, Blender 0.001/0.01/0.2.
   - On the normalised maps, 2.5e-4 excludes about 1 % of entries, so the loss is plain L2. The paper explicitly adds a
     tolerance, and with a full-view dense prior the risk of over-constraining is larger than in 3-view LLFF.
   - 0.2 would drop about 95 % of image-term entries. The choice therefore decides how strong the dominant (η=1) term is.
   - Fix: don't present 2.5e-4 as "the DNG value". Either:
     - pick DNG's code default 0.2 (or 0.01, its value for real-scene MVS-init configs) as the primary and justify it, or
     - keep 2.5e-4, but run a labelled sensitivity ablation (e.g. garden, 2.5e-4 / 0.01 / 0.2) and report every run.
   - Do not select the margin on test PSNR without labelling the result "tuned". Record the choice as fidelity gap #1
     with the numbers above.

### Minor
2. **Depth vs disparity space** (`losses.py:63-66`). `-disp` vs z is DNG-faithful. Patch normalisation is affine-invariant
   but not inversion-invariant, so it matters most for the image-level term in unbounded 360 scenes.
   - The current official 3DGS framework (which DET-GS says it builds on) renders *inverse* depth and regularises it
     against DA-V2 disparity directly.
   - The data do not favour either space (correlation 0.44 vs 0.28). Keep the current choice, but add a `space: depth|disparity`
     option and name this ambiguity as a fidelity gap.
3. **Edge loss magnitude** (`losses.py:107-110`). The logged L_edge is ~1e-4, i.e. ~1e-5 after β, against a photometric
   loss of ~0.1. The paper's ablation credits L_edge with about 0.2 dB.
   - Depth is in COLMAP units, the same as 3DGS, so this is faithful *if* the paper used z/distance depth. It would be
     much larger in inverse-depth units.
   - Before claiming C2 has any effect, log its gradient norm on the means and opacities relative to the photometric loss,
     or run an ablation.
4. **Optimiser step structure** (`losses.py:74-84`). DNG back-propagates hard depth with its *own* `optimizer.step()` (an extra
   Adam step on the means). The plugin sums it into the shared step. DET-GS does not say which it uses. This is a small
   difference; note it.
5. **TV channel normalisation** (`losses.py:116-123`). The loss divides by H·W·3, so channels are averaged. Eq. 15 uses
   1/|Ω| with Ω = pixels. Summing the channels instead would give an effective ϕ of 2.4. This is a documented interpretation;
   keep it.
6. **z-depth instead of ‖μ−o‖** (Eq. 3/4; `renderer.py:28`). This is documented. It is ≤1/cos(FOV/2) at the corners and
   partly absorbed by normalisation.
7. **Logging artefact.** `--log-every 500` (and 100 in the smoke run) always falls on a step with step%5==0, so the logged
   `det-gs-depth` and total loss overstate the average objective by including the depth term every time. Cosmetic.

### Checked and OK
- Hard depth: Eq. 4 = gsplat "D" with every opacity ω=0.95. It is not alpha-normalised. Scales and quats are detached and
  the opacity is constant, so only the means get gradient (the probe asserts this). The rendered gsplat opacity·G ≤ 0.95,
  so the gsplat 0.999 gradient-drop bug does not apply.
- Schedule: depth on step%5==0, steps 5–30000 (DNG starts at 0). Edge and TV run every step. The paper gives no start or stop.
- Edge (Eq. 8–11):
  - Canny(20,200) is computed on the grayscale GT train image at training resolution (`prepare_detgs.py:327`), the image the loader decodes.
  - The non-edge mask, the cross 3×3 with zero padding, ε=1e-8, P = number of non-edge pixels and β=0.1 are all correct.
  - It uses the soft depth D = ED·alpha = Σwz (Eq. 3, not normalised). The probe is a per-pixel loop.
- TV (Eq. 13–15): forward differences, the per-channel GT mask |∇I_gt| < 1e-2 in [0,1] units, hinge at 1e-4 and ϕ=0.8 are correct.
- Photometric: 0.8·L1 + 0.2·D-SSIM (3DGS λ=0.2). Eq. 16 "following 3DGS" is read correctly.
- The probes would fail if their mechanism were removed (independent loops, mask ablations, schedule and gradient-leak checks).
- Packaging: the entry point `det-gs = ss_det_gs:method` matches the name passed to make_method.
- Priors are found through `--data` in argv (bench passes it). The GPU cache holds about 0.9 GB (garden) or 1.7 GB (counter)
  of priors; the default `--min-free-gb 22` covers this.

## Launch (after resolving the major issue)
```
mkdir -p /bigdata/sejain/runs/det-gs-mipnerf360 && ln -s /bigdata/sejain/runs/det-gs-mipnerf360 /home/sejain/splatify/runs/det-gs-mipnerf360
cd /home/sejain/splatify && python3 tools/bench.py launch --exp det-gs-mipnerf360 --methods det-gs \
    --dataset mipnerf360-detgs --scenes garden counter --steps 30000 --recipe 3dgs
```
Compare by hand against `baseline3dgs-mipnerf360-30000` (same protocol_id; the dataset key differs). The paper's gains are
+0.36 dB on garden and +0.95 dB on counter. /bigdata has 115 GB free (99 % full).
