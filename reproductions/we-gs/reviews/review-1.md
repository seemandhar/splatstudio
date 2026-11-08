# WE-GS review 1 (splat-reviewer)

**Verdict: NOT approved.** 1 critical, 2 major, 6 minor.

What I read: paper.md, plan.md, paper_results.json, bench_request.json, all of `plugin/ss_wegs/`,
`source/latex/section/{3method,4experiments}.tex`, released WE-GS inference model
(`source/code_langsplatw/in_the_wild_renderer/*`, `arguments/__init__.py`), Ha-NeRF `losses.py`
(rover-xingyu/Ha-NeRF@main, the regulariser the paper says Eq. 20 follows), runtime
`splatstudio/runtime.py`/`data.py`, and smoke evidence.

Reviewer probes: 11 short runs, at most 1500 steps each, Brandenburg, downscale 2, GPU 6. Script:
`/tmp/claude-1013/-home-sejain/6ad4135e-056e-4bd4-9844-6b8764bea041/scratchpad/chk/variant.py`.
It monkey-patches the plugin and does not edit it. All runs use the quarter-res `L_regM` sum unless
`mean` is noted.

| variant | step-600 Gaussians | photometric @600 | mask | val PSNR @end |
|---|---|---|---|---|
| quarter (=reading A, 256² encoder input, 64×64 M) | 3.3k | 0.0000 | **M=0** | 6.5 |
| quarter, L_regTS off | 100k | 0.0000 | **M=0** | 8.5 |
| quarter, U-Net bias init +3 (M≈0.95 at start) | 4.1k | 0.0000 | **M=0** | 7.7 |
| native aspect (encoder on 336×512 → M 84×128 = paper W/4×H/4) | 3.4k | 0.0000 | **M=0** | 8.5 |
| native, L_regTS off | 100k | 0.0000 | **M=0** | 8.5 |
| mean reduction, λ=0.1 | 3.4k | 0.0000 | **M=0** | 7.4 |
| mean, λ=0.4 / λ=1.0 (tuned, not paper) | 100k | 0.09 | active | 18.7 / 18.1 @1k |
| native, **no SSIM term**, L_regTS off | 82k | 0.029 | active, mean(1−M)²≈0.2 | 17.4 @1k |
| quarter, **M detached inside SSIM** | 3.5k | 0.0000 | **M=0** | 7.3 |
| native, **M detached inside SSIM** (L_regTS on) | 15k | 0.086 | active, mean(1−M)²≈0.14–0.18 | 18.0 @1k |
| reference: plugin default (reading B), evidence/smoke-2 | 205k@2k | 0.07–0.15 | inert, mean M 0.97 | 19.5 @2k (gsplat 19.7) |

## Critical

**C1. Eq. 19 + Eq. 20 as implemented have a degenerate optimum. The shipped default ("full") avoids
it only by making the transient mask inert, which turns contributions C3/C4 into no-ops.**
`losses.py:23-26` (masked_photometric), `losses.py:11,29-32,76-85` (mask_reg).

- **No sign or lr bug.** `(1−M)²` with M = static mask penalises masking. That matches Ha-NeRF,
  where `mask_regularize` is `mean(mask_transient²)·1.5e-6`. The U-Net lr is 1e-5, as in the code
  (`itw_TEParams.mask_net_lr`). Neither L_regTS nor the mask initialisation causes the collapse:
  turning L_regTS off still collapses, and so does a bias init with M≈0.95.
- **The cause is the D-SSIM term on masked images.** `SSIM(M⊗I_r, M⊗I_gt) → 1` as M→0. That is a
  gain of about 0.2·(1−SSIM) ≈ 0.1 per step. The largest possible cost of L_regM is
  `1e-5·W'H'` = 0.04 at 64×64, or 0.11 at 84×128. The L1 term alone is held in check: per mask
  pixel the equilibrium is `1−M ≈ 4e4·ē/(W'H')`, and the "native, no SSIM" run keeps an active,
  non-degenerate mask.
- **Reading B (sum over R(1−M) at W×H, ~170k px) is not literal and makes the mask inert.**
  Eq. 20 is written on `M_k ∈ R^{1×W'×H'}`; `R(1−M)` only defines the transient mask. Reading B
  multiplies the weight by about 40×. Per pixel `1−M ≈ 0.2·ē`, so M stays near 0.97 and L_regTS
  is exactly 0 from step 200 (smoke-2). A 150k run with this default is effectively
  "3DGS + residual SH, no transient handling". Any PSNR gap would then be blamed on the wrong
  component.
- **Most literal form that does not collapse (checked: native + SSIM-detach row):**
  - Eq. 20 as the squared Frobenius sum over `M_k` at W/4×H/4 of the training image, λ1=1e-5.
  - Eq. 19 unchanged in value, but M detached inside the SSIM term only, so the mask learns from
    the L1 term and L_regM.
  - Record this as deviation "SSIM-term mask stop-gradient: the paper is silent; the literal form
    collapses (review-1 table)".
  - Keep "full" only as a labelled ablation. Do not launch 150k runs with the current default.

## Major

**M1. Encoder input is resized to 256×256, but the paper uses W'=W/4, H'=H/4 of I_k.**
`networks.py:66-69`, `bank.py:21`. References: paper §4.2 Eq. 14/17 and §5.1 "W' and H' to be 1/4
W and H". The 256² resize comes from the later TVCG code (`te_unet.ConvMaskNet`, `ae_cnn.encode`),
not from the paper. It changes the aspect ratio and the mask pixel count (4096 vs ~10.7k at
downscale 2). Because L_regM is a sum, that is a 2.6× change in the effective λ1. Even with the
SSIM fix, the 256² variant collapses (quarter + SSIM-detach row).

- Fix: run Enc¹ on the training-resolution image, resized to a multiple of 16 for the two stride-2
  U-Net convolutions.
- Update the `attention-encoder` probe, which currently asserts W/4 of the 256 input
  (`probes.py:132`).

**M2. L_regTS ε=0.5 with a sigmoid-initialised mask (M≈0.47 at step 1).** `losses.py:13,50-63`.
References: Eq. 21; ε is not given in the paper.

- At step 1 every visible Gaussian counts as transient (`wegs-transient-opacity-reg` 0.088 in
  smoke-1/2).
- In the working native + SSIM-detach variant, L_regTS still prunes 100k → 15k Gaussians by step
  600.
- Under reading B it is identically 0.
- Fix: make ε a recorded choice. A lower ε (e.g. 0.1–0.2), or an indicator over the mask once it
  has trained, needs a short check. Report how many Gaussians L_regTS touches over training
  (`counters.transient_gaussians_last` already exists; log it).

## Minor

1. **`bank.py:39`: `torch.manual_seed(seed)` inside `density.initialize`.** It resets the global
   torch RNG to bank_seed=0 after the runtime seeded it with `--seed`, so torch-side randomness is
   identical across seeds. View sampling uses Python `random` and is unaffected. Use a local
   `torch.Generator` for the initialisation.
2. **`evaluate.py:83`: `right_half` is not the NeRF-W protocol.** The appearance is still encoded
   from the full test image. Keep the label "reference only" and do not compare it to the SWAG or
   WildGaussians right-half numbers.
3. **`losses.py:35-38`: L_regSH is a mean over Gaussians, not the paper's Σ_j (Eq. 13).** This is a
   documented reduction choice. Fine, but list it in the report.
4. **`networks.py:89-111`: G_φ has 4 layers (paper) while the released model has 8 layers with
   skips.** Documented; keep it in the fidelity gaps.
5. **`density.py:73-75`: the position lr is set for step s+1 from `step`.** It is off by one,
   which is negligible. It does correctly override the runtime's 150k decay with the code's 30k
   `position_lr_max_steps`.
6. **Black background (`renderer.py:19`) vs runtime protocol `"background": "white"`.**
   PhotoTourism JPEGs have no alpha, so GT is unaffected and this matches the code. Mention it in
   the report.

## Checked and OK

- **Evaluation protocol** (`evaluate.py`):
  - Appearance l_k is encoded feed-forward from the full test image, and the full image is scored
    with no test-time mask and no optimisation. This is the Ha-NeRF / CR-NeRF / GS-W protocol that
    WE-GS §5.3 compares against; the paper notes that only NeRF-W and K-Planes rows use
    optimisation.
  - The evaluator applies no masking and does not leak beyond what the paper's protocol does.
  - `itw_evalutate_with_gtmask` in the TVCG code is unused there and is not a PhotoTourism
    protocol.
  - Global state is handled correctly: there is a guard against a live bank, `ACTIVE` is reset in
    `finally`, `from_export` freezes grads and sets `eval()`, and the specification/protocol ids
    are checked.
- **Data:**
  - NeRF-W tsv split: Brandenburg 763/10, Sacré-Cœur 830/21. The validation files are copies of
    the test images, not used for selection.
  - Downscale 2 with LANCZOS matches the commented-out §5.1 text, Ha-NeRF `img_downscale=2`,
    CR-NeRF and GS-W.
  - SWAG's reproduction uses full resolution and WildGaussians uses the right-half protocol, so
    their numbers are not directly comparable; compare only via each paper's 3DGS delta.
- **3DGS parts:** densification 500–15000 every 100, grad 2e-4, reset 3000, base lrs and
  position_lr_max_steps 30000 all match the code's `OptimizationParams`.
- **Model details:** SH residual (tanh, PE_sh=0, PE_xyz=4 freqs matching the code's
  `PositionalEncoding(n−1, n)`); `sh` not detached (as in the code). Densification goes through
  gsplat ops, so the "requires_grad" pitfall from TODO-runtime does not apply. Probes verify
  passed.

## Equation coverage

| Eq. | Status |
|---|---|
| (9)(10) | exact (paper depth; code width/tanh) |
| (13) | deviates (mean reduction) |
| (14)–(18) | deviate (256² input, see M1; CAM/U-Net widths assumed) |
| (19) | exact value, but degenerate with mask gradient through SSIM (C1) |
| (20) | deviates (default sums at W×H, see C1) |
| (21) | exact form, ε assumed (M2) |
| (22) | exact |
