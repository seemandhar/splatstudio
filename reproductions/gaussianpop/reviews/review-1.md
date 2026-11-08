# GaussianPOP — review 1

**Verdict: APPROVED for benchmark** (0 critical, 0 major, 6 minor). Plugin not edited.

Read: paper.md, paper_results.json, plan.md, all of plugin/, source/latex/sec/{3_proposed,4_results,suppl}.tex,
splatstudio runtime.py / backends/gsplat.py. Ran: `pytest` (6 passed); an extra kernel-vs-brute-force check
on 3 fixtures (below).

## 1. Error formula (C1/C2) — exact
- Eq. 2–4 + Supp. "Derivation of the Background Term": b_{k+1} = Σ_{j>k} (T_j/T_{k+1}) α_j c_j =
  (C_render − P_k)/T_{k+1}; Alg. 1: `b ← (C − P_k)/(T_{k+1}+ε)`, ε=1e-9, `Δ_SE[ID] += ||T_k α_k (c_k − b)||²`.
  `csrc/pop_error.cu:111-118`: P is updated *before* use (inclusive P_k), `next_T` = T_{k+1}, `w = T_k α_k`,
  squared L2 over RGB, no normalisation, atomic sum into the global buffer. Matches operator by operator.
- Summation: all pixels of all training views (`kernels.py:scene_error`, cameras = full train split at
  training resolution, runtime.py:155-160) = §3.4 "over the complete set of training views". No subset, no
  normalisation (paper has none). ✔
- Background: the paper's Eq. 1 has no background term (Inria black ⇒ bg contributes 0). The kernel adds
  T_final·bg to C_render (`pop_error.cu:126`), so b_{k+1} contains the white background seen through the
  remainder. For a model trained/evaluated over white this is the correct counterfactual (removing k reveals
  white), and it reduces to the paper exactly for black. Keep white. ✔ (fidelity gap 4 correctly documented)
- c_k = SH colour +0.5 clamp_min 0, degree 3, dirs from camera centre (`kernels.py:project`) = gsplat
  `rasterization`; px = j+0.5, 1/255 threshold, 0.999 clamp, exclusive 1e-4 termination = gsplat 1.5.3.

## 2. Kernel correctness / probe independence
- `probes.py:delta_se_oracle` brute force uses gsplat `rasterization` (independent code path, not the
  kernel) with one Gaussian deleted, Σ_pix ||C − C'||², + discriminators (black bg, n_max=2, blend weight).
  Independent ✔. Weakness: fixture has α ≤ 0.3, n=24 (no termination, no N_max truncation).
- My extra check (scratchpad chk.py, GPU 3):

  | fixture | n_max | C_render err | rel. err max / median | Spearman vs oracle |
  |---|---|---|---|---|
  | n=24, α∈[.05,.3] | 64 | 2.4e-7 | 1.2e-6 / 3.9e-7 | 1.000 |
  | n=60, α∈[.3,.95] (termination) | 64 | 2.4e-7 | 1.3e-3 / 9.8e-7 | 1.000 |
  | n=150, α∈[.05,.6] | 64 | 3.0e-7 | 0.42 / 1.6e-5 | 0.99999 |
  | n=150 | 1000 | 3.0e-7 | 0.11 / 1.0e-5 | 0.99999 |

  Residuals come only from removals that change the termination set (an approximation inherent to Eq. 4) and
  from N_max truncation; totals agree to 1e-6. Kernel is correct.
- Optimizer state: gsplat `ops.remove` (keeps `requires_grad`, slices exp_avg/exp_avg_sq, skips `step`, slices
  strategy state grad2d/count/radii) then `opt.state.clear()`; `density.py:110-112` asserts every field is a
  trainable leaf → the ResGS no_grad pitfall cannot occur. Strategy state sizes are consistent because no
  densification happens after 15k. ✔

## 3. Schedule / fine-tune
- Cycles: paper gives no ratio split for C=8 at fixed P (Tab. 4 splits an error budget B/C; the authors'
  commented supp. text uses r_step = 1−(1−P)^{1/N}). Compound (`schedule.py:15`) is the best-supported choice;
  final count is exactly (1−P)N0 like Tab. 8 (0.973 = 0.2·4.87M). ✔
- N_max=64 is from the paper (Supp. Impl. details: "fixed N_max to 64; 128 no meaningful difference").
  Semantics (only the first 64 contributors scored, C_render full) is an interpretation; measured effect is
  tiny (table above). ✔
- Fine-tune: paper states only "optional 5k iterations" (§4.1) — LR/optimizer unstated. Fresh Adam +
  means LR clamped at lr0·0.01 (= 3DGS get_expon_lr_func past max_steps, = LightGaussian-style continuation
  to 35k) is reasonable. Order is right: runtime sets its 35k decay (runtime.py:209) and the density
  overrides it afterwards (`density.py:72-74`), so steps 1–30000 have the identical schedule to the 30k
  baseline. Densification stops at 15k (gsplat refine_stop_iter), opacity reset only for step < 15000
  (backends/gsplat.py:85-87) → both off in fine-tune ✔. SH degree 3 throughout fine-tune ✔.

## 4. Targets / protocol
- paper_results.json = Supp. Tab. 8 ("Post-training pruning … 5,000 iterations with C=8") P=0.8 row:
  bicycle 25.12 / counter 29.00; P=0.7 25.17/29.14; P=0.9 24.32/28.40. ✔ Not the on-training Tab. 7.
- Protocol: 30k Inria 3DGS + 8 cycles + 5k FT = 35k total ✔; images_4/images_2 pre-downsampled (Supp.) =
  our data ✔; llffhold8 ✔.

## Minor issues
1. `reviews`/report: base-model size differs — our 30k gsplat bicycle has **6.59M** Gaussians vs the paper's
   3DGS 4.87M (Tab. 5a; implied by Tab. 8 0.973/0.2), counter 1.22M vs ~1.08M. At fixed P we keep 1.32M on
   bicycle (35% more than the paper). Report #G next to PSNR and state this; Δ-vs-unpruned is still the key
   metric.
2. `probes.py:delta_se_oracle` fixture never exercises termination or N_max; add a high-opacity fixture
   asserting median rel. err < 1e-3 and Spearman > 0.999 (numbers above).
3. Paper-faithful alternative for N_max (C_render and b from the truncated 64-list) not offered; record as
   ambiguity only (effect negligible).
4. Pre-prune model (step 30000) is not evaluated in the run; compare that the run's step-30000 loss /
   Gaussian count matches `baseline3dgs-mipnerf360-30000` (6,590,201 / 1,215,973) as a sanity check of
   identical pretraining; record cycle logs (`pop_prune_step` lines) in the report.
5. Fine-tune alternatives (means LR restarting at lr0 as in a fresh `train.py` from a .ply) are a plausible
   reading; if Δ is far from the paper, try it as a labelled variant.
6. plan.md lists `finetune_means_lr_ratio` but code uses `lr_final_ratio` (density.py:35) — doc mismatch only.

## Launch
```
mkdir -p /bigdata/sejain/runs/gaussianpop-mipnerf360 && ln -s /bigdata/sejain/runs/gaussianpop-mipnerf360 runs/gaussianpop-mipnerf360
python3 tools/bench.py launch --exp gaussianpop-mipnerf360 --methods gaussianpop --dataset mipnerf360 --scenes bicycle counter --steps 35000
```
(optional: add `gaussianpop-p07 gaussianpop-p09` to `--methods`; post-prune-only Tab. 3 check: separate exp
with `--steps 30000`, target bicycle 20.71 dB.)
