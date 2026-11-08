# Sort-free Gaussian Splatting via Weighted Sum Rendering — implementation spec

arXiv 2410.18931 (ICLR 2025, Qualcomm AI Research). Sources: `source/latex/` (camera-ready
LaTeX incl. appendix and rebuttal), `source/paper.pdf`. **No official code.** Unofficial
re-implementation (linked by the first author as reference code): `source/code_unofficial/`
(github.com/LiYukeee/sort-free-gs, 3DGS fork) + `source/code_unofficial_raster/`
(LiYukeee/sort-free-diff-gaussian-rasterization). Read only, never run. The paper is the
authority; differences are listed in §7.

## 1. Method summary
- Replace 3DGS's sorted alpha compositing with an order-independent **Weighted Sum Rendering
  (WSR)**: pixel colour = normalised weighted sum over all Gaussians plus a learnable-weight
  background term. No depth sort, no transmittance, no early termination.
- Per-Gaussian weight depends on the Gaussian's camera depth through a learnable weight function;
  the paper's main/best variant is **LC-WSR** `w = max(0, 1 - d/σ) v_i` (global learnable σ,
  per-Gaussian learnable v_i). DIR-WSR (w=1) and EXP-WSR (w = exp(-σ d^β)) are ablations.
- **View-dependent opacity**: the 3DGS max-opacity t_i ∈ [0,1] is replaced by an unclamped
  value u_i evaluated from a per-Gaussian spherical-harmonic expansion of the view direction.
- Loss, densification (clone/split) and other training unchanged from 3DGS; opacity-threshold
  pruning removed.

Contributions
- C1 Weighted Sum Rendering (Sec. 3.2, Eq. 7) with learnable background weight — custom
  forward/backward kernel (appendix "derivative" text, not \input in camera-ready).
- C2 Learnable depth-weight function LC-WSR (Sec. 3.2, Eq. 10), learnable σ and v_i.
- C3 View-dependent opacity via SH (Sec. 3.3, Eq. 11).
- C4 Training recipe: 3DGS loss, 3DGS densification, no opacity pruning (Sec. 3.4, App. "More
  discussions").

## 2. Equations
Eq. 7 (WSR):
$$\mathbf{C} = \frac{\mathbf{c}_Bw_B + \sum_{i=1}^{\mathcal{N}} \mathbf{c}_i \alpha_i w(d_i) }{ w_B + \sum_{i=1}^{\mathcal{N}} \alpha_i w(d_i)}$$
c_B, w_B: background colour and learnable background weight; d_i depth; w(·) learnable weight
function. α_i is the 3DGS per-pixel alpha (Eq. 2: α_i = t_i · exp(-½ Δᵀ Σ'^{-1} Δ)), with t_i
replaced by u_i (Eq. 11). "α_i serves as a learnable parameter … constraints [α∈[0,1]] are not
necessary". Rebuttal: d is "the same depth … used for sorting in 3DGS" (view-space z of the centre).

Eq. 8 DIR-WSR: $w(d_i) = 1$.
Eq. 9 EXP-WSR: $w(d_i) = \exp(-\sigma d_i^\beta)$, σ, β learnable (init σ=0.1, β=0.8).
Eq. 10 LC-WSR: $w(d_i) = \max\left(0, 1 - \frac{d_i}{\sigma}\right) v_i$, σ, v_i learnable
(init σ = 10, v_i = 0.1). (Derivative appendix rewrites it as max(0, 1-σd) v_i "for
simplification"; the camera-ready main text and the unofficial code use d/σ.)
Eq. 11: $u_i = Y(\lVert f - p_i \rVert, H_i)$ — SH Y of the view direction (camera centre f to
Gaussian p_i) with learnable coefficients H_i; "u_i is not clamped".

Derivatives (appendix text, w_s = w_B + Σ α_i w(d_i) saved in forward):
∂L/∂α_i = w(d_i)(c_i − C)/w_s · ∂L/∂C;  ∂L/∂c_i = α_i w(d_i)/w_s · ∂L/∂C;
∂L/∂w_B = (c_B − C)/w_s · ∂L/∂C;  ∂L/∂w(d_i) = α_i (c_i − C)/w_s · ∂L/∂C;
LC: ∂L/∂v_i = max(0, 1−σd_i) ∂L/∂w; ∂L/∂σ = −Σ v_i d_i ∂L/∂w where 1−σd_i>0 (in the 1−σd form).
These are exactly the chain rule of Eq. 7 — our kernel computes the two linear sums and lets
autograd differentiate the quotient, which yields the same expressions.

Loss (Sec. 3.4): 3DGS L = 0.8 L1 + 0.2 D-SSIM. Final image clamped to [0,1] (rebuttal).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| LC σ init | 10 | Sec. 3.4 |
| LC v_i init | 0.1 | Sec. 3.4 |
| EXP σ, β init | 0.1, 0.8 | Sec. 3.4 (not implemented, ablation only) |
| u_i init | not stated; 0.1 (3DGS t_i init; unofficial SH-DC = 0.1/C0) | code_unofficial scene/gaussian_model.py:181-183 |
| w_B init | not stated; 0.02 | code_unofficial gaussian_model.py:198 |
| c_B | background colour (3DGS: black; our runtime: white) | Eq. 7; runtime convention |
| SH degree of u_i | not stated; same active degree as colour (0→3 every 1000) | code_unofficial gaussian_renderer:56 |
| lr σ | not stated; 0.1·extent → 0.001·extent exponential over 30k | code_unofficial arguments:87-90 |
| lr w_B | not stated; 0.01 | arguments:91 |
| lr v_i | not stated; 0.01 (unofficial: sigmoid param) | arguments:85 |
| lr u SH DC / rest | not stated; 0.05 / 0.0025 | gaussian_model.py:210-211 |
| other lrs, densify 500–15000 every 100, grad 2e-4, percent_dense 0.01 | 3DGS defaults | App. "Densification … maintained without modification" |
| opacity pruning | removed | App. "More discussions" |
| opacity reset | not stated (meaningless for unclamped u); disabled in unofficial | train.py:142 |
| iterations | 30k (3DGS protocol) | Sec. 4 "followed the evaluation setting of 3DGS" |

## 4. Schedules
SH degree +1 every 1000 steps (colour and opacity SH). σ LR exponential decay (unofficial).
Densification 500–15000. Means LR 3DGS schedule.

## 5. Builds on
Original 3DGS (Inria). Unchanged: projection/EWA, SH colour, loss, clone/split densification,
optimiser. Changed: compositing (WSR), opacity (view-dependent, unclamped), pruning (no opacity
prune).

## 6. Protocol
13 scenes: Mip-NeRF 360 (all 9), T&T truck/train, DB drjohnson/playroom; 3DGS evaluation setting
(llffhold 8, images_4 outdoor / images_2 indoor). PSNR/SSIM/LPIPS; 3DGS numbers in Table 1
are those of the 3DGS paper. Per-scene: appendix Tables 6–8 (LC-WSR only). Speed measured in
Vulkan on Snapdragon 8 Gen 3 (not reproducible here, not targeted). "For convenience, we also
evaluate the quality of our method with a PyTorch inference simulation."

## 7. Ambiguities and paper vs unofficial code
1. **Roles of u_i and v_i**: paper α_i = u_i·G (u from SH, unclamped), w = max(0,1−d/σ)·v_i.
   Unofficial: α_i = sigmoid(o_i)·G, w = max(0,1−d/σ)·SH(dir) — names swapped, but only the
   product u_i v_i max(0,1−d/σ) G enters Eq. 7, so the two are the same model up to the range
   constraint on the scalar factor (sigmoid vs raw). We follow the paper's naming with v_i =
   sigmoid(logit) ∈ (0,1) (unofficial numerics; u is unconstrained so the product is too).
2. **Sign**: paper says u_i is unconstrained; unofficial culls Gaussians with u ≤ 0 or v ≤ 0
   (and d ≥ σ). Negative weights could make the denominator vanish. Default: cull u ≤ 0
   (config `cull_nonpositive`), recorded as a gap.
3. **Alpha threshold / footprint**: not stated. Unofficial skips |α·w| < 1/255 (threshold on the
   weighted value) inside a 3σ box. We skip α = u·G < 1/255 (3DGS threshold on α, Eq. 2, not on
   α·w which is scale-dependent) within 3.33σ (gsplat footprint).
4. **Densification**: "3DGS densification maintained without modification; removed pruning
   based on opacity threshold". Unofficial removes all pruning; we keep the 3DGS world-size
   prune (scale > 0.1·extent after step 3000; the screen-size prune is a no-op in 3DGS).
5. Opacity reset not mentioned; disabled (as unofficial).
6. Denominator guard: unofficial replaces w_s == 0 by 1e-4; we clamp w_s ≥ 1e-4.
7. Learning rates of new parameters, u init and w_B init are not in the paper (unofficial values).
8. σ in world units; init 10 independent of scene scale. d ≥ σ ⇒ weight 0.
