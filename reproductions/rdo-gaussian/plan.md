# RDO-Gaussian native plan

Method `rdo-gaussian` (highest rate point, λ_GS 0.0005 / λ_SH 0.005; the paper's per-scene
tables), extra entry points `rdo-gaussian-rate2` … `rdo-gaussian-rate6` (App. Tab. A.1–A.3 λ
pairs) for the RD curve, `rdo-gaussian-smoke` (2000-step compressed schedule, diagnostic only).
Module `ss_rdo_gaussian`.

## Architecture
Global learned state (6 codebooks + 6 logit vectors, their Adam and LR milestone, index cache,
activation flags) lives in `bank.RDOBank` (nn.Module) **owned by the density** and read by the
renderer through `bank.ACTIVE` — the EAGLES / Compact-3DGS pattern
(reproductions/eagles/plugin/ss_eagles/density.py, reproductions/compact-3dgs/plugin/...).
Per-Gaussian masks are FieldSpecs (runtime Adam, eps 1e-15 = official). Final encode →
bytes → decode → replace params is the Reduced-3DGS pattern
(reproductions/reduced-3dgs/plugin/ss_reduced_3dgs/density.py `_encode`). A fresh process
(`ss-eval`) has no active bank → plain gsplat render of the decoded params.

| component | kind | factory | config (paper value) |
|---|---|---|---|
| rdo-renderer | renderer | `renderer:create_renderer` | — (reads bank); provides rgb, depth, gradient.screen.signed |
| rdo-density | density | `density:create_density` | base 3DGS density (refine 500–15000, reset 3000), total_steps 30000, gs/sh/vq start 15001, rate_constrain 20001, reactivate period 1000 / p 1e-5, thresholds 0.1/0.1, cb sizes 8192×3 / 4096×3, λ_s 32768, λ_r,dc,sh 256, cb lr 2e-4, logits lr 2e-3, milestone 12000 ×0.1, patch 65536, kmeans 10 iters, rANS precision 16, opacity bits 8, evaluate "decoded" |
| photometric | loss | splatstudio | (1−0.2) L1 + 0.2 D-SSIM (Eq. 12) |
| rdo-rate | loss | `losses:create_rate` | weight 1 (Eq. 11, code normalisation /N/55) |
| rdo-vq | loss | `losses:create_vq` | weight 1 |
| rdo-gs-prune | loss | `losses:create_gs_prune` | weight λ_GS = 0.0005 |
| rdo-sh-prune | loss | `losses:create_sh_prune` | weight λ_SH = 0.005 |

Fields: `rdo_gs_mask` () lr 0.01 init 0; `rdo_sh_mask` (3,) lr 0.05 init 0. Gradients are None
before activation (renderer does not read them) → Adam does not step them, as official
(mask optimizers only stepped once active). Densification copies them (gsplat ops), as official
`densification_postfix`.

## Per-step order (native vs official)
Official iteration: [15001: k-means init] [20001: rate_constrain] render → loss → backward →
3DGS/VQ/mask optimizer steps (skipped at 30000) → [30000: mask apply + encode] → [it%1000==0 ∧
it<20001: reactivate] → densification (<15000).
Native: render → losses → backward → runtime Adam (all per-Gaussian fields incl. masks) →
`after_backward`: bank Adam step (if VQ active and step < total; LR ×0.1 after 12000 bank
steps) → base 3DGS density → reactivate → [step == start−1: k-means init + activate] →
[step == 20000: rate_constrain on] → [step == total: encode/decode/replace].

## Renderer (training, bank active, not baked) — official `gaussian_renderer/__init__.py`
s = exp(scales), α = σ(opacities); GS: hard = STE(σ(m) > 0.1); ŝ = hard·s, α̂ = hard·α;
SH: hard_l STE per degree expanded to 3/5/7 coefficients; shN̂ = hard·shN; VQ on rows
gs-bit (scale ŝ, rot raw quats, dc sh0) and sh_l-bit ∧ gs-bit (sh1/2/3 of shN̂):
index = argmin cdist(x, CB) (+ log2-pmf/λ once rate_constrain), recomputed when
(step−1) % 10 == 0 else from the global cache; x_q = sg(x̂ − x) + x; rot normalised.
Outputs `outputs["rdo"]`: rate = Σ bits/λ /N/55, vq = Σ‖x̂−x‖₂ /N/55, gs_prune = mean σ(m) over N,
sh_prune = mean expanded soft mask over gs-kept rows, plus diagnostics. Rasterisation by gsplat
with the activated attributes (cannot go through log-scale params: codewords may be ≤ 0).

## Encode (official `apply_and_save_vq` + `decode`)
remove ¬gs-bit Gaussians → shN ·= sh hard mask → negative scale codewords → 1e-8 →
rate-constrained full assignment (no clamp, no_grad as official) → codeword attributes →
stable sort by SH-mask integer (sh1 = bit 2) + official 8-entry boundaries → shrink codebooks →
fp16 xyz, 8-bit logit-opacity quantisation with histogram pmf → vendored CompressAI rANS
(16-bit CDFs) per attribute → files `index_bitstream.bin`, `header.bin`, `codebook.npz`,
`logits.npz`, `position.npz` built byte-exactly in memory. Size = Σ bytes / 1e6 MB. Decode the
bytes (rANS decode, header parse, npz load), check indices round-trip, replace params with
the decoded model, as the official render.py does (`evaluate: "official"` = diagnostic undecoded mode (fp32 xyz, unquantised opacity) that no official evaluation mirrors). Bitstream kept as uint8 tensors in density state; `report_size.py`
prints the report / writes the files.

## Probes
| probe | measures |
|---|---|
| gs-mask-ste | masked Gaussians (σ ≤ 0.1) render as removed; mask gradient = (∂L/∂ŝ·s + ∂L/∂α̂·α)σ' (independent chain); L_GS = mean σ |
| sh-mask-ste | SH degree masks zero exactly their 3/5/7 coefficients; L_SH = (2l+1)/15-weighted mean over gs-kept rows (independent formula) |
| ecvq | assignment = independent loop argmin ‖x−c‖₂ + bits/λ (rate on) / NN (off); bits, rate/vq loss normalisation /N/55; STE: render grad reaches x not codebook; codebook grad from vq loss, logit grad from rate |
| kmeans-reactivate | k-means init equals an independent Lloyd loop with same seeds; logits = log(bins/‖bins‖₂); reactivation revives p ≤ 1e-5 codewords only |
| rans-codec | C++ pmf_to_quantized_cdf = independent Python port; encode/decode lossless; bytes ≈ ideal entropy |
| encode-decode | fixture encode → bytes → decode: decoded params = codewords (log-scale, raw rot, dc, SH zeros on pruned degrees), sort/boundaries = independent computation, size = Σ official file sizes written with np.savez to disk; fresh renderer reproduces baked render |
| schedule | activation 15000→15001, reactivation 16000–20000, rate on at 20001, bank LR ×0.1 from bank step 12000 (it. 27001), no bank step at total, encode at total |
| rd-losses | the four RD losses are 0 (but attached) before activation and weight × renderer term after |

## Fidelity gaps
1. Runtime still applies one Adam step at step 30000 (official skips the last optimizer step) — negligible.
2. Runtime differences shared with all ports: white background (official black), random view sampling with replacement, SSIM valid conv, gsplat rasterizer; LPIPS-VGG from `lpips`.
3. Evaluation default = decoded bitstream (fp16 xyz, 8-bit opacity), which is also what the official render.py evaluates (Scene.__init__ → gaussians.decode); not a gap. `evaluate: "official"` is a diagnostic undecoded mode (fp32 xyz, unquantised opacity) that no official evaluation mirrors; reviewer's estimate of the fp16-position/8-bit-opacity effect vs the undecoded model: 0.01-0.05 dB. (Corrected after review-1.)
4. k-means / reactivation randomness differs from torch-on-their-GPU seeds (stochastic anyway).
5. Exactly-zero scale codewords would give log(0); clamped at 1e-12 (official stores −inf → 0 scale); essentially unreachable.
6. Dataset: `mipnerf360-1600` (official -r -1 loader); our gsplat baseline must also run there for the improvement comparison.
7. An attribute with zero coded symbols (no Gaussian keeps an SH degree) is stored as an empty
   rANS string (4-byte length prefix); the official CompressAI encoder asserts on it (found in
   the smoke test).
8. Run time: VQ assignment over all Gaussians every 10 steps + 6 codebooks; smoke 839 s vs 286 s
   baseline at 2k steps on contended GPUs — the full 30k run will be noticeably slower than gsplat.
