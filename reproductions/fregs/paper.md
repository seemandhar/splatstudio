# FreGS (arXiv 2403.06908, CVPR 2024) — implementation spec

Sources: `source/latex/` (main.tex, sec/3_method.tex, sec/4_experiment.tex), `source/paper.txt`.
Official code: NOT found (project page https://rogeraigc.github.io/FreGS-Page/ links only arXiv; GitHub search / web search found no official repo;
`SaiJoshika/FreGS-3D-Gaussian-Splatting` is a 2025 third-party student repo built on mip-splatting, unreviewed, not cloned; treat as unofficial).
The arXiv source has NO supplementary (`\input{sec/X_suppl}` is commented out and the file does not exist), so no per-scene tables, no loss weights, no D0/T0 values.

## 1. Method summary / contributions
- Adds a Fourier-space loss on top of 3DGS's L1 + D-SSIM (which stays for the whole training): amplitude and phase discrepancies between rendered and GT image.
- C1 (Sec 3.2) Frequency regularization: amplitude + phase discrepancies over all frequency components.
- C2 (Sec 3.3) Frequency annealing: low-pass always on; a dynamic high-pass band that grows from D0 to D between T0 and T; coarse-to-fine densification.
- Densification claim (Sec 3.2, Fig 2): frequency loss yields larger per-pixel gradients in over-reconstructed regions, so the standard 3DGS view-space-gradient densification (split) triggers there. NO separate densification rule is described; the effect is only through gradient flow. To keep the Gaussian count similar to 3DGS the authors INCREASE the gradient threshold (Sec 4.2, Table 2 caption; value not given).

## 2. Equations (verbatim from sec/3_method.tex)
(DFT, Eq.2)  `F(u, v) = \sum_{x=0}^{H-1} \sum_{y=0}^{W-1} I(x, y) \cdot e^{-i2\pi(u\frac{x}{H}+v\frac{y}{W})}`
`|F(u,v)| = \sqrt{Re(u,v)^2 + Im(u,v)^2}`; `\angle F(u,v) = arctan(Im(u,v)/Re(u,v))`
(Eq.5 amplitude) `d_a = \frac{1}{\sqrt{HW}} \sum_{x=0}^{H-1}\sum_{y=0}^{W-1} \big| |F(u,v)| - |\hat F(u,v)| \big|`
(Eq.6 phase) `d_p = \frac{1}{\sqrt{HW}} \sum_{x,y} \big| \angle F(u,v) - \angle \hat F(u,v) \big|`
Filtered: `LF = F H_l`, `HF = F H_h`; `d_{la}, d_{lp}, d_{ha}, d_{hp}` are Eq.5/6 applied to LF and HF (same 1/sqrt(HW) and sum, abs difference).
(Band) `D_0 < D_t < \frac{(t-T_0)(D-D_0)}{T-T_0} + D_0`, with D0 = max range of low-pass, D = max range of the spectrum, origin at spectrum center (H/2, W/2); t current iter, T0/T start/end of introducing high frequency.
(Loss)
```
L_f = w_l (d_la + d_lp)                         , 0 < t <= T0
L_f = w_l (d_la + d_lp) + w_h (d_ha + d_hp)     , t > T0
```
Total = L1 + D-SSIM (3DGS, lambda 0.2) + L_f. The paper never writes how L_f is added (assumed plain sum with weights w_l, w_h).

Note on "Euclidean metric" + abs: per-element absolute difference, summed over all (u,v) (the loop index is written x,y but means u,v), scaled by 1/sqrt(HW) — it is a SUM, not a mean, so magnitude scales with sqrt(HW). Channel handling is unstated (rendered image is HxWxC; natural reading: per-channel FFT, sum over channels).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| w_l, w_h | NOT GIVEN | — |
| D0 (low-pass radius), D (max radius) | NOT GIVEN (D = max spectrum radius) | — |
| T0 (start of high-freq), T (end of annealing) | NOT GIVEN | — |
| freq-loss stop iteration | 15000 (frequency regularization terminates when densification ends) | Sec 4.1 |
| densify stop | 15000 (as 3DGS) | Sec 4.1 |
| densify gradient threshold | raised vs 3DGS 0.0002 to match Gaussian count; value NOT GIVEN | Sec 4.2 |
| resolution schedule | start 4x downsampled, upsample to full after 500 iters (as 3DGS as written) | Sec 4.1 |
| optimizer | Adam, PyTorch, original 3DGS CUDA rasterizer | Sec 4.1 |
| iterations | 30000 (Fig. 5 shows 7k/15k/30k) | Fig 5 |
Everything else = 3DGS defaults.

## 4. Schedules
Low-frequency terms active for t in (0, 15000]; high-pass band radius grows linearly from D0 at T0 to D at T (upper bound formula above; the paper writes an inequality, the natural implementation is D_t equal to the upper bound, mask = D0 < r <= D_t, r = distance from fftshifted center). Frequency loss off after 15000.

## 5. Builds on
3DGS unchanged otherwise (cloning/splitting, opacity reset, L1+0.2 D-SSIM, 30k iters). Contributions only touch the loss (and indirectly the densification threshold).

## 6. Protocol
All 9 Mip-NeRF360 scenes + 2 T&T scenes (text) and Deep Blending column in Table 1; llffhold 8 test split; 3DGS resolutions; metrics PSNR/SSIM/LPIPS (LPIPS net unstated). Seeds unstated.

## 7. Ambiguities (no official code to resolve them)
1. w_l, w_h, D0, T0, T values absent. Must be chosen and labelled as such (tuned/assumed).
2. Mask shape (circular vs square), hard vs soft, whether D0<r strict, whether H_l is ideal (hard) filter. Whether low-pass is fixed radius D0 and high-pass is the annulus D0..D_t (paper's text suggests so).
3. Whether the filters are applied before |.|/angle (as written: LF = F*H_l, then amplitude and phase of LF). Phase of zeroed bins is atan2(0,0)=0 -> contributes 0 for both images; amplitude 0 too.
4. Phase difference: raw angle difference, no wrapping stated (Eq. 6 uses plain |angle - angle|); arctan written, atan2 presumably. Gradient of angle is singular at zero amplitude.
5. Per-channel vs gray; sum vs mean (1/sqrt(HW) with sum is written); FFT normalisation ("backward" unnormalised DFT is as written; torch.fft.fft2 default matches).
6. How freq loss gradients change densification beyond raising the gradient threshold: unspecified.
7. Table 1's 3DGS row is the 3DGS paper's reported numbers, and FreGS gains partly come from lower gradient threshold with matched counts; "Base" in ablation also equals those numbers.
8. Sec 4.1 says 11 scenes (9 + 2 T&T) while Table 1 reports Deep Blending too.
