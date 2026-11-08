# splatstudio-method-we-gs

Native SplatStudio port of **WE-GS** (Wang, Wang, Qi, arXiv 2406.02407). Paper fidelity unverified.

Mechanism: per-image appearance embedding `l_k` (dim 16) and static mask `M_k` (1/4 res) from a
spatial-attention CNN on the image itself (Enc¹ → CAM → U-Net mask → `M⊗F'` → Enc²); a 4-layer
MLP `G_φ(sh, γ(μ), l_k)` predicts residual SH added to every Gaussian's SH; masked L1/SSIM,
`L_regM`, `L_regSH`, `L_regTS`. Test views are encoded feed-forward.

Training code is unreleased. Values the paper omits come from the authors' released WE-GS
inference model (`github.com/yuzewang1998/takinglangsplatw`, `in_the_wild_renderer/`, Inria
licence; read, not copied): encoder conv shapes, G_φ width 256 + tanh,
PE frequencies (sh 0, xyz 4), lrs (G_φ 1e-4, encoder 1e-4, mask 1e-5).

| config | value | source |
|---|---|---|
| wegs-masked-photometric l1/ssim | 0.8 / 0.2, M detached in D-SSIM (deviation) | §5.1; paper.md §8 |
| wegs-mask-reg weight, reduction | 1e-5, sum over M_k at W/4×H/4 | §5.1 Eq. 20 |
| wegs-sh-reg weight, reduction | 0.1, mean over Gaussians of ‖Δsh‖₂ | §5.1; reduction assumed |
| wegs-transient-opacity-reg weight, eps | 1e-5, 0.1 | §5.1; eps chosen (paper.md §8) |
| density | 3DGS (100 / 500-15000 / reset 3000 / 2e-4), means lr decay over 30k | §5.1, code args |

Commands:
```bash
ss-train we-gs --data ~/datasets/phototourism/native/brandenburg-gate --downscale 2 \
  --steps 150000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000 --output RUN
python -m ss_wegs.evaluate --run RUN      # RUN/test_wegs.json, headline metrics.full
```
Plain `ss-eval` cannot render (no bank) and raises. Fidelity gaps: `../plan.md`.
