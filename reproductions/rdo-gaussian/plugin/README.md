# splatstudio-method-rdo-gaussian

Native SplatStudio port of **RDO-Gaussian** — *End-to-End Rate-Distortion Optimized 3D Gaussian
Representation* (H. Wang et al., ECCV 2024, arXiv 2406.01597), following the official code
https://github.com/USTC-IMCL/RDO-Gaussian (commit 710ebab8). Paper fidelity unverified.

## Mechanism
3DGS training for 15k iterations; from 15001 to 30000 the rendered Gaussians are
1. masked by a learnable STE Gaussian mask (σ(φ) > 0.1) on scale and opacity (Eq. 2–5),
2. masked per SH degree by three learnable STE masks (Eq. 6–8),
3. quantised by entropy-constrained VQ (Eq. 9–11) of scale / rotation / DC / SH1 / SH2 / SH3 with
   six codebooks (8192 / 8192 / 8192 / 4096 / 4096 / 4096) and learned softmax entropy models
   (k-means init at 15001, dead-codeword reactivation every 1000 until 20000, rate-constrained
   assignment from 20001, indices refreshed every 10 steps),
trained with `L = L_render + L_VQ + L_rate + λ_GS L_GS + λ_SH L_SH` (Eq. 13, official
normalisation /N/55). At step 30000 the model is encoded into the official bitstream (masked
Gaussians removed, SH-mask sorting with 8 boundaries, shrunk codebooks, rANS with the learned
pmfs — vendored CompressAI/ryg_rans C++ in `ss_rdo_gaussian/csrc`, fp16 positions, 8-bit
opacity), the bytes are decoded and the parameters replaced by the decoded model.

## Entry points
| name | λ_GS | λ_SH | paper |
|---|---|---|---|
| `rdo-gaussian` | 0.0005 | 0.005 | Rate 1 (highest; per-scene tables A.5–A.7) |
| `rdo-gaussian-rate2` … `-rate6` | 0.002 … 0.05 | 0.02 … 0.5 | Tables A.1–A.3 |
| `rdo-gaussian-smoke` | 0.0005 | 0.005 | 2000-step compressed schedule, diagnostic only |

All other values (`density.DEFAULTS`): thresholds 0.1, mask lr 0.01 / 0.05 (fields), codebook lr
2e-4, logits lr 2e-3 (×0.1 after 12000 VQ steps), λ_s 32768, λ_r = λ_DC = λ_SH1..3 = 256, patch
65536, 3DGS densification unchanged (App. A1, `run_real_world.sh`, `arguments/__init__.py`).
Default `evaluate: "decoded"` renders the decoded bitstream, exactly what the official `render.py`
evaluates (`Scene.__init__` → `gaussians.decode`). `evaluate: "official"` is a misnamed diagnostic undecoded mode (fp32 xyz, unquantised opacity) that no official evaluation mirrors
(reviewer's estimate of the fp16-position/8-bit-opacity effect vs the undecoded model: 0.01-0.05 dB).

## Size
`report_size.py <run>` → `<run>/rdo_size.json`; `size_MB` = bytes of
`index_bitstream.bin + header.bin + codebook.npz + logits.npz + position.npz` / 1e6, exactly the
official `bitstream/` directory (results csv "Size [Bytes]"). `--write` also writes the files.

## Fidelity gaps
See `../plan.md` (one extra Adam step at 30000; runtime white background / sampling / SSIM;
the diagnostic undecoded `evaluate` mode; RNG of k-means and reactivation).

## Commands
```bash
tools/ssenv pip install -e reproductions/rdo-gaussian/plugin
tools/ssenv ss-verify rdo-gaussian --device cuda
tools/ssenv ss-train rdo-gaussian --data ~/datasets/mipnerf360/w1600/garden --output runs/x \
    --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv python reproductions/rdo-gaussian/plugin/report_size.py runs/x
```
