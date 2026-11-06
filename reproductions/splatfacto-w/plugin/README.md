# splatstudio-method-splatfacto-w

Native SplatStudio port of **Splatfacto-W** (Xu, Kerr, Kanazawa — arXiv
[2407.12306](https://arxiv.org/abs/2407.12306)), written against the paper and the official
Nerfstudio code (<https://github.com/KevinXu02/splatfacto-w>, Apache-2.0). Paper fidelity unverified.

Methods: `splatfacto-w` (full), `splatfacto-w-a` (appearance only), `splatfacto-w-t`
(appearance + robust mask).

Mechanism (code values, see `../paper.md`):
- C1 per-Gaussian features f_i [72] + per-image embeddings l_j [48] → MLP (2×256 + heads) → SH
  degree-3 coefficients; colour = max(SH + 0.5, 0) (gsplat), SH degree +1 every 2000 steps.
- C2 robust mask: residual percentile 0–40 % interpolated from the running min/max image L1,
  upper 40 % never masked, 5×5 box > 0.4, from step 6000.
- C3 background: l_j → MLP (2×128 + heads) → degree-4 SH at infinity, composite
  C + (1 − α) C_bg; alpha loss 0.15·mean α where box3(|gt − bg| < 0.003) > 0.6.
- Splatfacto base: resolution 1/4 → 1/2 → 1 every 3000 steps, absgrad densification
  (0.0008), cull α < 0.1, reset every 2500, split stop 20000, Nerfstudio LR schedules and scene
  normalisation (thresholds in Nerfstudio units).

Global learned state (embeddings, MLPs) lives in `ss_splatfactow.bank.ACTIVE`, owned and stepped by
the density, exported in the checkpoint's `density_state["bank"]`.

## Data (official-code protocol)
`python ../tools/prepare_sfw.py --src ~/datasets/phototourism/native/<scene> --out ~/datasets/phototourism/sfw/<scene>`
adds a left-half crop of every test image to the training split (the code trains on all images and
masks the test images' right halves).

## Commands
```bash
ss-train splatfacto-w --data ~/datasets/phototourism/sfw/brandenburg-gate --output RUN \
  --steps 55000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
python -m ss_splatfactow.evaluate --run RUN      # RUN/test_sfw.json, headline metrics.nerfw
```
`ss-eval` scores full images and is **not** the paper protocol.

## Fidelity gaps
View sampling with replacement; masked views padded to 2·(W//2); alpha loss only on the left half of
masked views; masked views recognised by cx > 0.75·width; 55000 (code) vs 65000 (paper) steps.
