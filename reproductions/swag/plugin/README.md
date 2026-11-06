# ss_swag — SWAG native SplatStudio plugin

SWAG: Splatting in the Wild images with Appearance-conditioned Gaussians (Dahmani et al.,
arXiv 2403.10427). **No official code exists**; this was written from the paper alone, and every
unspecified choice is listed as an assumption A1–A11 in `../plan.md`.

Methods: `swag` (full), `swag-a` (no transient opacity, Tab. 3), `swag-t` (no colour conditioning, Tab. 3).

Mechanism (renderer `ss_swag.renderer`, bank `ss_swag.bank`):
- c = 3DGS SH colour; (c^I, Δα^I) = MLP(c, HashGrid(x), l_I) (Eq. 7/8); MLP 51→64→64→64→4.
- Δα̃ = sigmoid((log|Δα| + log U − log(1−U))/0.1), U~U(0,1) training / 0.5 eval (Eq. 9/10).
- α^I = max(α − Δα̃, 0) (Eq. 11). Transient Gaussians: Var_I[Δα̃] > 0 (reported by evaluate).
- Hash grid T=2^19, L=12, F=2, N_min 16, N_max 2048 (pure PyTorch); embeddings dim 24; all lr 0.0025.
- 3DGS density with densify every 2000 until 30000; 60000 iterations; 3DGS position-LR schedule
  over 30000 steps.

Train / evaluate (NeRF-W protocol: optimise the test embedding on the left half, metrics on the right):
```
ss-train swag --data ~/datasets/phototourism/native/brandenburg-gate --output RUN --steps 60000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
python -m ss_swag.evaluate --run RUN          # -> RUN/test_swag.json (headline metrics.nerfw)
```
`ss-eval` has no bank and renders plain 3DGS colour/opacity: not the paper's number.
