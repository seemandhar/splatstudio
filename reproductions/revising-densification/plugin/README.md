# splatstudio-method-revising-densification

Native SplatStudio port of **Revising Densification in Gaussian Splatting** (Rota Bulò, Porzi,
Kontschieder; ECCV 2024, [arXiv 2404.06109](https://arxiv.org/abs/2404.06109)).
No official code was released; the method is re-derived from the paper. Paper fidelity has not
been verified.

## Mechanism → code
| paper | code | value (source) |
|---|---|---|
| Sec. 3.2 error score E^π_k = Σ_u 𝓔(u) ω_k(u), E_k = max_π | `csrc/err_stats.cu`, `density.accumulate` | 𝓔 = 1 − SSIM(u) (main results use SSIM) |
| Sec. 4.2 growth threshold | `error_threshold` | 0.1 |
| Sec. 3.4 growth control | `growth_fraction`, `max_primitives` | 5 % per run; cap = baseline count (table in `core.BASELINE_CAP`) |
| Sec. 3.3 clone opacity | `opacity_correction` | 1 − √(1 − α) on source + copy; split keeps α |
| Sec. 3.5 opacity regularisation | `opacity_mode="decrement"`, `opacity_decrement` | −0.001 per ADC run, no reset |
| Sec. 3.5 residual transmittance | `revdens-aux.transmittance_weight` | 0.1 · mean(1 − alpha) |
| Sec. 4.2 ADC length | `densify_until` | 27000 (runs at 600..26900, every 100) |

The per-pixel error map is computed by the `revdens-aux` loss (it sees the target) and handed to
the density through `outputs["revdens_error"]`. The CUDA kernel re-traverses gsplat's tile lists
with gsplat's alpha numerics and is checked against a brute-force alpha-compositing loop and
against the paper's own auxiliary-loss gradient route (`error-attribution-oracle` probe).

Ablation switches: `score="gradient"`, `growth_control=False`, `opacity_correction=False`,
`opacity_mode="reset"`, `revdens-aux.error="l1"`.

## Fidelity gaps
See `../plan.md`. In short: the SSIM error formula and the prune/grow order are our
interpretation; the cap is our gsplat baseline's count rather than the authors' baseline; the
runtime uses a white background and gsplat numerics; results come from a single run.

## Commands
```bash
tools/ssenv pip install -e reproductions/revising-densification/plugin
tools/ssenv ss-verify revising-densification --device cuda
tools/ssenv ss-train revising-densification --data ~/datasets/mipnerf360/native/garden \
  --output runs/x --steps 30000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
