# splatstudio-method-omg

This is a native SplatStudio port of **Optimized Minimal 3D Gaussian Splatting (OMG)** by Lee, Ko
and Park (NeurIPS 2025, [arXiv 2503.16924](https://arxiv.org/abs/2503.16924)). Paper fidelity is
unverified.

- **Official code:** https://github.com/maincold2/OMG. The repository has no licence file, so it
  was read and re-implemented, not copied.
- **Base:** our matched Mini-Splatting port. `ms_density.py`, `ms_core.py`, `kernels.py`, `csrc/`
  and `ms_probes.py` are copied from `reproductions/mini-splatting` at freeze a6428eb7. The CUDA
  kernels are derived from 3DGS / Mini-Splatting under the Inria licence (see
  `csrc/LICENSE-3DGS.md`).
- **Patterns reused from our Compact-3DGS port** (same group): the dahuffman port, the
  global-bank pattern, the closed-form scheduler and the encode → decode → bake step.

## Methods
| name | variant | τ | importance metric |
|---|---|---|---|
| `omg` | OMG-XS | 0.96 | outdoor (M360 outdoor, T&T) |
| `omg-indoor` | OMG-XS | 0.96 | indoor (M360 indoor, DB) |
| `omg-m` | OMG-M | 0.99 | outdoor |

## Mechanism (official train.py schedule, 30k steps)
1. **0–15k: Mini-Splatting.** Densification with blur split, and depth re-initialisation every
   5k. At 15k: importance sampling, then a reinit from the centres.
2. **15k (C2): construct_net.** Per-Gaussian T (initialised to the SH DC) and V (zeros), lr 0.0025.
   - Space feature: F = MLP_s(Frequency16(contract(p))).
   - sh0 = MLP_t([F,T]), opacity = MLP_o([F,T]), shN = MLP_v([F,V]).
   - These four tcnn-layout MLPs have 15,360 parameters.
   - Net Adam: lr 0.01, 100-step warm-up, ×0.33 at +1k / +3.5k / +6k.
   - SH degree goes 0 → 3 at 16k / 17k / 18k.
3. **20k (C1): local distinctiveness.**
   - I = Ibar · (mean |T_i − T_j|₁ over 2 Morton-order neighbours)^2, using the official
     neighbour construction.
   - Keep the Gaussians above the τ-CDF split, then reset the Gaussian Adam.
4. **29k (C3): SVQ.** K-means once:
   - scale: 3 × 64 codes, length 1;
   - rotation: 2 × 512 codes, length 2;
   - [T,V]: 3 × 1024 codes, length 2.

   The indices are then frozen, and only the codebooks train (Adam lr 1e-8).
5. **30k (C4): storage.** The model is encoded and decoded, and the state is replaced by the
   decoded plain SH3 3DGS model. Both validation.json and ss-eval measure this decoded model.
   - fp16 positions → uint16 → G-PCC (`tmc3`);
   - Huffman-coded indices;
   - fp16 codebooks and MLPs;
   - LZMA container.

## Storage report
`checkpoint.pt` → `density_state["omg"]["encoded"]` holds:
- `official_bytes`: the official `write_storage`, i.e. G-PCC stream length with the official
  flags, codebooks ×2 B, dahuffman stream bytes, MLP params ×2 B;
- `official_total_mb` (MB = 2^20);
- `container_bytes`: the real LZMA file that was decoded;
- `decode_drift_vs_live_fp32`: the appearance change from fp16 positions and weights.

```bash
tools/ssenv python reproductions/omg/plugin/report_size.py <run>
```

## Fidelity gaps
- **Runtime:** white background, gsplat numerics, random-with-replacement view sampling.
- **MLPs:** fp32 instead of tcnn fp16 (`__sinf`).
- **K-means:** torch k-means++/Lloyd instead of cuML.
- **G-PCC:** the container uses the bypass-flag-0 stream, because TMC13 v14 cannot decode the
  official flag. The geometry is identical; both sizes are reported.
- **Container:** ours, not an LZMA pickle.
- **Final step:** one extra per-Gaussian Adam step at 30k.
- **Run length:** 30k-step runs only. Other lengths are refused at step 1. Use `smoke_schedule.py`
  for a compressed diagnostic run.

## Commands
```bash
cd reproductions/omg/plugin && ../../../tools/ssenv pip install -e .
tools/ssenv ss-verify omg --device cuda
tools/ssenv ss-train omg --data ~/datasets/mipnerf360/native/bicycle --output <run> --steps 30000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run <run> --split test
```
