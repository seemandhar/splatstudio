# splatstudio-method-mesongs

Native SplatStudio port of **MesonGS: Post-training Compression of 3D Gaussians via Efficient
Attribute Transformation** (Xie et al., ECCV 2024, arXiv 2409.09756), written against the paper
and the official code https://github.com/ShuzhaoXie/MesonGS (Gaussian-Splatting licence). Ported
with attribution, not imported: RAHT (`raht_torch.py`, from 3DAC), octree coding, VanillaQuan block
quantisation, Euler conversion, `save_npz`/`load_npz` layout; the VQ + `weighted_distance` kernel
are copied from the c3dgs plugin (MesonGS vendors the same c3dgs files). Paper fidelity unverified.

## Mechanism (one run of 30000 + 8000 steps)
| steps | what | code |
|---|---|---|
| 1–30000 | unchanged SplatStudio 3DGS density, means LR of a 30000-step run (= `baseline3dgs-*-30000`) | `density.py`, `schedule.py` |
| 30000 | C1 importance I_d (Σ α T over train pixels, gsplat gradient trick) × (V/V_90%)^β; prune τ | `importance.py` |
| 30000 | C2 octree keys at depth d, mean merge, lower-corner positions (frozen) | `octree.py` |
| 30000 | C4 RAHT tables (official 3DAC port, mask-based set differences); C6 SH-rest VQ (2048, 1000×262144, decay 0.8) | `raht.py`, `vq.py` |
| 30001–38000 | C8 fine-tune: renderer renders q→Euler (C3) → RAHT[opacity, Euler, DC] → DC float, AC 8-bit block fake-quant (C5) → inverse RAHT; log scales block fake-quant; covariance from Euler; SH rest = codebook[idx]. Official fine-tune LRs × lr_scale; last iteration frozen | `bank.py`, `renderer.py` |
| 38000 | C7 official bins (6 `np.savez_compressed` files) zipped; size = zip bytes; decoded from the bytes into the base fields → `ss-eval` measures the decoded model | `codec.py` |

Per-scene hyper-parameters (official code chooses them by scene name) are separate methods:
`mesongs-<scene>` (config3 = released `scripts/mesongs_block.sh`), `mesongs-c1-<scene>`
(universal config), `mesongs-noft-<scene>` (no fine-tune = paper "Our"), `mesongs-smoke`.

## Commands
```bash
tools/ssenv ss-train mesongs-garden --data ~/datasets/mipnerf360/w1600/garden --output runs/x \
    --steps 38000 --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
tools/ssenv ss-eval --run runs/x --split test
tools/ssenv python reproductions/mesongs/plugin/report_size.py runs/x [--export]
```

## Fidelity gaps
See `reproductions/mesongs/plan.md` (base model = our gsplat 3DGS; 8000 FT iterations from the
released mic script; no best-on-test selection; views with replacement; gsplat rasteriser;
Euler sqrt clamp; white background; decoded quaternions from the Euler matrix).
