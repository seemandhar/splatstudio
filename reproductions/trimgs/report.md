# Trim 3DGS reproduction report

**Paper:** *Trim 3D Gaussian Splatting for Accurate Geometry Representation* (arXiv 2406.07499).

**Methods:** `trimgs` (outdoor scenes) and `trimgs-indoor` (indoor scenes). Each run takes 37k steps: 30k steps of flat-3DGS pre-training (official `train.py` with `FlattenGaussianModel`), then 7k steps of Trim fine-tuning (official `tune.py`, using the Mip-NeRF 360 command lines). The plugin is in `plugin/` and is frozen at `4e184ab95770e797`.

**Outcome: matched.** All 4 scenes are within ±0.5 dB of the paper's per-scene PSNR: −0.47, −0.32, −0.32 and −0.11.

Trim changes PSNR relative to the flat pre-train it starts from. The paper measures the same thing against its own 3DGS* (also flat). The two agree in sign on 4 of 4 scenes and in size to within 0.2 dB.

## Results
Setup: Mip-NeRF 360 images_4 (outdoor) and images_2 (indoor), test views every 8th image.

The pre-train column is our 30k flat pre-train, our stand-in for the paper's 3DGS*. It comes from separate 30k runs (`trimgs-*pre30k*`), because the 37k run does not save its step-30k model.

| scene | **Trim ours / paper (Δ)** | SSIM ours / paper | flat pre-train ours / paper 3DGS* | Trim − pre-train, ours / paper | #G |
|---|---|---|---|---|---|
| garden | **26.81 / 27.28 (−0.47)** | 0.856 / 0.864 | 26.93 / 27.39 | −0.12 / −0.11 | 2.88M |
| stump | **26.27 / 26.59 (−0.32)** | 0.760 / 0.770 | 26.13 / 26.53 | +0.14 / +0.06 | 2.85M |
| counter | **28.57 / 28.89 (−0.32)** | 0.905 / 0.907 | 28.78 / 28.96 | −0.21 / −0.07 | 1.37M |
| room | **30.94 / 31.05 (−0.11)** | 0.916 / 0.917 | 31.49 / 31.40 | −0.55 / −0.35 | 1.36M |

Our flat pre-train sits 0.0–0.5 dB below the paper's 3DGS*. Two things cause this. It uses gsplat `DefaultStrategy` instead of 3DGS's densifier (gap 3), and the runtime has known offsets (calibration). Our gsplat baseline gives garden 26.89, stump 26.23, counter 28.85, room 31.52.

Trim prunes about 40 % of the pre-trained Gaussians on the outdoor scenes. Counter is different: it ends with more Gaussians after the 7k fine-tune (1.37M against 1.21M), because indoor fine-tuning keeps densifying.

## Run notes
- `stump` ran out of GPU memory 3 times on shared GPUs; other users' processes held up to 33 GB. It completed on the 4th attempt with `--per-gpu 1 --min-free-gb 36`. The failed attempts are kept as `runs/trimgs-mipnerf360/trimgs/mipnerf360-stump.oom-*` and `.stale-*`.
- Review-1 raised 2 majors, both about framing and benchmark scope. They were addressed by adding room (an indoor scene) and the mandatory 30k flat pre-train reference runs.

## Fidelity gaps (kept)
1. The background is white; the official code uses black.
2. gsplat rasterizer: alpha clamp 0.999, eps2d 0.3. gsplat also drops gradients at saturated alpha (see `TODO-runtime.md`).
3. The pre-train uses gsplat DefaultStrategy instead of 3DGS's exact densification. Our flat pre-train is therefore an assumed stand-in for the authors' 3DGS*.
4. Views are sampled at random with replacement.
5. Densification and kNN run after the optimizer step.
6. Geometry metrics (DTU Chamfer, TnT F-score) are not reproduced. We have no DTU data and no TSDF/Chamfer pipeline. Only the paper's rendering-quality claim is tested.

## Reproduce
```bash
python3 tools/bench.py launch --exp trimgs-mipnerf360 --methods trimgs --dataset mipnerf360 --scenes garden stump --steps 37000
python3 tools/bench.py launch --exp trimgs-indoor-mipnerf360 --methods trimgs-indoor --dataset mipnerf360 --scenes counter room --steps 37000
# 30k flat pre-train references: same with --exp trimgs[-indoor]-pre30k-mipnerf360 --steps 30000
```
