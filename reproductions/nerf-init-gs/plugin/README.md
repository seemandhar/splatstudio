# splatstudio-method-nerf-init-gs

Native SplatStudio port of **Evaluating Alternatives to SFM Point Cloud Initialization for
Gaussian Splatting** (Foroutan, Rebain, Yi, Tagliasacchi; arXiv 2404.12547). No official
training code exists (the linked repo is the project page). Paper fidelity unverified.

## Mechanism
1. **NeRF pre-training** (Sec. 3.2): NerfAcc proposal-network iNGP, 5000 iterations, training
   views only. The iNGP (tcnn-layout JIT CUDA hash grid, nerfacc proposal sampler/pdf loss and
   training loop) is copied from the SPLATIFY ConeGS plugin
   (`reproductions/conegs/plugin/ss_conegs/{ingp,hashgrid}.py`), set to the nerfacc Mip-NeRF 360
   example defaults (far 1e3, weight decay 0).
2. **Volumetric initialization** (Eq. 2-4): 500k rays drawn uniformly from all training rays, one
   inverse-CDF sample of the volume-rendering weights per ray, colour = NeRF radiance there.
   Loaded like an SfM cloud (exact 3-NN scales, opacity 0.1, 3DGS camera-extent scene scale).
3. **Depth distillation** (Eq. 5-7, Supp. Eq. 13): `lambda(i) * mean|D_NeRF - D_GS|`,
   `lambda(i) = 0.9 * 0.9^(i/100)`, D_NeRF precomputed for every training pixel.

| config | value | source |
|---|---|---|
| iNGP iterations | 5000 | Sec. 4.3 |
| init points | 500,000 (`--init random-box`: 50,000 in a 50^3 box) | Sec. 4.3 / 4.2 |
| lambda_init, decay, decay_steps | 0.9, 0.9, 100 | Sec. 3.4, Supp. Eq. 13 |
| 3DGS training | unchanged (built-in photometric + adaptive density) | Sec. 4.3 |

## Commands
```bash
T=/home/sejain/splatify/tools/ssenv
$T ss-nerfinit-prep --source ~/datasets/mipnerf360/native/garden \
                    --output ~/datasets/nerfinit-gs/mipnerf360/garden
$T ss-train nerf-init-gs --data ~/datasets/nerfinit-gs/mipnerf360/garden --output runs/x \
    --steps 30000 --recipe splatstudio --num-gaussians 10000000
```
`--recipe splatstudio` is safe: the density restores the 3DGS camera extent and the exact 3-NN
scales (the runtime's `--recipe 3dgs` kNN needs ~16 GB at 500k points). `--num-gaussians` must be
at least the size of points.npz (SHA-checked). NeRF preprocessing time is in
`<copy>/nerfinit/nerfinit.json` and belongs to the method's training time.

## Fidelity gaps
- GS depth uses gsplat's centre camera-z per Gaussian (un-normalised sum w z), not the paper's
  ray-conditional Gaussian mean; NeRF ray distance is converted to camera z.
- fp32 PyTorch iNGP vs tcnn fp16; indoor NeRF trained at images_2 (nerfacc: factor 4).
- Inverse CDF evaluated in nerfacc's lindisp s-space; opaque-background (sky) samples kept.
- Depth sidecar stored as float16.
- Runtime-wide: white background, view sampling with replacement, SSIM valid convolution.
