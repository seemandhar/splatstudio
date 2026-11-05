# splatstudio-method-atomgs

AtomGS: Atomizing Gaussian Splatting for High-Fidelity Radiance Field (Liu et al., BMVC 2024,
arXiv 2405.12369). Native SplatStudio port of the official code
(github.com/RongLiu-Leo/AtomicGS @5b9ce904). Paper fidelity unverified.

## Mechanism → code
| Contribution | Code | Values (source) |
|---|---|---|
| Atomized proliferation (§3.1, supp. Alg. 1) | `density.py` | S0 = P1 of 3-NN dists, init scales max(d,S0); atomize every 100 it < 7000 (min s ≤ S → isotropic S, scale moments zeroed, S×0.99); split first: grad ≥ min(2e-4·i/7000, 2e-4) & max s > S (2 children, s/1.6); clone grad ≥ 2e-4 (index-shifted padded grads, official quirk); prune α<0.005 or max s>0.1·extent; opacity reset every 300 to ≤0.1; densify 200..14900; xyz LR 1.6e-4→1.6e-6 ×extent over 7000 (train.py / arguments / gaussian_model.py) |
| Edge-aware normal loss (§3.2) | `losses.py:edge_aware_normal_loss` + `raster.py:depth_to_normal` | λ=0.1, until 7000, Sobel/4, (|∇I|−1)^500, normals from unnormalised mean depth (loss_utils.py, image_utils.py) |
| Feature rasterizer (supp.) | `csrc/` (vendored unmodified, JIT) + `raster.py` | mean depth Σ z w, median depth, alpha, depth/alpha backward |
| MS-SSIM photometric | `losses.py:create_photometric` | 0.9 L1 + 0.1 (1 − mean of 3 SSIM scales) |

Renderer: Inria FoV camera (centred principal point, znear 0.01/zfar 100), black background.
The vendored CUDA is under the Inria/MPII Gaussian-Splatting licence (non-commercial research;
`ss_atomgs/csrc/LICENSE-AtomGS.md`); glm headers MIT (`csrc/third_party/glm/copying.txt`).

## Fidelity gaps
1. Runtime optimizer step happens before density ops (official densifies first; replaced tensors
   skip that step).
2. Random-with-replacement view sampling; SH degree increments one step later.
3. Runtime SSIM/LPIPS evaluation; baseline is gsplat (white bg) vs Inria kernel (black bg) here.
4. No Poisson mesh extraction / DTU chamfer evaluation.

## Commands
```bash
pip install -e .   # via tools/ssenv; CUDA builds on first render (JIT, ~2-3 min)
ss-inspect atomgs && ss-verify atomgs --device cuda
ss-train atomgs --data ~/datasets/mipnerf360/native/garden --output runs/x --steps 30000 \
  --recipe 3dgs --num-gaussians 10000000 --max-gaussians 12000000
```
`--recipe 3dgs` is required (3-NN initial scales; guarded).
