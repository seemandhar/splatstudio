# MesonGS — implementation spec

Xie et al., "MesonGS: Post-training Compression of 3D Gaussians via Efficient Attribute
Transformation", ECCV 2024, arXiv 2409.09756. Official code: https://github.com/ShuzhaoXie/MesonGS
(cloned to `source/code`, HEAD e491557 "update block version"; first commit 9ce6f30 = lseg version).
Survey entry: `source/survey/` (w-m/3dgs-compression-survey), which takes the repo CSVs.

## 1. Summary
- Post-training codec for an already trained 3DGS model (30k iterations), followed by a short
  quantisation-aware fine-tune of the attributes (positions fixed).
- Prune by importance `I_g = I_d * I_i`, voxelise positions on an octree grid (depth d), average
  attributes of Gaussians sharing a voxel, replace quaternions by Euler angles, RAHT-transform
  the "important" attributes (opacity, Euler angles, SH DC = 7 channels), block-quantise the AC
  coefficients (8 bit) and the log-scales (8 bit, no RAHT), vector-quantise the 45 SH-rest
  coefficients, pack with DEFLATE (npz + zip).

Contributions
- C1 Universal pruning (§3.3, Eq. 3): view-dependent `I_d = Σ_p α_i Π(1-α_j)` over all training
  pixels, view-independent `I_i = (V/V_90%)^β`, prune the τ lowest.
- C2 Octree geometry coding (§3.4): voxel keys at depth d, mean merge of duplicates.
- C3 Euler replacement (§3.5 "Replacement", Eq. 4/5).
- C4 RAHT on important attributes (§3.5 "Region adaptive hierarchical transform", Eq. 6).
- C5 Block quantisation (§3.5, Eq. 7/8) of AC coefficients and scales.
- C6 VQ of 1+-degree SH (§3.5 "Clustering"), batched k-means (c3dgs code).
- C7 Encoding (§3.5 "Encoding"): LZ77 (zlib/DEFLATE) of all parts; size = zip bytes.
- C8 Fine-tune (§3.5 "Finetune"): positions fixed, encode/decode simulated in forward, STE.

## 2. Equations (verbatim from `sec/3_method.tex`)
- Eq. 3: `I_d = \sum_{p \in \mathcal{P}} \alpha_i \prod_{j=1}^{i-1}(1 - \alpha_j)`;
  `I_g = I_d I_i`, `I_i = (V_{\rm norm})^{\beta}`, V = product of scale vector, normalised by
  "the 90% largest", clipped to [0,1] (code: no clip; `sorted_desc[int(0.9 N)]`).
- Euler (Eq. 4): `[atan2(2(wx+yz), 1-2(x^2+y^2)), -π/2 + 2 atan2(√(1+2(wy-xz)), √(1-2(wy-xz))),
  atan2(2(wz+xy), 1-2(y^2+z^2))]`; Eq. 5 rotation matrix with C/S of φ, θ, ψ
  (code `build_rotation_from_euler(euler[:,2], euler[:,1], euler[:,0])`).
- RAHT (Eq. 6): `T = 1/√(w1+w2) [[√w1, √w2], [-√w2, √w1]]`, w = number of leaves.
- Block quantisation (Eq. 7/8): `c_q = round(clamp(c/S + Z, 0, 2^b-1))`,
  `S = (max-min)/2^b`, `Z = round(2^b - max/S)`. **Code**: `VanillaQuan`: running max (≥0) /
  min (≤0) updated in every forward, `S = (max-min)/(2^b-1)`, `Z = clamp(round(2^b-1 - max/S),
  0, 2^b-1)`, STE round (`round_pass`), gradient zero where clamped.
- Blocks: each channel of length L is split into `n_block` blocks by `split_length` (first
  `L mod n` blocks one longer).

## 3. Hyper-parameters
| name | value | source |
|---|---|---|
| bits | 8 | paper §4, code `num_bits` |
| merge | mean | code `oct_merge` |
| VQ codebook | 2048 (kitchen 8192, train/truck 4096) | code `config3['cb']` |
| VQ batch / steps / decay | 262144 / 1000 / 0.8 | code `batch_size`, script `--steps 1000`, `vq.py` |
| depth | 20 outdoor + db + tandt, 19 bonsai/counter/kitchen/room | `config3['depth']` |
| n_block | 57 (c3), 66 (c1) | `config3/universal['n_block']` |
| prune τ (c3) | bicycle .40, garden .28, stump .30, flowers .50, treehill .50, room .32, counter .22, kitchen .24, bonsai .34, train .22, truck .40, drjohnson .41, playroom .20 | `config3['prune']` |
| prune τ (c1) | bicycle .30, garden .18, counter .12, ... | `universal_config['prune']` |
| β | bicycle .03, stump .01, other mip360 .1 (default 0.1) | `cal_imp beta_list` |
| fine-tune LR scale | `lr_scale_list[scene] * 4`: bicycle .6, garden .4, counter .4, stump .04, flowers/treehill .2, room/kitchen/bonsai .4, drjohnson/playroom .4, train .2, truck .8 | `mesongs.py training()` |
| fine-tune LRs | f_dc 0.0025 s, f_rest (codebook) 0.0025/20 s, opacity 0.05 s, scaling 0.001·extent·s, rotation 0.001 s, xyz frozen | `finetuning_setup` (spatial_lr_scale = cameras_extent from load_ply) |
| Adam | fresh, eps 1e-15 | `finetuning_setup` |
| fine-tune iterations | 8000 (released script, run on mic); default 30000 | `scripts/mesongs_block.sh`, `arguments` |
| loss | 0.8 L1 + 0.2 D-SSIM | `mesongs.py` |
| SH degree in FT | 3 from the start | `load_ply` sets active = max |
| background | white (paper §4); released block script has no `-w` (black) | paper vs script |

## 4. Schedule
3DGS 30k (standard) → importance over train views → prune → octree → RAHT tables → init
quantisers → VQ → offline eval (iteration 0) → fine-tune `iterations` steps (no optimizer step
at the last iteration, `if iteration < opt.iterations`) → save bins.zip. Test evaluated every 400
iterations and "best" model saved by **test PSNR** (`if cur_psnr > psnr_train`) — test-set
model selection we do not reproduce.

## 5. Builds on
3DGS; everything of 3DGS training kept for the base model. VQ and weighted_distance kernel from
c3dgs (identical file).

## 6. Protocol
Mip-NeRF 360 at width 1600 ("1600×1063"; code `-r -1` loads `images/` and rescales to width
1600), llffhold 8 (`--eval`), T&T, DB. Metrics PSNR/SSIM/LPIPS-VGG (`lpipsPyTorch`), size =
zip bytes/2^20 (MB). Table 1 M360 excludes flowers and treehill.

## 7. Ambiguities / code facts
- Fine-tune length for M360 is not stated; released script uses 8000 (mic). We use 8000.
- VQ importance after octree merge is misaligned in the code (`imp` keeps the pre-merge order and
  length; features are sorted by voxel key) — reproduced as `vq_importance: official`.
- Positions decode to the voxel's lower corner (`xletra[code]`), not the centre.
- The "octree" is stored as the sorted int64 voxel keys `ki = x·4^d + y·2^d + z` inside
  `oct.npz` (DEFLATE), not as occupancy bytes.
- Bitstream: `oct.npz{points, params(6 f64)}`, `ntk.npz{int32 indices}`, `um.npz{f32 codebook
  [K,45]}`, `orgb.npz{f: f32 DC[7], i: uint8 AC (7 channel-major, each N-1)}`,
  `ct.npz{uint8 scales channel-major}`, `t.npz{f64 [depth, n_block, (scale, zp) per block...]}`,
  all `np.savez_compressed`, then `zip -j bins.zip`.
- Quaternion fed to Euler is the raw parameter normalised (`r/‖r‖`); Euler is differentiable in
  FT (no `FakeEuler`).
- RAHT is a pure-Python port of 3DAC (`raht_torch.py`); no external library.
