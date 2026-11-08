# Review 1 — beta-splatting (Deformable Beta Splatting, arXiv 2501.18630)

**Verdict: APPROVED for benchmark** — 0 critical, 0 major, 5 minor.

Reviewed: plugin/ss_beta_splatting/{__init__,renderer,kernels,density,losses,probes}.py, csrc/beta_raster.cu,
against official `source/code` @431d575 (train.py, scene/beta_model.py, arguments/__init__.py, benchmark.py,
utils/camera_utils.py, submodules/gsplat/{rendering.py, cuda/_wrapper.py, cuda/csrc/{rasterize_to_pixels_fwd.cu,
fully_fused_projection_fwd.cu, spherical_beta.cuh}}), the runtime (splatstudio runtime.py/state.py/losses.py/data.py),
tools/make_pil_resized.py. Checks run: plugin pytest (7 passed, GPU 1); smoke-1 checkpoint inspected.

## Verified to match the official code
| Item | Official | Plugin |
|---|---|---|
| Beta kernel | fwd.cu:147-155 sigma = a dx²+c dy²+2b dxdy (no ½), skip sigma<0 or ≥1, alpha=min(.999, o(1-sigma)^beta), stop T≤1e-4, no 1/255 | beta_raster.cu:88-103 identical; bwd gate o·vis≤.999, v_beta = v_α o vis ln(1-σ) (:221-230) |
| b activation | beta_model.py:52 beta=4e^b, init b=0 | renderer.py:144, density.py:128 |
| Support radius | fwd proj :147-167 1σ circle ceil(sqrt(λmax)) | renderer.py:154-165 1σ per-axis box ceil(sqrt(Σxx/yy)) — exact support, same pixels |
| Spherical Beta | spherical_beta.cuh: c0 + Σ[dot>0] dot^(4e^{b_m}) c_m, dir=normalize(μ−campos), lobe (θ,φ)→(sθcφ,sθsφ,cθ); rendering.py:400-414 c0=clamp_min(SH0+.5,0), clamp_min after SB | renderer.py:189-195; kernels.py:307-322 |
| SB layout/activation | [N,2,6]=(r,g,b,θ,φ,b_m); softplus(β=10 ln2) on rgb only (beta_model.py:47-50); θ~U[0,π], φ~U[0,2π] | renderer.py:149-151, density.py:129-134 |
| View-dir grad | fork bwd allocates v_dirs zeros, never written → 0 | dirs detached (renderer.py:193) — equivalent |
| MCMC relocation | beta_model.py:700-760 opacity-only (no scale adjust), o'=clamp(1-(1-o)^{1/(k+1)}, .005, 1-eps), Adam moments zeroed only at reinit_idx | density.py:50-76 identical incl. which moments are zeroed |
| Add | beta_model.py:762-800 min(cap, int(1.05N))−N, multinomial(o), sources' opacity updated, new rows zero moments | density.py:79-99 — **yes official adds +5%/100 steps up to cap** |
| Schedule | train.py:136-152 500<it<25000, it%100==0, dead o≤0.005 | density.py:138-140 |
| Noise | train.py:143-151 randn·(1−o)^100·5e4·xyz_lr, then Σ=LLᵀ (L=RS) bmm; **only on densification steps**, after reloc/add | density.py:102-110,153-154 identical |
| Regularisers | train.py:109-111 0.01·mean|σ(o)| + 0.01·mean|e^s|, 500<it<25000 | losses.py:169-173 |
| LRs | arguments: beta .001, sb_params .0025, sh .0025, opacity .05, scale .005, rot .001, xyz 1.6e-4→1.6e-6 ×extent (no delay: lr_delay_steps=0) ; Adam eps 1e-15 | __init__.py:73-74, state.py:36-49, runtime.py:209 |
| SH 0 / black bg / init opacity 0.5 / 3-NN scales / identity quats / all SfM points | ModelParams sh_degree 0, bg [0,0,0], create_from_pcd | renderer bg 0.0, density.initialize, recipe 3dgs + num_gaussians 1e7 (smoke config) |
| Caps | benchmark.py:6-20 garden 5M, room 1.5M, bicycle 6M, default 1M | __init__.py:25-26 |
| Resolution | benchmark.py:82/87 `-r 4`/`-r 2`; camera_utils.py:24-27 int(W/r+0.5), PIL default (BICUBIC) resize of full-res `images/` | pilr garden 1297×840 (5187×3361/4), room 1557×1038 (3114×2075/2) — match |
| Trainable leaves | — | smoke-1 checkpoint: all fields requires_grad, betas/sb_params moved (b std .10, b_m std .30); probe mcmc-opacity-only asserts leaves after reloc/add |

Kernel fwd/bwd vs float64 dense autograd ~3e-7 (verify.json); probes would fail on Gaussian kernel, missing v_beta,
scale-adjusting/logit-noise MCMC, sum-vs-mean regulariser, or schedule off-by-one.

## Minor issues
1. **Room delta target misleading** (bench_request.json `notes`): paper's 3DGS room 30.63 vs our gsplat baseline 31.69
   (+1.06). The +2.20 room delta is not attainable by construction; judge room on absolute PSNR (32.83) and the paper's
   own 3DGS-MCMC row (32.48, paper_results.json `mcmc_3dgs`) as a sanity reference.
2. **bicycle pilr off by 1 px** (tools/make_pil_resized.py:47 `round()` banker's → 1236) vs official
   camera_utils.py:25 `int(W/4+0.5)` → 1237. Affects only beta-splatting-bicycle; not in this benchmark. Record if bicycle is run.
3. **Step order** (runtime.py:204-212 vs train.py:136-157): official relocates/adds/noises *before* Adam step and then
   skips the step on those ~245 iterations (re-wrapped Parameters have no .grad); ours steps first, then densifies, and
   reads the next step's xyz lr for noise. Already gap #1/#4 in plan.md; negligible.
4. **shN (15×3) allocated and copied by MCMC** but unused (renderer.py:229-230): ~0.9 GB at 5M, no Adam state (grad None).
   Harmless; just memory.
5. **Paper vs code on noise frequency**: paper Sec. 4.4 describes MCMC-style per-step noise; official applies it only on
   densification steps — plugin follows the code (correct choice); note in report.md.

## 30K vs full column
Correct to compare with the 30K sb=2 column (garden 28.15, room 32.83). Released code with `--eval` runs an unbounded
loop with test-PSNR checkpoint selection every 500 its after 15K and patience 20 (train.py:83,180-188) — test-set
selection, not reproducible honestly; the "full" column (garden 28.11, room 33.03) should not be the target.

## Launch (put runs on /bigdata per CLAUDE.md)
```
mkdir -p /bigdata/sejain/runs/beta-splatting-garden-pilr /bigdata/sejain/runs/beta-splatting-room-pilr
ln -s /bigdata/sejain/runs/beta-splatting-garden-pilr runs/beta-splatting-garden-pilr
ln -s /bigdata/sejain/runs/beta-splatting-room-pilr runs/beta-splatting-room-pilr
python3 tools/bench.py launch --exp beta-splatting-garden-pilr --methods beta-splatting-garden --dataset mipnerf360-pilr --scenes garden --steps 30000
python3 tools/bench.py launch --exp beta-splatting-room-pilr --methods beta-splatting-room --dataset mipnerf360-pilr --scenes room --steps 30000
```
Baseline: baseline3dgs-mipnerf360-pilr-30000 (garden 27.351, room 31.687).
