# Review 1: student-splatting (SSS, arXiv 2503.10148)

**Verdict: approved after one cheap fix.** 0 critical, 1 major (cheap, fix before launch), 4 minor.

The SGHMC sampler, the relocation formula, the signed-opacity blending, the partial dν, the
regularisers, the initialisation, the learning rates and the per-scene configs all match the
official code. I confirmed the sampler and the relocation numerically (below). The one real
deviation is the ν-dependent radius table: its interpolation runs in the opposite direction to
the official one, and the probe hard-codes the plugin's version.

## Checks run (read-only, CPU, no training)
- **SGHMC vs the official `AdamSGHMC` class.** I imported `source/code/utils/sghmc.py`, ran 11
  steps with burn-in B=5 (covering burn-in, the momentum reset at B+1, and the steps after it),
  random sig/cov, a decaying lr and the same RNG seed. Then I ran torch Adam with lr 0 followed by
  `density.sghmc_update`. Max abs diff 7e-9 (rel 2e-9): **exact**.
- **`relocation_student_t` vs a float64 double loop of `utils.cu compute_relocation_student_t`**
  (signed o, ν ∈ {1, 3.5, 100, 1500, 7}, N ∈ {1, 2, 5, 51, 10}). Opacity matches to 1e-8 and the
  scale factor (coeff², squared as in the official code) to 3e-5 relative. N=1 gives the identity.
- **Smoke checkpoint (garden, 2k).** `degrees` and `opacities` are still trainable leaves after
  recycling/growth. The means Adam state holds {step, exp_avg, exp_avg_sq, momentum}, means lr = 0.0,
  4.8 % of components are negative. ν: 0.9 % are at ν < 2 (clamped at 1), the median is ≈ 2400.
- **Runtime interaction** (`splatstudio/src/splatstudio/runtime.py:192-211`). `initial_lr` is read
  after `density.initialize` has set means lr to 0, so the runtime's means lr override stays at 0.
  torch Adam with lr=0, weight_decay=0 and eps 1e-15 leaves the means unchanged (the step adds
  −0·m/denom). The order is `opt.step()` (moments updated with this step's gradient) → `after_backward`
  (SGHMC). This is the same as the official order, where the moments are updated inside the same
  `step()`. sig and cov come from the pre-step parameters, as in `train.py:143-148`.

## Major
1. **`projection.py:28-37` `empirical_factor`: interpolation reversed vs the official code.**
   - Official (`forward.cu:246-281`), case k: `(T[k-1] - T[k])*(ν-k) + T[k]`. This gives 9.925 at
     ν=1.0, rising to 63.66 as ν→2, and then 5.841 at ν=2. It is a sawtooth that starts each
     interval at T[k].
   - Plugin: `(T[k] - T[k-1])*frac + T[k-1]`. This gives 63.657 at ν=1.0.
   - Effect: the ~0.9 % of components clamped at ν=1 get a radius of 63.7σ instead of 9.9σ
     (~41× the tile area). That means more intersections and more time/memory at the 3M cap.
   - Rendering effect: for ν∈[1,3) both radii lie beyond the 1/255 cutoff, so the image is the same.
     For ν∈[3,13) with high |o|, the official code truncates the tail at T[k] while the plugin does
     not. Example: at ν=3 the official cut is 4.60σ, but alpha only falls to 1/255 at 4.95σ.
   - The `t-projection` probe (`probes.py:205-210`) hard-codes the plugin's values (63.657 at ν=1,
     3.463 at 7.25, where the official value is 3.391), so it confirms the deviation instead of
     catching it.
   - Fix: `lo, hi = table[k], table[k-1]; interp = (hi - lo)*frac + lo`, and update the probe's
     expected values to the official formula.

## Minor
2. **`kernels.py` / `t_raster.cu` header** says "official partial dν". This is correct: official
   `backward.cu:582` `gdv` keeps only the (ν+2)/2·b/ν²·q^-(ν+4)/2 term and drops −½ ln(q)·G, and the
   kernel ports exactly that. Suggest stating in report.md that the official gradient is not the true
   derivative, and that this may be why ν drifts to 10³ (the dropped term is negative).
3. **Fidelity gap 8 / comparison:** room runs 40k steps (official `configs/room.json`; the paper
   says per-scene values are "in our code", supp. l.798). The gsplat baseline runs 30k. Note in
   report.md that the room delta includes 10k extra steps. Paper deltas: bicycle +0.43 dB, room +1.94 dB.
4. **Plan/README:** the common `student-splatting` config is described as "40k". That holds for
   garden, but `configs/stump.json` uses 20000 iterations and counter uses cap 1.2M. If more scenes
   are run, add per-scene methods or `--steps`.
5. Already-listed gaps confirmed: SH degree steps up at 1001 vs 1000; the official code skips the
   optimiser step at the last iteration; sampling is with replacement; there is no 8-bit PNG round
   trip; 1 seed vs a 3-run mean. Principal point: the plugin uses K's cx; pilr bicycle has
   cx=618=W/2, so it matches Inria's centred convention.

## Confirmed exact (code ↔ official)
- **Kernel** (`t_raster.cu` ↔ `forward.cu:396-430` and `backward.cu:500-585`):
  - G=(1+b/ν)^-(ν+2)/2; skip when `base<=0` and when ν<1.
  - Signed alpha clamped to ±0.99; |α|<1/255 skip; T<1e-4 stop (that splat is excluded).
  - Straight-through clamp gradient: no gate. dL/do = G·dL/dα.
  - Conic-y gradient: the kernel computes the full derivative, and projection.py halves it before the
    Inria backward (1/(det²+1e-7)). Consistent.
- **Projection:** no +0.3 low-pass (`forward.cu:113` is commented out); J uses the ±1.3·tan_fov
  clamp; near plane 0.2; radius uses max(0.1, mid²−det).
- **Activations:** tanh opacity with init atanh(0.5); ν = hardtanh(raw, 1, 1e4) with init 100
  (configs); scales 0.1·√(3-NN d²) under `--recipe 3dgs`. Black background (official
  `white_background=False`).
- **Learning rates:** f_dc 0.0025, f_rest /20, opacity 0.005, scaling 0.005, rotation 0.001;
  degree 10 ×0.5 every 5000, min 0.01, applied at iteration `step`; Adam eps 1e-15 for all groups.
- **Means lr:** sqrt(base·s/7.4518) exp-decay over 30k (kept at 30k for 40k runs) with
  s = 1.1·camera radius (`train.py:59-66`).
- **SGHMC:** burn-in uses friction lr·(1/lr)·r·sig (cancels) and noise Σ·(N(0,√(2lr·C_b))·sig).
  Momentum is reset at B+1. After burn-in: friction lr·C·r·sig, noise ·sig.
  sig = σ(100((1−|o|)−0.995)).
  - Bicycle: C_b 5e8, B 15000, cap 3M. Room: 5e5, 7000, 1.5M. C 120 for both.
- **Recycling** (`train.py:161-167`, `nt_model.py` recycle/add):
  - Runs every 100 steps for 500<it<25000; dead means |o|<0.005; at most int(0.05N) recycled,
    choosing the lowest |o|.
  - Sources are drawn by multinomial on |o| over alive components; N clamped to [1,51]; o_new
    clamped to ±(1−eps) with a ±0.005 floor.
  - Source opacity/scale are overwritten as in the official code. Adam m,v are zeroed at the
    sources and the momentum at the dead (the dead keep their old m,v, as in the official code).
  - Growth: min(cap, int(1.05N)). New rows get zero m, v and momentum; the sources' m, v are zeroed
    and their momentum kept. `step` is preserved.
- **Regulariser:** 0.01·mean|tanh o| + 0.01·mean exp s at every step (`train.py:124-125`).
- **Data:** pilr bicycle is 1236×822, matching the official `-r 4` `round(4946/4)`; room uses `-r 2`.
  Budget: bench passes `--max-gaussians 12000000` (> 3M cap) and `--num-gaussians 1e7`.

## Launch (after fixing issue 1)
```
mkdir -p /bigdata/sejain/runs/student-splatting-bicycle-pilr && ln -s /bigdata/sejain/runs/student-splatting-bicycle-pilr runs/
python3 tools/bench.py launch --exp student-splatting-bicycle-pilr --methods student-splatting-bicycle --dataset mipnerf360-pilr --scenes bicycle --steps 30000
mkdir -p /bigdata/sejain/runs/student-splatting-room-pilr && ln -s /bigdata/sejain/runs/student-splatting-room-pilr runs/
python3 tools/bench.py launch --exp student-splatting-room-pilr --methods student-splatting-room --dataset mipnerf360-pilr --scenes room --steps 40000
```
Compare against `baseline3dgs-mipnerf360-pilr-30000` (gsplat bicycle 25.447, room 31.687). The
targets are deltas of +0.43 (bicycle) and +1.94 (room) over that baseline.
