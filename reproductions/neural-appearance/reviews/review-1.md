# Review 1 — neural-appearance (arXiv 2609.05255), module `ss_neural_appearance`

**Verdict: approved for benchmark** — 0 critical, 0 major, 6 minor. (Reviewer spec asks for JSON; the caller asked for markdown.)

## Questions asked

1. **Is the paper's per-scene "Neural" row really run with PPISP + distillation? Is the test protocol leakage-free?**
   Yes. Table 1 caption (`tables/0_training.tex:3`) says "We employ PPISP on real scenes". Supp. §B.1 (`sections/X_suppl.tex:94,186,189`) says the controller distillation "runs after optimization and is identical for all models". `paper_configs/default/{neural,sh}/<scene>.yaml` have `PPISP.USE: true`, `CONTROLLER_TRAINING_STEPS: 5000` and `CONTROLLER_DISTILLATION: true`. At test time the plugin calls PPISP with `frame_idx=-1` (`renderer.py:86`, `bank.py:79`). The vendored PPISP forward (`ppisp_vendor/__init__.py:774-799`) then takes exposure and colour from `controller(rendered_rgb)`, and vignetting and CRF from camera 0. Only the **rendered** radiance reaches the controller. The test GT is never read and nothing is optimised at test time, so there is no leakage. The distillation also matches the official code: the official `Trainer.finalize` (`Trainer.py:255-276`) freezes the scene, zeroes the opacity and scale regulariser weights, and trains only the controller (main ISP parameters frozen, input detached) on training views for 5000 steps. The plugin does the same: steps 30001..35000 rasterise under `no_grad` (`renderer.py:62-64`), and PPISP freezes its own parameters at `__init__.py:764-770`. One thing I could not check: NeRFICG's `Optim/ppisp.py::PPISPWrapper` is not in `source/`. Its view→frame mapping, `num_cameras` and the `controller_activation_ratio` arithmetic (`bank.py:26-35`) are reconstructed, not copied. The ratio gives an activation step of exactly 30000 (I checked: `int(30000/35000*35000)=30000`).
2. **Decoder.** It matches `Appearance/Neural.py` and `utils.py`. Input = `[sin/cos interleaved features (16), SH deg 0..3 (16)]`, 32→16→16→3, ReLU, no biases, 816 params. The output layer is zero-initialised and the latents start at 0. Xavier-uniform init uses padded fan-in. Residual = tanh. The residual is off at degree 0 (official `Base.py:92`). Degree-0 SH is never gated. d = normalize(μ − campos) and its gradient reaches μ. The colour, ReLU(0.5 + c0 + A), comes out exactly right from the degree-0 rasteriser path fed `sh0 + A/Y0` (probe error 2e-7). Learning rates: base 0.00025, features 0.008, cosine 7k→37k. MLP: 0.003, β2 0.99, eps 1e-15, LRDecayPolicy with a 3000-step delay at 1e-8 and decay to 0.1× at 30k. The step↔iteration mapping is consistent with `Trainer.training_iteration` (`update_learning_rate(i+1)`).
3. **MCMC.** It matches `Model.mcmc_densification` (`Model.py:351-445`) and `kernels_mcmc.cuh`:
   - Dead = logit ≤ logit(0.005) or a degenerate quaternion.
   - Sampling is multinomial on alive opacities. n = per-sample count + 1, clamped to [1, 50].
   - The relocation coefficients are identical, and the result is clamped to [0.005, 1−eps] before logit.
   - All fields are copied to the dead rows. Only the sampled rows get their moments reset (as the official code does).
   - Adds +5% up to the cap.
   - Noise = `lr·sigmoid-gate(100o−0.5)·Σ·ε` with Σ = R diag(e^{2s}) Rᵀ (the covariance, not R·S, as in the official kernel). lr = 5e5·lr_means of the step just taken.
   - Densification runs over [600, 24900] every 100 steps. Morton runs at 0 and then every 5000 steps until 25000. The final prune is opacity < 1/255 or a degenerate quaternion, followed by Morton.
   - Caps (`__init__.py:32-35`) equal `MAX_PRIMITIVES` in `paper_configs/default/*/<scene>.yaml` for all 13 scenes. The SH and Neural caps are identical.
   - The MCMC init (opacity 0.5, 0.1 × RMS-3NN) is applied on top of `--recipe 3dgs` scales.
4. **`na_blob`.** It is robust:
   - `export()` (`density.py:227-241`) runs as the **last** operation of every `after_backward`, so stale rows copied or permuted by MCMC, Morton or prune are always overwritten.
   - The blob never enters an autograd graph: training reads the live `ACTIVE` bank, and evaluation uses `blob.detach()`.
   - It has no gradient and lr 0, so FusedAdam skips it (`optim.py:26`).
   - Pack size ≈ 244k floats, almost all of it the PPISP controller CNN. That needs N ≥ 61k rows, and the final step raises if it does not fit.
   - Memory: 5M×4 fp32 = 80 MB in the checkpoint, plus an 80 MB zero buffer allocated every step (cheap). There is no memory blow-up.
   - The ss-eval reload was verified: reload vs live max diff 1e-5 on `evidence/distill-check`, and ss-eval test == in-process validation on smoke-1.
5. **Background and step count.** The background is black (`renderer.py:207`, `__init__.py:87`, = `DATASET.BACKGROUND_COLOR`). `--steps 35000` is required. With fewer steps, `export(required=False)` and `unpack` fall back to identity PPISP for novel views, so the run does not crash but gets the wrong protocol.
6. **requires_grad pitfall.** I found none. `append_rows`, `remove_rows` (`density.py:83,96`) and `permute_gaussians` (`ss_faster_gs/density.py:87`) build `p.detach()[...]` with `requires_grad=p.requires_grad`. PPISP's own `requires_grad=False` on its main parameters during distillation is intended (official behaviour).
7. **Is it justified to run both the Neural and SH twins?** Yes. There is no 3DGS row in the paper. The SH row is the same pipeline with only the appearance model swapped, and it is the paper's own comparison. Our gsplat baseline uses 3DGS ADC, a white background and no PPISP, so it cannot be used to interpret a +0.10/+0.25 dB gain. Caveat: the paper's numbers are 5-run means with std 0.02–0.06. A single-seed Neural−SH gain on garden (+0.10) cannot be resolved, so use room's gain (+0.25) and the absolute PSNR as the primary criteria.

## Issues (all minor)

| # | where | problem | fix |
|---|---|---|---|
| m1 | `bank.py:26-35`, `:63-64` | The NeRFICG `PPISPWrapper` (frame mapping, num_cameras, activation ratio, any exposure prior) is not in `source/`, so it is reconstructed. Assuming one camera is right for M360, T&T and DB. | Record it as a fidelity gap in report.md. Fetch `nerficg/src/Optim/ppisp.py` if possible. |
| m2 | `bank.py:70` | Every render does a GPU→CPU sync of the viewmat (16 floats) for the frame key. Some cost per step. Collisions checked: garden has 161/161 unique train keys and room 272/272, and 0 test views collide with any train view. Resolutions are 1297×840 and 1557×1038 (= images_4 and images_2). | Precompute keys, or pass a frame index if the runtime ever exposes one. Optional. |
| m3 | `renderer.py:233` | The module-global `ACTIVE` wins over `na_blob`. Evaluating another checkpoint in the same process after training would use the wrong bank. bench.py runs ss-eval in a fresh process, so this is safe in practice. | Key `ACTIVE` on the params identity, or clear it at the end of training. |
| m4 | `probes.py` | No probe covers the distillation phase (scene frozen after T, controller used and trained only after T, regularisers off). Coverage exists only as the evidence log `evidence/distill-check.log`. | Add a tiny-T probe: after T, the scene params get no gradient and the controller weights change. |
| m5 | `density.py:222-224` | Learning rates are zeroed at T. The runtime then rewrites the means lr every step (`runtime.py:209`). This is harmless only because no gradient flows. | Note only. |
| m6 | `bench_request.json` "extra" | It repeats `--num-gaussians/--max-gaussians`, which bench.py already passes. bench.py has no method-per-scene mapping, so one launch per scene is needed. Otherwise the garden-cap method would run on room. | Use the per-scene launches below. |

Fidelity gaps confirmed (all already in plan.md): fp32 unfused MLP vs fused fp16 tcnn; tcnn and Philox RNG; runtime SSIM vs fused-ssim; sampling with replacement; single seed; images_4/images_2 loader.

## Launch
```
mkdir -p /bigdata/sejain/runs/neural-appearance-garden && ln -s /bigdata/sejain/runs/neural-appearance-garden runs/neural-appearance-garden
mkdir -p /bigdata/sejain/runs/neural-appearance-room && ln -s /bigdata/sejain/runs/neural-appearance-room runs/neural-appearance-room
python3 tools/bench.py launch --exp neural-appearance-garden --methods neural-appearance-garden neural-appearance-sh-garden --dataset mipnerf360 --scenes garden --steps 35000 --min-free-gb 28
python3 tools/bench.py launch --exp neural-appearance-room --methods neural-appearance-room neural-appearance-sh-room --dataset mipnerf360 --scenes room --steps 35000 --min-free-gb 24
```
