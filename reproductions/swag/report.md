# SWAG (plan #85) — build report
- **Code and sources:** no official code exists (the project page says "Code (TBA)"). Built from the arXiv 2403.10427 LaTeX: Eq. 8–11 and supp. Table 7 hyper-parameters.
- **Assumptions:** A1–A11 are in plan.md. They cover MLP activations, hash-grid bbox and alignment, embedding init, opacity reset, the test-time optimisation (128 Adam steps, lr 0.1, mean-embedding init) and resolution.
- **Methods:**
  - swag (full)
  - swag-a (no transient opacity)
  - swag-t (no colour conditioning)
- **Custom CUDA:** JIT hash-encoding kernel, matching the PyTorch reference to about 2e-7.
- **Smoke test** (Brandenburg Gate, 2k steps, NeRF-W protocol): swag 16.07, swag-a 18.44, swag-t 18.32.

## Review resolution (review-1, approved, 6 minor)
- **Transient opacity:** no port bug. Eq. 9–11 match term by term, with U~U(0,1) in training and U=0.5 at test. At 2k steps the MLP learns to switch off 25–40% of Gaussians per view. The paper has no regulariser for this, and none is added. If full swag is still more than 1 dB below swag-a at 60k, this is reported as an unresolved fidelity gap.
- **Fixed:** the NeRF-W half split now uses [:w//2] and [w//2:]; before, the middle column was both optimised on and scored for odd widths (evaluate.py:53-54).
- **Labels:**
  - The test-time lr (0.01 → 0.1) was chosen on test images, using the left-half loss only.
  - Resolution is full-res. The paper never states it, and its comparison baselines are at downscale 2, so absolute PSNR may be offset.
  - Hash-grid alignment differs from tiny-cuda-nn.
  - Embeddings start at N(0,1); their mean is out of distribution for the MLP, which explains the low no-optimisation PSNR.
- **Probe gap:** no probe checks that training renders sample U from U(0,1). Recorded, not fixed.

## Benchmark (launched)
- **swag-pt:** swag on brandenburg-gate (paper 26.33) and sacre-coeur (21.16), 60k steps.
- **swag-a-pt:** swag-a on brandenburg-gate (paper 26.44).
- **Headline numbers:** `python -m ss_swag.evaluate --run RUN`, metrics.nerfw.

## Result: swag-a on Brandenburg Gate (swag-a-pt, 60k) — within tolerance
NeRF-W protocol (right half, embedding optimised on the left): **26.22** / 0.930 / LPIPS-Alex 0.137, vs paper SWAG-A 26.44 (Δ −0.22).
- No test-time optimisation: 18.76.
- Full image with the optimised embedding: 27.00.
The full swag runs (swag-pt) are still pending.
