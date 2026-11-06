# 3DGRT (plan #82) — BLOCKED: needs user approval to build OptiX tracer

## Feasibility
- Hardware/runtime: OK. 8× RTX A6000 (Ampere, RT cores), driver 580.159.03 ships
  `libnvoptix.so.1` + `libnvidia-rtcore` (OptiX runtime).
- OptiX SDK headers: OK, no NVIDIA login needed. The official repo pins submodule
  `threedgrt_tracer/dependencies/optix-dev` → `NVIDIA/optix-dev` branch v7.5.0 (public). Fetched to
  `reproductions/3dgrt/source/deps/optix-dev/include`.
- Official tracer build (`threedgrt_tracer/setup_3dgrt.py`, nv-tlabs/3dgrut @ bb203b1, already in
  `reproductions/3dgut/source/code`): `torch.utils.cpp_extension` JIT of `optixTracer.cpp`,
  `particlePrimitives.cu`, `bindings.cpp` — fits `tools/ssenv` (CUDA 12.1, gcc 11) — **but first** it runs
  `slangc` to generate `include/3dgrt/kernels/slang/gaussianParticles.cuh` from `.slang` sources.
  `slangc` is not in the env (no `slangtorch` either). Downloading the prebuilt slangc release
  was **denied by the permission classifier** ("Code from External"). Building/loading the vendored
  OptiX tracer itself is also external code and may need the same approval.

## What the user needs to run (to unblock)
```bash
mkdir -p /home/sejain/splatify/reproductions/3dgrt/source/deps/slang
cd /home/sejain/splatify/reproductions/3dgrt/source/deps/slang
curl -sSL -o slang.tar.gz https://github.com/shader-slang/slang/releases/download/v2026.5.2/slang-2026.5.2-linux-x86_64.tar.gz
tar xzf slang.tar.gz && rm slang.tar.gz && ./bin/slangc -version   # expect 2026.5.2
```
(or `tools/ssenv pip install slangtorch==1.3.18`, which bundles slangc), and allow Claude to JIT-build
and import the vendored `threedgrt_tracer` (nv-tlabs/3dgrut, Apache-2.0) through `tools/ssenv`
(e.g. a permission rule for `tools/ssenv python*` running the plugin build). Then re-run this paper.

## Plan once unblocked (not started)
Plugin renderer wrapping the unmodified OptiX tracer (BVH of stretched icosahedra / AABB proxies,
k-buffer any-hit, max-response along ray) for training + eval; probes vs an independent
float64 ray-marching reference (alpha, ordering, gradients); 3DGRT density (official GSStrategy,
shared with `reproductions/3dgut`); benchmark 2 reported Mip-NeRF 360 scenes (garden + counter)
per the paper's supplementary, 30k steps, vs `baseline3dgs-mipnerf360-30000`.
