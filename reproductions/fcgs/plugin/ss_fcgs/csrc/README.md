Vendored CUDA sources (unmodified).

- `gridcreater/`, `gridencoder/`, `freqencoder/`: official FCGS repository
  (github.com/YihangChen-ee/FCGS @ 31e59a46, `submodules/<name>/src/`). FCGS is Apache-2.0
  with a non-commercial clause (`LICENSE-FCGS.md`); grid/freq encoders derive from torch-ngp
  (MIT).
- `arithmetic/`: CUDA arithmetic coder, byte-identical in FCGS (`submodules/arithmetic`) and
  HAC; copied from `reproductions/hac/plugin/ss_hac/csrc/arithmetic` (HAC commit 110dcf0a),
  based on torchac (github.com/fab-jul/torchac, GPL-3.0).

Research use only. Built JIT by `ss_fcgs/kernels.py`.
