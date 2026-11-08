# splatstudio-method-dwtgs

DWTGS (arXiv 2507.15690): wavelet-space frequency regularisation for sparse-view 3DGS, built on
DropGaussian (the in-repo `ss_dropgaussian` port; its renderer, photometric loss and density are
reused unchanged). No official code exists; this follows the paper (see `../paper.md`).

| term | config (paper) | source |
|---|---|---|
| `dwtgs-lf` Eq. 5: Σ_{n≤N} λ_LL L1(LL_n, LL_n_gt) | weight 0.5, levels 2, averaging Haar (LL = block mean) | §IV-B, §V-A |
| `dwtgs-hf` Eq. 6: L1(HH_1(novel render), 0) | weight 1, every 10 its, until 5000; novel views slerp/lerp between a random training pair (Fig. 2 note 3) | §IV-B, §V-A |

Methods: `dwtgs` (full), `dwtgs-lf` (Table III "+LF 2-level" ablation).
Fidelity gaps: Haar normalisation (averaging chosen from §IV-B; `haar_norm: orthonormal` is a
labelled option), interpolation details unstated (random pair, t~U(0,1)), novel view rendered
without dropout, HF weight 1, inherited DropGaussian gaps. Labelled option `novel_views: fsgs_llff`
uses FSGS poses (code and near/far table copied from `reproductions/fsgs`).

```bash
python3 tools/bench.py launch --exp dwtgs-llff-r8 --methods dwtgs --dataset llff-fewview3-r8 --steps 10000
```
