# Paper reproductions as native SplatStudio plugins

This directory holds the reproductions of the SPLATIFY 100-paper plan. Each paper is ported as a native SplatStudio method plugin, then trained and evaluated with the paper's protocol on this runtime and compared against the paper's reported numbers. Live status is in **[PROGRESS.md](PROGRESS.md)**.

## Layout
Each paper has its own directory, `<slug>/`:
- `plugin/`: pip-installable method package (`pip install -e <slug>/plugin`, then `ss-train <method>`).
- `report.md`: outcome, results against the paper and against our gsplat baseline, every fidelity gap, and reproduce commands.
- `paper.md`, `paper_results.json`: extracted paper protocol and target numbers.
- `plan.md` / `plan.json`: mapping from paper contributions to components and probes.
- `reviews/`: independent review of the port against the official code.
- `bench_request.json`, `state.json`: benchmark request and pipeline state.
- `tools/`: dataset builders specific to that paper, where needed.

Other entries:
- `_tools/`: benchmark scheduler (`bench.py`), dataset registry, protocol helpers (PIL resize, multi-scale builder), reference 3DGS numbers, and known runtime pitfalls (`TODO-runtime.md`).
- `_pipeline/`: the Claude Code agents and skills that ran the pipeline: read, plan, implement, test, review, benchmark and gap analysis.

## Not included
- Downloaded paper sources and official repositories (each report cites the repo and commit).
- Datasets, model weights and priors.
- Training runs, checkpoints and build artefacts.
- glm documentation and tests.

## Licences
Many plugins vendor the authors' CUDA rasterizers **unmodified**. These are mostly under the Inria/MPII Gaussian-Splatting licence, which permits non-commercial research use only. Some carry other non-commercial or MIT licences.

Each vendored directory keeps its original licence file, and the plugin README or `csrc/` notes cite it. Those licences, not this repository's licence, govern the vendored code.

## Protocol notes
- **Comparison:** results compare each method's gain over our own gsplat 3DGS baseline, on the same images, with the paper's gain over its own 3DGS. Absolute PSNR is compared as well.
- **Tolerance:** ±0.5 dB PSNR.
- **Scene selection:** for papers that report per-scene numbers, 2–3 reported scenes. For average-only papers, the cheapest complete dataset average.
- **Honesty rules:** no tuning on test images, and failed or weak results are kept.
