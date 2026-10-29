# Research and engineering roadmap

SplatStudio is the execution and evidence layer for Gaussian splatting methods. SPLATIFY is the paper-to-code agent pipeline. They can work together, but this release does not implement a new agent or establish an autonomous discovery result.

## Available in version 0.1

- Independently installable Python plugins, renderer/density/loss factories, and serializable specifications.
- Capability validation, executable diagnostic probes, and Gaussian/optimizer lifecycle checks.
- Native 3D and 2D rasterization, standard/absolute-gradient/MCMC density strategies.
- A separately installable 33-configuration Nerfstudio research collection with source lineage and status cards.
- Disjoint data splits, source and protocol hashes, paired evaluation, and fresh execution records.

## A research question worth testing

**Can executable mechanism specifications make automatically implemented rendering methods more faithful and safely composable than strong coding agents given the same scaffold, papers, code access, and compute?**

A plugin registry alone is infrastructure. Evidence for the research question needs an independently authored specification set, hard renderer-changing cases, matched agent baselines, and held-out experiments. Correctness of a context-free grammar or a successful training process does not establish semantic fidelity.

| Next milestone | Evidence needed |
| :--- | :--- |
| Independent mechanism benchmark | Paper equations mapped to tests, authored independently of generated implementations; mutation tests for missing mechanisms |
| Renderer coverage | Explicit representation/rasterizer strata; unsupported cases counted as failures or abstentions |
| Faithful reproduction | Rerun original-author code under the same protocol; per-scene absolute gaps, worst gaps, PSNR/SSIM/LPIPS, uncertainty across seeds |
| Fair agent comparison | Strong coding agents and adapted NeRFify with the same template, retrieval, tools, tokens, wall time, GPU budgets, and repair opportunities |
| Controlled composition | Compatible component graphs, fixed Gaussian/memory/time budgets, validation-only selection, frozen held-out scenes, negative transfer reported |
| Vision contribution | A concrete rendering/geometry mechanism with independent physical or geometric tests, relevant domain baselines, and multiple scenes |
| Portable experiments | Clean environment locks, original datasets/checkpoints, exact resume, author-code launchers, and reproducible artifact export |

## Scope that remains open

The native runtime does not yet support anchor-owned/global learned fields, deformation models, custom optimizer factories, or automatic composition search. The imported ports still need individual scientific audits and migration into native contracts. Short execution checks do not exercise late pruning, all densification branches, geometry warmup, compression/decompression, or external teacher pipelines.

Nebula, bubble, and diamond configurations are exploratory appearance experiments. Their existence and RGB quality do not establish physical accuracy, priority over earlier work, or a new scientific method. Any future physical claim needs an independent reference simulator, mechanism measurements, and relevant prior methods.
