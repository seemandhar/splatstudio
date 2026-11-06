# HDA-GS (plan id 97): BLOCKED

**Outcome: blocked (no build attempted).** Nothing was implemented or benchmarked.

## Why it is blocked: the full paper can't be obtained
- The only copy is IEEE Xplore (ICASSP 2025, DOI 10.1109/ICASSP49660.2025.10887967, 5 pages). OpenAlex reports it as **closed access** (`oa_status: closed`, no repository full text). ieeexplore.ieee.org returns HTTP 418 to scripted access for both the abstract page and the stamp/getPDF endpoints.
- There is **no arXiv version**. arXiv API searches turned up nothing: by title, "HDA-GS", the author "Zhanke Wang" (5 unrelated GS papers), "Ronggang Wang" + density, "anisotropic opacity", and "hierarchical density".
- There is **no official code or project page**. A GitHub search on the name, the title keywords and the author found nothing. papers_100.json lists no code_url. ResearchGate has a listing with no full text.
- What is public is only the abstract-level mechanism. Citing work (AD-GS) adds one line: HDA-GS classifies Gaussians into several categories by size and applies a different densification strategy to each.

## What is missing (all of it is needed)
1. **Hierarchical density control**: the size-category thresholds, how "sparse initial regions" are detected, the per-category split/clone/gradient thresholds, and the schedules.
2. **View-dependent anisotropic opacity**: its parameterisation (for example SH or direction-dependent opacity), where it enters alpha compositing, any regularisers, and its initialisation. This decides whether a renderer adapter or custom CUDA is needed.
3. **Result tables**: per-scene or average PSNR/SSIM/LPIPS, and which datasets and scenes. Without them there is no benchmark target, and the batch scene rule (per-scene vs. complete average) can't be applied.
4. **Protocol details**: iterations, resolution, and loss weights.

Building from the abstract alone would rest entirely on guessed values, and there would be no number to check it against. That is exactly what the batch rules forbid.

## What would unblock it
The user supplies the PDF, for example via institutional IEEE access. Save it to `reproductions/hda-gs/source/paper.pdf` and rerun the batch prompt for plan id 97 from the read stage.

Files: state.json, source/metadata.json.
