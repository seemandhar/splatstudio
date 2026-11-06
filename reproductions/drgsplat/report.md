# DRGSplat: Depth-Regularized 3D Gaussian Splatting (plan id 99) — BLOCKED

**Outcome: blocked (no build attempted).** Nothing was implemented or benchmarked.

## What was found
- **Paper identity:** Barath, Tateno, Pollefeys, Tombari. Listed on Google Research as BMVC 2026. It was also an ICLR 2026 submission, OpenReview forum `BpwRgbmTW9` (submission 7179, rejected, CC BY 4.0). The OpenReview metadata, including the abstract, is saved in `source/openreview_note.json`.
- **Abstract-level mechanism:** rendered-depth regularization with three losses: (1) a monocular depth loss for global consistency, (2) a surface-normal loss for local detail, and (3) an uncertainty-aware curvature/flatness loss that penalizes high-gradient regions. Headline result: ETH3D accuracy/completeness/F1 of 3DGS improve by +15/+25/+17 pp (ICLR version) or +13/+25/+16 pp (BMVC page).
- **Code:** none. The Google Research page and the abstract both say "code will be made publicly available". GitHub repo search for `drgsplat` returns 0 hits, and no author page links code.
- **arXiv:** no preprint (the arXiv API search for "DRGSplat" is empty). There are no BMVC 2026 proceedings on bmva-archive yet (404), and Semantic Scholar has no open-access copy.

## What is missing (why blocked)
1. **Full paper text.** The only full text is the OpenReview PDF (`/pdf/8e2eab8d82ecedca8de99376885480238a54424f.pdf`). OpenReview serves a browser verification challenge (HTTP 403 `ChallengeRequiredError`) to every non-interactive request, from curl, the API host and WebFetch. Getting around a bot check is not allowed, so without the paper I'm missing:
   - the exact loss equations: the monocular depth loss (scale/shift alignment? inverse or metric depth? which network?), the normal loss (how normals are derived from rendered depth, and in which frame), and the curvature/flatness loss (gradient operator, what the uncertainty is and where it comes from, and the masking rule);
   - the loss weights, schedules (start iteration), and which prior networks are used;
   - the result tables: PSNR/SSIM/LPIPS per dataset or scene (Mip-NeRF 360? T&T? DB?), plus ETH3D Acc/Comp/F1 per scene and the thresholds used.
2. **No code** to recover any of this from.
3. **Geometry benchmark data.** ETH3D high-res multi-view (with GT laser scans and the ETH3D evaluator) is not on disk. It is downloadable publicly, so this alone would not block, but it would still need setting up.

With equations, weights, the prior source and the uncertainty definition all unknown, a build would be a generic "mono depth + normal + edge-aware smoothness" method that just carries the paper's name. That breaks the fidelity rule.

## What would unblock it
- **The user downloads the PDF in a browser** from https://openreview.net/forum?id=BpwRgbmTW9 (or the BMVC 2026 camera-ready once it is published) and drops it at `reproductions/drgsplat/source/paper.pdf`. Then re-run the build from the paper-reader stage.
- Or: the authors release code, or the paper appears on arXiv.
