# Planar Gaussian Splatting (plan id 65) — BLOCKED

**Outcome: blocked (no build attempted).** Nothing was implemented or benchmarked.

Why it is blocked:
1. **No benchmark target we can evaluate.** The paper reports **no PSNR/SSIM/LPIPS anywhere**, main text or supplement. Its only numbers are 3D plane instance segmentation scores (VOI / RI / SC) on ScanNetv2 and Replica (Tables 1–2), plus Accuracy/Completeness against the ScanNet ground-truth mesh (supplement). It has no results on mipnerf360, tandt, db, llff, blender, DTU or MuSHRoom.
2. **The data and evaluator can't be obtained here.** ScanNetv2 needs a signed Terms of Use form and a manual approval from the dataset owners, which an agent can't do in reasonable time. The ground-truth plane labels have to be generated with the PlanarRecon/NMF plane-annotation pipeline. The scene list is incomplete: only 4 of the 10 ScanNet scenes can be identified (via the Manhattan-SDF reference), and the Replica scenes are not named. So even the averages couldn't be compared like for like.
3. **No official code, and the method is underspecified.** The merge thresholds eps_B and eps_z, the mean-shift rate eta, the weights of the normal and descriptor losses, the exact descriptor/regression loss, and the step from planes to evaluated labels are all missing. On top of that it needs SAM and Omnidata preprocessing. Any re-derivation would rest on guessed values that could only be checked against plane metrics we can't compute.

What would unblock it: ScanNetv2 access granted by the user, plus PlanarRecon's GT-plane generation and RI/VOI/SC evaluator. It would also need the authors' scene list and hyper-parameters, or their code if it is ever released.

Files: paper.md, paper_results.json, source/ (LaTeX + PDF).
