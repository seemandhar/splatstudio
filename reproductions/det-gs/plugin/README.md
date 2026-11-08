# splatstudio-method-det-gs

DET-GS (arXiv 2508.04099) native SplatStudio port. 3DGS + three losses:
`det-gs-depth` (C1, Eq. 4-7: hard depth ω=0.95, patch+image normalisation, γ=0.1, η=1, every 5
steps, patch size U[5,20], DNGaussian δ, error tolerance 0.2 = DNGaussian code default, `space` depth|disparity), `det-gs-edge` (C2, Eq. 8-11:
Canny 20/200 non-edge mask, cross 3×3 masked mean, β=0.1), `det-gs-tv` (C3, Eq. 12-15:
τ_edge=1e-2, τ_smooth=1e-4, ϕ=0.8). Photometric loss = 3DGS (0.8 L1 + 0.2 D-SSIM).

Priors: `reproductions/det-gs/tools/prepare_detgs.py` writes `detgs_priors.npz`
(Depth Anything V2-Large disparity + Canny) into a hard-linked dataset copy
(`~/datasets/detgs-depth/mipnerf360/<scene>`); losses find it via `--data` or `DETGS_DATA`.

Logged det-gs-depth is from depth steps only (log steps are multiples of 5); average ≈ logged/5.

Fidelity gaps: no official code (δ, λ, ω, error tolerance assumed); camera-z instead of ray
distance; TV averaged over channels; edge loss on the soft (Eq. 3) depth. See
`reproductions/det-gs/plan.md`.

    ss-train det-gs --data ~/datasets/detgs-depth/mipnerf360/garden --recipe 3dgs ...
