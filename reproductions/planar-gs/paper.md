# Planar Gaussian Splatting (PGS) — arXiv 2412.01931, WACV 2025 (Zanjani et al., Qualcomm)

Official code: none (paper, arXiv, WACV supplement, GitHub search; `zju3dv/PGSR` is a different paper).

## Contributions
- C1 (§3.2) Gaussian Mixture Tree: leaf nodes = Gaussians fit to 3D-lifted boundary points of RAG-merged SAM masks (supp §A.2); greedy merge if Bhattacharyya D_B <= eps_B and 1-<z_i,z_j> <= eps_z; parent via Eq.(merge) (product-of-Gaussians form). eps_B, eps_z NOT given.
- C2 (§3.3) Per-Gaussian normal n (R^3) supervised by rendered-normal cosine loss vs Omnidata normals (Eq. normal_rendering). Loss weight NOT given.
- C3 (§3.3) Per-Gaussian unit descriptor z (k=3); rendered descriptor image; per-image closed-form linear regression W from [Z|1] to one-hot labels Y of RAG-merged SAM segments (32x32 grid prompts; high-normal-variance segments dropped). RAG edges cut at 10 cm planar distance (Eq. d_p) / 20 deg normal. Training loss on the regression residual and its weight NOT given.
- C4 (§3.4) Local planar alignment: every 500 it, project centers onto KNN (K=30) tangent plane; Laplacian smoothing of n and z over KNN.
- C5 (§3.5) Recurrent vMF mean-shift Z <- Z(eta K D^-1 + (1-eta) I), gamma=60, 10 steps, every 100 it after 2k; subsampled kernel (supp §A.1). eta NOT given.

## Protocol
3DGS hyper-params except densify grad thresh 0.001, SH degree 2, 15k iterations, SfM init.
Eval: 3D plane instance segmentation RI / VOI / SC on ScanNetv2 (10 scenes, 4 = 0050/0084/0580/0616 from Manhattan-SDF, 6 unnamed) and Replica (scenes unnamed), GT planes per PlanarRecon/NMF protocol; supp: Accuracy/Completeness vs ScanNet GT mesh. No NVS metrics at all.

## Ambiguities (blocking)
eps_B, eps_z, eta, weights of normal and descriptor losses, exact form of the descriptor loss, how planes are turned into the evaluated point/mesh labels, full scene list.
