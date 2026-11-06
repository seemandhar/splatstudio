# Reproduction progress

75/100 started: 4 blocked, 11 built, 5 built (benchmark pending), 5 in progress, 24 matched, 1 matched (3/3 scenes), 1 matched (T&T, DB), 1 matched (absolute; gain over Splatfacto not reproduced), 1 matched (absolute; indoor gain over 3DGS not reproduced), 2 matched (iter 1), 3 not-matched, 1 not-matched (borderline), 1 not-matched (iter 1), 1 partial, 2 partial (2/3 scenes), 11 ready_for_benchmark, 1 skipped

| # | paper | stage | outcome | ours vs paper (PSNR) |
|---:|---|---|---|---|
| 1 | [AbsGS: Recovering Fine Details for 3D Gaussian Splatting](absgs/report.md) | plan | skipped |  |
| 2 | Micro-splatting: Multistage Isotropy-informed Covariance Regularizatio | — | — | |
| 3 | Gradient-Driven Natural Selection for Compact 3D Gaussian Splatting | — | — | |
| 4 | Multi-Sample Anti-Aliasing and Constrained Optimization for 3D Gaussia | — | — | |
| 5 | Optimized 3D Gaussian Splatting using Coarse-to-Fine Image Frequency M | — | — | |
| 6 | Decomposing Densification in Gaussian Splatting for Faster 3D Scene Re | — | — | |
| 7 | BOGausS: Better Optimized Gaussian Splatting | — | — | |
| 8 | LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction an | — | — | |
| 9 | StableGS: A Floater-Free Framework for 3D Gaussian Splatting | — | — | |
| 10 | VoD-3DGS: View-opacity-Dependent 3D Gaussian Splatting | — | — | |
| 11 | Relaxing Accurate Initialization Constraint for 3D Gaussian Splatting | — | — | |
| 12 | Trick-GS: A Balanced Bag of Tricks for Efficient Gaussian Splatting | — | — | |
| 13 | CLoD-GS: Continuous Level-of-Detail via 3D Gaussian Splatting | — | — | |
| 14 | 2D Gaussian Splatting for Geometrically Accurate Radiance Fields | — | — | |
| 15 | Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Sp | — | — | |
| 16 | GaussianShader: 3D Gaussian Splatting with Shading Functions for Refle | — | — | |
| 17 | Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering | — | — | |
| 18 | DWTGS: Rethinking Frequency Regularization for Sparse-view 3D Gaussian | — | — | |
| 19 | LP-3DGS: Learning to Prune 3D Gaussian Splatting | — | — | |
| 20 | CompGS: Compact Gaussian Splatting with Vector Quantization | — | — | |
| 21 | CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization | — | — | |
| 22 | FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting | — | — | |
| 23 | FastGS: Training 3D Gaussian Splatting in 100 Seconds | — | — | |
| 24 | 3D Gaussian Splatting for Real-Time Radiance Field Rendering | — | — | |
| 25 | 3D Gaussian Splatting as Markov Chain Monte Carlo | — | — | |
| 26 | [Mip-Splatting: Alias-free 3D Gaussian Splatting](mip-splatting/report.md) | benchmark | matched | 3 scenes: −0.17/−0.33/−0.19 dB after PIL-protocol iteration; gain over 3DGS ≥ paper |
| 27 | [FreGS: 3D Gaussian Splatting with Progressive Frequency Regularization](fregs/report.md) | calibrate_threshold | not-matched | 27.12 vs 27.85 (−0.73) after 1 iteration; Δ vs our 3DGS −0.16 vs paper +0.64 |
| 28 | [Mini-Splatting: Representing Scenes with a Constrained Number of Gauss](mini-splatting/report.md) | benchmark | matched | 5 scenes: mean gap −0.09 dB (all within ±0.5) |
| 29 | [Mini-Splatting2: Building 360 Scenes within Minutes via Aggressive Gau](mini-splatting2/report.md) | benchmark | matched | 5 scenes: mean gap −0.21 dB (4/5 within ±0.5; garden −0.56) |
| 30 | [Taming 3DGS: High-Quality Radiance Fields with Limited Resources](taming-3dgs/report.md) | benchmark | matched | 9 scenes: 27.60 vs 27.79 (−0.19); Δ vs 3DGS +0.32 vs paper +0.33 |
| 31 | [DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds](dashgaussian/report.md) | gap_analysis | matched | 7 scenes: mean −0.35 dB (5/7 within ±0.5) after 1 fix iteration |
| 32 | [EfficientGS: Streamlining Gaussian Splatting for Large-Scale High-Reso](efficientgs/report.md) | benchmark | matched | 9 scenes: 26.99 vs 27.38 (−0.39); 0.30× Gaussians |
| 33 | [EDGS: Eliminating Densification for Efficient Convergence of 3DGS](edgs/report.md) | benchmark | matched | 7 scenes: mean −0.20 dB (6/7 within ±0.5); +0.22..+0.73 over 3DGS without densification |
| 34 | [AD-GS: Alternating Densification for Sparse-Input 3D Gaussian Splattin](ad-gs/report.md) | gap_analysis | not-matched | LLFF 3-view, 3 seeds (iter 1): horns 19.03 (-0.76), fortress 23.10 (-0.77), room 22.02 (-0.37) vs paper; no remaining code-grounded fix for ~0.4 dB; LPIPS gap is mostly the official [0,1]-input lpipsPyTorch convention |
| 35 | [DropGaussian: Structural Regularization for Sparse-view Gaussian Splat](dropgaussian/report.md) | report | matched (iter 1) |  |
| 36 | [PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting](pup-3dgs/report.md) | benchmark | matched | 3 scenes at 90% pruned: −0.42/−0.33/−0.14 dB; pruning loss matches paper |
| 37 | [Trim 3D Gaussian Splatting for Accurate Geometry Representation](trimgs/report.md) | report | matched |  |
| 38 | [Reducing the Memory Footprint of 3D Gaussian Splatting](reduced-3dgs/report.md) | report | not-matched (borderline) |  |
| 39 | [EAGLES: Efficient Accelerated 3D Gaussians with Lightweight EncodingS](eagles/report.md) | benchmark | matched | Decoded-model PSNR vs paper: bicycle 24.80/25.04 (-0.24), garden 26.52/26.91 (-0.39), counter 28.09/28.40 (-0.31); delta vs gsplat -0.12/-0.37/-0.76 vs paper -0.09/-0.41/-0.71; size() 105.7/64.5/23.6 MB vs 104/74/25 MB (real coded bytes within 0.01 MB) |
| 40 | [Compact 3D Scene Representation via Self-Organizing Gaussian Grids](sog/report.md) | benchmark | matched | SOG decoded JPEG XL model, Mip-NeRF360 30k: PSNR vs official results CSV bicycle 24.53/24.54 (-0.01), garden 26.91/27.03 (-0.12), counter 28.51/28.46 (+0.05); vs our gsplat -0.39/+0.02/-0.34 dB; size 61.06/53.20, 82.26/83.49, 18.56/17.62 MB (+14.8/-1.5/+5.3%, tracking Gaussian count 3.36M/4.45M/1.01M; bytes/Gaussian within 4%). Paper v2 avg 27.64 is pre-fix; CSV avg 27.08 is the decoded-model target. |
| 41 | [HAC: Hash-grid Assisted Context for 3D Gaussian Splatting Compression](hac/report.md) | report | matched |  |
| 42 | [Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis](c3dgs/report.md) | report | matched |  |
| 43 | [Compact 3D Gaussian Representation for Radiance Field](compact-3dgs/report.md) | benchmark | matched | Decoded Ours+PP within 0.5 dB on 3/3 mip360 scenes (bicycle 24.63/24.73, garden 26.35/26.72, bonsai 31.94/31.98); delta vs gsplat -0.29/-0.55/+0.05 vs paper vs 3DGS* -0.37/-0.53/-0.18. Storage Ours+PP 53.99/43.35/19.67 MB vs paper 42.42/43.26/16.40 (7-35% more Gaussians). Gaps: fp32 vs tcnn fp16, white bg/gsplat, update order, global state via fields; no G-PCC (paper uses fp16 positions). |
| 44 | [ContextGS: Compact 3D Gaussian Splatting with Anchor Level Context Mod](contextgs/report.md) | report | matched |  |
| 45 | [End-to-End Rate-Distortion Optimized 3D Gaussian Representation](rdo-gaussian/report.md) | report | matched |  |
| 46 | [MesonGS: Post-training Compression of 3D Gaussians via Efficient Attri](mesongs/report.md) | report | matched |  |
| 47 | [Fast Feedforward 3D Gaussian Splatting Compression](fcgs/report.md) | report | matched |  |
| 48 | [SafeguardGS: 3D Gaussian Primitive Pruning While Avoiding Catastrophic](safeguardgs/report.md) | report | matched |  |
| 49 | [GaussianSpa: An “Optimizing-Sparsifying” Simplification Framework for ](gaussianspa/report.md) | report | partial (2/3 scenes) |  |
| 50 | [SplatFields: Neural Gaussian Splats for Sparse 3D and 4D Reconstructio](splatfields/report.md) | report | partial (2/3 scenes) |  |
| 51 | [DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields with Gl](dngaussian/report.md) | report | matched |  |
| 52 | [SparseGS: Sparse View Synthesis using 3D Gaussian Splatting](sparsegs/report.md) | report | not-matched (iter 1) |  |
| 53 | [FatesGS: Fast and Accurate Sparse-View Surface Reconstruction Using Ga](fatesgs/report.md) | report | matched (iter 1) |  |
| 54 | [MVGS: Multi-view-regulated Gaussian Splatting for Novel View Synthesis](mvgs/report.md) | review | ready_for_benchmark |  |
| 55 | [CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians](coherentgs/report.md) | report | not-matched |  |
| 56 | [SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Recons](sugar/report.md) | report | matched |  |
| 57 | [Gaussian Opacity Fields: Efficient Adaptive Surface Reconstruction in ](gof/report.md) | benchmark | matched (absolute; indoor gain over 3DGS not reproduced) |  |
| 58 | [PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity ](pgsr/report.md) | review | built (benchmark pending) |  |
| 59 | [RaDe-GS: Rasterizing Depth in Gaussian Splatting](radegs/report.md) | review | built (benchmark pending) |  |
| 60 | [DOF-GS: Adjustable Depth-of-Field 3D Gaussian Splatting for Post-Captu](dofgs/report.md) | report | matched (3/3 scenes) |  |
| 61 | [Improving Adaptive Density Control for 3D Gaussian Splatting](improving-adc/report.md) | report | matched (T&T, DB) |  |
| 62 | [High-quality Surface Reconstruction using Gaussian Surfels](gaussian-surfels/report.md) | benchmark | matched |  |
| 63 | [DN-Splatter: Depth and Normal Priors for Gaussian Splatting and Meshin](dn-splatter/report.md) | report | matched (absolute; gain over Splatfacto not reproduced) |  |
| 64 | [AGS-Mesh: Adaptive Gaussian Splatting and Meshing with Geometric Prior](ags-mesh/report.md) | review | built (benchmark pending) |  |
| 65 | [Planar Gaussian Splatting](planar-gs/report.md) | paper-reader | blocked |  |
| 66 | [AtomGS: Atomizing Gaussian Splatting for High-Fidelity Radiance Field](atomgs/report.md) | report | matched |  |
| 67 | [Revising Densification in Gaussian Splatting](revising-densification/report.md) | review | ready_for_benchmark |  |
| 68 | [Steepest Descent Density Control for Compact 3D Gaussian Splatting](steepgs/report.md) | review | ready_for_benchmark |  |
| 69 | [Improving Densification in 3D Gaussian Splatting for High-Fidelity Ren](improving-densification/report.md) | report | matched |  |
| 70 | [Analytic-Splatting: Anti-Aliased 3D Gaussian Splatting via Analytic In](analytic-splatting/report.md) | review | ready_for_benchmark |  |
| 71 | [StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Re](stopthepop/report.md) | review | ready_for_benchmark |  |
| 72 | [Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse](speedy-splat/report.md) | review | ready_for_benchmark |  |
| 73 | [FlashGS: Efficient 3D Gaussian Splatting for Large-scale and High-reso](flashgs/report.md) | report | partial |  |
| 74 | [A Hierarchical 3D Gaussian Representation for Real-Time Rendering of V](hierarchical-3dgs/report.md) | review | ready_for_benchmark |  |
| 75 | [Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured ](octree-gs/report.md) | review | in progress |  |
| 76 | [DeblurGS: Gaussian Splatting for Camera Motion Blur](deblurgs/report.md) | review | built |  |
| 77 | [SpotLessSplats: Ignoring Distractors in 3D Gaussian Splatting](spotlesssplats/report.md) | review | in progress |  |
| 78 | [3D-HGS: 3D Half-Gaussian Splatting](3dhgs/report.md) | review | ready_for_benchmark |  |
| 79 | [GaussianPro: 3D Gaussian Splatting with Progressive Propagation](gaussianpro/report.md) | review | built (benchmark pending) |  |
| 80 | [Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images](gradient-upscaling/report.md) | review | built |  |
| 81 | [3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splat](3dgut/report.md) | review | ready_for_benchmark |  |
| 82 | [3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes](3dgrt/report.md) | feasibility | blocked |  |
| 83 | [Laplacian Frequency Hierarchies for Efficient 3D Gaussian Splatting Tr](laplacian-gs/report.md) | tester | built |  |
| 84 | [WildGaussians: 3D Gaussian Splatting In the Wild](wildgaussians/report.md) | tester | in progress |  |
| 85 | [SWAG: Splatting in the Wild images with Appearance-conditioned Gaussia](swag/report.md) | tester | built |  |
| 86 | [Splatfacto-W: A Nerfstudio Implementation of Gaussian Splatting for Un](splatfacto-w/report.md) | tester | built |  |
| 87 | [RobustSplat: Decoupling Densification and Dynamics for Transient-Free ](robustsplat/report.md) | tester | in progress |  |
| 88 | [Deblurring 3D Gaussian Splatting](deblurring-3dgs/report.md) | tester | built |  |
| 89 | [BAGS: Blur Agnostic Gaussian Splatting through Multi-Scale Kernel Mode](bags/report.md) | tester | built |  |
| 90 | [Spec-Gaussian: Anisotropic View-Dependent Appearance for 3D Gaussian S](spec-gaussian/report.md) | tester | ready_for_benchmark |  |
| 91 | [GS-IR: 3D Gaussian Splatting for Inverse Rendering](gs-ir/report.md) | report | built |  |
| 92 | [FisherRF: Active View Selection and Uncertainty Quantification for Rad](fisherrf/report.md) | tester | ready_for_benchmark |  |
| 93 | [A Step to Decouple Optimization in 3DGS](adamw-gs/report.md) | test | built |  |
| 94 | [Faster and Better 3D Splatting via Group Training](group-training/report.md) | test | built (benchmark pending) |  |
| 95 | [MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic M](maskgaussian/report.md) | test | built |  |
| 96 | [ResGS: Residual Densification of 3D Gaussian for Efficient Detail Reco](resgs/report.md) | test | built |  |
| 97 | [HDA-GS: Hierarchical Density-Controlled for Anisotropic 3D Gaussian Sp](hda-gs/report.md) | started | blocked |  |
| 98 | [Multi-Scale 3D Gaussian Splatting for Anti-Aliased Rendering](ms-gs/report.md) | started | in progress |  |
| 99 | [DRGSplat: Depth-Regularized 3D Gaussian Splatting](drgsplat/report.md) | started | blocked |  |
| 100 | Faster-GS: Analyzing and Improving Gaussian Splatting Optimization | — | — | |
