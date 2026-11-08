# Reproduction progress

126/130 started: 4 blocked, 10 built, 5 built (benchmark pending), 1 built (fix pending re-benchmark), 5 in progress, 1 iterating, 55 matched, 1 matched (3/3 scenes), 1 matched (T&T, DB), 1 matched (absolute; gain over Splatfacto not reproduced), 1 matched (absolute; indoor gain over 3DGS not reproduced), 3 matched (iter 1), 1 not matched (test split: -0.65/-0.85 dB; control 3DGS -0.18/-0.51; gain over control not reproduced), 14 not-matched, 1 not-matched (borderline), 1 not-matched (iter 1), 5 partial, 2 partial (2/3 scenes), 14 ready_for_benchmark

| # | paper | stage | outcome | ours vs paper (PSNR) |
|---:|---|---|---|---|
| 1 | [AbsGS: Recovering Fine Details for 3D Gaussian Splatting](absgs/report.md) | report | matched |  |
| 2 | Micro-splatting: Multistage Isotropy-informed Covariance Regularizatio | — | — | |
| 3 | Gradient-Driven Natural Selection for Compact 3D Gaussian Splatting | — | — | |
| 4 | [Multi-Sample Anti-Aliasing and Constrained Optimization for 3D Gaussia](msaa-gs/report.md) | iter1-implement | iterating |  |
| 5 | [Optimized 3D Gaussian Splatting using Coarse-to-Fine Image Frequency M](opti3dgs/report.md) | report | matched | T&T 23.46 vs paper 23.15 (+0.31); Δ vs our 3DGS +0.10 vs paper −0.54; #G −41% vs paper −64% |
| 6 | [Decomposing Densification in Gaussian Splatting for Faster 3D Scene Re](decomp-gs/report.md) | gap_analysis | not-matched | T&T mean 23.32 vs paper 24.06 (-0.74); delta vs our gsplat -0.04 vs paper +0.44; Gaussians 0.51x, 2.3x faster (paper 0.55x) |
| 7 | [BOGausS: Better Optimized Gaussian Splatting](bogauss/report.md) | report | not-matched |  |
| 8 | [LightGaussian: Unbounded 3D Gaussian Compression with 15x Reduction an](lightgaussian/report.md) | report | matched |  |
| 9 | [StableGS: A Floater-Free Framework for 3D Gaussian Splatting](stablegs/report.md) | test | ready_for_benchmark |  |
| 10 | [VoD-3DGS: View-opacity-Dependent 3D Gaussian Splatting](vod3dgs/report.md) | report | not-matched |  |
| 11 | [Relaxing Accurate Initialization Constraint for 3D Gaussian Splatting](rain-gs/report.md) | report | matched |  |
| 12 | [Trick-GS: A Balanced Bag of Tricks for Efficient Gaussian Splatting](trick-gs/report.md) | report | not-matched | T&T mean 22.51 (iter0) / 22.41 (iter1) vs paper 23.48; method gap -0.66 + runtime -0.31 |
| 13 | CLoD-GS: Continuous Level-of-Detail via 3D Gaussian Splatting | — | — | |
| 14 | [2D Gaussian Splatting for Geometrically Accurate Radiance Fields](2dgs/report.md) | report | matched |  |
| 15 | [Pixel-GS: Density Control with Pixel-aware Gradient for 3D Gaussian Sp](pixel-gs/report.md) | report | matched |  |
| 16 | GaussianShader: 3D Gaussian Splatting with Shading Functions for Refle | — | — | |
| 17 | [Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering](scaffold-gs/report.md) | report | matched |  |
| 18 | [DWTGS: Rethinking Frequency Regularization for Sparse-view 3D Gaussian](dwtgs/report.md) | report | not-matched |  |
| 19 | [LP-3DGS: Learning to Prune 3D Gaussian Splatting](lp-3dgs/report.md) | benchmark | matched | garden 26.99 (-0.30), room 31.30 (-0.19); prune ratio 0.576/0.735 vs paper 0.59/0.74 |
| 20 | [CompGS: Compact Gaussian Splatting with Vector Quantization](compgs/report.md) | report | matched | Decoded CompGS 32K: garden 26.56/26.822 (-0.26), room 31.32/31.131 (+0.18); delta vs gsplat -0.33/-0.21 vs paper -0.42/-0.22; paper-formula size 27.53/9.12 MB vs 30/9 MB |
| 21 | [CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](cor-gs/report.md) | benchmark | matched |  |
| 22 | [FSGS: Real-Time Few-shot View Synthesis using Gaussian Splatting](fsgs/report.md) | report | matched |  |
| 23 | [FastGS: Training 3D Gaussian Splatting in 100 Seconds](fast-gs/report.md) | implement | in progress |  |
| 24 | [3D Gaussian Splatting for Real-Time Radiance Field Rendering](3dgs/report.md) | report | matched |  |
| 25 | [3D Gaussian Splatting as Markov Chain Monte Carlo](3dgs-mcmc/report.md) | report | matched |  |
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
| 88 | [Deblurring 3D Gaussian Splatting](deblurring-3dgs/report.md) | gap-analysis | built (fix pending re-benchmark) |  |
| 89 | [BAGS: Blur Agnostic Gaussian Splatting through Multi-Scale Kernel Mode](bags/report.md) | tester | built |  |
| 90 | [Spec-Gaussian: Anisotropic View-Dependent Appearance for 3D Gaussian S](spec-gaussian/report.md) | tester | ready_for_benchmark |  |
| 91 | [GS-IR: 3D Gaussian Splatting for Inverse Rendering](gs-ir/report.md) | report | built |  |
| 92 | [FisherRF: Active View Selection and Uncertainty Quantification for Rad](fisherrf/report.md) | tester | ready_for_benchmark |  |
| 93 | [A Step to Decouple Optimization in 3DGS](adamw-gs/report.md) | test | built |  |
| 94 | [Faster and Better 3D Splatting via Group Training](group-training/report.md) | test | built (benchmark pending) |  |
| 95 | [MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic M](maskgaussian/report.md) | test | built |  |
| 96 | [ResGS: Residual Densification of 3D Gaussian for Efficient Detail Reco](resgs/report.md) | test | built |  |
| 97 | [HDA-GS: Hierarchical Density-Controlled for Anisotropic 3D Gaussian Sp](hda-gs/report.md) | started | blocked |  |
| 98 | [Multi-Scale 3D Gaussian Splatting for Anti-Aliased Rendering](ms-gs/report.md) | test | ready_for_benchmark |  |
| 99 | [DRGSplat: Depth-Regularized 3D Gaussian Splatting](drgsplat/report.md) | started | blocked |  |
| 100 | [Faster-GS: Analyzing and Improving Gaussian Splatting Optimization](faster-gs/report.md) | test | ready_for_benchmark | smoke garden 2k: 23.42 vs gsplat 22.99 dB, 65 s vs 150 s |
| 101 | [Unified Loss-Aware Density Control for 3D Gaussian Splatting](loss-aware-dc/report.md) | benchmark | matched |  |
| 102 | [Efficient Confidence-Aware Densification for 3D Gaussian Splatting](confidence-densify/report.md) | gap_analysis | not-matched |  |
| 103 | [Pushing Rendering Boundaries: Hard Gaussian Splatting](hard-gs/report.md) | report | matched |  |
| 104 | [Improving Gaussian Splatting with Localized Points Management](lpm/report.md) | report | matched |  |
| 105 | [ConeGS: Error-Guided Densification Using Pixel Cones for Improved Reco](conegs/report.md) | report | matched (iter 1) | garden 27.46 vs 27.42 (iter 1, post-hoc black bg); room 32.15 vs 32.30 (iter 0) |
| 106 | [Evaluating Alternatives to SFM Point Cloud Initialization for Gaussian](nerf-init-gs/report.md) | gap_analysis | not-matched |  |
| 107 | [Beyond Gaussians: Fast and High-Fidelity 3D Splatting with Linear Kern](linear-splatting/report.md) | gap_analysis | not-matched |  |
| 108 | [3DGabSplat: 3D Gabor Splatting for Frequency-adaptive Radiance Field R](gabsplat/report.md) | gap_analysis | not-matched |  |
| 109 | [HDGS: Textured 2D Gaussian Splatting for Enhanced Scene Rendering](hdgs/report.md) | report | matched |  |
| 110 | [Augmented Radiance Field: A General Framework for Enhanced Gaussian Sp](augs/report.md) | report | matched |  |
| 111 | [DET-GS: Depth- and Edge-Aware Regularization for High-Fidelity 3D Gaus](det-gs/report.md) | gap-analysis-1 | not-matched |  |
| 112 | [GaussianPOP: Principled Simplification Framework for Compact 3D Gaussi](gaussianpop/report.md) | benchmark | matched | P=0.8 post-training: bicycle 25.01 (-0.11), counter 28.87 (-0.13) vs paper |
| 113 | [WE-GS: An In-the-wild Efficient 3D Gaussian Representation for Unconst](we-gs/report.md) | gap-analysis-1 | not-matched |  |
| 114 | [CRiM-GS: Continuous Rigid Motion-Aware Gaussian Splatting from Motion-](crim-gs/report.md) | gap-analysis | partial |  |
| 115 | [Denoising-GS: Gaussian Splatting with Spatial-aware Denoising](denoising-gs/report.md) | report | partial | counter -0.21 (matched), bicycle -0.74 (backbone: control -0.69 vs paper MCMC); method-control -0.04/-0.03 vs paper +0.02/+0.06 |
| 116 | [Sort-free Gaussian Splatting via Weighted Sum Rendering](sort-free-gs/report.md) | gap_analysis | partial |  |
| 117 | [Turbo-GS: Accelerating 3D Gaussian Fitting for High-Quality Radiance F](turbo-gs/report.md) | report | matched |  |
| 118 | [Metropolis-Hastings Sampling for 3D Gaussian Reconstruction](mh-3dgs/report.md) | report | matched |  |
| 119 | [Perceptual-GS: Scene-adaptive Perceptual Densification for Gaussian Sp](perceptual-gs/report.md) | benchmark | matched | pilr 30k: bicycle 25.79 (−0.17), counter 29.21 (−0.24) vs paper; both within ±0.5 |
| 120 | [LiteGS: A High-performance Framework to Train 3DGS in Subminutes via S](litegs/report.md) | benchmark | matched | garden 27.78 vs 28.15 (−0.37), room 32.79 vs 32.69 (+0.10); Δ vs 3DGS +0.89/+1.27 vs paper +0.76/+1.22 |
| 121 | [Optimized Minimal Gaussians](omg/report.md) | benchmark | matched | OMG-XS decoded: bicycle 24.68/24.95 (-0.27), counter 27.94/28.40 (-0.46); 421k/299k G (paper 481k/310k); 4.18/2.94 MB (paper 4.61/2.95) |
| 122 | [Volumetrically Consistent 3D Gaussian Rasterization](vol3dgs/report.md) | benchmark | matched |  |
| 123 | [On the Error Analysis of 3D Gaussian Splatting and an Optimal Projecti](optimal-projection/report.md) | report | matched |  |
| 124 | [Compact Neural Appearance Models for Efficient Gaussian Splatting](neural-appearance/report.md) | analysis | matched | Neural garden 28.05 (paper 28.06), room 33.05 (33.23); Neural-SH +0.10/+0.25 = paper |
| 125 | [Deformable Beta Splatting](beta-splatting/report.md) | report | matched |  |
| 126 | [3D Student Splatting and Scooping](student-splatting/report.md) | benchmark | matched |  |
| 127 | [AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering](aaa-gaussians/report.md) | report | matched |  |
| 128 | [Low-Frequency First: Eliminating Floating Artifacts in 3D Gaussian Spl](efa-gs/report.md) | benchmark | matched | pilr 30k black bg: garden 27.95 (-0.07), bonsai 32.19 (-0.37); control efa-gs-base 27.90/32.39; LFCF effect +0.05/-0.20 vs paper -0.02/+0.22 (single seed, unresolved); 0.66-0.69x Gaussians |
| 129 | [BillBoard Splatting (BBSplat): Learnable Textured Primitives for Novel](bbsplat/report.md) | report | partial |  |
| 130 | [Effective Rank Analysis and Regularization for Enhanced 3D Gaussian Sp](erank-gs/report.md) | report | not matched (test split: -0.65/-0.85 dB; control 3DGS -0.18/-0.51; gain over control not reproduced) |  |
