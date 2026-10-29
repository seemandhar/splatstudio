# Method collection

The collection contains **33 research configurations**, including baseline and capacity variants and exploratory extensions. These are imported implementations from [SPLATIFY](https://github.com/seemandhar/splatify), packaged for standard Nerfstudio discovery. They are not 33 independent, verified paper reproductions.

Install with `pip install -e packages/nerfstudio-methods` in a Nerfstudio environment, then run `ns-train ss-METHOD --data /path/to/scene`. Native configurations use `ss-train`; see the [root quickstart](../README.md).

All 33 configuration imports passed on stock Nerfstudio 1.1.5. Short training checks and their exact scope are listed below. **Passed** means ten steps on a 32×32 analytic fixture with finite state and checkpoint writing. **Not run** identifies missing auxiliary inputs. These checks leave paper reproduction unverified.

| Configuration | Area | Training check | Historical records |
| :--- | :--- | :--- | ---: |
| [gaussian-shader](methods/gaussian-shader.md) | Appearance | passed | 0 |
| [vod3dgs-l](methods/vod3dgs-l.md) | Appearance | passed | 9 |
| [vod3dgs-s](methods/vod3dgs-s.md) | Appearance | passed | 0 |
| [gs-template](methods/gs-template.md) | Baseline | passed | 0 |
| [compgs](methods/compgs.md) | Compression | passed | 0 |
| [compgs-enhanced](methods/compgs-enhanced.md) | Compression | passed | 0 |
| [light-gs](methods/light-gs.md) | Compression | passed | 0 |
| [lp3dgs](methods/lp3dgs.md) | Compression | passed | 0 |
| [trick-gs](methods/trick-gs.md) | Compression | passed | 2 |
| [absgs](methods/absgs.md) | Densification | passed | 0 |
| [bogauss](methods/bogauss.md) | Densification | passed | 9 |
| [compact-gs](methods/compact-gs.md) | Densification | passed | 0 |
| [decomp-gs](methods/decomp-gs.md) | Densification | passed | 9 |
| [pixel-gs](methods/pixel-gs.md) | Densification | passed | 0 |
| [clod-gs](methods/clod-gs.md) | Efficiency | passed | 0 |
| [fast-gs](methods/fast-gs.md) | Efficiency | passed | 0 |
| [fast-gs-big](methods/fast-gs-big.md) | Efficiency | passed | 0 |
| [rain-gs](methods/rain-gs.md) | Efficiency | passed | 1 |
| [bubble-gs](methods/bubble-gs.md) | Exploratory | passed | 0 |
| [diamond-gs-3d](methods/diamond-gs-3d.md) | Exploratory | passed | 0 |
| [nebula-gs](methods/nebula-gs.md) | Exploratory | passed | 0 |
| [feature-gs](methods/feature-gs.md) | Features | not run | 0 |
| [micro-splatting](methods/micro-splatting.md) | Filtering | passed | 7 |
| [msaa-gs](methods/msaa-gs.md) | Filtering | passed | 2 |
| [msaa-gs-plus](methods/msaa-gs-plus.md) | Filtering | passed | 1 |
| [opti3dgs](methods/opti3dgs.md) | Filtering | passed | 9 |
| [scaffold-gs](methods/scaffold-gs.md) | Geometry | passed | 0 |
| [stablegs](methods/stablegs.md) | Geometry | not run | 0 |
| [twodgs](methods/twodgs.md) | Geometry | passed | 0 |
| [cor-gs](methods/cor-gs.md) | Sparse views | passed | 0 |
| [depth-reg-gs](methods/depth-reg-gs.md) | Sparse views | not run | 0 |
| [dwtgs](methods/dwtgs.md) | Sparse views | passed | 9 |
| [fsgs](methods/fsgs.md) | Sparse views | not run | 0 |

## Known boundaries

The imported `twodgs` port uses the shared 3D rasterizer with geometric regularizers. It does not implement the original paper's 2D ray-splat intersection. The native `gsplat-surfel` recipe does call the actual gsplat 2D rasterizer, with its separate L1 distortion convention.

GaussianShader uses a PyTorch approximation of its shading path. CompGS includes approximated pruning cadence. DepthRegGS/FSGS require depth models, StableGS requires DUSt3R priors, and FeatureGS needs feature targets. See individual cards for details. Nebula, bubble, and diamond configurations are exploratory appearance models; no physical fidelity or scientific priority is asserted.

Historical records are inherited exports with unverified protocols, not results from this release. The [results report](results.md) explains their scope and the new native experiments. Machine-readable metadata ships in [catalog.json](../src/splatstudio/catalog.json).
