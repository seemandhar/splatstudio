# Installation

Choose the runtime that owns the method you want to run. The native SDK has no Nerfstudio dependency. The research collection installs into Nerfstudio through its standard method entry points.

## Native SDK

Use Linux, Python 3.10+, an NVIDIA GPU, and a CUDA-compatible PyTorch installation. The metadata-only SDK works without a GPU:

```bash
git clone https://github.com/seemandhar/splatstudio.git
cd splatstudio
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
# Install your CUDA-compatible PyTorch here: https://pytorch.org/get-started/locally/
pip install -e '.[runtime,data,dev]'
ss-methods
ss-verify gsplat --device cuda
```

For the exact Python package versions used for the native release checks, use `pip install -c constraints-tested.txt -e '.[runtime,dev]'` after installing PyTorch. The tested native environment used Python 3.11, PyTorch 2.10.0+cu128, gsplat 1.5.3, NumPy 1.26.4, Pillow 12.1.0, and an RTX A6000. `constraints-tested.txt` records versions; it is not a cross-platform CUDA environment lock.

If gsplat needs to compile an extension, install a compatible CUDA toolkit and C++ toolchain. A CUDA-enabled PyTorch wheel alone does not provide the `nvcc` compiler. Follow [gsplat's installation documentation](https://docs.gsplat.studio/main/INSTALL.html) for supported wheel/build choices. Check `python -c 'import gsplat; print(gsplat.__file__)'` when multiple editable gsplat forks are installed.

Core discovery and scaffolding need only `pip install -e .`. CPU verification cannot establish renderer correctness: skipped CUDA probes produce `unverified` and exit code 2.

## Nerfstudio research collection

Use a separate environment with [Nerfstudio 1.1.5](https://docs.nerf.studio/quickstart/installation.html), its PyTorch/CUDA dependencies, and gsplat. From this repository:

```bash
pip install -e packages/nerfstudio-methods
ns-train ss-absgs --help
ns-train ss-absgs --data /path/to/scene
```

The release import/training audit used **stock Nerfstudio 1.1.5**, Python 3.8.20, PyTorch 2.1.2+cu118, gsplat 1.4.0, and NumPy 1.24.4. The native and collection environments were tested separately. The collection supports Python 3.8+; the native SDK requires Python 3.10+. Installing both in one newer Python environment is not part of the release validation.

`ss-` prefixes prevent collisions with existing method packages. All collection commands start with `ns-train ss-...`, and use Nerfstudio's normal viewer, checkpoint, export, and `ns-eval --load-config ...` workflows. `ss-train` and `ss-eval` apply to native plugins.

DepthRegGS and FSGS need depth model weights and preprocessing; StableGS needs DUSt3R priors; FeatureGS needs feature targets. Consult their [method cards](methods.md) before a full run. Import success does not establish that auxiliary models or datasets are installed.

## Input data and output files

Native input is a directory containing `transforms.json`, images, and optionally `points.npz` or a declared PLY point cloud. Cameras use the Nerfstudio/OpenGL convention. Images must be undistorted pinhole images; `prepare-data` supports OpenCV radial/tangential distortion and records its conversion settings. It preserves poses and point-cloud coordinates, and rejects auxiliary depth/mask inputs that need separate resampling rules.

Supply all three lists `train_filenames`, `val_filenames`, and `test_filenames` for a fixed protocol. Otherwise, every eighth image is test, the fourth in each group is validation, and the rest train. For 3–7 images, the first is test and second validation. Splits must be disjoint and cover all images. Initialization uses `points.npz` XYZ and RGB arrays in [0,1], a PLY cloud, or deterministic random points.

Native runs contain `config.json`, `training.jsonl`, `checkpoint.pt`, `validation.json`, and `status.json`. Explicit `ss-eval` writes `test.json`; `--save-images` writes numbered target/render pairs whose order matches `per_view`. Checkpoints support evaluation, not exact training resume. Evaluation rejects changed source files, method specifications, or datasets.

Native metrics currently include RGB PSNR and SSIM. LPIPS, an interactive native viewer, export to external Gaussian viewers, exact resume, and native dynamic/anchor representations remain future work. The Nerfstudio collection retains that framework's viewer and evaluation features.
