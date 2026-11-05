---
name: splat-setup
description: Check and repair the SplatStudio reproduction environment — splatstudio conda env, gsplat CUDA kernels, lpips, datasets (Mip-NeRF 360, Tanks&Temples, Deep Blending) and their native conversions, runtime calibration.
---

# /splat-setup

Run each check; fix only what fails; report a short table at the end.

1. **Env**: `tools/ssenv python -c "import torch,gsplat,splatstudio,lpips;print(torch.__version__,gsplat.__version__,torch.cuda.is_available())"`.
   If the env is missing or broken, recreate it (there is no system nvcc, so gsplat must be
   the prebuilt wheel, which exists only for cp310):
   ```bash
   source ~/miniconda3/etc/profile.d/conda.sh
   conda create -y -n splatstudio python=3.10 && conda activate splatstudio
   pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
   pip install "gsplat==1.5.3+pt24cu121" -f https://docs.gsplat.studio/whl/gsplat/
   pip install -e "/home/sejain/splatify/splatstudio[runtime,data,reports,dev]" lpips
   ```
2. **CUDA kernels**: `CUDA_VISIBLE_DEVICES=<free gpu> tools/ssenv ss-verify gsplat --device cuda`
   must exit 0.
3. **SplatStudio tests**: `cd splatstudio && ../tools/ssenv python -m pytest -q`.
4. **Data**: for `mipnerf360`, `tandt`, `db`: `python3 tools/bench.py prepare --dataset <d>`.
   Missing raw data:
   - Mip-NeRF 360: `http://storage.googleapis.com/gresearch/refraw360/360_v2.zip` and
     `https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip` → unzip into
     `~/datasets/mipnerf360/360_v2/`.
   - T&T + DB (3DGS release): `https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip`
     → unzip into `~/datasets/tandt_db/`.
5. **Installed plugins**: `tools/ssenv ss-methods` and list `reproductions/*/plugin`.
6. **Calibration**: report `reproductions/_calibration/report.md` if present; otherwise say a
   baseline calibration has not been run (`/splat-bench gsplat`).
