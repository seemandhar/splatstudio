"""Launch upstream BoostingMonocularDepth (compphoto, cloned to ~/models/bmd/repo) under numpy 2.

SparseGS (official prepare_gt_depth.py -> BoostingMonocularDepth/run.get_monodepth) runs BMD with
depthNet=2 (LeReS res101 + pix2pix merge), Final=True, output_resolution=1; its fork only changes
write_depth to np.save. Upstream writes 16-bit PNG (min-max to [0, 65535]); the SparseGS loader
min-max normalises after resizing, so the 16-bit quantisation is the only difference.
Usage: run_bmd.py <image_dir> <out_dir>   (cwd must be the BMD repo, for its relative paths)
"""
import runpy
import sys

import numpy as np

if not hasattr(np, "RankWarning"):
    np.RankWarning = np.exceptions.RankWarning  # removed in numpy 2; run.py only filters it

data_dir, out_dir = sys.argv[1], sys.argv[2]
sys.argv = ["run.py", "--Final", "--data_dir", data_dir, "--output_dir", out_dir,
            "--depthNet", "2", "--output_resolution", "1"]
sys.path.insert(0, ".")
runpy.run_path("run.py", run_name="__main__")
