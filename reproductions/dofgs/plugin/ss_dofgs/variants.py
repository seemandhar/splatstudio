"""Official per-scene configurations of DOF-GS on the real defocus (Deblur-NeRF) scenes.

Transcribed from DOFGS_implementation @cfd635c config_files/config_real_<scene>.yaml (the
resolution-4 configs used for the paper's 600x400 evaluation). Across the nine real scenes only
two keys differ: ``band_limit`` (Mip-Splatting 3D filter in rendering + filter-aware opacity
reset, training.py:286-289, renderer/__init__.py:63-82) and ``lambda_mask_loss`` (L_mk weight;
the entropy regulariser weight is hard-coded as 0.1 * lambda_mask_loss, training.py:212). All
other keys are shared and are the plugin component defaults (pinned by tests/test_dofgs.py).
"""

# scene -> (band_limit, lambda_mask_loss)
OFFICIAL_REAL = {
    "cake": (True, 0.002),
    "daisy": (True, 0.002),
    "cisco": (True, 0.001),
    "sausage": (True, 0.001),
    "seal": (True, 0.001),
    "tools": (True, 0.001),
    "caps": (False, 0.001),
    "cupcake": (False, 0.001),
    "cups": (False, 0.001),
}
REG_FACTOR = 0.1  # lambda_reg = 0.1 * lambda_mask_loss (training.py:212)

# method name -> (band_limit, lambda_mask_loss)
VARIANTS = {
    "dofgs": (True, 0.002),          # cake, daisy
    "dofgs-bl-m001": (True, 0.001),  # cisco, sausage, seal, tools
    "dofgs-nobl": (False, 0.001),    # caps, cupcake, cups
}
SCENE_VARIANT = {scene: next(n for n, v in VARIANTS.items() if v == cfg)
                 for scene, cfg in OFFICIAL_REAL.items()}

# keys shared by every config_real_<scene>.yaml -> (component, plugin option)
SHARED_KEYS = {
    "densify_until_iter": ("density", "densify_until_iter"),
    "iterations": ("density", "total_steps"),
    "num_extra_points": ("density", "num_extra_points"),
    "fdist_lr": ("density", "fdist_lr"),
    "position_lr_init": ("density", "position_lr_init"),
    "position_lr_final": ("density", "position_lr_final"),
    "position_lr_max_steps": ("density", "position_lr_max_steps"),
    "scaling_lr": ("density", "scaling_lr"),
    "min_opacity": ("density", "min_opacity"),
    "percent_dense": ("density", "percent_dense"),
    "add_point_iterations": ("density", "add_point_iterations"),
    "iter_defocus_start": ("density", "iter_defocus_start"),
    "coc_max": ("density", "coc_max"),
    "shrink_factor": ("density", "shrink_factor"),
    "use_compatnes_density": ("density", "use_compactness"),
    "densify_compatnes_interval": ("density", "compactness_interval"),
    "compatnes_K": ("density", "compactness_K"),
    "mask_nn_lr": ("density", "mask_nn_lr"),
    "extra_points_factor": ("density", "extra_points_factor"),
    "iter_stop_cam_gd": ("density", "iter_stop_cam_gd"),
    "infocus_start_iter": ("objective", "infocus_start_iter"),
    "lambda_infocus": ("objective", "lambda_infocus"),
    "lambda_dssim": ("objective", "lambda_dssim"),
    "use_mask_loss": ("mask", "use_mask_loss"),
    "maskNet_start_iter": ("mask", "infocus_start_iter"),
}

# their values (identical in all nine resolution-4 real configs)
SHARED_VALUES = {
    "densify_until_iter": 25000, "iterations": 40000, "num_extra_points": 30000,
    "fdist_lr": 0.05, "position_lr_init": 0.0005, "position_lr_final": 0.000005,
    "position_lr_max_steps": 80000, "scaling_lr": 0.01, "min_opacity": 0.005,
    "percent_dense": 0.03, "add_point_iterations": [2000], "iter_defocus_start": 100,
    "coc_max": 15, "shrink_factor": 1.5, "use_compatnes_density": True,
    "densify_compatnes_interval": 1000, "compatnes_K": 3, "mask_nn_lr": 0.0005,
    "extra_points_factor": 0.25, "iter_stop_cam_gd": 15000, "infocus_start_iter": 5000,
    "lambda_infocus": 1.0, "lambda_dssim": 0.2, "use_mask_loss": True, "maskNet_start_iter": 5000,
}
# yaml keys with no plugin option, and why that is faithful
NOT_PORTED = {
    "band_limit": "per-variant (OFFICIAL_REAL)",
    "lambda_mask_loss": "per-variant (OFFICIAL_REAL)",
    "use_size_prune": "false in every config -> the plugin has no size/world prune",
    "use_scale_regularization": "false -> unused", "max_gauss_ratio": "only used with scale reg",
    "use_depth_estimator": "false -> unused", "depth_end_iter": "depth estimator only",
    "depth_reslution": "depth estimator only", "depth_update_freq": "depth estimator only",
    "tonemap_type": "'none' -> identity", "resolution": "4 -> data converter (600x400)",
    "model_folder": "bookkeeping", "model_path": "bookkeeping", "model_version": "bookkeeping",
    "scene_name": "bookkeeping", "source_path": "bookkeeping",
    "save_iterations": "bookkeeping", "test_iterations": "logging only (no selection)",
}
