"""Official per-scene BAGS configurations (github.com/snldmt/BAGS @ed04144 scripts/*.sh).

Every Deblur-NeRF script shares ``--min_opacity 0.1 --eval -r 4 --kernel_size 0.1``; the keys
below are the ones that differ. ``depth_alpha`` is only used with ``use_depth_loss``
(default --depth_loss_alpha 0.001); ``bpn_lr`` is the KERLR environment variable (default 5e-4;
scripts/defocustools.sh exports 1e-3).
"""

# scene: (dataset, llffhold, init_dgt, iterations, ms_steps, use_depth, depth_alpha,
#         use_rgbtv, use_another_mlp, bpn_lr)
OFFICIAL = {
    "ball": ("motion", 7, 0.0008, 49000, 9000, False, 0.001, False, False, 5e-4),
    "basket": ("motion", 7, 0.0006, 40000, 6000, True, 0.001, False, False, 5e-4),
    "buick": ("motion", 7, 0.0008, 38000, 9000, False, 0.001, False, False, 5e-4),
    "coffee": ("motion", 6, 0.0006, 12000, 6000, False, 0.001, False, False, 5e-4),
    "decoration": ("motion", 6, 0.0008, 49000, 9000, False, 0.001, False, False, 5e-4),
    "girl": ("motion", 7, 0.0008, 49000, 9000, True, 0.01, True, False, 5e-4),
    "heron": ("motion", 8, 0.0008, 49000, 9000, True, 0.01, True, False, 5e-4),
    "parterre": ("motion", 6, 0.0008, 35000, 9000, False, 0.001, False, False, 5e-4),
    "puppet": ("motion", 6, 0.0008, 28000, 9000, False, 0.001, False, False, 5e-4),
    "stair": ("motion", 6, 0.0008, 25000, 9000, False, 0.001, False, False, 5e-4),
    "cake": ("defocus", 9, 0.0008, 58000, 12000, True, 0.001, True, True, 5e-4),
    "caps": ("defocus", 5, 0.0006, 12000, 9000, True, 0.001, True, False, 5e-4),
    "cisco": ("defocus", 7, 0.0006, 49000, 6000, True, 0.001, False, False, 5e-4),
    "coral": ("defocus", 10, 0.0006, 22000, 18000, False, 0.001, False, True, 5e-4),
    "cupcake": ("defocus", 6, 0.0006, 30000, 10000, False, 0.001, False, False, 5e-4),
    "cups": ("defocus", 8, 0.0006, 22000, 9000, False, 0.001, False, False, 5e-4),
    "daisy": ("defocus", 8, 0.0002, 18000, 9000, True, 0.01, False, True, 5e-4),
    "sausage": ("defocus", 6, 0.0006, 12000, 9000, False, 0.001, False, False, 5e-4),
    "seal": ("defocus", 6, 0.0002, 10000, 7000, True, 0.01, True, False, 5e-4),
    "tools": ("defocus", 7, 0.0006, 40000, 6000, True, 0.001, True, False, 1e-3),
}
FIELDS = ("dataset", "llffhold", "init_dgt", "iterations", "ms_steps", "use_depth_loss",
          "depth_loss_alpha", "use_rgbtv_loss", "use_another_mlp", "bpn_lr")


def scene_config(scene):
    return dict(zip(FIELDS, OFFICIAL[scene]))


# Official arguments/__init__.py defaults (no per-scene script): the generic ``bags`` method.
GENERIC = {"init_dgt": 0.0006, "iterations": 60000, "ms_steps": 6000, "use_depth_loss": False,
           "depth_loss_alpha": 0.001, "use_rgbtv_loss": False, "use_another_mlp": False,
           "bpn_lr": 5e-4, "min_opacity": 0.005}

# Compressed schedule for a <= 2000-step smoke test that crosses every phase: resolution
# switches at 400 / 800, phase-2 restart at 1200 (ball-like flags). Not a paper configuration.
SMOKE = {"init_dgt": 0.0008, "iterations": 2000, "ms_steps": 1200, "use_depth_loss": True,
         "depth_loss_alpha": 0.01, "use_rgbtv_loss": True, "use_another_mlp": False,
         "bpn_lr": 5e-4, "upsample_steps": [400, 800]}
