"""Per-scene arguments of the official scripts (train_base.sh / train_big.sh @ 44e02a5).

The native runtime does not tell components which scene they train, so the density looks the
scene up by its initial SfM point count (all points are kept with --num-gaussians 10000000;
counts copied from the SplatStudio Taming-3DGS port and re-checked against
~/datasets/mipnerf360/w1600/*/points.npz). Unknown counts -> official argument defaults.
Keys: grad_abs_thresh, dense, highfeature_lr, lowfeature_lr, loss_thresh, mult.
"""

SCENE_SFM_COUNTS = {
    "bicycle": 54275, "flowers": 38347, "garden": 138766, "stump": 32049, "treehill": 52363,
    "room": 112627, "counter": 155767, "kitchen": 241367, "bonsai": 206613,
    "truck": 136029, "train": 182686, "drjohnson": 80861, "playroom": 37005,
}

# arguments/__init__.py OptimizationParams defaults
ARG_DEFAULTS = {"grad_abs_thresh": 0.0012, "dense": 0.001, "highfeature_lr": 0.005,
                "lowfeature_lr": 0.0025, "loss_thresh": 0.1, "mult": 0.5}

BASE = {  # train_base.sh (FastGS, --densification_interval 500)
    "bicycle": {"grad_abs_thresh": 0.0012},
    "flowers": {"dense": 0.005, "grad_abs_thresh": 0.0015},
    "garden": {"highfeature_lr": 0.02, "loss_thresh": 0.06, "grad_abs_thresh": 0.0008},
    "stump": {"dense": 0.004, "grad_abs_thresh": 0.0015},
    "treehill": {"dense": 0.01, "grad_abs_thresh": 0.002},
    "room": {"highfeature_lr": 0.02, "grad_abs_thresh": 0.0008},
    "counter": {"highfeature_lr": 0.02, "grad_abs_thresh": 0.0008},
    "kitchen": {"highfeature_lr": 0.02, "grad_abs_thresh": 0.0006},
    "bonsai": {"highfeature_lr": 0.02, "grad_abs_thresh": 0.0006},
    "truck": {"highfeature_lr": 0.04, "grad_abs_thresh": 0.0009, "mult": 0.7},
    "train": {"highfeature_lr": 0.042, "grad_abs_thresh": 0.0015, "dense": 0.01, "mult": 0.7},
    "playroom": {"highfeature_lr": 0.0015, "dense": 0.003, "mult": 0.7},
    "drjohnson": {"highfeature_lr": 0.0025, "grad_abs_thresh": 0.0012, "dense": 0.013,
                  "mult": 0.7},
}

BIG = {  # train_big.sh (FastGS-Big, --densification_interval 100)
    "bicycle": {"grad_abs_thresh": 0.0008},
    "flowers": {"dense": 0.005, "grad_abs_thresh": 0.001},
    "garden": {"highfeature_lr": 0.02, "loss_thresh": 0.06, "grad_abs_thresh": 0.0003},
    "stump": {"dense": 0.004, "grad_abs_thresh": 0.001},
    "treehill": {"dense": 0.01, "grad_abs_thresh": 0.0018},
    "room": {"highfeature_lr": 0.02, "grad_abs_thresh": 0.0004},
    "counter": {"highfeature_lr": 0.02, "grad_abs_thresh": 0.0004},
    "kitchen": {"highfeature_lr": 0.02, "grad_abs_thresh": 0.0002},
    "bonsai": {"highfeature_lr": 0.02, "grad_abs_thresh": 0.0002},
    "truck": {"highfeature_lr": 0.04, "grad_abs_thresh": 0.0004, "mult": 0.7},
    "train": {"highfeature_lr": 0.042, "grad_abs_thresh": 0.0004, "dense": 0.015, "mult": 0.7},
    "playroom": {"highfeature_lr": 0.0015, "dense": 0.003, "mult": 0.7, "grad_abs_thresh": 0.0005},
    "drjohnson": {"highfeature_lr": 0.0025, "lowfeature_lr": 0.0005, "grad_abs_thresh": 0.0005,
                  "dense": 0.005, "mult": 0.7},
}

TABLES = {"base": BASE, "big": BIG}


def scene_for_count(count):
    for name, n in SCENE_SFM_COUNTS.items():
        if n == int(count):
            return name
    return None


def resolve(count, table):
    """(scene name or None, full argument dict) for an initial point count."""
    scene = scene_for_count(count)
    args = dict(ARG_DEFAULTS)
    if scene is not None:
        args.update(TABLES[table][scene])
    return scene, args
