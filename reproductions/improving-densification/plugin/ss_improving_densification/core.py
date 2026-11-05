"""Pure (CPU-testable) pieces of ImprovedGS, each mirroring a function of the official code
(github.com/XiaoBin2001/Improved-GS @ 20b5db3, paths relative to that repository)."""

import math
import random

import torch
import torch.nn.functional as F

# scene/methods/densification_methods.py FIND_EDGES_KERNEL (PIL ImageFilter.FIND_EDGES)
FIND_EDGES_KERNEL = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]])

# Paper Sec. 4.2 / configs/improvedgs.json: per-scene Gaussian budget N_max.
SCENE_BUDGETS = {
    "bicycle": 3_000_000, "garden": 3_000_000, "stump": 3_000_000,
    "flowers": 1_500_000, "treehill": 1_500_000, "drjohnson": 1_500_000, "truck": 1_500_000,
    "bonsai": 1_000_000, "counter": 1_000_000, "kitchen": 1_000_000, "room": 1_000_000,
    "playroom": 1_000_000, "train": 1_000_000,
}


def minmax01(x):
    """normalize_to_unit_range: nan/inf -> 0, then (x - min) / (max - min); zeros if flat."""
    x = torch.nan_to_num(x.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    if x.numel() == 0:
        return x
    lo, hi = x.amin(), x.amax()
    scale = hi - lo
    if float(scale) <= 0.0:
        return torch.zeros_like(x)
    return (x - lo) / scale


def edge_map(image):
    """compute_edge_map: HWC float image in [0,1] -> [H,W] edge weights in [0,1].

    8-bit grayscale approximation (ITU-R 601 integer weights, rounded), 3x3 Laplacian with zero
    padding, clamp to [0,255]/255, min-max normalisation (Sec. 3.2 "Laplacian operator")."""
    rgb = torch.round(image[..., :3].detach().float().permute(2, 0, 1)[None] * 255.0)
    gray = torch.round((299.0 * rgb[:, 0:1] + 587.0 * rgb[:, 1:2] + 114.0 * rgb[:, 2:3]) / 1000.0)
    kernel = FIND_EDGES_KERNEL.to(gray.device)[None, None]
    response = F.conv2d(gray, kernel, padding=1)
    return minmax01(torch.clamp(response, 0.0, 255.0) / 255.0)[0, 0]


def budget(step, n_max, densify_from=500, densify_until=15000, warmup_offset=500):
    """_compute_current_budget (paper Eq. 9, growth control): N_max * sqrt(progress)."""
    n_max = int(n_max)
    start, end = int(densify_from), int(densify_until) - int(warmup_offset)
    if end <= start:
        return n_max
    progress = min(max((step - start) / float(end - start), 0.0), 1.0)
    if progress >= 1.0:
        return n_max
    return max(int(progress**0.5 * n_max), 1)


def expon_lr(step, lr_init, lr_final, max_steps=30000):
    """utils/general_utils.get_expon_lr_func with lr_delay_steps = 0 (no delay)."""
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        return 0.0
    t = min(max(step / max_steps, 0.0), 1.0)
    return math.exp(math.log(lr_init) * (1 - t) + math.log(lr_final) * t)


def mu_update(step, total_steps=30000, start=20000, interval=5, second_start=24000,
              second_interval=20, enabled=True):
    """training_loop.should_run_parameter_update (paper Sec. 3.5 multi-view update)."""
    if step >= int(total_steps):
        return False
    if not enabled:
        return True
    start = 20000 if int(start) < 0 else int(start)
    interval = 5 if int(interval) <= 1 else int(interval)
    second_interval = max(int(second_interval), 1)
    second_start = int(second_start)
    if second_start <= start:
        second_start = start + 1
    return (
        step < start
        or (start <= step < second_start and step % interval == 0)
        or (step >= second_start and step % second_interval == 0)
    )


def quat_to_rotmat(q):
    """utils/general_utils.build_rotation (wxyz, normalised)."""
    q = F.normalize(q, dim=-1)
    r, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
            2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
            2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def long_axis_children(means, log_scales, quats, opacity_logits, split_distance=0.45,
                       opacity_reduction=0.6):
    """long_axis_split geometry (paper Sec. 3.3, Eq. 8) for k selected parents.

    Returns 2k children ordered [+offset children, -offset children] (official order):
    centres at +-3*d*s_max along the rotated longest axis (L0 = 3 sigma, d = 0.45 L0);
    long semi-axis (1-d) s_max (L_s = L0 - d); short axes s*sqrt(1-d^2) (Eq. 8);
    opacity = logit(sigmoid(o) * 0.6)."""
    d = float(split_distance)
    stds = log_scales.exp()
    max_values, max_idx = stds.max(dim=1, keepdim=True)
    axis_mask = torch.zeros_like(stds, dtype=torch.bool).scatter(1, max_idx, True)
    offsets = stds * axis_mask * 3.0 * d
    offsets = torch.cat([offsets, -offsets], dim=0)
    rot = quat_to_rotmat(quats).repeat(2, 1, 1)
    new_means = torch.bmm(rot, offsets[..., None])[..., 0] + means.repeat(2, 1)
    rate_w = max(1.0 - d, 1e-6)
    rate_h = math.sqrt(max(1.0 - d * d, 1e-6))
    new_scales = stds.scatter(1, max_idx, max_values * rate_w / rate_h).repeat(2, 1) * rate_h
    p = torch.sigmoid(opacity_logits) * float(opacity_reduction)
    new_opac = torch.log(p / (1 - p)).repeat(2)  # Inria inverse_sigmoid
    return new_means, torch.log(new_scales), new_opac


class EdgeViewPool:
    """_sample_improvedgs_views: views are popped from the end of the (shuffled) training list;
    the pool refills when it is empty at the start of a call, or inside the loop after its last
    element was taken. `sample_count` is fixed before the loop, so a call that starts with fewer
    than `count` views left returns only those (official behaviour)."""

    def __init__(self, n_views, count=10, seed=0):
        self.n = int(n_views)
        self.count = int(count)
        order = list(range(self.n))
        random.Random(seed).shuffle(order)  # Scene: random.shuffle(train_cameras)
        self.order = order
        self.pool = list(order)

    def all_views(self):
        return list(self.order)

    def sample(self):
        if self.count == -1 or self.count >= self.n:
            return list(self.order)
        if not self.pool:
            self.pool = list(self.order)
        out = []
        for _ in range(min(self.count, len(self.pool))):
            out.append(self.pool.pop())
            if not self.pool and len(out) < self.count:
                self.pool = list(self.order)
        return out
