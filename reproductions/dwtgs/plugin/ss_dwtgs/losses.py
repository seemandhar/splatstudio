"""DWTGS wavelet-space losses (paper §IV-B).

Eq. 5  L_LF = sum_{n=1}^{N} lambda_LL * L1(X^n_LL, X^n_LL_gt), N = 2, lambda_LL = 0.5, on the
       training render (the DropGaussian dropout render the main loss also uses).
Eq. 6  L_HF = L1(X^1_HH, 0) on a novel-view render (no ground truth; weight 1 — Eq. 6 has none),
       every 10 iterations, stopped after 5000 (§V-A).
L1 = mean absolute difference (3DGS l1_loss). Haar normalisation default "average" (LL = 2x2
block mean: §IV-B "LL subband is in the same space as the original image, only a coarser
version"; Figs. 2-3 show LL at image brightness); "orthonormal" is a labelled option only.
Novel views (default novel_views="interp"): spherical interpolation between a random pair of
training views (Fig. 2 note 3, see interp.py). "fsgs_llff" (FSGS generate_random_poses_llff,
10000-pose shuffled stack) is a labelled option only. Rendered with all Gaussians (official
render() default is_train=False), black background, clamped, at the runtime's SH degree.
"""

from types import SimpleNamespace

import numpy as np
import torch
from ss_dropgaussian.renderer import Renderer

from .bounds import SCENE_BOUNDS, official_bounds
from .interp import interpolate_viewmat
from .pseudo import depth_bounds, random_poses_llff
from .wavelet import DIVISORS, haar_dwt, ll_pyramid

LF_DEFAULTS = {"weight": 0.5, "levels": 2, "haar_norm": "average"}
HF_DEFAULTS = {
    "weight": 1.0,
    "interval": 10,
    "start": 1,
    "stop": 5000,
    "novel_views": "interp",
    "n_pseudo_poses": 10000,
    "pseudo_seed": 0,
    "pseudo_bounds": "official",
    "background": "black",
    "haar_norm": "average",
}


def _check(cfg, defaults, name):
    unknown = set(cfg) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {name} options: {sorted(unknown)}")
    return {**defaults, **cfg}


def chw(img):
    return img.permute(2, 0, 1)


def lf_loss(render, gt, weight, levels, norm="average"):
    """Eq. 5 on CHW images."""
    total = render.new_zeros(())
    for r, g in zip(ll_pyramid(render, levels, norm), ll_pyramid(gt, levels, norm)):
        total = total + weight * (r - g).abs().mean()
    return total


def hf_loss(render, norm="average"):
    """Eq. 6 on a CHW image: mean |X^1_HH|."""
    return haar_dwt(render, norm)[3].abs().mean()


def hf_active(step, interval, start, stop):
    return start <= step <= stop and step % interval == 0


def sh_degree_at(step):
    """The native trainer's SH schedule (runtime.train: min((step - 1) // 1000, 3))."""
    return min((int(step) - 1) // 1000, 3)


def create_lf(spec):
    cfg = _check(dict(spec.config), LF_DEFAULTS, "dwtgs-lf")
    weight, levels, norm = float(cfg["weight"]), int(cfg["levels"]), cfg["haar_norm"]
    if levels < 1 or norm not in DIVISORS:
        raise ValueError("levels must be >= 1 and haar_norm orthonormal|average")

    def loss(params, outputs, target):
        return lf_loss(chw(outputs["rgb"]), chw(target), weight, levels, norm)

    return loss


class InterpViews:
    """Fig. 2 note 3: slerp/lerp between a random pair of distinct training views, t ~ U(0, 1)."""

    def __init__(self, cfg):
        self.rng = np.random.default_rng(int(cfg["pseudo_seed"]))
        self.viewmats = None
        self.info = {}

    def build(self, cameras, initial_points):
        if len(cameras) < 2:
            raise ValueError("interpolated novel views need >= 2 training cameras")
        sizes = {(int(c["width"]), int(c["height"])) for c in cameras}
        if len(sizes) != 1:
            raise ValueError("interpolated novel views need training views of one size")
        self.cameras = cameras
        self.viewmats = [c["viewmat"].detach().cpu().double().numpy() for c in cameras]
        self.info = {"source": "interp", "n_train": len(cameras)}

    def next(self):
        i, j = (int(v) for v in self.rng.choice(len(self.viewmats), size=2, replace=False))
        t = float(self.rng.random())
        vm = interpolate_viewmat(self.viewmats[i], self.viewmats[j], t)
        a, b = self.cameras[i], self.cameras[j]
        K = (1 - t) * a["K"] + t * b["K"]
        self.last = {"pair": (i, j), "t": t}
        return {"viewmat": torch.tensor(vm, dtype=torch.float32, device=a["K"].device), "K": K,
                "width": int(a["width"]), "height": int(a["height"])}


class NovelViews:
    """FSGS LLFF pseudo poses; official pseudo_stack: pop a random pose, refill when empty."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg["pseudo_seed"]))
        self.order = []
        self.viewmats = None
        self.info = {}

    def build(self, cameras, initial_points):
        viewmats = [c["viewmat"].detach().cpu().double().numpy() for c in cameras]
        mode = self.cfg["pseudo_bounds"]
        if mode == "official":
            scene, bounds = official_bounds(SCENE_BOUNDS, viewmats)
        elif mode == "points":
            Ks = [c["K"].detach().cpu().double().numpy() for c in cameras]
            sizes = [(int(c["width"]), int(c["height"])) for c in cameras]
            scene, bounds = None, depth_bounds(initial_points, viewmats, Ks, sizes)
        else:
            raise ValueError("pseudo_bounds must be 'official' or 'points'")
        poses, info = random_poses_llff(viewmats, bounds, int(self.cfg["n_pseudo_poses"]),
                                        self.rng)
        cam0 = cameras[0]
        w, h = int(cam0["width"]), int(cam0["height"])
        K = cam0["K"].detach().clone()
        K[0, 2], K[1, 2] = w / 2.0, h / 2.0  # official PseudoCamera: FoV of view 0, centred
        self.viewmats = torch.tensor(poses, dtype=torch.float32, device=K.device)
        self.K, self.size = K, (w, h)
        self.info = {"scene": scene, "bounds": np.asarray(bounds).tolist(), "bounds_source": mode,
                     **info}

    def next(self):
        if not self.order:
            self.order = list(self.rng.permutation(len(self.viewmats)))
        i = int(self.order.pop())
        w, h = self.size
        return {"viewmat": self.viewmats[i], "K": self.K, "width": w, "height": h}


def novel_renderer(background):
    """DropGaussian renderer with gamma = 0: all Gaussians, unscaled opacity (is_train=False)."""
    return Renderer(SimpleNamespace(config={"gamma": 0.0, "background": background}))


def create_hf(spec):
    cfg = _check(dict(spec.config), HF_DEFAULTS, "dwtgs-hf")
    weight, norm = float(cfg["weight"]), cfg["haar_norm"]
    if norm not in DIVISORS:
        raise ValueError("haar_norm must be average or orthonormal")
    interval, start, stop = int(cfg["interval"]), int(cfg["start"]), int(cfg["stop"])
    if cfg["novel_views"] == "interp":
        views = InterpViews(cfg)
    elif cfg["novel_views"] == "fsgs_llff":
        views = NovelViews(cfg)
    else:
        raise ValueError("novel_views must be 'interp' or 'fsgs_llff'")
    renderer = novel_renderer(cfg["background"])
    stats = {"used": 0}

    def loss(params, outputs, target):
        step = int(outputs["step"])
        if not hf_active(step, interval, start, stop):
            return outputs["rgb"].sum() * 0.0
        if views.viewmats is None:
            owner = outputs.get("dwtgs")
            if owner is None or owner.training_cameras is None:
                raise RuntimeError("dwtgs-hf needs the dwtgs density (training cameras)")
            views.build(owner.training_cameras, owner.initial_points)
        camera = views.next()
        out = renderer.render(params, camera, sh_degree=sh_degree_at(step))
        stats["used"] += 1
        stats["last_camera"] = camera
        return weight * hf_loss(chw(out["rgb"]), norm)

    loss.stats = stats
    loss.views = views
    return loss
