"""MiDaS DPT-Hybrid depth estimator exactly as official FSGS `utils/depth_utils.py`.

Official: `torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")` = `DPTDepthModel(path=None,
backbone="vitb_rn50_384", non_negative=True)` + `dpt_hybrid_384.pt` (MiDaS v3 release), eval mode,
parameters frozen. `estimate_depth(img)`: (img - 0.5) / 0.5, bicubic to 384x512, DPT, bicubic back
to H x W. mode='test' runs under no_grad (training views, at load); mode='train' keeps the
autograd graph so the pseudo-view loss back-propagates through DPT into the rendered image.

Local copies (no network, no pip): MiDaS repo `~/models/dpt/midas_repo`, weights
`~/models/dpt/dpt_hybrid_384.pt`, timm 0.6.12 in `~/models/dpt/pylib` (the env's timm 1.x lacks
the ViT-hybrid layout the checkpoint expects). Loaded lazily on first use.
"""

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HOME = Path.home()
DEFAULT_PATHS = {
    "dpt_repo": str(HOME / "models/dpt/midas_repo"),
    "dpt_weights": str(HOME / "models/dpt/dpt_hybrid_384.pt"),
    "dpt_pylib": str(HOME / "models/dpt/pylib"),
    "dpt_sha256_prefix": "501f0c75",
}
_MODELS = {}


class DPTUnavailable(RuntimeError):
    pass


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dpt(device, paths=None):
    paths = {**DEFAULT_PATHS, **(paths or {})}
    key = (str(device), paths["dpt_weights"])
    if key in _MODELS:
        return _MODELS[key]
    for k in ("dpt_repo", "dpt_weights", "dpt_pylib"):
        if not Path(paths[k]).exists():
            raise DPTUnavailable(f"{k} not found: {paths[k]}")
    if "timm" in sys.modules and not sys.modules["timm"].__version__.startswith("0.6."):
        raise DPTUnavailable(f"timm {sys.modules['timm'].__version__} already imported; MiDaS "
                             "DPT_Hybrid needs timm 0.6.12 (~/models/dpt/pylib)")
    for p in (paths["dpt_pylib"], paths["dpt_repo"]):
        if p not in sys.path:
            sys.path.insert(0, p)
    import timm

    if not timm.__version__.startswith("0.6."):
        raise DPTUnavailable(f"MiDaS DPT_Hybrid needs timm 0.6.12, got {timm.__version__}")
    from midas.dpt_depth import DPTDepthModel

    digest = _sha256(paths["dpt_weights"])
    if not digest.startswith(paths["dpt_sha256_prefix"]):
        raise DPTUnavailable(f"unexpected DPT weights sha256 {digest}")
    model = DPTDepthModel(path=None, backbone="vitb_rn50_384", non_negative=True)
    model.load_state_dict(torch.load(paths["dpt_weights"], map_location="cpu", weights_only=True))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    _MODELS[key] = model
    return model


def estimate_depth(model, img_chw, mode="test"):
    """Official estimate_depth: img [3,H,W] in [0,1] (not clamped) -> DPT output [H,W]."""
    h, w = img_chw.shape[1:3]
    norm = (img_chw[None] - 0.5) / 0.5
    norm = F.interpolate(norm, size=(384, 512), mode="bicubic", align_corners=False)

    def run():
        pred = model(norm)
        return F.interpolate(pred.unsqueeze(1), size=(h, w), mode="bicubic",
                             align_corners=False).squeeze()

    if mode == "test":
        with torch.no_grad():
            return run()
    return run()
