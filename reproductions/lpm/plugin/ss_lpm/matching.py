"""SuperPoint + LightGlue matching (official `lpm.py:LPM.initialize_*`, `utils.get_matched_points`).

LightGlue (Apache-2.0, https://github.com/cvg/LightGlue @5a650c4) is imported from
``LPM_LIGHTGLUE`` (default ~/models/lightglue/src). Weights are loaded by LightGlue through
torch.hub (`superpoint_v1.pth`, `superpoint_lightglue_v0-1_arxiv.pth`; the files live in
~/models/lightglue/weights and are linked into ~/.cache/torch/hub/checkpoints). SuperPoint's
weights carry Magic Leap's non-commercial licence.
"""

import os
import sys
from pathlib import Path

import torch

LIGHTGLUE_SRC = os.environ.get("LPM_LIGHTGLUE", str(Path.home() / "models/lightglue/src"))
WEIGHTS = ("superpoint_v1.pth", "superpoint_lightglue_v0-1_arxiv.pth")


def weights_available():
    hub = Path(torch.hub.get_dir()) / "checkpoints"
    return Path(LIGHTGLUE_SRC, "lightglue").is_dir() and all((hub / w).exists() for w in WEIGHTS)


def _import():
    if LIGHTGLUE_SRC not in sys.path:
        sys.path.insert(0, LIGHTGLUE_SRC)
    import lightglue  # noqa: F401  (raises with a clear message when absent)
    from lightglue import LightGlue, SuperPoint
    from lightglue.utils import rbd

    return LightGlue, SuperPoint, rbd


class Matcher:
    """Official settings: SuperPoint(max_num_keypoints=None) (default resize 1024),
    LightGlue(features='superpoint', depth_confidence=0.9, width_confidence=0.95)."""

    def __init__(self, device):
        if not weights_available():
            raise RuntimeError(
                f"LightGlue source/weights missing (LPM_LIGHTGLUE={LIGHTGLUE_SRC}, "
                f"{WEIGHTS} in {torch.hub.get_dir()}/checkpoints)"
            )
        LightGlue, SuperPoint, self._rbd = _import()
        self.device = device
        self.extractor = SuperPoint(max_num_keypoints=None).eval().to(device)
        self.matcher = LightGlue(
            features="superpoint", depth_confidence=0.9, width_confidence=0.95
        ).eval().to(device)

    @torch.no_grad()
    def features(self, image_hwc):
        """image: HWC float [0,1] (any device). Returns batched feature dict on CPU."""
        img = image_hwc.to(self.device, torch.float32).permute(2, 0, 1).contiguous()
        feats = self.extractor.extract(img)
        return {k: v.cpu() for k, v in feats.items()}

    @torch.no_grad()
    def match(self, feats0, feats1):
        """Matched keypoint pixel coordinates (points0 [M,2], points1 [M,2]) on CPU."""
        f0 = {k: v.to(self.device) for k, v in feats0.items()}
        f1 = {k: v.to(self.device) for k, v in feats1.items()}
        m01 = self.matcher({"image0": f0, "image1": f1})
        f0, f1, m01 = (self._rbd(x) for x in (f0, f1, m01))
        m = m01["matches"]
        return f0["keypoints"][m[..., 0]].cpu(), f1["keypoints"][m[..., 1]].cpu()
