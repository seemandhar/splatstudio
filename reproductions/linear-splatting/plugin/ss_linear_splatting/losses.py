"""3DLS objective (supp. Sec. A, Eq. 1-4): 0.6 L1 + 0.2 L2 + 0.2 (1 - SSIM).

Replaces the base `photometric` loss (0.8 L1 + 0.2 D-SSIM). Each weighted term is its own loss
component so that training.jsonl logs the three terms separately (the runtime sums losses).
"""

import torch.nn.functional as F
from splatstudio.losses import ssim

TERMS = {
    "l1": lambda rgb, target: F.l1_loss(rgb, target),        # supp. Eq. 2
    "l2": lambda rgb, target: F.mse_loss(rgb, target),       # supp. Eq. 3
    "ssim": lambda rgb, target: 1.0 - ssim(rgb, target),     # supp. Eq. 4
}
PAPER_WEIGHTS = {"l1": 0.6, "l2": 0.2, "ssim": 0.2}          # supp. Eq. 1, alpha/beta/gamma


def combined_loss(rgb, target, weights=None):
    """Independent single-expression form of supp. Eq. 1 (used by probes/tests)."""
    w = PAPER_WEIGHTS if weights is None else weights
    return (w["l1"] * (rgb - target).abs().mean() + w["l2"] * (rgb - target).square().mean()
            + w["ssim"] * (1.0 - ssim(rgb, target)))


def create_loss(spec):
    cfg = dict(spec.config)
    unknown = set(cfg) - {"term", "weight"}
    if unknown:
        raise ValueError(f"Unknown 3DLS loss options: {sorted(unknown)}")
    term = cfg.get("term")
    if term not in TERMS:
        raise ValueError(f"term must be one of {sorted(TERMS)}")
    weight = float(cfg.get("weight", PAPER_WEIGHTS[term]))
    if weight < 0:
        raise ValueError("loss weight must be non-negative")
    fn = TERMS[term]

    def loss(params, outputs, target):
        return weight * fn(outputs["rgb"], target)

    return loss
