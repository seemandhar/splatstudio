"""SpotLessSplats losses (arXiv 2406.20055; numerics of the official
``examples/spotless_trainer.py``, commit 0caae3c).

``sls-robust``  masked L1 (Eq. 1) with the RobustNeRF-style mask (Eq. 2), the residual-quantile
                histogram (§4.2.2), the SD-feature MLP mask (Eq. 5) or the agglomerative-cluster
                vote (Eq. 3-4), and the Bernoulli warm-up (Eq. 9, supp. B). Replaces 3DGS's
                L1 + D-SSIM (official ``ssim_lambda = 0``).
``sls-mlp``     hinge supervision of the MLP between the two residual masks (Eq. 6-8). It reads
                the step-local values ``sls-robust`` stores in ``outputs["sls"]`` and must
                therefore follow it in the component list.

Step convention: the runtime's 1-based ``outputs["step"]`` k is the official 0-based t = k - 1.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .context import Shared

ROBUST_DEFAULTS = {
    "mask_mode": "mlp",          # mlp (SLS-mlp) | agg (SLS-agg) | filter (RobustFilter)
    "robust_percentile": 0.7,    # code robust_percentile
    "lower_bound": 0.5,          # Eq. 8 tau = .5 (code lower_bound)
    "upper_bound": 0.9,          # Eq. 8 tau = .9 (code upper_bound)
    "bins": 10000,               # code bin_size (histogram on [0, 1])
    "hist_decay": 0.95,          # code update_running_stats
    "stats_stop": 15000,         # histogram only updated while t < refine_stop_iter
    "schedule": True,            # Eq. 9 Bernoulli warm-up
    "schedule_beta": -3e-3,      # code schedule_beta (paper supp. B beta_1, see paper.md §7)
    "schedule_div": 1.5,         # supp. B beta_2
    "mlp_hidden": 16,            # utils.SpotLessModule
    "mlp_lr": 1e-3,              # paper §5 / code
    "pe_freqs": 20,              # code get_positional_encodings(H, W, 20)
    "feature_dim": 1280,         # SD 2.1 up-block-1 channels
}
MLP_DEFAULTS = {"reg_weight": 0.5}


# ----------------------------------------------------------------------------- pure functions
def robust_mask(error, threshold):
    """``Runner.robust_mask``: error [H,W,3] -> inlier mask [H,W] (float 0/1).

    inlier = mean_c(error) < threshold; neighbours = 3x3 box mean of inliers (zero padding) > 0.5;
    mask = (neighbours + inlier) > 1e-3, i.e. inlier OR majority-inlier neighbourhood (Eq. 2).
    """
    e = error.mean(dim=-1)[None, None]
    inlier = (e < threshold).float()
    window = torch.ones((1, 1, 3, 3), dtype=e.dtype, device=e.device) / 9.0
    neighbours = (F.conv2d(inlier, window, padding=1) > 0.5).float()
    return ((neighbours + inlier) > 1e-3).float()[0, 0]


def histogram_threshold(hist, fraction):
    """``update_running_stats``: left bin edge of the first bin whose cumulative count reaches
    ``fraction`` of the total (``linspace(0, 1, bins + 1)[k]``)."""
    bins = hist.numel()
    target = hist.sum() * fraction
    k = torch.nonzero(torch.cumsum(hist, 0) >= target)[0, 0]
    return k.to(hist.dtype) / bins  # == torch.linspace(0, 1, bins + 1)[k]


def schedule_alpha(t, beta, div):
    """Supp. B: alpha = exp(-beta_1 floor((t + 1) / beta_2)) (code: beta = -beta_1)."""
    return math.exp(beta * math.floor((1 + t) / div))


def positional_encoding(height, width, num_frequencies, device):
    """``utils.get_positional_encodings`` -> [80, H, W] (y sin, y cos, x sin, x cos)."""
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device),
                          indexing="ij")
    y = y / (height - 1)
    x = x / (width - 1)
    freqs = torch.pow(2, torch.arange(num_frequencies, device=device)).float() * torch.pi
    ye = torch.cat([torch.sin(freqs * y[..., None]), torch.cos(freqs * y[..., None])], dim=-1)
    xe = torch.cat([torch.sin(freqs * x[..., None]), torch.cos(freqs * x[..., None])], dim=-1)
    return torch.cat([ye, xe], dim=-1).permute(2, 0, 1)


def cluster_vote(inlier, onehot):
    """Eq. 3-4 (``robust_cluster_mask``): inlier [H,W], onehot [K,H,W] -> mask [H,W]."""
    size = onehot.sum(dim=(-1, -2), keepdim=True)
    inl = (inlier[None] * onehot).sum(dim=(-1, -2), keepdim=True)
    keep = ((inl / size) > 0.5).float()
    return (onehot * keep).sum(dim=0)


class SpotLessMLP(nn.Module):
    """``utils.SpotLessModule``: Linear(C+80, 16) - ReLU - Linear(16, 1) - Sigmoid, applied per
    pixel to [bilinear-upsampled SD features ; positional encoding]."""

    def __init__(self, num_features, hidden=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(num_features, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1), nn.Sigmoid())

    def regularizer(self):
        """Code: max|W1| * max|W2| on ``.data`` (no gradient)."""
        return torch.max(self.mlp[0].weight.detach().abs()) * torch.max(
            self.mlp[2].weight.detach().abs())

    def forward_naive(self, feat, pe):
        """Official order: upsample all C channels to H x W, concatenate PE, MLP per pixel."""
        H, W = pe.shape[-2:]
        up = F.interpolate(feat[None], size=(H, W), mode="bilinear")[0]
        x = torch.cat([up, pe], dim=0).reshape(up.shape[0] + pe.shape[0], -1).T
        return self.mlp(x).reshape(H, W)

    def forward(self, feat, pe):
        """Same function, first Linear applied before the (linear, weight-sum-1) bilinear
        upsampling: W1 up(F) + b = up(W1 F) + b exactly, but 80x less memory."""
        H, W = pe.shape[-2:]
        lin1 = self.mlp[0]
        c = feat.shape[0]
        wf, wp = lin1.weight[:, :c], lin1.weight[:, c:]
        low = torch.einsum("oc,chw->ohw", wf, feat)
        hid = F.interpolate(low[None], size=(H, W), mode="bilinear")[0]
        hid = hid + torch.einsum("oc,chw->ohw", wp, pe) + lin1.bias[:, None, None]
        hid = torch.relu(hid)
        lin2 = self.mlp[2]
        out = torch.einsum("oc,chw->ohw", lin2.weight, hid) + lin2.bias[:, None, None]
        return torch.sigmoid(out[0])


# ----------------------------------------------------------------------------- sls-robust
class RobustLoss:
    def __init__(self, spec):
        cfg = {**ROBUST_DEFAULTS, **spec.config}
        unknown = set(cfg) - set(ROBUST_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown sls-robust options: {sorted(unknown)}")
        if cfg["mask_mode"] not in ("mlp", "agg", "filter"):
            raise ValueError("mask_mode must be mlp, agg or filter")
        self.cfg = cfg
        self.hist = None
        self.thresholds = None  # (avg_err, lower_err, upper_err), official initial (1, 0, 1)
        self.mlp = None
        self.opt = None
        self.pe_cache = {}

    def _ensure(self, device):
        if self.hist is None:
            self.hist = torch.zeros(int(self.cfg["bins"]), device=device)
            one = torch.ones((), device=device)
            self.thresholds = (one, torch.zeros((), device=device), one.clone())
        if self.cfg["mask_mode"] == "mlp" and self.mlp is None:
            n_in = int(self.cfg["feature_dim"]) + 4 * int(self.cfg["pe_freqs"])
            self.mlp = SpotLessMLP(n_in, int(self.cfg["mlp_hidden"])).to(device)
            self.opt = torch.optim.Adam(self.mlp.parameters(), lr=float(self.cfg["mlp_lr"]))

    def _pe(self, H, W, device):
        key = (H, W, str(device))
        if key not in self.pe_cache:
            self.pe_cache[key] = positional_encoding(H, W, int(self.cfg["pe_freqs"]), device)
        return self.pe_cache[key]

    @torch.no_grad()
    def update_histogram(self, err_pp):
        """hist <- decay * hist + histogram(err, bins, [0,1]); new quantile thresholds."""
        cfg = self.cfg
        new = torch.histc(err_pp.float(), bins=int(cfg["bins"]), min=0.0, max=1.0)
        self.hist = cfg["hist_decay"] * self.hist + new
        self.thresholds = tuple(histogram_threshold(self.hist, float(cfg[k])) for k in
                                ("robust_percentile", "lower_bound", "upper_bound"))

    def __call__(self, params, outputs, target):
        cfg = self.cfg
        rgb = outputs["rgb"]
        t = int(outputs["step"]) - 1
        device = rgb.device
        self._ensure(device)
        if self.opt is not None and self.mlp.mlp[0].weight.grad is not None:
            # MLP update of the previous step (official: right after spot_loss.backward()).
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        colors = rgb.clamp(0.0, 1.0)
        error = (colors - target).abs()                                   # [H,W,3]
        avg_err, lower_err, upper_err = self.thresholds
        with torch.no_grad():
            mask = robust_mask(error, avg_err)                            # [H,W]
        shared = {"step": int(outputs["step"])}
        H, W = error.shape[:2]
        if cfg["mask_mode"] == "agg":
            onehot = Shared.cluster_table().lookup(outputs["camera"]["viewmat"], device)
            onehot = F.interpolate(onehot[None], size=(H, W), mode="nearest")[0]
            with torch.no_grad():
                mask = cluster_vote(mask, onehot)
        elif cfg["mask_mode"] == "mlp":
            feat = Shared.feature_table().lookup(outputs["camera"]["viewmat"], device)
            if feat.shape[0] != int(cfg["feature_dim"]):
                raise ValueError(f"SD features have {feat.shape[0]} channels, expected "
                                 f"{cfg['feature_dim']}")
            pred = self.mlp(feat, self._pe(H, W, device))                 # [H,W], graph -> MLP
            with torch.no_grad():
                shared["lower"] = robust_mask(error, lower_err)
                shared["upper"] = robust_mask(error, upper_err)
            shared["pred"] = pred
            shared["reg"] = self.mlp.regularizer()
            mask = pred.detach()
        if cfg["schedule"]:
            alpha = schedule_alpha(t, float(cfg["schedule_beta"]), float(cfg["schedule_div"]))
            mask = torch.bernoulli(torch.clip(alpha + (1 - alpha) * mask, 0.0, 1.0))
        outputs["sls"] = shared
        loss = (mask.detach()[..., None] * error).mean()
        if t < int(cfg["stats_stop"]):
            self.update_histogram(error.detach().mean(dim=-1))
        return loss


def create_robust(spec):
    return RobustLoss(spec)


# ----------------------------------------------------------------------------- sls-mlp
def hinge(pred, lower, upper):
    """Code ``spotless_loss(p, minimum=upper, maximum=lower)``:
    mean(relu(p - upper) + relu(lower - p)) (Eq. 7)."""
    return torch.mean(torch.relu(pred - upper) + torch.relu(lower - pred))


def create_mlp_loss(spec):
    cfg = {**MLP_DEFAULTS, **spec.config}
    unknown = set(cfg) - set(MLP_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown sls-mlp options: {sorted(unknown)}")
    weight = float(cfg["reg_weight"])

    def loss(params, outputs, target):
        shared = outputs.get("sls")
        if shared is None or shared.get("step") != int(outputs["step"]) or "pred" not in shared:
            raise RuntimeError("sls-mlp must follow sls-robust (mask_mode mlp) in the components")
        return hinge(shared["pred"], shared["lower"], shared["upper"]) + weight * shared["reg"]

    return loss
