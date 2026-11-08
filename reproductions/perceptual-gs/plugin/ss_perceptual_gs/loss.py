"""C2 dual-branch rendering loss (Eqs. 8-10; official train.py:106-117).

While step <= bce_until (densify_until_iter = 15000):
    r   = render(colors = 1 - sigma(eps), background black)            (Eq. 8 branch)
    L_S = BCE(1 - r, I^S)                                              (Eq. 9, torch BCELoss)
    L   = (1 - lambda_S) (0.8 L1 + 0.2 D-SSIM) + lambda_S L_S           (Eq. 10)
afterwards the plain 3DGS photometric loss.
The sensitivity render shares geometry with the RGB branch (gradients reach means, scales,
quats and opacities as in the official code); it has its own screen-space means, so the
densification statistics come from the RGB branch only (scaled by 1 - lambda_S, as official).
"""

from gsplat import rasterization
from splatstudio.losses import create_photometric

from . import core, sidecar

DEFAULTS = {"lambda_s": 0.1, "bce_until": 15000, "ssim_weight": 0.2,
            "sidecar_root": sidecar.DEFAULT_SIDECAR_ROOT}


def render_sensitivity(params, camera, values=None):
    """[H,W] render of per-Gaussian `values` (default 1 - sigma(eps)) over black, gsplat classic."""
    if values is None:
        values = 1 - params["sensitivity"].sigmoid()
    render, _, _ = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=params["scales"].exp(),
        opacities=params["opacities"].sigmoid(),
        colors=values[:, None],
        viewmats=camera["viewmat"][None],
        Ks=camera["K"][None],
        width=int(camera["width"]),
        height=int(camera["height"]),
        sh_degree=None,
        packed=False,
        render_mode="RGB",
        rasterize_mode="classic",
    )
    return render[0, ..., 0]


def _config(spec):
    unknown = set(spec.config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown Perceptual-GS loss options: {sorted(unknown)}")
    return {**DEFAULTS, **spec.config}


class WeightedPhotometric:
    """(1 - lambda_S) x 3DGS photometric loss while the sensitivity branch is active (Eq. 10)."""

    def __init__(self, spec):
        self.cfg = _config(spec)
        self.photometric = create_photometric(
            type(spec)(spec.name, "loss", spec.factory, config={"ssim_weight": self.cfg["ssim_weight"]})
        )

    def __call__(self, params, outputs, target):
        photo = self.photometric(params, outputs, target)
        if outputs["step"] > self.cfg["bce_until"]:
            return photo
        return (1 - self.cfg["lambda_s"]) * photo


class SensitivityBCE:
    """lambda_S x BCE(1 - R[1 - sigma(eps)], I^S) (Eqs. 8-10); zero after bce_until."""

    def __init__(self, spec):
        self.cfg = _config(spec)
        self.table = None  # sidecar.Table; set by probes, else loaded lazily from --data

    def _table(self):
        if self.table is None:
            self.table = sidecar.table(sidecar_root=self.cfg["sidecar_root"])
        return self.table

    def __call__(self, params, outputs, target):
        if outputs["step"] > self.cfg["bce_until"]:
            return params["sensitivity"].sum() * 0.0  # inactive; keeps the scalar attached
        camera = outputs["camera"]
        gt = self._table().sens(camera["viewmat"])
        if tuple(gt.shape) != (int(camera["height"]), int(camera["width"])):
            raise ValueError("Sensitivity sidecar resolution differs from the training image")
        pred = core.predicted_sensitivity(render_sensitivity(params, camera))
        return self.cfg["lambda_s"] * core.bce(pred, gt)


def create_photometric_weighted(spec):
    return WeightedPhotometric(spec)


def create_bce(spec):
    return SensitivityBCE(spec)
