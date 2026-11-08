"""2DGS regularisers as used by HDGS stage 1 (official train.py):

    lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
    lambda_dist   = opt.lambda_dist   if iteration > 3000 else 0.0
    normal_error  = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
    loss = Lc + lambda_dist * rend_dist.mean() + lambda_normal * normal_error.mean()

(Eq. Ld / Ln; run_360.py: lambda_dist 100, lambda_normal 0.01). Stage 2 (train_texture.py)
computes them but optimises ``total_loss = loss`` only, so the term is zero once the renderer
runs textured. The photometric L_c is the runtime's ``photometric`` loss (0.8 L1 + 0.2 D-SSIM).
"""

GEOMETRY_DEFAULTS = {
    "lambda_dist": 100.0,
    "lambda_normal": 0.01,
    "dist_start": 3000,
    "normal_start": 7000,
}


def geometry_terms(out):
    normal_error = 1 - (out["rend_normal"] * out["surf_normal"]).sum(dim=0)
    return normal_error.mean(), out["rend_dist"].mean()


def create_geometry(spec):
    unknown = set(spec.config) - set(GEOMETRY_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown HDGS geometry options: {sorted(unknown)}")
    cfg = {**GEOMETRY_DEFAULTS, **spec.config}

    def loss(params, outputs, target):
        step = outputs["step"]
        if outputs.get("textured", False):
            return outputs["rgb"].sum() * 0.0  # stage 2: total_loss = loss (no regularisers)
        normal, dist = geometry_terms(outputs["hdgs"])
        w_n = cfg["lambda_normal"] if step > cfg["normal_start"] else 0.0
        w_d = cfg["lambda_dist"] if step > cfg["dist_start"] else 0.0
        return w_d * dist + w_n * normal

    return loss
