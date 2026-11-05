"""gsplat renderer that renders the MVGS multi-view batch of a training step.

With gradients enabled and an active MVGS density controller, the renderer ignores the runtime's
single training camera and renders the M views the density controller sampled for this step (the
official shuffled-stack sampler, train.py:91-94) at the cross-intrinsic stage resolution. The last
view is returned as `rgb`/`info` (the official densification-statistics view, train.py:124-126);
all views go to `mv_views` for the summed loss (Eq. 4) and cross-ray densification.
Evaluation renders (`torch.no_grad()`) and renders without an active schedule are single-view at
the given camera's resolution.
"""

import torch
from gsplat import rasterization
from torch.utils.checkpoint import checkpoint

from .link import LINK

BATCH_CAPABILITY = "render.multiview_batch"
CAPABILITIES = ("render.rgb", "render.depth.expected_z", "gradient.screen.signed",
                BATCH_CAPABILITY)


ACT_KEYS = ("means", "quats", "scales", "opacities", "colors")


def activations(params):
    """View-independent rasterizer inputs. A multi-view step computes them once and shares them
    across its M views: the [N, 16, 3] SH concatenation alone is 192 B per Gaussian, and building
    it per view kept M copies alive in the autograd graph until the single backward (the indoor
    stage-1 OOM: 48 views x 2.5-3M Gaussians). Sharing changes only the order in which autograd
    sums the per-view gradients."""
    return {
        "means": params["means"],
        "quats": params["quats"],
        "scales": params["scales"].exp(),
        "opacities": params["opacities"].sigmoid(),
        "colors": torch.cat((params["sh0"], params["shN"]), dim=1),
    }


def _rasterize(camera, sh_degree, means, quats, scales, opacities, colors):
    render, alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=camera["viewmat"][None],
        Ks=camera["K"][None],
        width=int(camera["width"]),
        height=int(camera["height"]),
        sh_degree=sh_degree,
        packed=False,
        render_mode="RGB+ED",
        rasterize_mode="classic",
    )
    return render[0], alpha[0], info


def _outputs(render, alpha, info):
    return {
        "rgb": render[..., :3] + (1 - alpha),  # white background (--white_background)
        "depth": render[..., 3:4],
        "alpha": alpha,
        "info": info,
    }


def render_view(params, camera, sh_degree, act=None):
    act = activations(params) if act is None else act
    return _outputs(*_rasterize(camera, sh_degree, *(act[k] for k in ACT_KEYS)))


def render_view_checkpointed(act, camera, sh_degree):
    """`render_view` whose rasterizer graph is recomputed during backward instead of stored.

    Only the image outputs (render, alpha) leave the checkpoint; the per-Gaussian projection/SH
    tensors (~115 B per Gaussian per view in the stored graph) are rebuilt from the shared
    activations when autograd reaches this view, so one view's graph is alive at a time. The
    gsplat forward is deterministic, so the recomputation reproduces the stored values exactly
    and loss/gradients are unchanged. No `info` is returned: these views are not statistics
    views (their screen-space gradient is never read)."""

    def fn(*tensors):
        render, alpha, _ = _rasterize(camera, sh_degree, *tensors)
        return render, alpha

    render, alpha = checkpoint(fn, *(act[k] for k in ACT_KEYS), use_reentrant=False)
    out = _outputs(render, alpha, None)
    out["info"] = {"width": int(camera["width"]), "height": int(camera["height"])}
    return out


class Renderer:
    capabilities = set(CAPABILITIES)

    def __init__(self, spec):
        if spec.config:
            raise ValueError("mvgs-renderer accepts no configuration")

    def render(self, params, camera, sh_degree=3):
        batch = LINK.batch() if torch.is_grad_enabled() else None
        if batch is None:
            return render_view(params, camera, sh_degree)
        act = activations(params)
        views = []
        for view in batch["views"]:
            # Statistics views (official: the last one) keep their graph and `info` so that the
            # density controller can read means2d.grad; the others are checkpointed.
            if view.get("stats", True):
                out = render_view(params, view["camera"], sh_degree, act)
            else:
                out = render_view_checkpointed(act, view["camera"], sh_degree)
            views.append({**out, "target": view["target"], "camera": view["camera"],
                          "index": view["index"]})
        last = views[-1]
        return {
            "rgb": last["rgb"],
            "depth": last["depth"],
            "alpha": last["alpha"],
            "info": last["info"],
            "mv_views": views,
            "mv_step": batch["step"],
            "mv_stage": batch["stage"],
        }


def create_renderer(spec):
    return Renderer(spec)
