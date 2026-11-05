"""State shared by the SuGaR renderer, density controller and losses of one built method.

`splatstudio.components.build` instantiates components in declaration order; the renderer is
declared first and creates the context, later factories bind to it.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class Mesh:
    verts: torch.nn.Parameter  # [V,3] learnable vertex positions
    faces: torch.Tensor  # [F,3] long
    bary: torch.Tensor  # [n,3] barycentric coordinates of the n Gaussians of every face
    log_thickness: float  # log of the fixed normal-axis scale (camera extent / 1e6)
    pairs: torch.Tensor  # [P,4] face pairs sharing an edge, for normal consistency
    optimizer: Any = None


@dataclass
class SugarContext:
    renderer: Any = None
    mesh: Optional[Mesh] = None
    knn: Optional[torch.Tensor] = None  # [N,K] long, 16-NN of each centre (incl. itself)
    cameras: list = field(default_factory=list)
    extent: Optional[float] = None  # 3DGS camera extent (1.1 x max centre distance)
    center: Optional[torch.Tensor] = None  # mean training camera centre
    sh_cap: int = 3  # refinement SH warm-up cap (official refine.py: +1 level / 1000 its)
    cache: dict = field(default_factory=dict)  # per-step SDF samples shared by R and R_Norm


_CURRENT = []


def new_context():
    ctx = SugarContext()
    _CURRENT[:] = [ctx]
    return ctx


def current_context():
    if not _CURRENT:
        raise RuntimeError("The sugar renderer must be declared before density and losses")
    return _CURRENT[0]
