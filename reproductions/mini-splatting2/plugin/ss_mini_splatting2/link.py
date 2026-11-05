"""Renderer <-> density link for one training run.

Visibility culling and the half-resolution warm-up are decided by the density (which owns the
per-view visibility masks and the schedule) but must act inside the renderer, before
rasterization. The runtime passes the same ParameterDict to both components, so the density
registers a `LinkTable` for that object and the renderer looks it up on every render. Parameter
dicts without a table (evaluation from a checkpoint, probes) render with stock gsplat.
"""

import weakref
from dataclasses import dataclass

import torch

_TABLES = {}


@dataclass
class LinkTable:
    viewmats: torch.Tensor | None = None  # [V,16] training viewmats (row-major, flattened)
    cull: torch.Tensor | None = None  # [N,V] bool, True = culled for that view
    half_res: bool = False  # warm-up: render training views at int(W/2) x int(H/2)

    def view_index(self, viewmat, tol=1e-5):
        """Index of the training camera with this viewmat, or None (not a training view)."""
        if self.viewmats is None:
            return None
        d = (self.viewmats - viewmat.reshape(1, -1)).abs().sum(dim=1)
        k = int(d.argmin())
        return k if float(d[k]) <= tol else None


def register(params, table):
    for key in [k for k, (ref, _) in _TABLES.items() if ref() is None]:
        del _TABLES[key]
    _TABLES[id(params)] = (weakref.ref(params), table)


def lookup(params):
    entry = _TABLES.get(id(params))
    if entry is None or entry[0]() is not params:
        return None
    return entry[1]
