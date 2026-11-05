import math

import torch

import ss_agsmesh as m
from ss_agsmesh import probes
from ss_agsmesh.density import v1_radius
from ss_agsmesh.losses import find_edges


def test_loss_probes_cpu():
    meth = m.method()
    assert probes.depth_schedule(meth, "cpu")["switch_ns"] == 7000
    probes.normal_anr(meth, "cpu")
    probes.scale_loss(meth, "cpu")
    probes.surface_normals(meth, "cpu")
    probes.sidecar_init(meth, "cpu")


def test_dnc_probe():
    r = probes.dnc_filter(m.method(), "cpu")
    assert r["rejected_bad_depth"] > 0.9 and r["kept_consistent"] > 0.97


def test_find_edges_constant_maps():
    # positive constant: zero padding gives a negative border Laplacian -> no edges at all
    assert not find_edges(torch.full((3, 6, 7), 0.5)).any()
    # negative constant: border Laplacian positive -> dilated border edges, clean interior
    e = find_edges(torch.full((3, 6, 7), -0.5))
    assert e[:, 0].all() and e[:, 1].all() and not e[:, 2:-2, 2:-2].any()


def test_v1_radius_matches_gsplat10_formula():
    cov = torch.tensor([[4.0, 1.0], [1.0, 2.0]])
    inv = torch.linalg.inv(cov)
    r = v1_radius(torch.tensor([[inv[0, 0], inv[0, 1], inv[1, 1]]]))
    b = 0.5 * (4 + 2)
    v1 = b + math.sqrt(max(0.1, b * b - (8 - 1)))
    assert r.item() == math.ceil(3 * math.sqrt(v1))


def test_specs():
    names = [c.name for c in m.method().components]
    assert names == ["ags-render", "splatfacto-density", "photometric", "ags-depth",
                     "ags-normal", "ags-scale"]
    assert [c.name for c in m.method_rgb().components][:3] == names[:3]
