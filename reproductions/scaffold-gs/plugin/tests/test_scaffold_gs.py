import math
from types import SimpleNamespace

import torch

from ss_scaffold_gs import method, method_db, method_tandt, model
from ss_scaffold_gs.density import DEFAULTS, ScaffoldDensity, merge
from ss_scaffold_gs.model import expon_lr
from ss_scaffold_gs.probes import neural_gaussians, visibility_filter
from ss_scaffold_gs.renderer import blob_floats, pack_blob, unpack_blob


def test_spec_values():
    for fn, v in ((method, 0.001), (method_tandt, 0.01), (method_db, 0.005)):
        m = fn()
        d = next(c for c in m.components if c.kind == "density")
        assert d.config["voxel_size"] == v
    m = method()
    r = next(c for c in m.components if c.kind == "renderer")
    assert r.config["opacity_cap"] == 0.999
    reg = next(c for c in m.components if c.name == "scaffold-scaling-reg")
    assert reg.config["weight"] == 0.01
    assert DEFAULTS["densify_grad_threshold"] == 0.0002 and DEFAULTS["min_opacity"] == 0.005
    assert (DEFAULTS["start_stat"], DEFAULTS["update_from"], DEFAULTS["update_until"]) == (500, 1500, 15000)


def test_no_hgs_options():
    for key in ("pghgs", "rehgs"):
        try:
            merge(DEFAULTS, {key: {"enabled": True}})
        except ValueError:
            continue
        raise AssertionError(key)


def test_lr_schedule():
    f = expon_lr(0.01, 0.0001, 30000)
    assert math.isclose(f(0), 0.01) and math.isclose(f(30000), 0.0001)
    assert math.isclose(f(15000), 0.001, rel_tol=1e-6)


def test_blob_roundtrip():
    mlps = model.scaffold_mlps(32, 10)
    n = blob_floats(32, 10) // 8 + 5
    blob = pack_blob(mlps, n, 32, 10, "cpu")
    out = unpack_blob(blob, 32, 10)
    x = torch.randn(4, 35)
    for k in model.RENDER_MLPS:
        assert torch.allclose(out[k](x), mlps[k](x))


def test_cpu_probes():
    neural_gaussians(None, "cpu")
    visibility_filter(None, "cpu")


def test_initialize_cpu():
    d = ScaffoldDensity(SimpleNamespace(config={"voxel_size": 0.001}))
    pts = torch.rand(2000, 3)
    params = torch.nn.ParameterDict({"means": torch.nn.Parameter(pts)})
    opts = {"means": torch.optim.Adam([params["means"]], lr=1.0)}
    d.initialize(params, opts, 2.0)
    n = len(params["means"])
    assert n <= 2000 and params["sgs_offset"].shape == (n, 10, 3)
    assert opts["means"].param_groups[0]["lr"] == 0.0
    assert math.isclose(opts["sgs_offset"].param_groups[0]["lr"], 0.02, rel_tol=1e-3)
    assert all(p.requires_grad and p.is_leaf for p in params.values())
    model.ACTIVE["bank"] = None
