import math
from types import SimpleNamespace

import torch

from ss_hard_gs import method, method_scaffold, model
from ss_hard_gs.density import DEFAULTS, HGSDensity, merge, push_topk, rehgs_hard, ssim_map
from ss_hard_gs.model import expon_lr
from ss_hard_gs.probes import neural_gaussians, visibility_filter
from ss_hard_gs.renderer import blob_floats, pack_blob, unpack_blob


def test_spec_values():
    m = method()
    d = next(c for c in m.components if c.kind == "density")
    assert d.config["voxel_size"] == 0.001
    assert d.config["pghgs"] == {"enabled": True, "k": 3, "lambda": 1.0}
    assert d.config["rehgs"] == {"enabled": True, "tau_large": 2e-4, "tau_ssim": 0.7,
                                 "min_views": 2}
    base = method_scaffold()
    d0 = next(c for c in base.components if c.kind == "density")
    assert not d0.config["pghgs"]["enabled"] and not d0.config["rehgs"]["enabled"]


def test_merge_rejects_unknown():
    try:
        merge(DEFAULTS, {"pghgs": {"kk": 1}})
    except ValueError:
        return
    raise AssertionError


def test_topk():
    buf = torch.zeros(3, 3)
    push_topk(buf, torch.tensor([0, 0, 1]), torch.tensor([1.0, 2.0, 3.0]))
    push_topk(buf, torch.tensor([0]), torch.tensor([0.5]))

    # duplicates inside one call: only one survives per row (one view = one gradient per
    # Gaussian in training, so ids are unique there)
    buf = torch.zeros(2, 3)
    for v in (0.1, 0.4, 0.3, 0.2):
        push_topk(buf, torch.tensor([1]), torch.tensor([v]))
    assert torch.allclose(buf[1], torch.tensor([0.4, 0.3, 0.2]))
    assert buf[0].abs().sum() == 0


def test_rehgs_hard():
    m2d = torch.tensor([[1.2, 1.7], [1.2, 1.7], [30.0, 1.0]])
    am = torch.tensor([10, 1, 10], dtype=torch.int32)
    s = torch.ones(4, 4)
    s[1, 1] = 0.5
    ok = torch.ones(3, dtype=torch.bool)
    assert rehgs_hard(am, m2d, ok, s, 4, 4, 0.2, 0.7).tolist() == [True, False, False]


def test_ssim_map():
    x = torch.rand(16, 16, 3)
    assert torch.allclose(ssim_map(x, x), torch.ones(16, 16), atol=1e-4)


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
    d = HGSDensity(SimpleNamespace(config={"voxel_size": 0.001}))
    pts = torch.rand(2000, 3)
    params = torch.nn.ParameterDict({"means": torch.nn.Parameter(pts)})
    opts = {"means": torch.optim.Adam([params["means"]], lr=1.0)}
    d.initialize(params, opts, 2.0)
    n = len(params["means"])
    assert n <= 2000 and params["hgs_offset"].shape == (n, 10, 3)
    assert opts["means"].param_groups[0]["lr"] == 0.0
    assert math.isclose(opts["hgs_offset"].param_groups[0]["lr"], 0.02, rel_tol=1e-3)
    assert all(p.requires_grad and p.is_leaf for p in params.values())
    model.ACTIVE["bank"] = None
