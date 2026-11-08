import math

import torch

from ss_compgs import codec, method, probes, schedule, smoke
from ss_compgs.density import DEFAULTS


def test_paper_defaults():
    d = next(c for c in method().components if c.kind == "density").config
    assert d["codebooks"] == {"dc": 4096, "sh": 4096, "scale": 32768, "rot": 32768}
    assert (d["vq_start"], d["total_steps"], d["assign_freq"], d["kmeans_iters"]) == (20000, 30000, 100, 5)
    assert (d["prune_start"], d["prune_stop"], d["prune_every"], d["prune_opacity"]) == (15000, 20000, 1000, 0.005)
    reg = next(c for c in method().components if c.name == "compgs-opacity-reg").config
    assert reg == {"lambda_reg": 1e-7, "start": 15000, "stop": 20000}


def test_schedule_probe_cpu():
    out = probes.schedule(method(), "cpu")
    assert out["assignments"] == 51


def test_smoke_schedule():
    cfg = next(c for c in smoke().components if c.kind == "density").config
    assigns, updates, prunes = probes.schedule_check(cfg)
    assert assigns == [1701, 1801, 1901] and prunes == [1600, 1700] and updates == 297
    assert cfg["base"]["refine_stop_iter"] == 1500 and cfg["base"]["grow_grad2d"] == 0.0002


def test_kmeans_and_reg_cpu():
    probes.kmeans_official(method(), "cpu")
    probes.opacity_reg(method(), "cpu")


def test_run_length():
    assert math.isclose(schedule.inferred_run_length(1.0, 0.01 ** (1 / 30000)), 30000)
    assert schedule.inferred_run_length(1.0, 1.0) is None


def test_paper_size_garden_order_of_magnitude():
    # paper supp.: garden 1,370,624 Gaussians -> 30 MB
    cbs = {"sh0": (4096, 3), "shN": (4096, 45), "scales": (32768, 3), "quats": (32768, 4)}
    mb = codec.paper_size(1370624, cbs)["MB"]
    assert 29 < mb < 32, mb


def test_quantizer_keep():
    from ss_compgs.vq import KMeansQuantizer

    q = KMeansQuantizer(4, 1)
    x = torch.randn(20, 3)
    q.assign(x)
    keep = torch.ones(20, dtype=torch.bool)
    keep[:5] = False
    idx = q.idx.clone()
    q.keep(keep)
    assert torch.equal(q.idx, idx[5:])
    assert set(DEFAULTS) >= {"seed", "chunk"}
