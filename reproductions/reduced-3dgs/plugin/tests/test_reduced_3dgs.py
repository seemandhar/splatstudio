"""CPU unit tests of the pure functions of the Reduced-3DGS port."""

import math

import torch

from ss_reduced_3dgs import codec, density, losses, method, probes, redundancy
from ss_reduced_3dgs.renderer import DEGREE_FIELD, coefficient_mask


def torch_assign(values, centers):
    return (values[:, None] - centers[None]).abs().argmin(1)


def test_spec_valid():
    from splatstudio.contracts import validate

    assert validate(method()) == []


def test_coefficient_mask_bands():
    deg = torch.tensor([0.0, 1.0, 2.0, 3.0])
    m = coefficient_mask(deg, 3)[..., 0]
    assert m.sum(1).tolist() == [0, 3, 8, 15]
    assert coefficient_mask(deg, 1)[..., 0].sum(1).tolist() == [0, 3, 3, 3]


def test_schedule_probe_passes():
    out = probes.schedule(method(), "cpu")
    assert out["cull_sh"] == 1 and out["encode"] == 1 and out["mercy"] == 23


def test_schedule_compressed_order():
    cfg = {**density.DEFAULTS, "total_steps": 2000, "densify_until": 1000, "cull_sh_step": 1000,
           "mercy_interval": 200, "mercy_finetune": 300, "save_steps": []}
    assert density.schedule_events(1000, cfg) == ["dead_prune", "mercy", "cull_sh"]
    assert density.schedule_events(2000, cfg) == ["dead_prune", "save_prune", "encode"]


def test_loss_gradients():
    probes.sh_sparsity_gradient(method(), "cpu")
    probes.opacity_l1_gradient(method(), "cpu")


def test_active_degree_matches_runtime():
    assert [losses.active_degree(s) for s in (1, 1000, 1001, 2001, 3001, 9000)] == [0, 0, 1, 2, 3, 3]


def test_quantile_matches_torch():
    x = torch.rand(1001)
    for q in (0.0, 0.03, 0.5, 1.0):
        assert torch.isclose(redundancy.quantile_linear(x, q), torch.quantile(x, q))


def test_mercy_mask_rule():
    score = torch.tensor([1, 1, 1, 2, 2, 10, 10, 10, 10])
    opacity = torch.tensor([0.5, 0.6, 0.01, 0.5, 0.5, 0.1, 0.2, 0.3, 0.4])
    mask, info = redundancy.mercy_mask(score, opacity)
    thr = max(float(score.float().mean() + score.float().std()), 3)
    assert math.isclose(info["used_threshold"], thr)
    # candidates: score 10 -> below their median opacity (0.2): only 0.1; plus opacity < min(q3%, .05)
    assert mask.tolist() == [False, False, True, False, False, True, False, False, False]


def test_footprint_landscape_and_frustum():
    cam = {"viewmat": torch.eye(4), "K": torch.tensor([[50.0, 0, 32], [0, 60.0, 24], [0, 0, 1]]),
           "width": 64, "height": 48}
    means = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0], [100.0, 0.0, 2.0], [0, 0, 150.0]])
    fp = redundancy.min_pixel_footprint(means, [cam])
    assert torch.isclose(fp[0], torch.tensor(2.0 / 50.0))
    assert fp[1:].tolist() == [10000.0, 10000.0, 10000.0]


def test_paper_memory_formula_bicycle_scale():
    # paper Table 7 bicycle: 2.41M Gaussians, 89/0.1/2.7/8.2 % bands -> ~48 MB
    p = 2_410_000
    counts = [int(p * f) for f in (0.89, 0.001, 0.027, 0.082)]
    mb = codec.paper_memory_bytes(counts) / codec.MIB
    assert 45 < mb < 52
    assert math.isclose(codec.baseline_memory_bytes(6_050_000) / codec.MIB, 1361.6, abs_tol=0.5)


def test_kmeans_fixed_point_and_empty_rule_cpu():
    v = torch.zeros(100)
    v[:10] = torch.arange(1, 11).float()
    g = torch.Generator().manual_seed(0)
    ids, centers = codec.kmeans_1d(v, 8, 1e-6, generator=g, assign=torch_assign)
    for c in range(8):
        members = v[ids == c]
        # used centre = mean of its members (Lloyd fixed point); unused centre = 0 (0/0 -> 0)
        assert torch.isclose(centers[c], members.mean() if len(members) else torch.tensor(0.0))
    assert torch.equal(ids, torch_assign(v, centers))


def test_encode_decode_roundtrip_cpu():
    n = 200
    params = probes._fake_params(n, "cpu", 3)
    books = codec.produce_codebooks(params, 16, 0, assign=torch_assign)
    data = codec.encode_ply(params, books)
    dec = codec.decode_ply(data, "cpu")
    deg = params[DEGREE_FIELD].detach().long()
    order = torch.cat([torch.where(deg == d)[0] for d in range(4)])
    assert torch.equal(dec["means"], params["means"].detach()[order].half().float())
    assert torch.equal(dec[DEGREE_FIELD].long(), deg[order])
    ids = books["scaling"][0][order].long()
    assert torch.equal(dec["scales"], books["scaling"][1].half().float()[ids])
    rid = books["features_rest_4"][0][order].long()  # coefficient index 4 belongs to band 2
    keep = deg[order] >= 2
    want = books["features_rest_4"][1].half().float()[rid]
    assert torch.equal(dec["shN"][keep, 4], want[keep])
    assert (dec["shN"][~keep, 4] == 0).all()


def test_encode_with_empty_band_group_cpu():
    params = probes._fake_params(50, "cpu", 4)
    params[DEGREE_FIELD].data = (params[DEGREE_FIELD].data >= 2).float() * 3  # only 0 and 3
    books = codec.produce_codebooks(params, 8, 0, assign=torch_assign)
    dec = codec.decode_ply(codec.encode_ply(params, books), "cpu")
    assert len(dec["means"]) == 50 and set(dec[DEGREE_FIELD].tolist()) <= {0.0, 3.0}


def test_variance_alias_quirk():
    out = probes.shcull_variance_alias(method(), "cpu")
    assert out["configured_quirks"] and out["aliased_var"] < out["west_var"]


def test_run_length_check():
    from splatstudio.specs import ComponentSpec

    spec = next(c for c in method().components if c.name == "reduced-density")
    d = density.ReducedDensity(ComponentSpec(spec.name, spec.kind, spec.factory, spec.requires,
                                             spec.provides, spec.config))
    d.lr0 = 1.0
    d.check_run_length(1, 0.01 ** (1 / 30000))  # matches total_steps: no error
    try:
        d.check_run_length(1, 0.01 ** (1 / 2000))
    except ValueError:
        pass
    else:
        raise AssertionError("steps != total_steps must raise")
