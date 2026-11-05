import math

import pytest
import torch

from ss_mvgs import geometry as G
from ss_mvgs import method, method_outdoor
from ss_mvgs.density import MVGSDensity
from ss_mvgs.losses import create_photometric, ssim_same
from ss_mvgs.probes import _official_window, look_at, mvad


def _density_spec(m):
    return next(c for c in m.components if c.kind == "density")


def test_presets():
    indoor, outdoor = method(), method_outdoor()
    assert G.total_steps(G.check_stages(_density_spec(indoor).config["stages"])) == 39000
    assert G.total_steps(G.check_stages(_density_spec(outdoor).config["stages"])) == 30000
    assert [s[0] for s in _density_spec(outdoor).config["stages"]] == [3]


def test_stage_at():
    st = G.check_stages([[48, 8, 3000], [24, 4, 3000], [12, 2, 3000], [8, -1, 30000]])
    assert G.stage_at(st, 3000) == (0, 3000)
    assert G.stage_at(st, 3001) == (1, 1)
    assert G.stage_at(st, 39000) == (3, 30000)
    with pytest.raises(RuntimeError):
        G.stage_at(st, 39001)


def test_stage_size_official():
    assert G.stage_size(1600, 1066, 8, 3115 / 1600)[0] == round(3115 / 8)
    assert G.stage_size(1600, 1066, 2, 3115 / 1600)[0] == round(3115 / 2)
    assert G.stage_size(1600, 1066, -1, 3115 / 1600) == (1600, 1066)


def test_scaled_camera():
    cam = {"K": torch.tensor([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]]), "width": 100,
           "height": 80, "viewmat": torch.eye(4)}
    s = G.scaled_camera(cam, 25, 20)
    assert torch.allclose(s["K"], torch.tensor([[25.0, 0, 12.5], [0, 25.0, 10], [0, 0, 1]]))


@pytest.mark.parametrize("hw", [(100, 200), (107, 173), (41, 81), (20, 40)])
def test_window_matches_official(hw):
    g = torch.Generator().manual_seed(hw[0])
    m = torch.rand(3, *hw, generator=g)
    ref = _official_window(m)
    got = G.max_loss_window(m.permute(1, 2, 0))
    assert got[1] == ref[1] and abs(got[0] - ref[0]) < 1e-5


def test_mvad_probe_cpu():
    mvad(method(), "cpu")


def test_mvad_threshold():
    c = torch.tensor([[1.0, 0, 0], [math.cos(1.2), math.sin(1.2), 0]])  # 68.8 deg
    assert G.mvad_threshold(c, 2e-4) == 1e-4
    c = torch.tensor([[1.0, 0, 0], [math.cos(0.9), math.sin(0.9), 0]])  # 51.6 deg
    assert G.mvad_threshold(c, 2e-4) == 2e-4
    assert G.mvad_threshold(c[:1], 2e-4) == 2e-4


def test_intersect_and_parallel():
    q = torch.tensor([1.0, 2.0, 3.0])
    o1, o2 = torch.zeros(3), torch.tensor([5.0, -1.0, 0.0])
    p, ok = G.intersect_lines(o1, q - o1, o2, q - o2)
    assert ok and torch.allclose(p, q, atol=1e-5)
    _, ok = G.intersect_lines(o1, q, o2, q)
    assert not ok


def test_sort_stable_desc():
    assert G.sort_views([0.1, 0.5, 0.5, 0.2]) == [1, 2, 3, 0]


def test_box_index_quirk():
    cams = [look_at((x, 0.0, 0.0), (0, 0, 5), "cpu", 128, 60, 50.0) for x in (-1.0, 0.0, 1.0)]
    boxes = [[0, 0, 20, 40], [20, 40, 40, 80], [0, 80, 20, 120]]
    official = G.cross_ray_boxes(cams, boxes, [0.3, 0.2, 0.1], "official")
    paper = G.cross_ray_boxes(cams, boxes, [0.3, 0.2, 0.1], "paper")
    assert official.shape == paper.shape == (3, 6)
    # pair (0, 1): official uses boxes[0] for the second camera, paper boxes[1]
    assert not torch.allclose(official[0], paper[0])
    # pair (1, 2): j = 0 -> official boxes[0] vs paper boxes[2]
    assert not torch.allclose(official[2], paper[2])


def test_in_boxes_strict():
    means = torch.tensor([[0.5, 0.5, 0.5], [1.0, 0.5, 0.5], [2.0, 2.0, 2.0]])
    boxes = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    assert G.in_boxes(means, boxes).tolist() == [True, False, False]


def test_loss_sum_cpu():
    spec = next(c for c in method().components if c.kind == "loss")
    loss = create_photometric(spec)
    a = torch.rand(16, 16, 3, requires_grad=True)
    t1, t2 = torch.rand(16, 16, 3), torch.rand(16, 16, 3)
    one = loss({}, {"rgb": a}, t1)
    two = loss({}, {"mv_views": [{"rgb": a, "target": t1}, {"rgb": a, "target": t2}]}, None)
    three = loss({}, {"rgb": a}, t2)
    assert torch.allclose(two, one + three)


def test_density_schedule():
    d = MVGSDensity(_density_spec(method()))
    assert not d.is_densify(500) and d.is_densify(600) and not d.is_densify(15000)
    assert d.is_reset(500) and d.is_reset(3000) and not d.is_reset(15000)
    with pytest.raises(ValueError):
        MVGSDensity(type(_density_spec(method()))(**{**_density_spec(method()).__dict__,
                                                    "config": {"bogus": 1}}))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_memory_path_matches_per_view_graphs():
    """Shared activations + checkpointed non-statistics views == the original M independent
    stored per-view graphs: same loss, parameter gradients and last-view statistics."""
    from splatstudio.state import sample_scene

    from ss_mvgs.probes import _parts, _targets, ring
    from ss_mvgs.renderer import render_view

    device = "cuda"
    renderer, density, loss = _parts(method(), stages=[[4, -1, 10]], full_width_ratio=1.0)
    params, opts = sample_scene(method(), device)
    cams = ring(6, device)
    density.initialize(params, opts)
    density.set_training_frames(cams, _targets(cams, device))
    batch = density.batch
    assert [v["stats"] for v in batch["views"]] == [False, False, False, True]

    out = renderer.render(params, cams[0], sh_degree=1)
    out["step"] = 1
    density.before_backward(params, opts, 1, out)
    value = loss(params, out, None)
    value.backward()
    grads = {k: p.grad.clone() for k, p in params.items()}
    density.after_backward(params, opts, 1, out)
    accum, count = density.grad_accum.clone(), density.count.clone()
    assert all("means2d" not in v.get("info", {}) for v in out["mv_views"][:-1])

    for p in params.values():
        p.grad = None
    ref_views = [render_view(params, v["camera"], 1) for v in batch["views"]]
    ref_info = ref_views[-1]["info"]
    ref_info["means2d"].retain_grad()
    ref = sum(
        0.8 * (r["rgb"] - v["target"]).abs().mean()
        + 0.2 * (1 - ssim_same(r["rgb"], v["target"]))
        for r, v in zip(ref_views, batch["views"])
    )
    ref.backward()
    assert torch.allclose(value, ref, rtol=1e-6, atol=0)
    for k, p in params.items():
        assert torch.allclose(grads[k], p.grad, rtol=1e-4, atol=1e-7 * float(p.grad.abs().max())), k
    density._reset_stats(params)
    density._accumulate(ref_info)
    assert torch.equal(count, density.count)
    assert torch.allclose(accum, density.grad_accum, rtol=1e-5, atol=1e-12)
