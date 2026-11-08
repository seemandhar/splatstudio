import math

import torch

from ss_hdgs import fisher, method, method_indoor
from ss_hdgs import texture as tex
from ss_hdgs.density import expon_lr
from ss_hdgs.reference import sh_rgb, tex2d


def test_methods_declare_paper_values():
    for m, res in ((method(), 200), (method_indoor(), 500)):
        comps = {c.name: c for c in m.components}
        assert comps["hdgs-raster"].config["sort_window"] == 16
        assert comps["hdgs-raster"].config["tex_res"] == res
        d = comps["hdgs-density"].config
        assert d["tex_res"] == res and d["stage1_steps"] == 40000 and d["prune_ratio"] == 0.1
        assert d["densification_interval"] == 150 and d["densify_until_iter"] == 20000
        assert d["opacity_cull"] == 0.01
        g = comps["hdgs-geometry"].config
        assert g["lambda_dist"] == 100.0 and g["lambda_normal"] == 0.01


def test_texture_index_and_init():
    s = torch.tensor([[0.0101, 0.0199], [3.0, 0.5], [0.001, 0.001]])
    idx, total = tex.texture_index(s, 200, 500)
    assert idx.tolist() == [[0, 3, 4], [12, 500, 83], [41512, 1, 1]] and total == 41513
    sh0 = torch.arange(9.0).reshape(3, 1, 3)
    buf = tex.initial_texture(sh0, idx, total)
    assert buf.shape == (total, 3)
    assert torch.equal(buf[0], sh0[0, 0]) and torch.equal(buf[11], sh0[0, 0])
    assert torch.equal(buf[12], sh0[1, 0]) and torch.equal(buf[-1], sh0[2, 0])


def test_lr_schedules():
    assert math.isclose(expon_lr(15000, 1e-4, 1e-6, 30000), 1e-5, rel_tol=1e-9)
    assert expon_lr(45000, 1e-4, 1e-6, 30000) == expon_lr(30000, 1e-4, 1e-6, 30000)
    assert tex.tex_lr(8000, 8e-3, 8000, 0.1) == 8e-3
    assert math.isclose(tex.tex_lr(8001, 8e-3, 8000, 0.1), 8e-4)


def test_bilinear_matches_grid_sample_inside():
    torch.manual_seed(0)
    Wt, Ht = 5, 4
    buf = torch.rand(Wt * Ht, 3, dtype=torch.float64)
    img = buf.reshape(Ht, Wt, 3).permute(2, 0, 1)[None]
    for u, v in ((0.1, -0.3), (-0.7, 0.55), (0.33, 0.9)):
        ref = torch.nn.functional.grid_sample(img, torch.tensor([[[[u, v]]]], dtype=torch.float64),
                                              align_corners=True)[0, :, 0, 0]
        assert torch.allclose(tex2d(buf, 0, Wt, Ht, torch.tensor(u, dtype=torch.float64),
                                    torch.tensor(v, dtype=torch.float64)), ref)


def test_sh_degree0_is_dc():
    sh = torch.zeros(1, 16, 3)
    sh[0, 0] = torch.tensor([1.0, -1.0, 0.0])
    rgb = sh_rgb(sh, torch.tensor([[0.0, 0.0, 1.0]]), 0)
    assert torch.allclose(rgb, torch.clamp_min(0.28209479177387814 * sh[:, 0] + 0.5, 0))


def test_hutchinson_unbiased_on_linear_map():
    torch.manual_seed(0)
    J = torch.randn(200, 5, dtype=torch.float64)
    exact = (J * J).sum(0)
    est = torch.zeros(5, dtype=torch.float64)
    for _ in range(4000):
        r = torch.randint(0, 2, (200,)).double() * 2 - 1
        est += (r @ J) ** 2
    est /= 4000
    assert torch.allclose(est, exact, rtol=0.08)
    mask = fisher.prune_mask(torch.tensor([3.0, 1.0, 2.0, 5.0, 4.0, 0.0, 9, 8, 7, 6]), 0.2)
    assert mask.tolist() == [False, True, False, False, False, True, False, False, False, False]
