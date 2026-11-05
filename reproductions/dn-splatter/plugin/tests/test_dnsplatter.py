import math

import torch

import ss_dnsplatter as m
from ss_dnsplatter import probes
from ss_dnsplatter.density import v1_radius
from ss_dnsplatter.renderer import gaussian_normals


def test_loss_probes_cpu():
    meth = m.method()
    assert probes.depth_loss(meth, "cpu")["effective_weight"] == 1.2
    probes.normal_loss(meth, "cpu")
    probes.scale_loss(meth, "cpu")
    probes.sidecar_init(meth, "cpu")


def test_v1_radius_matches_gsplat10_formula():
    cov = torch.tensor([[4.0, 1.0], [1.0, 2.0]])
    inv = torch.linalg.inv(cov)
    r = v1_radius(torch.tensor([[inv[0, 0], inv[0, 1], inv[1, 1]]]))
    b = 0.5 * (4 + 2)
    v1 = b + math.sqrt(max(0.1, b * b - (8 - 1)))
    assert r.item() == math.ceil(3 * math.sqrt(v1))


def test_gaussian_normals_shortest_axis_and_flip():
    params = {
        "quats": torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
        "scales": torch.log(torch.tensor([[1.0, 1.0, 0.1], [0.1, 1.0, 1.0]])),
        "means": torch.tensor([[0.0, 0, 2], [0.0, 0, -2]]),
    }
    n = gaussian_normals(params, torch.eye(4))  # camera at the origin
    torch.testing.assert_close(n, torch.tensor([[0.0, 0, -1], [1.0, 0, 0]]))


def test_specs():
    for meth in (m.method(), m.method_rgb()):
        names = [c.name for c in meth.components]
        assert names[:3] == ["dn-render", "splatfacto-density", "photometric"]
    assert {"dn-depth", "dn-normal", "dn-scale"} <= {c.name for c in m.method().components}


def test_splatfacto_control_spec():
    sf = m.method_splatfacto()
    comps = {c.name: c for c in sf.components}
    assert set(comps) == {"dn-render", "splatfacto-density", "photometric"}
    assert comps["splatfacto-density"].config["sidecar_init"] == "isotropic"
    assert comps["dn-render"].config["num_downscales"] == 2
    assert comps["dn-render"].config["normal_pass"] is False
    assert m.method_rgb().components[1].config["sidecar_init"] == "oriented"
    probes.isotropic_init(sf, "cpu")


def test_downscale_factor_and_resize():
    from ss_dnsplatter.losses import resize_image
    from ss_dnsplatter.renderer import downscale_factor

    assert [downscale_factor(k, 2, 3000) for k in (0, 2999, 3000, 5999, 6000)] == [4, 4, 2, 2, 1]
    img = torch.arange(2 * 8 * 9 * 3, dtype=torch.float32).reshape(2 * 8, 9, 3)
    out = resize_image(img, 4)
    assert out.shape == (4, 2, 3)
    torch.testing.assert_close(out[0, 0], img[:4, :4].mean(dim=(0, 1)))
