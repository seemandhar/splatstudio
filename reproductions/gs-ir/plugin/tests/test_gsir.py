import math

import torch

from ss_gsir import baking, cubemap, geometry, light, volumes
from ss_gsir.losses import create_color, create_light, create_material, create_normal


class Spec:
    def __init__(self, name, config=None):
        self.name = name
        self.config = config or {}


def test_cube_round_trip_and_faces():
    n = 8
    d = cubemap.texel_dirs(n, "cpu")
    face, s, t = cubemap.dir_to_face(d)
    assert torch.equal(face, torch.arange(6)[:, None, None].expand(6, n, n))
    xs = torch.arange(n).float()
    assert torch.allclose(s[0, 0], (xs + 0.5) / n)
    assert torch.allclose(t[0, :, 0], (xs + 0.5) / n)
    tex = torch.rand(6, n, n, 2)
    assert torch.allclose(cubemap.sample(tex, d), tex, atol=1e-5)
    assert torch.equal(cubemap.sample(tex, d, "nearest"), tex)


def test_face_rotations_match_cube_layout():
    n = 16
    rot = baking.face_rotations("cpu")
    ys, xs = torch.meshgrid(torch.arange(n).float(), torch.arange(n).float(), indexing="ij")
    ray = torch.stack(((xs - n / 2 + 0.5) / (n / 2), (ys - n / 2 + 0.5) / (n / 2),
                       torch.ones_like(xs)), -1)
    d = cubemap.texel_dirs(n, "cpu")
    for f in range(6):
        w = torch.nn.functional.normalize(ray @ rot[f].T, dim=-1)
        assert torch.allclose(w, d[f], atol=1e-6)


def test_face_cameras_are_rigid():
    pos = torch.tensor([[0.5, -1.0, 2.0]])
    view, K = baking.face_cameras(pos, 64)
    c2w = torch.linalg.inv(view[0])
    assert torch.allclose(c2w[:, :3, 3], pos.expand(6, 3), atol=1e-6)
    assert torch.allclose(K[0, 0], torch.tensor([[32.0, 0, 32], [0, 32, 32], [0, 0, 1]]))


def test_latlong_solid_angles_sum_to_4pi():
    _, solid = cubemap.latlong_dirs(256, 512, "cpu", centered=False)
    assert abs(float(solid.double().sum()) - 4 * math.pi) < 1e-3


def test_sh_components_orthonormal():
    dirs, solid = cubemap.latlong_dirs(512, 1024, "cpu", centered=False)
    y = volumes.sh_components(4, dirs.double()).reshape(-1, 16)
    gram = (y * solid.reshape(-1, 1).double()).T @ y
    assert torch.allclose(gram, torch.eye(16, dtype=torch.float64), atol=2e-2)


def test_probe_mask_corners_and_clamp():
    m = baking.probe_mask(torch.tensor([[0.01, 0.01, 0.01], [50.0, 0.0, 0.0]]), 9, 1.0)
    assert bool(m[4:6, 4:6, 4:6].all())
    assert bool(m[8].any())  # outside the box -> clamped to the boundary
    assert int(m.sum()) == 8 + 4


def test_masked_trilinear_masks_back_corners():
    res = 3
    ids = torch.arange(27, dtype=torch.int32).reshape(3, 3, 3)
    coeffs = torch.zeros(27, 1)
    coeffs[ids[1, 1, 1].long()] = 1.0
    coeffs[ids[0, 1, 1].long()] = 5.0
    aabb = torch.tensor([-1.0, -1, -1, 1, 1, 1])
    p = torch.tensor([[-0.5, 0.0, 0.0]])  # between x-cells 0 and 1
    n_pos = torch.tensor([[1.0, 0.0, 0.0]])
    out = volumes.masked_trilinear(coeffs, ids, aabb, p, n_pos)
    assert torch.allclose(out, torch.tensor([[1.0]]))  # only the +x corner is in front
    out = volumes.masked_trilinear(coeffs, ids, aabb, p, -n_pos)
    assert torch.allclose(out, torch.tensor([[5.0]]))
    assert res == 3


def test_dilation_fills_from_neighbours():
    ids = torch.full((4, 4, 4), -1, dtype=torch.int32)
    ids[0, 0, 0] = 2
    out = volumes.dilate_ids(ids)
    assert bool((out == 2).all())


def test_irradiance_query_constant_band0():
    coeffs = torch.zeros(3, 3, 3, 9, 1)
    coeffs[..., 0, 0] = 2.0
    aabb = torch.tensor([-1.0, -1, -1, 1, 1, 1])
    p = torch.rand(10, 3) * 4 - 2
    n = torch.nn.functional.normalize(torch.randn(10, 3), dim=-1)
    out = volumes.query_irradiance(coeffs, aabb, p, n)
    assert torch.allclose(out, torch.full((10, 1), 2.0 * 0.28209479177387814), atol=1e-6)


def test_normal_from_depth_fronto_parallel():
    K = torch.tensor([[30.0, 0, 16], [0, 30.0, 16], [0, 0, 1]])
    depth = torch.full((32, 32), 2.0)
    n = geometry.normal_from_depth(depth, K, torch.eye(3))
    valid = n.norm(dim=-1) > 0
    # zero-padded median filter zeroes the 4 image corners -> (2,2)-type pixels invalid too
    assert int(valid.sum()) == 28 * 28 - 4
    assert torch.allclose(n[valid], torch.tensor([0.0, 0.0, -1.0]).expand(780, 3), atol=1e-5)
    depth[10, 10] = 0.0  # an isolated hole is removed by the median filter
    assert torch.equal(geometry.normal_from_depth(depth, K, torch.eye(3)), n)
    depth[9:12, 9:12] = 0.0  # a 3x3 hole survives and invalidates its 5x5 neighbourhood
    n = geometry.normal_from_depth(depth, K, torch.eye(3))
    assert float(n[10, 10].norm()) == 0.0 and float(n[12, 10].norm()) == 0.0
    assert float(n[10, 14].norm()) > 0.0


def test_median_blur_removes_spike():
    d = torch.ones(9, 9)
    d[4, 4] = 10.0
    assert float(geometry.median_blur3(d)[4, 4]) == 1.0


def test_tv_losses():
    gt = torch.zeros(3, 4, 4)
    pred = torch.zeros(1, 4, 4)
    pred[:, :, 2:] = 1.0
    v = geometry.tv_loss(gt, pred)
    assert abs(float(v) - 4 / 12) < 1e-6  # horizontal diffs: 4 of 12 are 1, vertical 0
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    assert float(geometry.masked_tv_loss(mask, gt, pred)) == 0.0


def test_mip_level_and_lut():
    r = torch.tensor([0.08, 0.5, 1.0, 0.29])
    lv = light.mip_level(r, 5)
    assert torch.allclose(lv, torch.tensor([0.0, 3.0, 4.0, 1.5]), atol=1e-6)
    lut = torch.zeros(1, 256, 256, 2)
    lut[0, :, :, 0] = torch.linspace(0, 1, 256)[None, :]  # varies along u (width)
    lut[0, :, :, 1] = torch.linspace(0, 1, 256)[:, None]  # varies along v (height)
    out = light.lut_lookup(lut, torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0]))
    assert torch.allclose(out, torch.tensor([[0.0, 1.0], [1.0, 0.0]]), atol=1e-6)


def test_brdf_lut_file():
    lut = light.brdf_lut("cpu")
    assert lut.shape == (1, 256, 256, 2) and bool(torch.isfinite(lut).all())
    assert 0.0 <= float(lut.min()) and float(lut.max()) <= 1.01


def test_losses_stage_gating():
    rgb = torch.rand(16, 16, 3, requires_grad=True)
    tgt = torch.rand(16, 16, 3)
    l1 = torch.nn.functional.l1_loss(rgb, tgt)
    color = create_color(Spec("photometric", {"ssim_weight": 0.2}))
    assert torch.allclose(color({}, {"rgb": rgb, "gsir_stage": 2}, tgt), l1)
    assert not torch.allclose(color({}, {"rgb": rgb, "gsir_stage": 1}, tgt), l1)
    out2 = {"rgb": rgb, "gsir_stage": 2}
    assert float(create_normal(Spec("n"))({}, out2, tgt)) == 0.0
    out1 = {"rgb": rgb, "gsir_stage": 1}
    assert float(create_material(Spec("m"))({}, out1, tgt)) == 0.0
    assert float(create_light(Spec("l"))({}, out1, tgt)) == 0.0
    v = create_normal(Spec("n"))({}, out2, tgt)
    assert v.requires_grad
