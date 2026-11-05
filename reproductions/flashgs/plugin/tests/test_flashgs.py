import math

import pytest
import torch

from ss_flashgs import core, method


def _camera(width=64, height=48):
    angle = 0.3
    R = torch.tensor([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0],
                      [-math.sin(angle), 0, math.cos(angle)]])
    V = torch.eye(4)
    V[:3, :3] = R
    V[:3, 3] = torch.tensor([0.1, -0.2, 0.5])
    K = torch.tensor([[50.0, 0, width / 2], [0, 55.0, height / 2], [0, 0, 1]])
    return {"viewmat": V, "K": K, "width": width, "height": height}


def test_rotmat_matches_reference_formula():
    q = torch.tensor([[0.9, 0.1, -0.3, 0.2]])
    R = core.quat_to_rotmat(q)[0]
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6)
    assert torch.isclose(torch.det(R), torch.tensor(1.0), atol=1e-6)
    # rotation about z by theta: q = (cos t/2, 0, 0, sin t/2)
    t = 0.7
    Rz = core.quat_to_rotmat(torch.tensor([[math.cos(t / 2), 0, 0, math.sin(t / 2)]]))[0]
    expect = torch.tensor([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
    assert torch.allclose(Rz, expect, atol=1e-6)


def test_prepare_covariance_layout_and_sh_mask():
    n = 5
    g = torch.Generator().manual_seed(0)
    params = {"means": torch.randn(n, 3, generator=g), "scales": torch.randn(n, 3, generator=g),
              "quats": torch.randn(n, 4, generator=g), "opacities": torch.randn(n, generator=g),
              "sh0": torch.randn(n, 1, 3, generator=g), "shN": torch.randn(n, 15, 3, generator=g)}
    p = core.prepare(params, sh_degree=1)
    R = core.quat_to_rotmat(params["quats"])
    S = torch.diag_embed(params["scales"].exp())
    cov = R @ S @ S @ R.transpose(1, 2)
    idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    for k, (i, j) in enumerate(idx):
        assert torch.allclose(p["cov3d"][:, k], cov[:, i, j], atol=1e-5)
    shs = p["shs"].view(n, 16, 3)
    assert torch.equal(shs[:, 0], params["sh0"][:, 0])
    assert torch.equal(shs[:, 1:4], params["shN"][:, :3])
    assert torch.all(shs[:, 4:] == 0)
    assert torch.allclose(p["opacity"], params["opacities"].sigmoid())


def test_camera_terms_centre_and_rotation():
    cam = _camera()
    pos, rot, fx, fy, W, H = core.camera_terms(cam)
    c2w = torch.linalg.inv(cam["viewmat"])
    assert torch.allclose(pos, c2w[:3, 3], atol=1e-6)
    assert torch.allclose(rot, c2w[:3, :3], atol=1e-6)  # 3DGS cameras.json 'rotation'
    assert (fx, fy, W, H) == (50.0, 55.0, 64, 48)


def test_off_centre_principal_point_rejected():
    cam = _camera()
    cam["K"] = cam["K"].clone()
    cam["K"][0, 2] += 3
    with pytest.raises(ValueError):
        core.camera_terms(cam)


def test_scaled_camera_keeps_fov():
    cam = _camera(980, 545)
    cam["K"][0, 0] = cam["K"][1, 1] = 580.0
    s = core.scaled_camera(cam, 1920)
    assert s["width"] == 1920 and s["height"] == round(545 * 1920 / 980)
    assert math.isclose(s["width"] / s["K"][0, 0].item(), 980 / 580.0, rel_tol=1e-6)
    assert s["K"][0, 2] == 960 and s["K"][1, 2] == s["height"] / 2


def test_projection_matches_3dgs_formula():
    W, H, fx, fy = 100, 60, 80.0, 90.0
    P = core.projection_3dgs(W, H, fx, fy)
    p = torch.tensor([0.3, -0.2, 2.0, 1.0])
    ndc = P @ p
    ndc = ndc[:2] / ndc[3]
    # ndc2Pix(v,S) = ((v+1)S-1)/2  must equal f*x/z + S/2 - 0.5
    px = ((ndc[0] + 1) * W - 1) / 2
    py = ((ndc[1] + 1) * H - 1) / 2
    assert math.isclose(px.item(), fx * 0.3 / 2 + W / 2 - 0.5, rel_tol=1e-5)
    assert math.isclose(py.item(), fy * -0.2 / 2 + H / 2 - 0.5, rel_tol=1e-5)


def test_quantize_truncates_like_kernel():
    x = torch.tensor([0.0, 0.999 / 255, 1.0 / 255, 0.5, 1.2, -0.1])
    assert core.quantize_trunc(x).tolist() == [0, 0, 1, 127, 255, 0]


def test_bbox_bound_counts_single_and_culled():
    cam = _camera(64, 48)
    cam["viewmat"] = torch.eye(4)
    params = {"means": torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 0.1], [0.0, 0.0, 3.0]]),
              "scales": torch.full((3, 3), math.log(1e-4)), "quats": torch.tensor([[1.0, 0, 0, 0]] * 3),
              "opacities": torch.logit(torch.tensor([0.9, 0.9, 0.001])),
              "sh0": torch.zeros(3, 1, 3), "shN": torch.zeros(3, 15, 3)}
    n = core.bbox_tile_counts(core.prepare(params), cam)
    # tiny Gaussian at the image centre (31.5, 23.5): rect covers tiles x 1, y 1 -> 1 tile...
    assert n[0] >= 1
    assert n[1] == 0  # behind the 0.2 near plane
    assert n[2] == 0  # 255 * alpha < 1


def test_method_spec_declares_flashgs_renderer():
    spec = method()
    r = next(c for c in spec.components if c.kind == "renderer")
    assert r.factory == "ss_flashgs.renderer:create_renderer"
    assert r.config["inference_backend"] == "flashgs"
    names = {p.name for p in spec.probes}
    assert {"flashgs-parity", "precise-intersection", "render-speed", "inference-switch"} <= names
