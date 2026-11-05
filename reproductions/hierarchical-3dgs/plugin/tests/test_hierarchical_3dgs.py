import math

import numpy as np
import torch

from ss_hierarchical_3dgs import hier, method, probes
from ss_hierarchical_3dgs.density import expon_lr


def test_qvec_roundtrip_gives_camera_centre():
    V = probes.look_at((3.0, -2.0, 1.5))
    q = hier.rotmat_to_qvec(V[:3, :3].numpy())
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    np.testing.assert_allclose(R, V[:3, :3].numpy(), atol=1e-5)
    centre = -R.T @ V[:3, 3].numpy()
    np.testing.assert_allclose(centre, [3.0, -2.0, 1.5], atol=1e-5)


def test_pack_unpack_roundtrip():
    nodes = torch.tensor([[2, -1, 0, 0, 1, 1, 2], [0, 0, 1, 1, 0, -1, 0], [0, 0, 2, 1, 0, -1, 0]],
                         dtype=torch.int32)
    boxes = torch.rand(3, 2, 4)
    blob = hier.pack(nodes, boxes, torch.tensor([1, 2]), 5)
    params = {hier.FIELD: blob}
    assert hier.is_built(params)
    n, b, a = hier.unpack(params)
    assert torch.equal(n, nodes)
    assert torch.allclose(b, boxes)
    assert a.tolist() == [False, True, True, False, False]
    assert not hier.is_built({hier.FIELD: torch.zeros(4, hier.WIDTH)})


def test_thresholds_and_schedules():
    cam = {"width": 100, "K": torch.tensor([[50.0, 0, 50], [0, 50, 50], [0, 0, 1]])}
    # tanfovx = 1, threshold = 2 (tau + .5) / 50
    assert math.isclose(hier.pixel_threshold(6.0, cam), 13 / 50)
    assert hier.pixel_threshold(-0.5, cam) == 0
    assert math.isclose(hier.sample_limit(0.005, 0.1, 0.0), 0.005)
    assert math.isclose(hier.sample_limit(0.005, 0.1, 0.5), math.sqrt(0.0005))
    assert math.isclose(expon_lr(15000, 2e-5, 2e-7, 30000), 2e-6, rel_tol=1e-9)


def test_ply_layout(tmp_path):
    p = probes.synthetic_leaves("cpu", n=7)
    hier.write_ply(tmp_path / "a.ply", p)
    raw = (tmp_path / "a.ply").read_bytes()
    head, body = raw.split(b"end_header\n")
    assert b"element vertex 7" in head.split(b"\n")[2]
    data = np.frombuffer(body, dtype="<f4").reshape(7, 62)
    np.testing.assert_allclose(data[:, 6:9], p["sh0"][:, 0].numpy())
    np.testing.assert_allclose(data[:, 9:24], p["shN"][:, :, 0].numpy())  # channel-major
    np.testing.assert_allclose(data[:, 54], p["opacities"].numpy())


def test_leaf_training_probe_cpu():
    probes.leaf_training(method(), "cpu")
