import struct

import numpy as np
import torch

import ss_stablegs
from ss_stablegs import pairs as P
from ss_stablegs import probes


def test_cpu_probes():
    m = ss_stablegs.method()
    for fn in (probes.depth_consistency_formula, probes.pair_selection, probes.pair_stack,
               probes.gof_qualifier):
        fn(m, "cpu")
    from ss_stablegs.link import LINK

    LINK.attach(None)


def test_images_bin_roundtrip(tmp_path):
    path = tmp_path / "images.bin"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<idddddddi", 7, 1, 0, 0, 0, 0.1, 0.2, 0.3, 1))
        f.write(b"img.jpg\x00")
        f.write(struct.pack("<Q", 2))
        f.write(struct.pack("<ddq", 1.5, 2.5, 11))
        f.write(struct.pack("<ddq", 3.0, 4.0, -1))
    im = P.read_images_bin(path)["img.jpg"]
    assert np.allclose(im["xys"], [[1.5, 2.5], [3.0, 4.0]])
    assert im["point3D_ids"].tolist() == [11, -1]
    assert np.allclose(im["tvec"], [0.1, 0.2, 0.3])


def test_rotation_angle():
    assert abs(P.rotation_angle_deg(probes._rot_y(0), probes._rot_y(37)) - 37) < 1e-6


def test_spec_defaults():
    m = ss_stablegs.method()
    f = {x.name: x for x in m.fields}["opacity_aux"]
    assert abs(torch.sigmoid(torch.tensor(f.initial_value)) - 0.99) < 1e-6
    cfg = {c.name: c.config for c in m.components}
    assert cfg["depth-consistency"]["lambda_consis"] == 0.05
    assert cfg["depth-normal"]["lambda_depth_normal"] == 0.05
    assert cfg["stablegs-density"]["dual_from"] == 15000
    assert cfg["stablegs-density"]["ho_iteration"] == 15000
    assert cfg["stablegs-density"]["budget_table"][str(136029)] == 2584171  # truck run.sh


def test_consistency_zero_depth_guard():
    from ss_stablegs.losses import _sample

    depth = torch.rand(6, 8, requires_grad=True)
    pts = torch.tensor([[0.1, -0.7, 0.0], [0.0, 0.2, -0.1], [2.0, 0.0, 0.0]],
                       requires_grad=True)  # third point: z exactly 0
    cam = {"K": torch.tensor([[5.0, 0, 4], [0, 5.0, 3], [0, 0, 1]])}
    out = _sample(depth, pts, cam)
    out.sum().backward()
    assert torch.isfinite(pts.grad).all() and torch.isfinite(depth.grad).all()
    assert float(out[2]) == 0.0
