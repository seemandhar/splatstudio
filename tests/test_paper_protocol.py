"""Paper-protocol additions: llffhold8 splits, COLMAP conversion, and the 3DGS recipe."""

import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from splatstudio.data import load_dataset
from splatstudio.demo import create_dataset
from splatstudio.runtime import camera_extent, knn_log_scales

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "colmap_to_transforms.py"
spec = importlib.util.spec_from_file_location("colmap_to_transforms", SCRIPT)
colmap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colmap)


def test_llffhold8_holds_out_every_eighth_sorted_name(tmp_path):
    create_dataset(tmp_path)
    path = tmp_path / "transforms.json"
    meta = json.loads(path.read_text())
    for key in ("train_filenames", "val_filenames", "test_filenames"):
        meta.pop(key)
    meta["split_protocol"] = "llffhold8"
    meta["frames"] = meta["frames"][::-1]  # order in the file must not matter
    path.write_text(json.dumps(meta))
    data = load_dataset(tmp_path)
    names = sorted(f["file_path"] for f in meta["frames"])
    assert data["splits"]["test"] == names[::8]
    assert data["splits"]["validation"] == data["splits"]["test"]
    assert set(data["splits"]["train"]) == set(names) - set(names[::8])
    assert data["protocol"]["split_protocol"] == "llffhold8"


def test_unknown_split_protocol_is_rejected(tmp_path):
    create_dataset(tmp_path)
    path = tmp_path / "transforms.json"
    meta = json.loads(path.read_text())
    for key in ("train_filenames", "val_filenames", "test_filenames"):
        meta.pop(key)
    meta["split_protocol"] = "every-other"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="split_protocol"):
        load_dataset(tmp_path)


def test_knn_scales_match_brute_force_mean_of_three_squared_distances():
    g = torch.Generator().manual_seed(0)
    points = torch.rand(300, 3, generator=g)
    d2 = torch.cdist(points, points).square()
    d2.fill_diagonal_(float("inf"))
    expected = d2.topk(3, largest=False).values.mean(1).sqrt().log()
    result = knn_log_scales(points, chunk=64)
    assert result.shape == (300, 3)
    torch.testing.assert_close(result[:, 0], expected)


def test_camera_extent_is_1p1_times_max_center_distance():
    frames, names = {}, []
    for i, center in enumerate(([0.0, 0, 0], [2.0, 0, 0], [1.0, 3, 0])):
        c2w = torch.eye(4)
        c2w[:3, 3] = torch.tensor(center)
        frames[str(i)] = {"viewmat": torch.linalg.inv(c2w)}
        names.append(str(i))
    dataset = {"frames": frames, "splits": {"train": names}}
    mean = torch.tensor([1.0, 1.0, 0.0])
    expected = 1.1 * max(float(torch.linalg.norm(torch.tensor(c) - mean)) for c in
                         ([0.0, 0, 0], [2.0, 0, 0], [1.0, 3, 0]))
    assert camera_extent(dataset) == pytest.approx(expected)


def write_colmap(root, poses, size=(64, 48), scale=2):
    """Minimal COLMAP binary model: one PINHOLE camera at `scale` x the image resolution."""
    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    w, h = size
    with open(sparse / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, w * scale, h * scale))
        f.write(struct.pack("<dddd", 50.0 * scale, 52.0 * scale, w * scale / 2, h * scale / 2))
    with open(sparse / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(poses)))
        for i, (q, t) in enumerate(poses):
            f.write(struct.pack("<idddddddi", i + 1, *q, *t, 1))
            f.write(f"img{i:02d}.png".encode() + b"\x00")
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<ddq", 1.0, 2.0, -1))
    with open(sparse / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", 3))
        for i in range(3):
            f.write(struct.pack("<QdddBBBd", i + 1, i, 2.0 * i, 5.0, 255, 128, 0, 0.5))
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<ii", 1, 0))
    images = root / "images_2"
    images.mkdir()
    for i in range(len(poses)):
        Image.fromarray(np.full((h, w, 3), 10 * i, np.uint8)).save(images / f"img{i:02d}.png")


def test_colmap_conversion_round_trips_poses_intrinsics_and_points(tmp_path):
    rng = np.random.default_rng(1)
    poses = []
    for _ in range(9):
        q = rng.normal(size=4)
        poses.append((tuple(q / np.linalg.norm(q)), tuple(rng.normal(size=3))))
    write_colmap(tmp_path / "scene", poses)
    result = colmap.convert(tmp_path / "scene", "images_2", tmp_path / "out", "llffhold8")
    assert result["frames"] == 9 and result["distorted_frames"] == 0
    data = load_dataset(tmp_path / "out")
    for i, (q, t) in enumerate(poses):
        w2c = np.eye(4)
        w2c[:3, :3] = colmap.rotation(q)
        w2c[:3, 3] = t
        frame = data["frames"][f"images/img{i:02d}.png"]
        np.testing.assert_allclose(frame["viewmat"].numpy(), w2c, atol=1e-5)
        # Intrinsics rescaled from the 2x COLMAP calibration to the stored image size.
        np.testing.assert_allclose(frame["K"].numpy()[:2, :3], [[50, 0, 32], [0, 52, 24]])
    assert data["splits"]["test"] == ["images/img00.png", "images/img08.png"]
    with np.load(tmp_path / "out" / "points.npz") as cloud:
        np.testing.assert_allclose(cloud["points"][2], [2, 4, 5])
        np.testing.assert_allclose(cloud["colors"][0], [1, 128 / 255, 0], atol=1e-6)


def test_opacity_reset_follows_3dgs_schedule():
    from splatstudio.backends.gsplat import Density, resets_opacity
    from splatstudio.specs import ComponentSpec

    density = Density(ComponentSpec("d", "density", "x:y", ("gradient.screen.signed",)))
    steps = [s for s in range(1, 30001) if resets_opacity(density.strategy, s)]
    assert steps == [3000, 6000, 9000, 12000]
    assert density.opacity_reset
    off = Density(ComponentSpec("d", "density", "x:y", (), (), {"opacity_reset": False}))
    assert not off.opacity_reset


def test_opacity_reset_clamps_and_clears_moments():
    from splatstudio.backends.gsplat import Density
    from splatstudio.methods import vanilla
    from splatstudio.specs import ComponentSpec
    from splatstudio.state import sample_scene

    params, optimizers = sample_scene(vanilla())
    density = Density(ComponentSpec("d", "density", "x:y", ("gradient.screen.signed",)))
    density.initialize(params, optimizers)
    with torch.no_grad():
        params["opacities"].fill_(3.0)
    params["opacities"].grad = torch.ones_like(params["opacities"])
    optimizers["opacities"].step()
    from gsplat.strategy.ops import reset_opa

    reset_opa(params, optimizers, density.state, value=density.strategy.prune_opa * 2.0)
    assert torch.sigmoid(params["opacities"]).max().item() <= 0.01 + 1e-6
    assert optimizers["opacities"].state[params["opacities"]]["exp_avg"].abs().max() == 0
