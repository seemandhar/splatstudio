import numpy as np
import pytest
import torch

from ss_fcgs import codec, xyz_codec


def test_morton_roundtrip():
    rng = np.random.default_rng(0)
    vox = rng.integers(0, 65536, size=(1000, 3))
    back = xyz_codec.unmorton(xyz_codec.morton(vox))
    assert np.array_equal(back.astype(np.int64), vox)


def test_octree_lossless_with_duplicates():
    rng = np.random.default_rng(1)
    vox = rng.integers(0, 65536, size=(3000, 3)).astype(np.float32)
    vox = np.concatenate([vox, vox[:100], vox[:100], vox[:3]])
    dec = xyz_codec.octree_decode(xyz_codec.octree_encode(vox))
    assert np.array_equal(xyz_codec.sorted_voxels(dec), xyz_codec.sorted_voxels(vox))


def test_octree_single_point_and_corners():
    vox = np.array([[0, 0, 0], [65535, 65535, 65535], [0, 65535, 0]], dtype=np.float32)
    dec = xyz_codec.octree_decode(xyz_codec.octree_encode(vox))
    assert np.array_equal(xyz_codec.sorted_voxels(dec), xyz_codec.sorted_voxels(vox))


def test_voxelize_official_extremes():
    means = np.array([[0.0, -1.0, 2.0], [1.0, 1.0, 3.0], [0.5, 0.0, 2.5]], dtype=np.float32)
    vox, lo, hi = xyz_codec.voxelize(means)
    assert vox.min() == 0 and vox.max() == 65535
    back = xyz_codec.devoxelize(vox, lo, hi)
    assert np.abs(back - means).max() <= (hi - lo).max() / 65535


def test_resolve_coder():
    assert xyz_codec.resolve_coder("octree") == "octree"
    assert xyz_codec.resolve_coder("auto") in ("gpcc", "octree")
    with pytest.raises(ValueError):
        xyz_codec.resolve_coder("zip")


def test_attribute_layout_matches_3dgs_ply():
    n = 5
    params = {"means": torch.zeros(n, 3), "sh0": torch.arange(n * 3.0).reshape(n, 1, 3),
              "shN": torch.arange(n * 45.0).reshape(n, 15, 3) + 1000,
              "opacities": torch.arange(n * 1.0) + 2000, "scales": torch.ones(n, 3) * 3,
              "quats": torch.ones(n, 4) * 4}
    fea = codec.params_to_fea(params)
    assert fea.shape == (n, 56)
    # 3DGS: f_dc (3) then features_rest [N,15,3].view(N,45) then opacity, scale(3), rot(4)
    assert torch.equal(fea[:, :3], params["sh0"].reshape(n, 3))
    assert torch.equal(fea[:, 3:48], params["shN"].reshape(n, 45))
    assert torch.equal(fea[:, 48], params["opacities"])
    back = codec.fea_to_params(params["means"], fea)
    for k in params:
        assert torch.equal(back[k].reshape(params[k].shape), params[k])


def test_official_step_size():
    assert codec.per_step_size(500_000) == 1_000_000
    assert codec.per_step_size(1_050_000) == 1_100_000
    assert codec.per_step_size(5_600_000) == 1_000_000


def test_weights_names():
    assert codec.weights_path(1e-4).name == "checkpoint_0.0001.pkl"
    assert codec.weights_path(16e-4).name == "checkpoint_0.0016.pkl"
