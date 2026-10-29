import json

import numpy as np
import pytest
from PIL import Image

from splatstudio.data import load_dataset
from splatstudio.demo import create_dataset

pytest.importorskip("cv2")
from splatstudio.preprocessing import prepare_dataset  # noqa: E402


def test_pinhole_conversion_preserves_pixels_poses_and_split_membership(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    create_dataset(source)
    before = (source / "transforms.json").read_bytes()
    prepare_dataset(source, output)
    old, new = load_dataset(source), load_dataset(output)
    for previous, current in zip(old["frames"].values(), new["frames"].values()):
        for key in ("image", "K", "viewmat"):
            np.testing.assert_allclose(previous[key].numpy(), current[key].numpy())
    assert [len(v) for v in old["splits"].values()] == [len(v) for v in new["splits"].values()]
    assert before == (source / "transforms.json").read_bytes()
    assert (output / "points.npz").read_bytes() == (source / "points.npz").read_bytes()


def test_distortion_is_rectified_and_recorded(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    create_dataset(source, 64)
    path = source / "transforms.json"
    metadata = json.loads(path.read_text())
    metadata.update(k1=0.25, k2=-0.03, p1=0.01)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="distortion"):
        load_dataset(source)
    report = prepare_dataset(source, output, 2)
    dataset = load_dataset(output)
    assert report["preprocessing"]["downscale"] == 2
    assert next(iter(dataset["frames"].values()))["width"] == 32
    assert Image.open(output / "images/000000.png").size == (32, 32)


def test_converter_rejects_overwriting_input(tmp_path):
    create_dataset(tmp_path)
    with pytest.raises(ValueError, match="separate output"):
        prepare_dataset(tmp_path, tmp_path)
