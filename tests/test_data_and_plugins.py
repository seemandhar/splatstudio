import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from splatstudio.data import load_dataset
from splatstudio.demo import create_dataset
from splatstudio.registry import discover
from splatstudio.scaffold import create


def test_demo_has_disjoint_splits_and_stable_protocol(tmp_path):
    create_dataset(tmp_path)
    data = load_dataset(tmp_path)
    groups = [set(v) for v in data["splits"].values()]
    assert sum(map(len, groups)) == len(set.union(*groups)) == 16
    assert data["protocol_id"] == load_dataset(tmp_path)["protocol_id"]
    assert data["protocol_id"] != load_dataset(tmp_path, downscale=2)["protocol_id"]


def test_split_overlap_is_rejected(tmp_path):
    create_dataset(tmp_path)
    p = tmp_path / "transforms.json"
    data = json.loads(p.read_text())
    data["test_filenames"].append(data["train_filenames"][0])
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="disjoint"):
        load_dataset(tmp_path)


def test_unhandled_distortion_is_rejected(tmp_path):
    create_dataset(tmp_path)
    p = tmp_path / "transforms.json"
    data = json.loads(p.read_text())
    data["k1"] = 0.1
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="distortion"):
        load_dataset(tmp_path)


def test_duplicate_plugin_names_are_rejected(monkeypatch):
    monkeypatch.setattr(
        "splatstudio.registry.metadata.entry_points",
        lambda **kw: [SimpleNamespace(name="gsplat", value="other:method")],
    )
    with pytest.raises(ValueError, match="Duplicate method"):
        discover()


def test_scaffold_is_a_real_external_package(tmp_path):
    create("test-extension", tmp_path)
    source = (tmp_path / "pyproject.toml").read_text()
    assert '[project.entry-points."splatstudio.methods"]' in source
    assert (tmp_path / "ss_test_extension/__init__.py").exists()
    with pytest.raises(ValueError, match="empty"):
        create("test-extension", tmp_path)


def test_discovery_does_not_import_torch():
    code = "import sys; from splatstudio.registry import discover; discover(); assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
