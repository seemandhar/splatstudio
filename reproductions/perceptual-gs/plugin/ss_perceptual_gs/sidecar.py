"""Ground-truth perceptual-sensitivity maps (C1, Eqs. 5-7) and their per-scene sidecar.

`ss-pgs-prep --data <native scene> --raw <COLMAP scene with full-res images/>` follows the
official `preprocess.py` + loader exactly:
  full-resolution image -> PIL convert("L") -> gamma point(255 (x/255)^(1/1.5)) -> ToTensor
  -> Sobel / threshold / pool (core.sensitivity_map) -> uint8 PNG semantics
  -> PIL resize to the training size (Pillow default BICUBIC, official PILtoTorch) -> /255.
The training-size maps are stored as uint8 in

    <sidecar_root>/<data path relative to ~/datasets>/
        viewmats.npy  [N,4,4] float32 OpenCV world-to-camera (splatstudio.data convention)
        names.json    frame file paths (transforms.json order)
        sens_u8.npz   "sens": [N,H,W] uint8
        prep.json     provenance

The runtime does not pass the dataset path to components; it is taken from `PGS_DATA` or
ss-train's `--data` argument, and frames are matched by viewmat (SPLATIFY sidecar pattern).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from . import core

DEFAULT_SIDECAR_ROOT = "/bigdata/sejain/perceptual-gs-sens"
DATASETS_HOME = Path.home() / "datasets"
_TABLES = {}


def data_root():
    if os.environ.get("PGS_DATA"):
        return Path(os.environ["PGS_DATA"]).expanduser()
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--data" and i + 1 < len(argv):
            return Path(argv[i + 1]).expanduser()
        if a.startswith("--data="):
            return Path(a.split("=", 1)[1]).expanduser()
    return None


def sidecar_dir(data, sidecar_root=DEFAULT_SIDECAR_ROOT):
    data = Path(data).expanduser().resolve()
    try:
        rel = data.relative_to(DATASETS_HOME.resolve())
    except ValueError:
        rel = Path(*data.parts[1:])
    return Path(sidecar_root) / rel


def viewmats_from_transforms(meta):
    """Same OpenGL->OpenCV conversion as splatstudio.data.load_dataset."""
    out = []
    for frame in meta["frames"]:
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        c2w[:3, 1:3] *= -1
        out.append(torch.linalg.inv(c2w))
    return torch.stack(out)


def load_luminance(path, gamma=1.5):
    """Official load_image_L: PIL L conversion + gamma via Image.point, ToTensor -> [1,1,H,W]."""
    from PIL import Image

    image = Image.open(path).convert("L")
    image = image.point(lambda x: 255 * (x / 255) ** (1 / gamma))
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)[None, None]


def map_u8_at(path, size, tau_e=0.05, tau_s=0.3, pool=5, gamma=1.5):
    """Full-res sensitivity map, saved/loaded as uint8 L, PIL-resized to `size` (w, h)."""
    from PIL import Image

    full = core.sensitivity_map(load_luminance(path, gamma), tau_e, tau_s, pool)
    u8 = (full * 255).clamp(0, 255).byte()[0, 0].numpy()  # official save_image_L
    return np.asarray(Image.fromarray(u8).convert("L").resize(size), dtype=np.uint8)


class Table:
    def __init__(self, viewmats, maps_u8, names=None, tol=1e-4):
        self.viewmats = torch.as_tensor(viewmats, dtype=torch.float32).reshape(-1, 16)
        self.maps_u8 = torch.as_tensor(maps_u8, dtype=torch.uint8)
        self.names = names
        self.tol = tol
        self._device = {}

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        viewmats = np.load(directory / "viewmats.npy")
        with np.load(directory / "sens_u8.npz") as d:
            maps = d["sens"]
        names = json.loads((directory / "names.json").read_text())
        if len(names) != len(viewmats) or len(maps) != len(names):
            raise ValueError(f"Inconsistent Perceptual-GS sidecar in {directory}")
        return cls(viewmats, maps, names)

    def _on(self, device):
        key = str(device)
        if key not in self._device:
            self._device[key] = (self.viewmats.to(device), self.maps_u8.to(device))
        return self._device[key]

    def index(self, viewmat):
        vm, _ = self._on(viewmat.device)
        err = (vm - viewmat.detach().float().reshape(1, 16)).abs().amax(dim=1)
        i = int(err.argmin())
        if float(err[i]) > self.tol:
            raise KeyError("Camera is not in the Perceptual-GS sensitivity sidecar")
        return i

    def sens(self, viewmat):
        """[H,W] float GT sensitivity in [0,1] (official PILtoTorch /255) for this camera."""
        _, maps = self._on(viewmat.device)
        return maps[self.index(viewmat)].float() / 255.0


def table(root=None, sidecar_root=DEFAULT_SIDECAR_ROOT):
    root = Path(root) if root is not None else data_root()
    if root is None:
        raise RuntimeError(
            "Perceptual-GS needs the dataset path (ss-train --data or PGS_DATA) to find its "
            "sensitivity sidecar"
        )
    directory = sidecar_dir(root, sidecar_root)
    key = str(directory)
    if key not in _TABLES:
        if not (directory / "sens_u8.npz").exists():
            raise FileNotFoundError(
                f"No Perceptual-GS sensitivity sidecar at {directory}; run "
                f"ss-pgs-prep --data {root} --raw <COLMAP scene with full-res images>"
            )
        _TABLES[key] = Table.load(directory)
    return _TABLES[key]


def build(data, raw, sidecar_root=DEFAULT_SIDECAR_ROOT, image_dir="images", **kw):
    data, raw = Path(data).expanduser().resolve(), Path(raw).expanduser().resolve()
    meta = json.loads((data / "transforms.json").read_text())
    raw_images = {p.stem: p for p in (raw / image_dir).iterdir() if p.is_file()}
    from PIL import Image

    maps, names = [], []
    for i, frame in enumerate(meta["frames"]):
        stem = Path(frame["file_path"]).stem
        if stem not in raw_images:
            raise FileNotFoundError(f"No full-resolution image for {frame['file_path']} in {raw}")
        with Image.open(data / frame["file_path"]) as im:
            size = im.size
        maps.append(map_u8_at(raw_images[stem], size, **kw))
        names.append(frame["file_path"])
        print(f"\r{i + 1}/{len(meta['frames'])} {stem}", end="", flush=True)
    print()
    out = sidecar_dir(data, sidecar_root)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "viewmats.npy", viewmats_from_transforms(meta).numpy())
    (out / "names.json").write_text(json.dumps(names))
    np.savez_compressed(out / "sens_u8.npz", sens=np.stack(maps))
    import PIL

    beta = core.scene_sensitivity([torch.from_numpy(m) for m in maps])
    (out / "prep.json").write_text(json.dumps(
        {"data": str(data), "raw": str(raw / image_dir), "pillow": PIL.__version__,
         "params": kw, "beta_all_frames": beta}, indent=1))
    return out, beta


def main(argv=None):
    p = argparse.ArgumentParser(prog="ss-pgs-prep", description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True, help="native scene (transforms.json)")
    p.add_argument("--raw", required=True, help="COLMAP scene holding full-resolution images/")
    p.add_argument("--image-dir", default="images")
    p.add_argument("--sidecar-root", default=DEFAULT_SIDECAR_ROOT)
    a = p.parse_args(argv)
    out, beta = build(a.data, a.raw, a.sidecar_root, a.image_dir)
    print(json.dumps({"sidecar": str(out), "beta_all_frames": beta}))


if __name__ == "__main__":
    main()
