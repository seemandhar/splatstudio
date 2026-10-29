"""Convert the native analytic fixture to Nerfstudio's PLY initialization format."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    if output.exists():
        raise ValueError("Choose a new output directory")
    shutil.copytree(source, output)
    data = np.load(source / "points.npz", allow_pickle=False)
    vertices = np.empty(
        len(data["points"]),
        dtype=[(k, "f4") for k in ("x", "y", "z")] + [(k, "u1") for k in ("red", "green", "blue")],
    )
    for i, key in enumerate(("x", "y", "z")):
        vertices[key] = data["points"][:, i]
    for i, key in enumerate(("red", "green", "blue")):
        vertices[key] = (data["colors"][:, i] * 255).round()
    PlyData([PlyElement.describe(vertices, "vertex")]).write(output / "points.ply")
    metadata = json.loads((output / "transforms.json").read_text())
    metadata.update(ply_file_path="points.ply", camera_model="OPENCV")
    (output / "transforms.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
