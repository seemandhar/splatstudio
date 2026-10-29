"""Independent NumPy sphere ray caster creates a tiny training fixture, not a benchmark."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


def create_dataset(output, size=32):
    root = Path(output)
    if size < 16:
        raise ValueError("Demo image size must be at least 16")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Demo output directory must be empty")
    (root / "images").mkdir(parents=True, exist_ok=True)
    centers = np.array([[-0.35, 0, 0], [0.35, 0.1, 0.1], [0, -0.25, -0.3]])
    radii = [0.35, 0.28, 0.24]
    colors = np.array([[0.8, 0.1, 0.15], [0.1, 0.65, 0.25], [0.15, 0.25, 0.85]])
    frames = []
    f = size * 1.2
    yy, xx = np.mgrid[:size, :size]
    local = np.stack(
        [(xx + 0.5 - size / 2) / f, -(yy + 0.5 - size / 2) / f, -np.ones_like(xx)], axis=-1
    )
    for i in range(16):
        theta = 2 * np.pi * i / 16
        origin = np.array([2.4 * np.sin(theta), 0.8, 2.4 * np.cos(theta)])
        back = origin / np.linalg.norm(origin)
        right = np.cross([0, 1, 0], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        rotation = np.stack([right, up, back], axis=1)
        rays = local @ rotation.T
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        nearest = np.full((size, size), np.inf)
        image = np.ones((size, size, 3))
        for center, radius, color in zip(centers, radii, colors):
            offset = origin - center
            b = np.sum(rays * offset, axis=-1)
            disc = b * b - (np.dot(offset, offset) - radius * radius)
            distance = -b - np.sqrt(np.maximum(disc, 0))
            hit = (disc >= 0) & (distance > 0) & (distance < nearest)
            nearest[hit] = distance[hit]
            image[hit] = color
        name = f"images/{i:03d}.png"
        Image.fromarray(np.round(image * 255).astype(np.uint8)).save(root / name)
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = rotation, origin
        frames.append({"file_path": name, "transform_matrix": c2w.tolist()})
    rng = np.random.default_rng(7)
    points, rgb = [], []
    for center, radius, color in zip(centers, radii, colors):
        directions = rng.normal(size=(128, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        points.append(center + directions * radius)
        rgb.append(np.tile(color, (128, 1)))
    np.savez(root / "points.npz", points=np.concatenate(points), colors=np.concatenate(rgb))
    meta = {
        "w": size,
        "h": size,
        "fl_x": f,
        "fl_y": f,
        "cx": size / 2,
        "cy": size / 2,
        "camera_model": "PINHOLE",
        "frames": frames,
        "train_filenames": [f["file_path"] for i, f in enumerate(frames) if i % 4 not in (0, 2)],
        "val_filenames": [f["file_path"] for i, f in enumerate(frames) if i % 4 == 2],
        "test_filenames": [f["file_path"] for i, f in enumerate(frames) if i % 4 == 0],
    }
    (root / "transforms.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {
        "path": str(root.resolve()),
        "frames": 16,
        "description": "Analytic sphere fixture; not paper reproduction evidence",
    }
