"""Convert a COLMAP scene (Mip-NeRF 360, Tanks & Temples, Deep Blending) for native training.

Reads sparse/0/{cameras,images,points3D}.bin, links the chosen image folder (for example
images_4) into the output directory, rescales intrinsics to the actual image size, and writes
transforms.json plus points.npz from the SfM points. Poses are written in the Nerfstudio/OpenGL
convention that splatstudio.data expects; no world reorientation or rescaling is applied.

The 3DGS benchmark protocol is selected with --split-protocol llffhold8 (sorted names, every
8th image held out for test). Distorted camera models are written with their coefficients;
run `splatstudio prepare-data` afterwards to undistort them.

    python scripts/colmap_to_transforms.py --scene data/360_v2/garden --images images_4 \
        --output data/native/garden
"""

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np
from PIL import Image

# model_id: (name, number of parameters)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
}


def read(stream, fmt):
    return struct.unpack("<" + fmt, stream.read(struct.calcsize("<" + fmt)))


def read_cameras(path):
    cameras = {}
    with open(path, "rb") as f:
        for _ in range(read(f, "Q")[0]):
            cam_id, model_id, width, height = read(f, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id}")
            name, count = CAMERA_MODELS[model_id]
            cameras[cam_id] = {
                "model": name,
                "width": width,
                "height": height,
                "params": read(f, "d" * count),
            }
    return cameras


def read_images(path):
    images = []
    with open(path, "rb") as f:
        for _ in range(read(f, "Q")[0]):
            _, qw, qx, qy, qz, tx, ty, tz, cam_id = read(f, "idddddddi")
            name = b""
            while (char := f.read(1)) != b"\x00":
                name += char
            f.seek(read(f, "Q")[0] * 24, os.SEEK_CUR)  # skip 2D keypoints (x, y, point id)
            images.append(
                {"name": name.decode(), "q": (qw, qx, qy, qz), "t": (tx, ty, tz), "camera": cam_id}
            )
    return images


def read_points(path):
    xyz, rgb = [], []
    with open(path, "rb") as f:
        for _ in range(read(f, "Q")[0]):
            values = read(f, "QdddBBBd")
            xyz.append(values[1:4])
            rgb.append(values[4:7])
            f.seek(read(f, "Q")[0] * 8, os.SEEK_CUR)  # skip track (image id, point2D idx)
    return np.asarray(xyz, np.float32), np.asarray(rgb, np.float32) / 255


def rotation(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def intrinsics(camera, sx, sy):
    p, model = camera["params"], camera["model"]
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    else:
        fx, fy, cx, cy = p[:4]
    entry = {"fl_x": fx * sx, "fl_y": fy * sy, "cx": cx * sx, "cy": cy * sy}
    distortion = {}
    if model == "SIMPLE_RADIAL":
        distortion = {"k1": p[3]}
    elif model == "RADIAL":
        distortion = {"k1": p[3], "k2": p[4]}
    elif model == "OPENCV":
        distortion = {"k1": p[4], "k2": p[5], "p1": p[6], "p2": p[7]}
    entry["camera_model"] = "OPENCV" if any(distortion.values()) else "PINHOLE"
    entry.update({k: v for k, v in distortion.items() if v})
    return entry


def link(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def convert(scene, images_dir, output, split_protocol):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    sparse = scene / "sparse" / "0"
    if not sparse.is_dir():
        sparse = scene / "sparse"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    (output / "images").mkdir(parents=True, exist_ok=True)
    cameras = read_cameras(sparse / "cameras.bin")
    frames = []
    for image in sorted(read_images(sparse / "images.bin"), key=lambda i: i["name"]):
        camera = cameras[image["camera"]]
        source = scene / images_dir / image["name"]
        if not source.exists():
            raise ValueError(f"Missing image {source}")
        with Image.open(source) as pixels:
            width, height = pixels.size
        target = output / "images" / image["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        link(source, target)
        w2c = np.eye(4)
        w2c[:3, :3] = rotation(image["q"])
        w2c[:3, 3] = image["t"]
        c2w = np.linalg.inv(w2c)
        c2w[:3, 1:3] *= -1  # OpenCV -> OpenGL; splatstudio.data reverses this.
        frame = {
            "file_path": f"images/{image['name']}",
            "transform_matrix": c2w.tolist(),
            "w": width,
            "h": height,
        }
        frame.update(intrinsics(camera, width / camera["width"], height / camera["height"]))
        frames.append(frame)
    points, colors = read_points(sparse / "points3D.bin")
    np.savez(output / "points.npz", points=points, colors=colors)
    meta = {
        "frames": frames,
        "source": {"scene": scene.name, "images": images_dir, "colmap": str(sparse.name)},
    }
    if split_protocol:
        meta["split_protocol"] = split_protocol
    (output / "transforms.json").write_text(json.dumps(meta, indent=2) + "\n")
    distorted = sum(f["camera_model"] == "OPENCV" for f in frames)
    return {
        "output": str(output),
        "frames": len(frames),
        "points": len(points),
        "image_size": [frames[0]["w"], frames[0]["h"]],
        "distorted_frames": distorted,
        "next": "run `splatstudio prepare-data` to undistort" if distorted else "ready",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scene", required=True, help="COLMAP scene root containing sparse/")
    p.add_argument("--images", default="images", help="Image folder, e.g. images_4 or images_2")
    p.add_argument("--output", required=True)
    p.add_argument("--split-protocol", default="llffhold8", choices=["llffhold8", "none"])
    args = p.parse_args()
    protocol = None if args.split_protocol == "none" else args.split_protocol
    print(json.dumps(convert(args.scene, args.images, args.output, protocol), indent=2))


if __name__ == "__main__":
    main()
