"""SOG quantization + JPEG XL attribute-grid codec (paper §3.4), with decoding.

Numerics copied from the official compression/compression_exp.py (`compress_attr`,
`decompress_attr`, `run_single_compression`) and compression/codec.py (thresholds,
`normalize_img`, `decode_with_normalization`); experiment `jxl_quant_sh` of
config/compression/umbrella_sh.yaml. Official code: Inria/MPII Gaussian-Splatting license
(LICENSE-SOG-official.md), (c) Fraunhofer HHI authors.

JPEG XL I/O runs in a subprocess (jxl_worker.py) against an isolated codec env with the
official pin imagecodecs==2023.9.18 (needs numpy<2; see install_codec_env.sh).
Size = sum of all files of the compressed directory (*.jxl + compression_info.csv +
compression_config.yml), as official compression/decompress.py:get_size_of_files_in_dir.
"""

import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

IMAGECODECS_VERSION = "2023.9.18"
DEFAULT_CODEC_ENV = Path(__file__).resolve().parents[2] / "codec_env"
WORKER = Path(__file__).resolve().parent / "jxl_worker.py"

# official config/compression/umbrella_sh.yaml, experiment "jxl_quant_sh" (attribute order kept)
EXPERIMENT = {
    "name": "jxl_quant_sh",
    "attributes": [
        {"name": "_xyz", "method": "jpeg-xl", "normalize": True, "quantize": 14,
         "params": {"level": 101}},
        {"name": "_features_dc", "method": "jpeg-xl", "normalize": True,
         "params": {"level": 100}},
        {"name": "_features_rest", "method": "jpeg-xl", "normalize": True, "quantize": 5,
         "params": {"level": 101}},
        {"name": "_scaling", "method": "jpeg-xl", "normalize": False, "contract": False,
         "quantize": 6, "params": {"level": 101}},
        {"name": "_rotation", "method": "jpeg-xl", "normalize": True, "quantize": 6,
         "params": {"level": 101}},
        {"name": "_opacity", "method": "jpeg-xl", "normalize": True, "contract": False,
         "quantize": 6, "params": {"level": 101}},
    ],
}

# official attribute -> SplatStudio field (same memory layout: [N,3], [N,1,3], [N,15,3], ...)
ATTR_TO_FIELD = {"_xyz": "means", "_features_dc": "sh0", "_features_rest": "shN",
                 "_scaling": "scales", "_rotation": "quats", "_opacity": "opacities"}

# official compression/codec.py
MIN_THRESHOLDS = {"_features_dc": -2, "_features_rest": -1, "_scaling": -13, "_rotation": -1,
                  "_opacity": -6}
MAX_THRESHOLDS = {"_features_dc": 4, "_features_rest": 1, "_scaling": 3, "_rotation": 2,
                  "_opacity": 12}


def resolve_codec_env(path=None):
    env = Path(path or os.environ.get("SOG_CODEC_ENV") or DEFAULT_CODEC_ENV)
    if not (env / "imagecodecs").is_dir() or not (env / "numpy").is_dir():
        raise RuntimeError(
            f"SOG codec env missing at {env}: run plugin/install_codec_env.sh "
            "(imagecodecs==2023.9.18 + numpy==1.26.4 in an isolated --target dir)")
    return env


def grid_numpy(tensor, side):
    """Official attr_as_grid_img: tensor.reshape(S, S, -1), float32 numpy."""
    return tensor.detach().reshape(side, side, -1).float().cpu().numpy()


def quantize(attr_np, bits):
    """Official compress_attr quantization (min-max, round(x * 2^q) / 2^q)."""
    min_val = attr_np.min()
    max_val = attr_np.max()
    val_range = max_val - min_val
    if val_range == 0:
        val_range = 1
    attr_np_norm = (attr_np - min_val) / (val_range)
    qpow = 2 ** bits
    attr_np_quantized = np.round(attr_np_norm * qpow) / qpow
    return (attr_np_quantized * (val_range) + min_val).astype(np.float32)


def normalize_img(img, min_val, max_val):
    """Official codec.normalize_img."""
    img = img.clip(min_val, max_val)
    return (img - min_val) / (max_val - min_val)


def normalize_to_thresholds(img, attr_name):
    """Official Codec.normalize_to_thresholds: own min/max for _xyz, fixed ranges otherwise."""
    if attr_name == "_xyz":
        lo, hi = img.min(), img.max()
    else:
        lo, hi = MIN_THRESHOLDS[attr_name], MAX_THRESHOLDS[attr_name]
    return normalize_img(img, lo, hi), lo, hi


def prepare(attr_config, attr_np):
    """Everything compress_attr does before the codec call -> (image, min, max)."""
    if attr_config.get("contract", False):
        raise ValueError("contract=True is not used by jxl_quant_sh")
    if "quantize" in attr_config:
        attr_np = quantize(attr_np, attr_config["quantize"])
    if attr_config.get("normalize", False):
        return normalize_to_thresholds(attr_np, attr_config["name"])
    return attr_np, None, None


def csv_cell(v):
    # pandas writes object cells with str(); None -> empty
    return "" if v is None else str(v)


def compression_info_csv(rows):
    """Official compression_info.csv: DataFrame([min, max, file], index=[...]).T.to_csv()."""
    lines = [",min,max,file"]
    lines += [f"{name},{csv_cell(lo)},{csv_cell(hi)},{f}" for name, lo, hi, f in rows]
    return "\n".join(lines) + "\n"


def csv_float(v):
    """What pandas.read_csv returns for a written cell (float64), as decompress uses it."""
    return None if v is None else float(str(v))


def run_worker(jobs, workdir, codec_env):
    job_file = Path(workdir) / "jobs.json"
    info_file = Path(workdir) / "worker_info.json"
    job_file.write_text(json.dumps({"expect_version": IMAGECODECS_VERSION, "jobs": jobs,
                                    "info": str(info_file)}))
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONPATH"] = str(codec_env)
    proc = subprocess.run([sys.executable, "-s", str(WORKER), str(job_file)], cwd=workdir,
                          env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"JPEG XL worker failed:\n{proc.stdout}\n{proc.stderr}")
    return json.loads(info_file.read_text())


@torch.no_grad()
def encode_decode(params, active_sh_degree=3, max_sh_degree=3, codec_env=None,
                  experiment=EXPERIMENT, out_dir=None):
    """Compress the square-grid params, decode them back.

    Returns (decoded {field: tensor shaped like params[field]}, report dict, files {name: bytes}).
    """
    codec_env = resolve_codec_env(codec_env)
    n = len(params["means"])
    side = int(round(n ** 0.5))
    if side * side != n:
        raise ValueError("encode needs a square grid (prune to square first)")
    with tempfile.TemporaryDirectory(prefix="sog-codec-") as tmp:
        folder = Path(out_dir) if out_dir is not None else Path(tmp) / experiment["name"]
        folder.mkdir(parents=True, exist_ok=True)
        jobs, rows, ranges = [], [], {}
        for attr in experiment["attributes"]:
            name = attr["name"]
            if attr["method"] != "jpeg-xl":
                raise ValueError("only the jpeg-xl codec of jxl_quant_sh is implemented")
            img, lo, hi = prepare(attr, grid_numpy(params[ATTR_TO_FIELD[name]], side))
            file_name = f"{name}.jxl"
            np.save(Path(tmp) / f"{name}.npy", np.ascontiguousarray(img, dtype=np.float32))
            jobs.append({"array": str(Path(tmp) / f"{name}.npy"), "file": str(folder / file_name),
                         "level": attr["params"]["level"],
                         "decoded": str(Path(tmp) / f"{name}_dec.npy")})
            rows.append((name, lo, hi, file_name))
            ranges[name] = (lo, hi)
        info = run_worker(jobs, tmp, codec_env)
        (folder / "compression_info.csv").write_text(compression_info_csv(rows))
        config = copy.deepcopy(experiment)
        config["max_sh_degree"] = max_sh_degree
        config["active_sh_degree"] = active_sh_degree
        config["disable_xyz_log_activation"] = False
        with open(folder / "compression_config.yml", "w") as stream:
            yaml.dump(config, stream)

        decoded = {}
        for attr in experiment["attributes"]:
            name = attr["name"]
            img = np.load(Path(tmp) / f"{name}_dec.npy")
            if attr.get("normalize", False):  # official decode_with_normalization
                lo, hi = (csv_float(v) for v in ranges[name])
                img = img * (hi - lo) + lo
            field = ATTR_TO_FIELD[name]
            ref = params[field]
            decoded[field] = torch.tensor(img.reshape(ref.shape), dtype=torch.float,
                                          device=ref.device)
        files = {p.name: p.read_bytes() for p in sorted(folder.iterdir()) if p.is_file()}
    size = sum(len(b) for b in files.values())
    report = {
        "experiment": experiment["name"],
        "grid_side": side,
        "gaussians": n,
        "size_bytes": size,
        "size_mb": size / 1e6,
        "file_bytes": {k: len(v) for k, v in files.items()},
        "codec": info,
    }
    return decoded, report, files
