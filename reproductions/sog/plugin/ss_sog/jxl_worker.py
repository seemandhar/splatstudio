"""JPEG XL write/read worker, executed in the isolated codec env (numpy 1.26 + imagecodecs
2023.9.18, see install_codec_env.sh). Imports only numpy and imagecodecs.

Usage: python -s jxl_worker.py <jobs.json>
jobs.json: {"expect_version": "2023.9.18", "jobs": [{"array": a.npy, "file": a.jxl,
            "level": 101, "decoded": a_dec.npy}, ...]}
Jobs without "array" only decode an existing file. Each array is written with
`imagecodecs.imwrite(file, array, level=level)` (official compression/jpeg_xl.py
JpegXlCodec.encode_image) and read back with `imagecodecs.imread`.
"""

import json
import sys

import imagecodecs
import numpy as np


def main(job_file):
    with open(job_file) as f:
        spec = json.load(f)
    if imagecodecs.__version__ != spec["expect_version"]:
        raise SystemExit(f"imagecodecs {imagecodecs.__version__} != {spec['expect_version']}")
    for job in spec["jobs"]:
        if "array" in job:  # encode, then read back; decode-only jobs omit "array"
            array = np.load(job["array"])
            imagecodecs.imwrite(job["file"], array, level=job["level"])
        np.save(job["decoded"], imagecodecs.imread(job["file"]))
    info = {"imagecodecs": imagecodecs.__version__, "numpy": np.__version__,
            "libjxl": imagecodecs.jpegxl_version()}
    with open(spec["info"], "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    main(sys.argv[1])
