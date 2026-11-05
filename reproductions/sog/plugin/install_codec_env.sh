#!/bin/bash
# Isolated JPEG XL codec runtime for the SOG plugin.
# The official code pins imagecodecs==2023.9.18 (later versions produce ~4x larger JXL files,
# github.com/fraunhoferhhi/Self-Organizing-Gaussians/issues/3). That wheel is built against
# numpy 1.x and cannot be imported next to the splatstudio env's numpy 2.2.6, so it lives in a
# separate --target directory that ss_sog/jxl_worker.py is run against (subprocess).
set -euo pipefail
DEST="${1:-$(dirname "$(readlink -f "$0")")/../codec_env}"
/home/sejain/splatify/tools/ssenv pip install --no-deps --target "$DEST" imagecodecs==2023.9.18 numpy==1.26.4
echo "codec env: $DEST"
