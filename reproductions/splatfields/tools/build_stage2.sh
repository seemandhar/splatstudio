#!/usr/bin/env bash
# Build blender-fewview<N>-sf scenes from a finished stage-1 bench experiment.
# Usage: build_stage2.sh <stage1-exp> <N> [scenes...]   (default scenes: chair hotdog ficus)
# Stage-1 runs are runs/<exp>/splatfields-3dgs/blender-fewview<N>-<scene> (bench.py layout).
set -euo pipefail
ROOT=/home/sejain/splatify
EXP=$1; N=$2; shift 2
SCENES=${*:-chair hotdog ficus}
for s in $SCENES; do
  "$ROOT/tools/ssenv" python -I "$ROOT/reproductions/splatfields/tools/make_stage2.py" \
    --stage1-data "$HOME/datasets/nerf_synthetic/fewview$N/$s" \
    --run "$ROOT/runs/$EXP/splatfields-3dgs/blender-fewview$N-$s" \
    --raw "$HOME/datasets/nerf_synthetic/raw" --out "$HOME/datasets/nerf_synthetic/fewview$N-sf"
done
