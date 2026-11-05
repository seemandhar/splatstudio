#!/bin/bash
# Build native scenes + dnsplat sidecars (copied DN-Splatter tool) + AGS-Mesh DNC masks for the
# 6 MuSHRoom rooms of the paper's NVS table. Usage: prepare_rooms.sh <gpu> room...
G=$1; shift
S=/home/sejain/splatify; T=$S/tools/ssenv
export PYTHONPATH=$HOME/models/dtueval-pylib:$HOME/models/dpt/pylib:$HOME/models/omnidata/pylib
for r in "$@"; do
  while [ ! -d $HOME/datasets/mushroom/raw/${r}_x/room_datasets ]; do sleep 30; done
  sleep 5
  echo "== $r $(date)"
  CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=16 $T python $S/reproductions/ags-mesh/tools/prepare_mushroom.py --room $r 2>&1 | grep -v "conda\|WARN" | grep -v "^omnidata [1-9]\|^tsdf [1-9]"
  OMP_NUM_THREADS=4 $T python $S/reproductions/ags-mesh/tools/prepare_dnc.py --room $r --workers 24 2>&1 | grep -v "conda\|WARN\|^dnc"
done
