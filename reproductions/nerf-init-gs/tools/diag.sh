#!/bin/bash
# usage: diag.sh <scene> <gpu> <mode: nerf|base>
set -e
S=$1; G=$2; M=$3; T=/home/sejain/splatify/tools/ssenv; W=/home/sejain/splatify/reproductions/nerf-init-gs
H=~/datasets/nerfinit-gs/holdout; D=$W/evidence/diag; export CUDA_VISIBLE_DEVICES=$G
run() { # method data name
  $T ss-train $1 --data $2 --output $D/$3 --steps 7000 --recipe $4 --num-gaussians 10000000 \
     --max-gaussians 12000000 --log-every 250 --device cuda:0 > $D/$3.log 2>&1
  $T ss-eval --run $D/$3 --split test --device cuda:0 > /dev/null 2>&1
  $T python $W/tools/train_view_eval.py $D/$3 >> $D/$3.log 2>&1
  echo "$3 done"
}
if [ $M = base ]; then
  run gsplat $H/$S-sfm $S-gsplat 3dgs
else
  [ -f $H/$S/nerfinit/nerfinit.json ] || $T ss-nerfinit-prep --source $H/$S-sfm --output $H/$S --device cuda:0 > $D/prep-$S.log 2>&1
  for m in nerf-init-gs-nodepth nerf-init-gs nerf-init-gs-nerfunits; do run $m $H/$S $S-$m splatstudio; done
fi
