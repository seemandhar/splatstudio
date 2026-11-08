"""Evaluate finished runs on the TRAIN split (reviewer: the official train.sh has no --eval,
so the paper's DTU PSNR may be train-view PSNR). Writes <run>/train.json."""
import json
import sys

from splatstudio import runtime
from splatstudio.registry import load

for run in sys.argv[2:]:
    r = runtime.evaluate(load(sys.argv[1]), run, split="train", device="cuda:0")
    print(json.dumps({"run": run, **r["metrics"]}), flush=True)
