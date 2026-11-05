"""CoherentGS density/state controller (no densification or pruning; official train.py).

initialize():
  * loads the dataset sidecar ``coherentgs.npz`` (tools/prepare_cgs_llff.py) and builds the Bank;
  * checks that params are the per-pixel points of points.npz in camera-major/row-major order
    (ss-train must keep every point: --num-gaussians >= N*H*W);
  * coarse alignment of the monocular depth (Sec. 4.3, Eq. 8; 1000 Adam its, official its
    6001-7000) -> D_init; isotropic radius from D_init (Eq. 9, set_scaling()); identity
    rotation; opacity logit(alpha0); means g(D_init);
  * base-field learning rates of the official training_setup: shN 5e-5 (feature_lr/10),
    scales 1e-4, quats 1e-5; means/opacities/sh0 are never optimised directly;
  * builds the two decoders (AdamW + CyclicLR).
after_backward(step): decoder steps (skipped on the final step, as the official
`if iteration < opt.iterations`), then the radius reset from the current depth every 100 official
iterations while < 15000 (applied before the next render), then advances the bank step.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

from . import bank as bank_mod
from .bank import DEPTH_NGF, INIT_OPACITY, OPAC_NGF, Bank

DEFAULTS = {
    "n_views": 3, "mask_channels": 5, "ngf_depth": DEPTH_NGF[3], "ngf_opac": OPAC_NGF[3],
    "init_opacity": INIT_OPACITY[3], "radius_mult": 1.2, "depth_scale": 50.0, "opac_scale": 100.0,
    "align_steps": 1000, "align_lr_scale": 1.0, "align_lr_offset": 5.0,
    "depth_lr_base": 1e-6, "depth_lr_max": 1e-4, "depth_lr_step": 1000,
    "opac_lr_base": 1e-6, "opac_lr_max": 1e-5, "opac_lr_step": 2000,
    "lr_shN": 0.0005 / 10, "lr_scales": 0.0001, "lr_quats": 0.00001,
    "reset_every": 100, "reset_stop_iteration": 15000, "iteration_offset": 7000,
    "total_steps": 13000, "sh_steps": [8000, 9000, 10000], "multisample_start": 5000,
    "multisample_max": 4, "flow_mask_threshold": 0.5, "seed": 0, "sidecar": "coherentgs.npz",
}


def data_root():
    if os.environ.get("COHERENTGS_DATA"):
        return Path(os.environ["COHERENTGS_DATA"])
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--data" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith("--data="):
            return Path(a.split("=", 1)[1])
    raise RuntimeError("CoherentGS: set COHERENTGS_DATA or run through `ss-train --data`")


def load_sidecar(root, name):
    path = Path(root) / name
    if not path.exists():
        raise FileNotFoundError(f"CoherentGS sidecar {path} not found (build it with "
                                "reproductions/coherentgs/tools/prepare_cgs_llff.py)")
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in ("viewmats", "K", "z_mono", "dep_mask", "flows", "flow_w")}


class Density:
    requires = frozenset()

    def __init__(self, spec):
        unknown = set(spec.config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown density options: {sorted(unknown)}")
        self.cfg = {**DEFAULTS, **spec.config}
        self.bank = None
        self.align_history = []

    @property
    def state(self):
        return None if self.bank is None else self.bank.state_dict()

    def initialize(self, params, optimizers, scene_scale=1.0, sidecar=None):
        cfg = self.cfg
        data = sidecar if sidecar is not None else load_sidecar(data_root(), cfg["sidecar"])
        bank = Bank(data["viewmats"], data["K"], data["z_mono"], data["dep_mask"], data["flows"],
                    data["flow_w"], cfg, params["means"].device)
        n = bank.n * bank.hw
        if len(params["means"]) != n:
            raise ValueError(f"CoherentGS needs one Gaussian per training pixel ({n}); got "
                             f"{len(params['means'])} (pass --num-gaussians >= {n})")
        expected = bank.xyz(bank.z_mono)
        err = (params["means"].detach() - expected).abs().max() / expected.abs().max()
        if float(err) > 1e-4:
            raise ValueError(f"params are not the per-pixel points of the sidecar (rel err {err})")
        self.align_history = bank.align()
        with torch.no_grad():
            params["means"].copy_(bank.xyz(bank.z))
            params["scales"].copy_(bank.log_scales(use_decoder=False))
            params["quats"].zero_()
            params["quats"][:, 0] = 1
            params["opacities"].fill_(bank.opac0_logit)
        for name, key in (("shN", "lr_shN"), ("scales", "lr_scales"), ("quats", "lr_quats")):
            for group in optimizers[name].param_groups:
                group["lr"] = float(cfg[key])
        for name in ("means", "opacities", "sh0"):  # derived / fixed (no gradient either)
            for group in optimizers[name].param_groups:
                group["lr"] = 0.0
        bank.build_decoders(cfg["seed"])
        bank.step = 1
        self.bank = bank
        bank_mod.ACTIVE["bank"] = bank

    def before_backward(self, params, optimizers, step, outputs):
        if self.bank.step != step:
            raise RuntimeError(f"CoherentGS bank step {self.bank.step} != runtime step {step}")

    def after_backward(self, params, optimizers, step, outputs):
        bank, cfg = self.bank, self.cfg
        if step < int(cfg["total_steps"]):
            bank.step_decoders()
        else:
            bank.zero_decoder_grads()
        nxt = step + 1
        it = bank.official_iteration(nxt)
        if it % int(cfg["reset_every"]) == 0 and it < int(cfg["reset_stop_iteration"]):
            with torch.no_grad():
                params["scales"].copy_(bank.log_scales(use_decoder=True))
        bank.step = nxt


def create_density(spec):
    return Density(spec)
