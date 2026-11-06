"""Global GS-IR state (component-owned bank): stage flag, baked occlusion volumes, learnable
cube-map environment light and irradiance volumes, and the light optimizer.

The SplatStudio runtime has no global learned fields, so ``gsir-density`` owns a ``GSIRBank``
and publishes it in ``ACTIVE`` for the renderer and losses. It is exported (CPU tensors) into
the checkpoint's ``density_state["gsir"]`` together with a fingerprint of the Gaussians. A fresh
process (``ss-eval``) restores it from ``<run>/checkpoint.pt``; see ``resolve``.

Official light optimizer (train.py:160-168, 437): one Adam (lr = opacity_lr 0.05, default eps)
over the cube-map base and the irradiance coefficients, stepped from the first decomposition
iteration, then ``cubemap.clamp_(min=0)``.
"""

import os
import sys
from pathlib import Path

import torch

from . import light

ACTIVE = {"bank": None}
_LOADED = {}


def fingerprint(params):
    m = params["means"].detach()
    return {"n": int(m.shape[0]), "means_sum": float(m.double().sum()),
            "means_abs_sum": float(m.double().abs().sum())}


class GSIRBank:
    def __init__(self, cfg, device):
        self.cfg = dict(cfg)
        self.device = torch.device(device)
        self.stage = 1
        self.base = None  # [6, R, R, 3] learnable
        self.irradiance = None  # [G, G, G, d2, 1] learnable
        self.irr_aabb = None
        self.occ_ids = None
        self.occ_coeffs = None
        self.occ_bound = float(cfg["occ_bound"])
        self.occ_bands = int(cfg["occ_bands"])
        self.optimizer = None
        self.light_steps = 0
        self.bake_stats = None
        self._mips = None
        self._mips_version = None

    # ---------------------------------------------------------------- stage switch
    def start_decomposition(self, baked, seed=0):
        c = self.cfg
        dev = self.device
        g = torch.Generator(device=dev).manual_seed(int(seed))
        self.base = light.init_base(int(c["cube_res"]), g, dev).requires_grad_(True)
        r, b = int(c["irr_res"]), int(c["irr_bands"])
        self.irradiance = torch.zeros(r, r, r, b * b, 1, device=dev, requires_grad=True)
        ib = float(c["irr_bound"])
        self.irr_aabb = torch.tensor([-ib] * 3 + [ib] * 3, device=dev)
        if baked is not None:
            self.occ_ids = baked["ids"].to(dev)
            self.occ_coeffs = baked["coefficients"].to(dev)
            self.bake_stats = baked.get("stats")
        lr = float(c["light_lr"])
        self.optimizer = torch.optim.Adam([self.base, self.irradiance], lr=lr)
        self.stage = 2

    @torch.no_grad()
    def step_light(self):
        if any(p.grad is not None for p in (self.base, self.irradiance)):
            self.optimizer.step()
            self.light_steps += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.base.clamp_(min=0.0)

    # ---------------------------------------------------------------- shading inputs
    def mips(self, training):
        """(specular levels, diffuse cube); rebuilt every training call (official build_mips per
        iteration), cached for evaluation while the base is unchanged."""
        if training:
            return light.build_mips(self.base)
        version = (self.base._version, self.light_steps)
        if self._mips is None or self._mips_version != version:
            with torch.no_grad():
                self._mips = light.build_mips(self.base.detach())
            self._mips_version = version
        return self._mips

    @property
    def indirect(self):
        return bool(self.cfg.get("indirect", True)) and self.occ_ids is not None

    # ---------------------------------------------------------------- persistence
    def export(self, params=None):
        out = {"stage": int(self.stage), "light_steps": int(self.light_steps),
               "cfg": {k: v for k, v in self.cfg.items() if isinstance(v, (int, float, bool))}}
        if params is not None:
            out["fingerprint"] = fingerprint(params)
        if self.stage == 2:
            out["base"] = self.base.detach().cpu().clone()
            out["irradiance"] = self.irradiance.detach().cpu().clone()
            out["irr_aabb"] = self.irr_aabb.cpu().clone()
            if self.occ_ids is not None:
                out["occ_ids"] = self.occ_ids.cpu().clone()
                out["occ_coeffs"] = self.occ_coeffs.cpu().clone()
            if self.bake_stats:
                out["bake_stats"] = {k: v for k, v in self.bake_stats.items()
                                     if isinstance(v, (int, float))}
        return out

    @classmethod
    def from_export(cls, data, cfg, device):
        bank = cls({**cfg, **data.get("cfg", {})}, device)
        bank.stage = int(data["stage"])
        bank.light_steps = int(data.get("light_steps", 0))
        bank.fp = data.get("fingerprint")
        if bank.stage == 2:
            bank.base = data["base"].to(device)
            bank.irradiance = data["irradiance"].to(device)
            bank.irr_aabb = data["irr_aabb"].to(device)
            if "occ_ids" in data:
                bank.occ_ids = data["occ_ids"].to(device)
                bank.occ_coeffs = data["occ_coeffs"].to(device)
        return bank


def _cli_run():
    """--run of an ``ss-eval`` / ``splatstudio eval`` command line, or env GSIR_RUN."""
    env = os.environ.get("GSIR_RUN")
    if env:
        return Path(env)
    argv = sys.argv
    prog = os.path.basename(argv[0]) if argv else ""
    if not (prog == "ss-eval" or (len(argv) > 1 and argv[1] == "eval")):
        return None
    for i, a in enumerate(argv):
        if a == "--run" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith("--run="):
            return Path(a.split("=", 1)[1])
    return None


def resolve(params, cfg):
    """Bank for rendering ``params``: the live training bank, else the bank stored in the
    checkpoint of the run being evaluated. Returns None when neither exists (stage-1 render,
    e.g. probes); raises if an evaluated run lacks GS-IR state or does not match."""
    bank = ACTIVE["bank"]
    if bank is not None:
        return bank
    run = _cli_run()
    if run is None:
        return None
    device = params["means"].device
    key = (str(run.resolve()), str(device))
    if key not in _LOADED:
        ckpt = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
        state = (ckpt.get("density_state") or {}).get("gsir")
        if state is None:
            raise RuntimeError(f"{run}: checkpoint has no GS-IR bank (density_state['gsir'])")
        _LOADED[key] = GSIRBank.from_export(state, cfg, device)
    bank = _LOADED[key]
    fp = getattr(bank, "fp", None)
    if fp is not None:
        now = fingerprint(params)
        tol = 1e-6 * max(1.0, fp["means_abs_sum"])
        if now["n"] != fp["n"] or abs(now["means_sum"] - fp["means_sum"]) > tol:
            raise RuntimeError("GS-IR bank fingerprint does not match the evaluated Gaussians")
    return bank
