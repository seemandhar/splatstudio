"""Per-surfel texture maps (paper §3.4; official texture_manager.init_texture + train_texture.py).

The runtime has no global learned fields, so the flat texture buffer [T,3] and its Adam live in
the module-level ``STORE`` shared by the density component (owner, optimiser, stage switch) and
the renderer. The density exposes the buffer in its ``state`` dict, which the runtime saves as
``checkpoint.pt["density_state"]``; a fresh ``ss-eval`` process reloads it from ``--run``.
Offsets are a deterministic function of the (frozen) surfel scales, recomputed on load and
checked against the stored key.
"""

import math
import sys
from pathlib import Path

import torch

STORE = {"texture": None, "index": None, "optimizer": None, "key": None}
FLAG = "hdgs_textured"


def reset_store():
    for k in STORE:
        STORE[k] = None


def texture_index(scales2, tex_res, max_tex):
    """init_texture: W = ceil(s_u T), H = ceil(s_v T); if max(W,H) > max_tex both sides are
    rescaled by max_tex // max (integer division, as official). Sides are clamped to >= 1 (the
    official code can produce a zero side for > max_tex:1 surfels, whose bilinear lookup would
    then read the previous surfel's texel). Returns int32 [N,3] = (start, W, H) and T."""
    s = scales2.detach().float() * float(tex_res)
    W = torch.ceil(s[:, 0]).long()
    H = torch.ceil(s[:, 1]).long()
    m = torch.maximum(W, H)
    over = m > max_tex
    W[over] = W[over] * max_tex // m[over]
    H[over] = H[over] * max_tex // m[over]
    W.clamp_(min=1)
    H.clamp_(min=1)
    area = W * H
    end = torch.cumsum(area, 0)
    total = int(end[-1])
    if total >= 2**31:
        raise RuntimeError(f"texture buffer of {total} texels exceeds the kernel's int32 offsets")
    start = end - area
    return torch.stack([start, W, H], dim=1).int().contiguous(), total


def initial_texture(sh0, index, total):
    """Every texel of surfel i starts at its SH DC coefficient (official init; the paper says
    'background colour', the code overwrites it with features_dc)."""
    area = (index[:, 1].long() * index[:, 2].long())
    return sh0[:, 0, :].detach().repeat_interleave(area, dim=0, output_size=total).contiguous()


def store_key(params):
    """(N, sum of log scales[:, :2]) identifies the frozen geometry the offsets came from."""
    s = params["scales"].detach()[:, :2].double()
    return torch.tensor([float(len(s)), float(s.sum())], dtype=torch.float64)


def tex_lr(stage2_step, lr, step_size, gamma):
    """torch StepLR(step_size, gamma) stepped once per iteration after the optimiser step:
    stage-2 iteration t (1-based) uses lr * gamma ** ((t - 1) // step_size)."""
    return lr * gamma ** ((stage2_step - 1) // step_size)


def _run_from_argv():
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--run" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if a.startswith("--run="):
            return Path(a.split("=", 1)[1])
    return None


def ensure_loaded(params, tex_res, max_tex):
    """Return (texture, index) for textured params, loading the checkpoint's density_state in a
    fresh evaluation process. Raises if the texture cannot be found (no untextured fallback)."""
    key = store_key(params)
    if STORE["texture"] is not None and STORE["key"] is not None and torch.allclose(
            STORE["key"], key, rtol=0, atol=1e-6 * max(1.0, abs(float(key[1])))):
        return STORE["texture"], STORE["index"]
    run = _run_from_argv()
    if run is None or not (run / "checkpoint.pt").exists():
        raise RuntimeError("HDGS params are textured (stage 2) but no texture is loaded and no "
                           "--run <dir>/checkpoint.pt is available to load it from")
    dev = params["means"].device
    ck = torch.load(run / "checkpoint.pt", map_location=dev, weights_only=True)
    st = ck.get("density_state") or {}
    if "texture" not in st:
        raise RuntimeError(f"{run}/checkpoint.pt has no HDGS texture in density_state")
    saved = st["texture_key"].double().cpu()
    if not torch.allclose(saved, key, rtol=0, atol=1e-6 * max(1.0, abs(float(key[1])))):
        raise RuntimeError("HDGS texture key does not match the evaluated params")
    index, total = texture_index(params["scales"].detach()[:, :2].exp(), tex_res, max_tex)
    if total != st["texture"].shape[0] or not math.isclose(float(st["tex_res"]), tex_res):
        raise RuntimeError("HDGS texture layout differs from the method configuration")
    STORE.update(texture=st["texture"].float().contiguous(), index=index, key=key,
                 optimizer=None)
    return STORE["texture"], STORE["index"]
