"""FreGS frequency regularization (arXiv 2403.06908, Sec. 3.2-3.3, Eq. 2-6) as a loss component.

L_f = w_l (d_la + d_lp)                       , 0 < t <= T0
L_f = w_l (d_la + d_lp) + w_h (d_ha + d_hp)   , T0 < t            (Sec. 3.3)
L_f = 0                                       , t > stop_iter     (Sec. 4.1)
With anneal=False (Base+FR ablation) both bands are active from step 1 with the same per-band
weights and no annealing: L_f = w_l (d_la + d_lp) + w_h (d_a + d_p over r > D0). With w_l = w_h this
is the paper's unfiltered w (d_a + d_p); keeping fregs' w_l/w_h makes the annealing schedule the
only difference between fregs-fr and fregs (deconfounded ablation, see README).
"""

import math

import torch

KEYS = (
    "w_l",
    "w_h",
    "d0_frac",
    "t0",
    "t_end",
    "stop_iter",
    "anneal",
    "mask",
    "radius_units",
    "channels",
    "phase",
    "phase_eps",
    "normalization",
)
PHASES = ("wrapped", "raw", "arctan")
UNITS = ("index", "normalized")


def parse_config(config):
    """Validate a fregs-frequency config; every key is required and unknown keys raise."""
    missing = set(KEYS) - set(config)
    unknown = set(config) - set(KEYS)
    if missing or unknown:
        raise ValueError(
            f"fregs-frequency config: missing {sorted(missing)}, unknown {sorted(unknown)}"
        )
    cfg = {
        "w_l": float(config["w_l"]),
        "w_h": float(config["w_h"]),
        "d0_frac": float(config["d0_frac"]),
        "t0": int(config["t0"]),
        "t_end": int(config["t_end"]),
        "stop_iter": int(config["stop_iter"]),
        "anneal": bool(config["anneal"]),
        "mask": str(config["mask"]),
        "radius_units": str(config["radius_units"]),
        "channels": str(config["channels"]),
        "phase": str(config["phase"]),
        "phase_eps": float(config["phase_eps"]),
        "normalization": str(config["normalization"]),
    }
    if cfg["w_l"] < 0 or cfg["w_h"] < 0 or not all(map(math.isfinite, (cfg["w_l"], cfg["w_h"]))):
        raise ValueError("w_l and w_h must be finite and nonnegative")
    if not 0 < cfg["d0_frac"] < 1:
        raise ValueError("d0_frac must be in (0, 1)")
    if not 0 <= cfg["t0"] < cfg["t_end"]:
        raise ValueError("Require 0 <= t0 < t_end")
    if cfg["stop_iter"] < 1:
        raise ValueError("stop_iter must be positive")
    if cfg["mask"] != "ideal_circular":
        raise ValueError("Only mask='ideal_circular' is implemented")
    if cfg["radius_units"] not in UNITS:
        raise ValueError(f"radius_units must be one of {UNITS}")
    if cfg["channels"] != "per_channel_sum":
        raise ValueError("Only channels='per_channel_sum' is implemented")
    if cfg["phase"] not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    if not cfg["phase_eps"] >= 0:
        raise ValueError("phase_eps must be nonnegative")
    if cfg["normalization"] != "paper":
        raise ValueError("Only normalization='paper' (1/sqrt(HW) * sum) is implemented")
    return cfg


_GRIDS = {}


def radius_grid(h, w, device, units="index"):
    """Distance of each fftshifted bin from the spectrum centre (H//2, W//2); cached.

    Returns (r [H,W] float64, D = max radius, i.e. "max range of the spectrum").
    """
    key = (h, w, str(device), units)
    if key not in _GRIDS:
        u = torch.arange(h, device=device, dtype=torch.float64) - h // 2
        v = torch.arange(w, device=device, dtype=torch.float64) - w // 2
        if units == "normalized":
            u, v = u / (h / 2), v / (w / 2)
        r = torch.sqrt(u[:, None] ** 2 + v[None, :] ** 2)
        _GRIDS[key] = (r, float(r.max()))
    return _GRIDS[key]


def band_radius(t, cfg, D):
    """Sec. 3.3: D_t = (t - T0)(D - D0)/(T - T0) + D0 (bound taken as equality), in [D0, D]."""
    d0 = cfg["d0_frac"] * D
    if t <= cfg["t0"]:
        return d0
    value = (t - cfg["t0"]) * (D - d0) / (cfg["t_end"] - cfg["t0"]) + d0
    return min(max(value, d0), D)


def low_mask(r, D, cfg):
    """Ideal low-pass H_l: r <= D0."""
    return r <= cfg["d0_frac"] * D


def high_mask(r, D, cfg, t):
    """Ideal dynamic high-pass H_h(t): D0 < r <= D_t (empty for t <= T0); r > D0 without
    annealing."""
    if not cfg["anneal"]:
        return r > cfg["d0_frac"] * D
    return (r > cfg["d0_frac"] * D) & (r <= band_radius(t, cfg, D))


_REAL_BINS = {}


def _real_bins(h, w, device):
    """Self-conjugate bins (u, v in {0, Nyquist}) whose DFT of a real image is exactly real."""
    key = (h, w, str(device))
    if key not in _REAL_BINS:
        rows = [h // 2] + ([0] if h % 2 == 0 else [])  # shifted index of u = 0 and u = -H/2
        cols = [w // 2] + ([0] if w % 2 == 0 else [])
        mask = torch.ones(h, w, device=device)
        for i in rows:
            for j in cols:
                mask[i, j] = 0
        _REAL_BINS[key] = mask
    return _REAL_BINS[key]


def spectrum(image):
    """Eq. 2 per channel: unnormalized DFT of an HWC image, centred with fftshift -> [C,H,W].

    The imaginary part of the self-conjugate bins is set to its exact value 0: float32 FFTs
    (cuFFT) leave +-1e-7 there, which flips the raw atan2 phase of negative bins by 2*pi.
    """
    x = image.permute(2, 0, 1)
    f = torch.fft.fftshift(torch.fft.fft2(x, norm="backward"), dim=(-2, -1))
    h, w = f.shape[-2:]
    return torch.complex(f.real, f.imag * _real_bins(h, w, f.device).to(f.real.dtype))


def _parts(z):
    # "+ 0.0" maps a signed zero imaginary part (exactly real bins) to +0, so atan2 is
    # deterministic (+pi, not -pi) for real negative bins.
    return z.real, z.imag + 0.0


def _amplitude(re, im):
    # |F| = sqrt(Re^2 + Im^2) (Eq. 3); tiny offset keeps the gradient finite at |F| = 0.
    return torch.sqrt(re * re + im * im + 1e-20)


def _phase(re, im, mode):
    if mode == "arctan":
        # arctan(Im/Re) written without division: identical for Re != 0, limit +-pi/2 at Re = 0.
        return torch.atan2(im * torch.sign(re), re.abs())
    return torch.atan2(im, re)  # Eq. 3, read as atan2 (= torch.angle)


def discrepancy(rendered, target, mask, cfg):
    """Eq. 5/6 on the masked bins: (d_a, d_p), each (1/sqrt(HW)) * sum over channels and bins.

    `rendered`, `target` are [C,H,W] complex spectra; bins outside the mask are excluded
    (they are exactly 0 for both filtered spectra and contribute 0).
    """
    h, w = rendered.shape[-2:]
    scale = 1.0 / math.sqrt(h * w)
    a_re, a_im = _parts(rendered[:, mask])
    b_re, b_im = _parts(target[:, mask])
    amp_a, amp_b = _amplitude(a_re, a_im), _amplitude(b_re, b_im)
    d_a = scale * (amp_a - amp_b).abs().sum()  # Eq. 5
    # Phase gradient is 1/|F|: exclude bins with near-zero rendered amplitude (phase_eps).
    floor = cfg["phase_eps"] * math.sqrt(h * w)
    keep = amp_a.detach() >= floor
    if cfg["phase"] == "wrapped":
        # Eq. 6 with the angle difference wrapped to (-pi, pi]: angle(F_hat * conj(F)).
        # Also needs |F| > 0 (target), where the angle of the product is defined.
        keep &= amp_b >= floor
        a_re, a_im, b_re, b_im = a_re[keep], a_im[keep], b_re[keep], b_im[keep]
        diff = torch.atan2(a_im * b_re - a_re * b_im, a_re * b_re + a_im * b_im)
    else:
        if cfg["phase"] == "arctan":
            keep &= a_re.detach().abs() >= floor
        diff = _phase(a_re[keep], a_im[keep], cfg["phase"]) - _phase(
            b_re[keep], b_im[keep], cfg["phase"]
        )  # 'raw': plain difference of atan2 angles, no wrap (literal Eq. 6)
    d_p = scale * diff.abs().sum()  # Eq. 6
    return d_a, d_p


def frequency_components(rgb, target, t, cfg):
    """Unweighted Eq. 5/6 discrepancies at step t, amplitude and phase separately:
    {'low': (d_la, d_lp), 'high': (d_ha, d_hp)}; 'high' is present for t > T0 (annealing) or
    always (anneal=False, band r > D0). The target spectrum carries no gradient."""
    rendered = spectrum(rgb)
    with torch.no_grad():
        reference = spectrum(target.detach().to(rgb.dtype))
    h, w = rgb.shape[:2]
    r, D = radius_grid(h, w, rgb.device, cfg["radius_units"])
    parts = {"low": discrepancy(rendered, reference, low_mask(r, D, cfg), cfg)}
    if t > cfg["t0"] or not cfg["anneal"]:
        parts["high"] = discrepancy(rendered, reference, high_mask(r, D, cfg, t), cfg)
    return parts


def frequency_terms(rgb, target, t, cfg):
    """Unweighted band sums: {'low': d_la + d_lp, 'high': d_ha + d_hp}."""
    return {k: a + p for k, (a, p) in frequency_components(rgb, target, t, cfg).items()}


def frequency_loss(rgb, target, t, cfg):
    """Weighted L_f at one-based step t (Sec. 3.3 piecewise loss, Sec. 4.1 stop)."""
    if t > cfg["stop_iter"]:
        return rgb.sum() * 0  # stays attached to the graph (runtime check_loss)
    terms = frequency_terms(rgb, target, t, cfg)
    value = cfg["w_l"] * terms["low"]
    if "high" in terms:
        value = value + cfg["w_h"] * terms["high"]
    return value


def create_frequency_loss(spec):
    cfg = parse_config(spec.config)

    def loss(params, outputs, target):
        # Unclamped, white-composited render: the image the photometric loss sees.
        return frequency_loss(outputs["rgb"], target, int(outputs["step"]), cfg)

    return loss
