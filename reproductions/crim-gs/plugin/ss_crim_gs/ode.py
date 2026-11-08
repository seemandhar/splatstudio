"""Adaptive Dormand-Prince 5(4) ODE solver (dopri5 / RK45) in plain torch.

CRiM-GS (Sec. 3.3) uses "the Dormand-Prince-Shampine fifth-order Runge-Kutta solver ...
following [Neural ODE]", i.e. torchdiffeq's ``dopri5``. torchdiffeq is not installed in the
shared env (and must not be pip-installed there), so this is an independent implementation of the
same scheme:

* Butcher tableau of Dormand & Prince (1980), FSAL, 5th-order solution with the embedded
  4th-order error estimate (torchdiffeq ``_DORMAND_PRINCE_SHAMPINE_TABLEAU``);
* torchdiffeq's error norm (RMS of err / (atol + rtol * max(|y0|, |y1|))), initial step selection
  (Hairer, Norsett & Wanner, II.4) and step-size controller (safety 0.9, ifactor 10, dfactor 0.2,
  exponent 1/5);
* differences: output times are hit exactly (the step is clipped to the next requested time)
  instead of torchdiffeq's 4th-order dense interpolation, and the step-size control is evaluated
  on detached values (no gradient through the adaptive step size).

Gradients flow through every RK stage (direct back-propagation, torchdiffeq ``odeint`` without
adjoint, as the CoMoGaussian code does with ``adjoint=False``).
"""

import torch

# Dormand-Prince 5(4) tableau.
C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
B4 = (5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40)
ERR = tuple(b5 - b4 for b5, b4 in zip(B5, B4))


def rk_step(func, t, y, dt, k1=None):
    """One Dormand-Prince step. Returns (y5, err, k7) where k7 = f(t + dt, y5) (FSAL)."""
    k = [func(t, y) if k1 is None else k1]
    for i in range(1, 7):
        yi = y + dt * sum(a * kj for a, kj in zip(A[i], k) if a != 0.0)
        k.append(func(t + C[i] * dt, yi))
    y5 = y + dt * sum(b * kj for b, kj in zip(B5, k) if b != 0.0)
    err = dt * sum(e * kj for e, kj in zip(ERR, k) if e != 0.0)
    return y5, err, k[6]


def _rms(x):
    return float(x.detach().pow(2).mean().sqrt())


def initial_step(func, t0, y0, f0, rtol, atol, order=5):
    """torchdiffeq _select_initial_step (Hairer II.4)."""
    scale = atol + y0.detach().abs() * rtol
    d0, d1 = _rms(y0 / scale), _rms(f0 / scale)
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    with torch.no_grad():
        f1 = func(t0 + h0, y0 + h0 * f0)
    d2 = _rms((f1 - f0) / scale) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1))
    return min(100 * h0, h1)


def error_ratio(err, y0, y1, rtol, atol):
    scale = atol + rtol * torch.maximum(y0.detach().abs(), y1.detach().abs())
    return _rms(err / scale)


def optimal_step(dt, ratio, safety=0.9, ifactor=10.0, dfactor=0.2, order=5):
    """torchdiffeq _optimal_step_size."""
    if ratio == 0:
        return dt * ifactor
    if ratio < 1:
        dfactor = 1.0
    factor = min(ifactor, max(safety / ratio ** (1.0 / order), dfactor))
    return dt * factor


def odeint_dopri5(func, y0, t, rtol=1e-4, atol=1e-5, max_steps=10000, stats=None):
    """Solve dy/dt = func(t, y) from t[0] with y(t[0]) = y0; returns [len(t), *y0.shape].

    ``t`` is a strictly monotonic 1-D sequence (python floats or a tensor); integration runs
    forward or backward accordingly. ``stats`` (optional dict) receives accepted/rejected counts.
    """
    ts = [float(x) for x in (t.tolist() if torch.is_tensor(t) else t)]
    if len(ts) < 1:
        raise ValueError("odeint_dopri5 needs at least one time")
    direction = 1.0 if len(ts) < 2 or ts[-1] >= ts[0] else -1.0
    for a, b in zip(ts, ts[1:]):
        if (b - a) * direction <= 0:
            raise ValueError("times must be strictly monotonic")
    out = [y0]
    if len(ts) == 1:
        return torch.stack(out)
    tc, y = ts[0], y0
    f = func(tc, y)
    dt = direction * initial_step(func, tc, y, f, rtol, atol)
    accepted = rejected = 0
    for target in ts[1:]:
        while (target - tc) * direction > 1e-12:
            if accepted + rejected >= max_steps:
                raise RuntimeError(f"dopri5 exceeded max_steps={max_steps}")
            step = dt
            hit = (tc + step - target) * direction >= 0
            if hit:
                step = target - tc
            y_new, err, f_new = rk_step(func, tc, y, step, f)
            ratio = error_ratio(err, y, y_new, rtol, atol)
            if ratio <= 1.0:
                tc = target if hit else tc + step
                y, f = y_new, f_new
                accepted += 1
            else:
                rejected += 1
            # Controller update from the attempted step (a clipped hit keeps the larger dt).
            if not (hit and ratio <= 1.0):
                dt = optimal_step(step, ratio)
            else:
                dt = direction * max(abs(dt), abs(optimal_step(step, ratio)))
        out.append(y)
    if stats is not None:
        stats["accepted"] = stats.get("accepted", 0) + accepted
        stats["rejected"] = stats.get("rejected", 0) + rejected
    return torch.stack(out)


def odeint_fixed(func, y0, t, substeps=1):
    """Fixed-step Dormand-Prince (5th order) for convergence-order probes."""
    ts = [float(x) for x in (t.tolist() if torch.is_tensor(t) else t)]
    out, y = [y0], y0
    for a, b in zip(ts, ts[1:]):
        h = (b - a) / substeps
        for j in range(substeps):
            y, _, _ = rk_step(func, a + j * h, y, h)
        out.append(y)
    return torch.stack(out)
