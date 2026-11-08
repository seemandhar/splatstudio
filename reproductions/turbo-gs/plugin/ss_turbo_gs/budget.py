"""Convergence-aware budget control (paper Sec. 4.2, Eq. 5-6, supp. Algorithm 1). Pure CPU code.

No official code is released. Interpretations (paper.md Sec. 7):
- t in Eq. 6 is training progress in percent, t = 100 * step / total_steps (Alg. 1 input
  "steps: Total steps"); the alternative densify-window reading (`t_mode: "densify_window"`)
  lets B reach 1.5 M and leaves the cap inert (smoke-1, review-1 M1).
- alpha_hist = -slope of log(EMA loss) vs log(step) over all records (positive decay exponent);
  alpha_base = mean of the historical alpha_hist fits ("average value from historical alpha");
  rate = -slope over the most recent `window` records (alpha_recent).
- Alg. 1 `deviation = -rate - alpha_base` is read with alpha_base as the fitted *slope*, i.e.
  deviation = alpha_base - rate (> 0 when recent convergence is slower than the power law),
  which raises alpha and slows densification, as the paper's prose states.
"""

import math

import numpy as np

# Official Taming-3DGS full_eval.py `big_budgets` (= 3DGS final counts). Turbo-3DGS's peak #G
# (DB 2.82 M) matches their DB mean (2.80 M); the paper only says M is "manually set".
# Keyed by the scene's initial SfM point count in our native datasets (unique per scene).
SCENE_SFM_COUNTS = {
    "bicycle": 54275, "flowers": 38347, "garden": 138766, "stump": 32049, "treehill": 52363,
    "room": 112627, "counter": 155767, "kitchen": 241367, "bonsai": 206613,
    "truck": 136029, "train": 182686, "drjohnson": 80861, "playroom": 37005,
}
BIG_BUDGETS = {
    "bicycle": 5987095, "flowers": 3618411, "garden": 5728191, "stump": 4867429,
    "treehill": 3770257, "room": 1548960, "counter": 1190919, "kitchen": 1803735,
    "bonsai": 1252367, "truck": 2584171, "train": 1085480, "playroom": 2326100,
    "drjohnson": 3273600,
}
BUDGET_TABLE = {str(SCENE_SFM_COUNTS[s]): v for s, v in BIG_BUDGETS.items()}


def resolve_budget(initial_count, budget, table):
    value = budget if budget is not None else table.get(str(int(initial_count)))
    if value is None:
        raise ValueError(
            f"No Turbo-GS budget M for a scene with {initial_count} initial points; set the "
            "density config 'budget' explicitly"
        )
    return int(value)


def budget_curve(t, alpha, n0, m):
    """Eq. 6: B(t) = N + (t^alpha - 1)/(100^alpha - 1) * (M - N), t in [1, 100]."""
    return n0 + (t**alpha - 1.0) / (100.0**alpha - 1.0) * (m - n0)


def training_progress(step, total_steps):
    """Default reading (review-1 M1): Alg. 1 takes `steps: Total steps` as input, so
    t = 100 * step / total_steps, clamped to [1, 100]. Densification stops at 15k of 30k, so
    t <= 50 and the final budget is B(50) < M_adaptive (the cap binds)."""
    return min(max(100.0 * step / float(total_steps), 1.0), 100.0)


def progress_index(step, start, stop):
    """Map a step in the densification window [start, stop] to t in [1, 100]."""
    frac = min(max((step - start) / float(stop - start), 0.0), 1.0)
    return 1.0 + 99.0 * frac


def loglog_slope(steps, values):
    x, y = np.log(np.asarray(steps, dtype=np.float64)), np.log(np.asarray(values, np.float64))
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / max((x * x).sum(), 1e-12))


class BudgetScheduler:
    def __init__(self, initial_count, budget, *, warmup=100, ema=0.1, lam=0.5, window=500,
                 rate_threshold=0.05, grow=1.1, shrink=0.9, m_range=(0.5, 1.5),
                 alpha_clip=(0.1, 2.0), alpha_default=1.0):
        self.n0, self.m = int(initial_count), int(budget)
        self.warmup, self.ema_w, self.lam, self.window = warmup, ema, lam, window
        self.rate_threshold, self.grow, self.shrink = rate_threshold, grow, shrink
        self.m_range, self.alpha_clip, self.alpha = m_range, alpha_clip, alpha_default
        self.m_adaptive = float(budget)
        self.steps, self.ema_values, self.alpha_hist = [], [], []
        self.rate = None
        self._ema = None

    def record(self, step, loss):
        """Record the training loss of `step` (only after the warm-up, Sec. 4.2)."""
        if step <= self.warmup:
            return
        loss = float(loss)
        if not math.isfinite(loss) or loss <= 0:
            raise ValueError(f"budget scheduler needs a positive finite loss, got {loss}")
        self._ema = loss if self._ema is None else self.ema_w * loss + (1 - self.ema_w) * self._ema
        self.steps.append(step)
        self.ema_values.append(self._ema)

    def update(self):
        """Alg. 1 body: refit alpha and M_adaptive from the recorded history."""
        if len(self.steps) <= max(self.warmup, 2):
            return self.alpha
        self.alpha_hist.append(-loglog_slope(self.steps, self.ema_values))
        alpha_base = float(np.mean(self.alpha_hist))
        k = min(self.window, len(self.steps))
        self.rate = -loglog_slope(self.steps[-k:], self.ema_values[-k:])
        if self.rate > self.rate_threshold:
            self.m_adaptive = min(self.m_adaptive * self.grow, self.m * self.m_range[1])
        elif self.rate < -self.rate_threshold:
            self.m_adaptive = max(self.m_adaptive * self.shrink, self.m * self.m_range[0])
        deviation = alpha_base - self.rate
        alpha = alpha_base + self.lam * math.tanh(deviation)
        self.alpha = min(max(alpha, self.alpha_clip[0]), self.alpha_clip[1])
        return self.alpha

    def budget(self, t):
        return budget_curve(t, self.alpha, self.n0, self.m_adaptive)

    def state(self):
        return {"alpha": self.alpha, "m_adaptive": self.m_adaptive, "rate": self.rate,
                "alpha_base": float(np.mean(self.alpha_hist)) if self.alpha_hist else None,
                "records": len(self.steps)}
