"""AD-GS alternating schedule (paper §3.3, supp. Table 7; official train.py:168-169, 384-401)."""

from dataclasses import dataclass

SCHEDULE_KEYS = ("warmup_iters", "cycle_len", "high_dens_iters")


@dataclass(frozen=True)
class Schedule:
    warmup_iters: int = 1500  # N_W
    cycle_len: int = 200  # N_H + N_L
    high_dens_iters: int = 100  # N_H

    @classmethod
    def from_config(cls, config):
        s = cls(**{k: int(config[k]) for k in SCHEDULE_KEYS if k in config})
        if not 0 < s.high_dens_iters <= s.cycle_len:
            raise ValueError("high_dens_iters must be in (0, cycle_len]")
        return s

    def low_phase(self, step):
        # train.py:168  in_low_dens_phase = (iteration % alt_cycle_len) >= alt_high_dens_iters
        return step % self.cycle_len >= self.high_dens_iters

    def regularize(self, step):
        # train.py:169  apply_alt_regularization = iteration > N_W and in_low_dens_phase
        return step > self.warmup_iters and self.low_phase(step)

    def phase(self, step):
        """'warmup', 'low' or 'high' — the regime whose thresholds a densification uses."""
        if step <= self.warmup_iters:
            return "warmup"
        return "low" if self.low_phase(step) else "high"


def schedule_config(config):
    return {k: config[k] for k in SCHEDULE_KEYS if k in config}
