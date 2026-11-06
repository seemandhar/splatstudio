"""Smoke test with *compressed* schedules (debug only, not a paper configuration): exercises the
full-resolution phase, SH degrees, robust mask, densification, opacity reset and post-split culling
inside 2000 steps."""

import sys
from dataclasses import replace

from splatstudio.runtime import train

from ss_splatfactow import method

OVERRIDES = {
    "sfw-renderer": {"resolution_schedule": 300, "sh_degree_interval": 200},
    "sfw-photometric": {"start_robust_mask_at": 600, "robust_mask_reset_interval": 600},
    "sfw-density": {"warmup_length": 200, "reset_alpha_every": 12, "stop_screen_size_at": 1200,
                    "stop_split_at": 1600},
}


def compressed():
    m = method()
    comps = tuple(replace(c, config={**c.config, **OVERRIDES.get(c.name, {})})
                  for c in m.components)
    return replace(m, name="splatfacto-w", components=comps)


if __name__ == "__main__":
    data, out = sys.argv[1], sys.argv[2]
    train(compressed(), data, out, steps=2000, num_gaussians=10000000, max_gaussians=12000000,
          device="cuda:0", log_every=100, recipe="3dgs")
