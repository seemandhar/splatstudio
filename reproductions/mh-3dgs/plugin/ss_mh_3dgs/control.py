"""Same-budget 3DGS-MCMC control: gsplat MCMCStrategy with cap = MH-3DGS's per-scene count
(paper App. Tab. full_360/full_rest). The cap is resolved in initialize() (training only), so
`ss-eval --run` can rebuild the method without --data."""

from pathlib import Path

from splatstudio.backends.gsplat import MCMCDensity

from .density import _argv_value


class _Spec:
    def __init__(self, config):
        self.config = config


class BudgetMCMCDensity(MCMCDensity):
    def __init__(self, spec):
        config = dict(spec.config)
        self.cap = config.pop("cap_max", None)
        super().__init__(_Spec({**config, "cap_max": 1}))  # replaced in initialize()

    def initialize(self, params, optimizers, scene_scale=1.0):
        from . import PAPER_COUNTS

        cap = self.cap
        if cap is None:
            root = _argv_value("--data")
            scene = Path(root).name if root else None
            if scene not in PAPER_COUNTS:
                raise RuntimeError("mcmc-budget-density: unknown scene; set config 'cap_max'")
            cap = PAPER_COUNTS[scene]
        self.strategy.cap_max = int(cap)
        super().initialize(params, optimizers, scene_scale)


def create_density(spec):
    return BudgetMCMCDensity(spec)
