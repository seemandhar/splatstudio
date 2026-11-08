"""Debug: find which StableGS loss first yields a non-finite parameter gradient (smoke only)."""
import sys

import torch

import ss_stablegs.losses as L

NAMES = ["create_pair_color", "create_geometric_color", "create_depth_normal",
         "create_depth_consistency"]
SWITCH = int(sys.argv[sys.argv.index("--switch") + 1])


def wrap(name, factory):
    def make(spec):
        fn = factory(spec)

        def loss(params, outputs, target):
            v = fn(params, outputs, target)
            if outputs["step"] > SWITCH and v.requires_grad:
                keys = ["means", "scales", "quats", "opacities", "opacity_aux", "sh0"]
                gs = torch.autograd.grad(v, [params[k] for k in keys], retain_graph=True,
                                         allow_unused=True)
                bad = [k for k, g in zip(keys, gs) if g is not None and not torch.isfinite(g).all()]
                if bad:
                    pair = outputs["pair"]
                    info = {"step": outputs["step"], "loss": name, "bad": bad,
                            "pair": pair["indices"], "in_P": pair["in_P"]}
                    for k, view in enumerate(pair["views"]):
                        g = view["geo"]
                        d = g["depth"]
                        info[f"v{k}"] = {"dmin": float(d.min()), "dmax": float(d.max()),
                                         "zero": int((d == 0).sum()),
                                         "alpha_min_pos": float(g["alpha"][g["alpha"] > 0].min()),
                                         "mdmin": float(g["median_depth"].min()),
                                         "nan_rgb": int((~torch.isfinite(g["rgb"])).sum())}
                    print("NONFINITE", info, flush=True)
                    torch.save({"outputs_pair_idx": pair["indices"], "info": info}, "nan_info.pt")
                    raise RuntimeError(f"non-finite gradient from {name}: {info}")
            return v

        return loss

    return make


for n in NAMES:
    setattr(L, n, wrap(n, getattr(L, n)))
sys.argv.remove("--switch")
sys.argv.remove(str(SWITCH))
sys.argv += ["--switch", str(SWITCH)]
exec(open(__file__.replace("debug_nan.py", "smoke_geo.py")).read())
