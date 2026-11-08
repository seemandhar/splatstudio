"""Shape statistics of finished runs: needles (erank < 1.04, paper Table 'iteration'),
erank barrier / thin terms (code form), opacity and count."""
import json
import sys

import torch

sys.path.insert(0, "reproductions/erank-gs/plugin")
from ss_erank_gs.losses import DEFAULTS, effective_rank, erank_terms  # noqa: E402

for run in sys.argv[1:]:
    p = torch.load(f"{run}/checkpoint.pt", map_location="cpu", weights_only=False)["params"]
    s = p["scales"].detach().exp().double()
    er = effective_rank(s)
    b, t = erank_terms(s, DEFAULTS)
    op = p["opacities"].detach().sigmoid()
    print(json.dumps({"run": run, "n": len(s), "needles_lt_1.04": int((er < 1.04).sum()),
                      "frac_lt_1.5": float((er < 1.5).double().mean()),
                      "mean_erank": float(er.mean()), "barrier": float(b), "thin": float(t),
                      "median_opacity": float(op.median())}))
