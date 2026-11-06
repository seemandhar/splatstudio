"""Mechanism probes for the AdamW-GS port. Each raises AssertionError when the mechanism is
removed or wrong. References are written independently of optim.py/density.py (scalar loops
transcribed from the official adam.cu / gaussian_model.py)."""

import math
from dataclasses import replace

import torch

from .density import AdamWGSDensity
from .optim import AdamWGSContext, SparseAdamWGS, img_scale

TOL = 5e-6  # float32 update vs float64 scalar reference (|theta| <= ~5)


def _err(got, ref, p0):
    """Max error relative to float32 resolution: |got-ref| / (TOL + 1e-5 |ref - p0|) (<= 1 ok)."""
    return float(((got - ref).abs() / (TOL + 1e-5 * (ref - p0).abs())).max())


def _density_spec(method, **config):
    spec = next(c for c in method.components if c.kind == "density")
    return replace(spec, config={**spec.config, **config})


def _ref_element(p, g, m, v, t, lr, reg, reg_scale, b1=0.9, b2=0.999, eps=1e-15, cap=10.0):
    """Scalar transcription of adamUpdateCUDA (one element of a visible primitive)."""
    ib1 = 1.0 / (1.0 - b1 ** t)
    ib2 = 1.0 / (1.0 - b2 ** t)
    m = b1 * m + (1.0 - b1) * g
    v = b2 * v + (1.0 - b2) * g * g
    denom = math.sqrt(v * ib2) + eps
    r = 0.0
    if reg == 1:
        o = 0.5 * (1.0 + math.tanh(0.5 * p))
        r = min(reg_scale * o * (1.0 - o) / denom, cap)
    elif reg == 2:
        r = min(reg_scale * math.exp(p) / denom, cap)
    return p - lr * (m * ib1 / denom + r), m, v


def _run_step(field, n, step, device, asystep, visible, seed=0, img=1089480, lr=0.05):
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (n,) if field == "opacities" else (n, 3)
    p0 = torch.randn(shape, generator=g, dtype=torch.float64) * 2 - (1 if field == "scales" else 0)
    m0 = torch.randn(shape, generator=g, dtype=torch.float64) * 1e-6
    v0 = torch.rand(shape, generator=g, dtype=torch.float64) * 1e-14
    gr = torch.randn(shape, generator=g, dtype=torch.float64) * 1e-6
    param = torch.nn.Parameter(p0.float().to(device))
    config = {"opacity_reg_begin": 3000, "opacity_reg_end": 15000, "scale_reg_begin": 1000,
              "scale_reg_end": 15000, "lambda_opacity": 1e-3, "lambda_scale": 1e-5,
              "opacity_reg_cap": 10.0, "scale_reg_cap": 10.0}
    state = {"asystep": (asystep - 1).to(torch.int32).to(device)}
    ctx = AdamWGSContext(config, state)
    ctx.set_image_pixels(img)
    opt = SparseAdamWGS(param, lr, ctx, field)
    opt.state[param]["step"] = torch.tensor(0.0)
    opt.state[param]["exp_avg"] = m0.float().to(device).clone()
    opt.state[param]["exp_avg_sq"] = v0.float().to(device).clone()
    ctx.begin_iteration(step, visible.to(device))
    param.grad = gr.float().to(device)
    opt.step()
    return (p0, m0, v0, gr), param.detach().cpu().double(), opt.state[param]


def _reference(field, data, asystep, visible, step, img=1089480, lr=0.05):
    p0, m0, v0, gr = data
    reg, scale = 0, 0.0
    if field == "opacities" and 3000 <= step <= 15000:
        reg, scale = 1, 1e-3 / img_scale(img)
    if field == "scales" and 1000 <= step <= 15000:
        reg, scale = 2, 1e-5 / img_scale(img)
    out_p, out_m, out_v = p0.clone(), m0.clone(), v0.clone()
    flat = [t.reshape(len(p0), -1) for t in (p0, m0, v0, gr, out_p, out_m, out_v)]
    for i in range(len(p0)):
        if not bool(visible[i]):
            continue
        for j in range(flat[0].shape[1]):
            a, b, c = _ref_element(float(flat[0][i, j]), float(flat[3][i, j]), float(flat[1][i, j]),
                                   float(flat[2][i, j]), int(asystep[i]), lr, reg, scale)
            flat[4][i, j], flat[5][i, j], flat[6][i, j] = a, b, c
    return out_p, out_m, out_v


def sparse_step(method, device):
    """Sparse Adam: invisible rows untouched; per-primitive bias correction (asystep)."""
    n = 64
    visible = torch.rand(n, generator=torch.Generator().manual_seed(1)) < 0.6
    asystep = torch.randint(1, 40, (n,), generator=torch.Generator().manual_seed(2))
    worst = 0.0
    for field in ("means", "opacities"):  # step 500: no DAR term for either field
        data, got, st = _run_step(field, n, 500, device, asystep, visible)
        ref_p, ref_m, _ = _reference(field, data, asystep, visible, 500)
        err = _err(got, ref_p, data[0])
        worst = max(worst, err)
        assert err <= 1.0, f"{field}: sparse step differs from adam.cu reference ({err})"
        inv = ~visible
        assert torch.equal(got[inv].float(), data[0][inv].float()), "invisible primitive moved"
        assert torch.equal(st["exp_avg"].cpu()[inv], data[1][inv].float()), "invisible moment decayed"
        # a synchronous (dense) Adam would also move invisible rows via their moments
        assert float((data[1][inv]).abs().max()) > 0
        # a global-step Adam (t = max step for every row) must be distinguishable
        glob = _reference(field, data, torch.full_like(asystep, int(asystep.max())), visible, 500)[0]
        assert float((glob - ref_p).abs().max()) > 100 * TOL, "bias correction not per-primitive"
    return {"max_normalized_error": worst, "visible": int(visible.sum()), "n": n}


def dar(method, device):
    """DAR: reg term equals reference, moments carry only grad(l), cap 10, schedule windows."""
    n = 48
    visible = torch.ones(n, dtype=torch.bool)
    visible[::5] = False
    asystep = torch.randint(1, 3000, (n,), generator=torch.Generator().manual_seed(3))
    out = {}
    for field, steps in (("opacities", (2999, 3000, 15000, 15001)),
                         ("scales", (999, 1000, 15000, 15001))):
        moved = {}
        for s in steps:
            data, got, st = _run_step(field, n, s, device, asystep, visible, seed=s)
            ref_p, ref_m, ref_v = _reference(field, data, asystep, visible, s)
            err = _err(got, ref_p, data[0])
            assert err <= 1.0, f"{field}@{s}: DAR mismatch {err}"
            # decoupling: moments equal the pure photometric EMA (no grad(R) inside)
            torch.testing.assert_close(st["exp_avg"].cpu().double(), ref_m, rtol=1e-5, atol=1e-14)
            torch.testing.assert_close(st["exp_avg_sq"].cpu().double(), ref_v, rtol=1e-5, atol=1e-20)
            no_reg = _reference("means", data, asystep, visible, s)[0]
            moved[s] = float((ref_p - no_reg).abs().max())
        lo, on, end, off = steps
        assert moved[lo] == 0.0 and moved[off] == 0.0, f"{field}: DAR active outside its window"
        assert moved[on] > 100 * TOL and moved[end] > 100 * TOL, f"{field}: DAR too small to test"
        out[field] = moved
    # cap: with v = 0 the uncapped term would be ~1e-8/1e-15; the step must be lr*10 exactly
    p = torch.nn.Parameter(torch.zeros(4, device=device))
    ctx = AdamWGSContext({"opacity_reg_begin": 0, "opacity_reg_end": 10, "scale_reg_begin": 0,
                          "scale_reg_end": 10, "lambda_opacity": 1e-3, "lambda_scale": 1e-5,
                          "opacity_reg_cap": 10.0, "scale_reg_cap": 10.0},
                         {"asystep": torch.zeros(4, dtype=torch.int32, device=device)})
    ctx.set_image_pixels(1089480)
    opt = SparseAdamWGS(p, 0.05, ctx, "opacities")
    ctx.begin_iteration(5, torch.ones(4, dtype=torch.bool, device=device))
    p.grad = torch.zeros_like(p)
    opt.step()
    torch.testing.assert_close(p.detach().cpu(), torch.full((4,), -0.5))
    out["cap_step"] = float(p[0])
    return out


def image_scale(method, device):
    """Official `_img_scale`: keep two significant digits of N_I and divide by 10."""
    cases = {1297 * 840: 100000, 1558 * 1038: 160000, 979 * 546: 53000, 1264 * 832: 100000}
    for n, want in cases.items():
        digits = str(n)
        indep = int(digits[:2]) * 10 ** (len(digits) - 3)
        assert img_scale(n) == want == indep, f"img_scale({n}) = {img_scale(n)} != {want}"
    return {str(k): v for k, v in cases.items()}


def _scene(method, n, device, seed=0):
    from splatstudio.state import initialize

    g = torch.Generator().manual_seed(seed)
    pts = (torch.rand(n, 3, generator=g) - 0.5)
    pts[:, 2] += 3
    cols = torch.rand(n, 3, generator=g) * 0.6 + 0.2
    params, opts = initialize(method, pts.to(device), cols.to(device))
    with torch.no_grad():
        params["scales"].copy_((torch.rand(n, 3, generator=g) * 2 - 5).to(device))
        params["opacities"].copy_(torch.logit(torch.rand(n, generator=g) * 0.8 + 0.1).to(device))
        q = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1)
        params["quats"].copy_(q.to(device))
    return params, opts


def _fill_moments(params, opts, seed=0):
    g = torch.Generator().manual_seed(seed)
    for name, p in params.items():
        st = opts[name].state[p]
        st["step"] = torch.tensor(1.0)
        st["exp_avg"] = torch.randn(p.shape, generator=g).to(p.device)
        st["exp_avg_sq"] = torch.rand(p.shape, generator=g).to(p.device) + 0.5


def rsr(method, device):
    """RSR: exactly int(ratio N) rows of every optimizer rescaled by (alpha1, alpha2), with the
    milestone StSS schedule (interval 100 inside [1000, 15000])."""
    from .density import rsr_active, rsr_ratio

    density = AdamWGSDensity(_density_spec(method))
    c = density.config
    n = 200
    out = {}
    for step in (900, 1000, 1050, 6900, 7000, 9900, 10000, 15000, 15100):
        params, opts = _scene(method, n, device)
        density.initialize(params, opts, 1.0)
        _fill_moments(params, opts)
        before = {k: (opts[k].state[p]["exp_avg"].clone(), opts[k].state[p]["exp_avg_sq"].clone())
                  for k, p in params.items()}
        k = density.restate(params, opts, rsr_ratio(c, step)) if rsr_active(c, step) else 0
        rows = None
        for name, p in params.items():
            m, v = opts[name].state[p]["exp_avg"], opts[name].state[p]["exp_avg_sq"]
            m0, v0 = before[name]
            changed = (m != m0).reshape(n, -1).any(-1)
            if rows is None:
                rows = changed
            assert torch.equal(changed, rows), "RSR rows differ between fields"
            torch.testing.assert_close(m[changed], m0[changed] * c["rsr_alpha1"])
            torch.testing.assert_close(v[changed], v0[changed] * c["rsr_alpha2"])
        indep_on = c["rsr"] and 1000 <= step <= 15000 and step % 100 == 0
        want = int(rsr_ratio_indep(c, step) * n) if indep_on else 0
        assert int(rows.sum()) == k == want, f"step {step}: {int(rows.sum())} rows, expected {want}"
        out[step] = k
    assert out[1000] > 0 and out[900] == out[1050] == out[15100] == 0
    return out


def rsr_ratio_indep(c, step):
    r = 0.0
    for i in range(len(c["rsr_shifts"])):
        if step >= c["rsr_shifts"][i]:
            r = c["rsr_ratios"][i]
    return r


def topology(method, device):
    """Clone/split/prune keep asystep and moments aligned; opacity correction on clone."""
    from splatstudio.state import check_state

    density = AdamWGSDensity(_density_spec(method, percent_dense=0.01))
    n = 10
    params, opts = _scene(method, n, device)
    with torch.no_grad():
        params["scales"][:5] = math.log(0.001)   # small -> clone
        params["scales"][5:] = math.log(0.2)     # large -> split
        params["opacities"][:] = torch.logit(torch.linspace(0.2, 0.9, n)).to(device)
    density.initialize(params, opts, 1.0)
    _fill_moments(params, opts)
    density.state["asystep"].copy_(torch.arange(1, n + 1, dtype=torch.int32))
    o_before = torch.sigmoid(params["opacities"]).detach().clone()
    density.state["grad2d"][:] = 1.0
    density.state["grad2d"][[1, 6]] = 0.0          # 1 (small) and 6 (large) stay unchanged
    density.state["count"][:] = 1.0
    density._densify(params, opts, 600)
    check_state(method, params, opts)
    asy = density.state["asystep"].cpu()
    clones = [0, 2, 3, 4]
    splits = [5, 7, 8, 9]
    # gsplat layout: [kept originals] + [clones] after duplicate, then split rest + children
    expected = [i + 1 for i in range(n) if i not in splits] + [i + 1 for i in clones]
    expected += [i + 1 for i in splits] * 2
    assert asy.tolist() == expected, f"asystep not inherited: {asy.tolist()} vs {expected}"
    o = torch.sigmoid(params["opacities"]).detach().cpu()
    keep = [i for i in range(n) if i not in splits]
    corr = 1 - torch.sqrt(1 - o_before.cpu())
    for pos, i in enumerate(keep):
        want = corr[i] if i in clones else o_before[i].cpu()
        torch.testing.assert_close(o[pos], want, rtol=1e-5, atol=1e-6)
    for j, i in enumerate(clones):
        torch.testing.assert_close(o[len(keep) + j], corr[i], rtol=1e-5, atol=1e-6)
    for name, p in params.items():
        m = opts[name].state[p]["exp_avg"]
        assert torch.count_nonzero(m[len(keep):]) == 0, f"{name}: children moments not zero"
        assert torch.count_nonzero(m[:len(keep)].reshape(len(keep), -1).abs().sum(-1)) == len(keep)
    # prune keeps alignment
    with torch.no_grad():
        params["opacities"][0] = -20.0
    density.state["grad2d"].zero_()
    density._densify(params, opts, 700)
    check_state(method, params, opts)
    assert len(density.state["asystep"]) == len(params["means"])
    return {"gaussians": len(params["means"]), "asystep": density.state["asystep"].tolist()}


def noise(method, device):
    """Exploration noise equals Sigma * n * op_sigmoid(1-o) * noise_lr * lr; schedule."""
    from gsplat.strategy.ops import inject_noise_to_position

    density = AdamWGSDensity(_density_spec(method))
    c = density.config
    params, opts = _scene(method, 32, device)
    with torch.no_grad():
        params["opacities"][:16] = torch.logit(torch.tensor(0.001))  # near-dead: full noise
    before = params["means"].detach().clone()
    torch.manual_seed(7)
    density.add_noise(params, 1e-5)
    got = params["means"].detach() - before
    with torch.no_grad():
        params["means"].copy_(before)
    torch.manual_seed(7)
    inject_noise_to_position(params, opts, {}, c["noise_lr"] * 1e-5)  # gsplat reference
    ref = params["means"].detach() - before
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-12)
    assert got[:16].abs().max() > 0, "noise missing on low-opacity Gaussians"
    from .density import noise_active

    sched = {s: noise_active(c, s) for s in (999, 1000, 14000, 14001)}
    assert sched == {999: False, 1000: c["noise"], 14000: c["noise"], 14001: False}
    return {"noise_on": bool(c["noise"]), "max_disp": float(got.abs().max()), "schedule": sched}


def install(method, device):
    """initialize must install SparseAdamWGS into the runtime's own optimizers dict."""
    density = AdamWGSDensity(_density_spec(method))
    params, opts = _scene(method, 40, device)
    runtime_dict = opts
    lr_means = opts["means"].param_groups[0]["lr"]
    density.initialize(params, opts, 1.0)
    density.set_training_cameras([{"width": 1297, "height": 840}])
    assert opts is runtime_dict
    ctxs = {id(o.context) for o in opts.values() if isinstance(o, SparseAdamWGS)}
    assert all(isinstance(o, SparseAdamWGS) for o in opts.values()), "runtime Adam not replaced"
    assert len(ctxs) == 1 and len(opts) == len(params), "optimizers must share one context"
    assert opts["means"].param_groups[0]["lr"] == lr_means
    visible = torch.zeros(40, dtype=torch.bool, device=device)
    visible[:25] = True
    density.context.begin_iteration(5000, visible)
    before = {k: p.detach().clone() for k, p in params.items()}
    for p in params.values():
        p.grad = torch.randn_like(p) * 1e-3
    for o in opts.values():  # exactly what runtime.train does
        o.step()
        o.zero_grad(set_to_none=True)
    for k, p in params.items():
        assert torch.equal(p[25:], before[k][25:]), f"{k}: invisible rows moved"
        assert not torch.equal(p[:25], before[k][:25]), f"{k}: visible rows not updated"
        assert torch.count_nonzero(opts[k].state[p]["exp_avg"][25:]) == 0
    return {"fields": sorted(opts), "asystep_max": int(density.state["asystep"].max())}
