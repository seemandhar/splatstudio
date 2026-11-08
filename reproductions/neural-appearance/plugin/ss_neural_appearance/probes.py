"""Mechanism probes; each raises when the mechanism is removed or wrong."""

import math

import torch

from . import appearance as A
from . import bank as bank_mod
from .mcmc import MAX_SAMPLES, mcmc_step, noise_delta, relocation_adjustment
from .schedule import cosine_decay, lr_decay

CFG = dict(A.DEFAULTS)


def _sh_columns_reference(dirs):
    """Independent SH basis: 3DGS eval_sh (ss_faster_gs.reference.sh_to_color) on one-hot shN."""
    from ss_faster_gs.reference import sh_to_color

    n = len(dirs)
    cols = []
    for j in range(15):
        shN = torch.zeros(n, 15, 3, dtype=dirs.dtype)
        shN[:, j, 0] = 1.0
        cols.append(sh_to_color(torch.zeros(n, 1, 3, dtype=dirs.dtype), shN, dirs, 3)[:, 0] - 0.5)
    return torch.stack(cols, dim=-1)


def neural_decoder(method, device):
    g = torch.Generator().manual_seed(0)
    n = 64
    dirs = torch.nn.functional.normalize(torch.randn(n, 3, generator=g, dtype=torch.float64), dim=-1)
    feats = torch.randn(n, 8, generator=g, dtype=torch.float64)
    # SH basis vs 3DGS formula
    basis = A.sh_basis(dirs, 3)
    ref = _sh_columns_reference(dirs)
    sh_err = float((basis[:, 1:] - ref).abs().max())
    if sh_err > 1e-9 or abs(float(basis[0, 0]) - A.SH_C0) > 1e-12:
        raise AssertionError(f"SH direction encoding differs from 3DGS eval_sh ({sh_err})")
    # degree gating: degree a keeps (a+1)^2 columns
    for a in range(4):
        b = A.sh_basis(dirs, a)
        k = (a + 1) ** 2
        if b[:, k:].abs().sum() > 0 or (b[:, :k] - basis[:, :k]).abs().sum() > 0:
            raise AssertionError(f"degree gating wrong at active degree {a}")
    # MLP input: [sin f0, cos f0, sin f1, ...] then SH
    x = A.mlp_input(feats, dirs, 3, CFG)
    manual = torch.stack([feats.sin(), feats.cos()], dim=-1).reshape(n, 16)
    if x.shape != (n, 32) or (x[:, :16] - manual).abs().max() > 0 or (x[:, 16:] - basis).abs().max() > 0:
        raise AssertionError("MLP input layout differs from the official [gamma_feat(f), gamma_dir(d)]")
    mlp = A.ResidualMLP(CFG, generator=torch.Generator().manual_seed(1)).double()
    if mlp.n_params() != 816 or any(w.abs().sum() > 0 for w in mlp.weights[-1:]):
        raise AssertionError("MLP must have 816 parameters and a zero output layer")
    campos = torch.zeros(3, dtype=torch.float64)
    means = dirs * 3.0
    zero = A.neural_residual(means, feats, campos, mlp, 3, CFG)
    if zero.abs().max() > 0:
        raise AssertionError("residual must be exactly 0 at initialisation")
    with torch.no_grad():
        mlp.weights[-1].normal_(0, 1, generator=torch.Generator().manual_seed(2))
    if A.neural_residual(means, feats, campos, mlp, 0, CFG) is not None:
        raise AssertionError("residual must be disabled before the first degree activates")
    r = A.neural_residual(means, feats, campos, mlp, 3, CFG)
    W1, W2, W3 = (w.detach() for w in mlp.weights)
    expected = torch.tanh(torch.relu(torch.relu(x @ W1.t()) @ W2.t()) @ W3.t())
    err = float((r - expected).abs().max())
    if err > 1e-12 or r.abs().max() >= 1:
        raise AssertionError(f"A(d) != tanh(MLP(x)) ({err})")
    return {"sh_basis_max_err": sh_err, "mlp_params": mlp.n_params(), "residual_err": err}


def _scene(device, n=6, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = (torch.rand(n, 3, generator=g) - 0.5) * 1.2
    means[:, 2] += 4.0
    params = {
        "means": means, "scales": torch.full((n, 3), math.log(0.35)) + 0.2 * torch.randn(n, 3, generator=g),
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "opacities": torch.full((n,), 1.0), "sh0": (torch.rand(n, 1, 3, generator=g) - 0.5) / A.SH_C0,
        "shN": torch.zeros(n, 15, 3), "na_features": torch.randn(n, 8, generator=g),
        "na_blob": torch.zeros(n, 4),
    }
    params = {k: v.to(device).requires_grad_(True) for k, v in params.items()}
    K = torch.tensor([[60.0, 0, 32.3], [0, 60.0, 23.7], [0, 0, 1]], device=device)
    camera = {"K": K, "viewmat": torch.eye(4, device=device), "width": 64, "height": 48}
    return params, camera


def _renderer(appearance="neural"):
    from splatstudio.specs import ComponentSpec

    from .renderer import create_renderer

    return create_renderer(ComponentSpec("r", "renderer", "x", (), (), {"appearance": appearance}))


def _bank(device, mode="neural", ppisp=False, cameras=None):
    b = bank_mod.Bank(mode, CFG, dict(bank_mod.PPISP_DEFAULTS, use=ppisp), cameras or [], 30000,
                      device=device)
    if b.mlp is not None:
        with torch.no_grad():
            w = b.mlp.weights[-1]
            w.copy_(torch.randn(w.shape, generator=torch.Generator().manual_seed(3)) * 0.5)
    return b


def render_composition(method, device):
    """Renderer (vendored rasterizer, degree-0 path fed sh0 + A/Y0) == independent float64 dense
    compositor fed relu(0.5 + c0 + A(d)) through the 3DGS colour formula."""
    from ss_faster_gs.reference import render as ref_render

    params, camera = _scene(device)
    previous, bank_mod.ACTIVE = bank_mod.ACTIVE, _bank(device)
    try:
        out = _renderer().render(params, camera, sh_degree=3)["rgb"]
        mlp = bank_mod.ACTIVE.mlp
        campos = torch.zeros(3, device=device)
        r = A.neural_residual(params["means"], params["na_features"], campos, mlp, 3, CFG).detach()
        p64 = {k: v.detach().double() for k, v in params.items()}
        p64["sh0"] = p64["sh0"] + (r.double() / A.SH_C0)[:, None, :]
        cam64 = {**camera, "K": camera["K"].double(), "viewmat": camera["viewmat"].double()}
        ref = ref_render(p64, cam64, 0, torch.zeros(3, dtype=torch.float64, device=device))
        ref = ref.permute(1, 2, 0) if ref.shape[0] == 3 else ref
        err = float((out.double() - ref).abs().max())
        base = _renderer().render({**params, "na_features": params["na_features"]}, camera,
                                  sh_degree=0)["rgb"]
        diff = float((out - base).abs().max())
    finally:
        bank_mod.ACTIVE = previous
    if err > 2e-4:
        raise AssertionError(f"rendered colour != relu(0.5 + c0 + A(d)) composite ({err})")
    if diff < 1e-3:
        raise AssertionError("residual has no effect on the image")
    return {"max_err_vs_dense_reference": err, "residual_effect": diff}


def direction_gradient(method, device):
    """dL/dmu includes the view-direction path through the MLP (DIRECTION_GRADIENT: true)."""
    params, camera = _scene(device)
    previous, bank_mod.ACTIVE = bank_mod.ACTIVE, _bank(device)
    try:
        r = _renderer()
        r.render(params, camera, 3)["rgb"].square().sum().backward()
        g_full = params["means"].grad.clone()
        feats_grad = params["na_features"].grad.abs().sum().item()
        mlp_grad = sum(w.grad.abs().sum().item() for w in bank_mod.ACTIVE.mlp.weights)
        for p in params.values():
            p.grad = None
        import ss_neural_appearance.renderer as R

        original = R.neural_residual

        def detached(means, *a, **k):
            return original(means.detach(), *a, **k)

        R.neural_residual = detached
        try:
            r.render(params, camera, 3)["rgb"].square().sum().backward()
        finally:
            R.neural_residual = original
        g_det = params["means"].grad
        delta = float((g_full - g_det).abs().max())
    finally:
        bank_mod.ACTIVE = previous
    if delta <= 1e-7 or feats_grad <= 0 or mlp_grad <= 0:
        raise AssertionError("appearance gradients (direction -> mean, features, MLP) missing")
    return {"direction_grad_delta": delta, "feature_grad_l1": feats_grad, "mlp_grad_l1": mlp_grad}


def _relocation_reference(o, s, n):
    o2 = 1 - (1 - o) ** (1 / n)
    den = sum(math.comb(i, k) * (-1) ** k / math.sqrt(k + 1) * o2 ** (k + 1)
              for i in range(n) for k in range(i + 1))
    return o2, o / den * s


def mcmc(method, device):
    o = torch.tensor([0.3, 0.9, 0.05, 0.6, 0.99])
    s = torch.rand(5, 3) + 0.1
    counts = torch.tensor([1, 2, 3, 7, 60])
    no, ns = relocation_adjustment(o, s, counts)
    err = 0.0
    for i in range(5):
        ro, rs = _relocation_reference(float(o[i]), s[i].double(), min(int(counts[i]), MAX_SAMPLES))
        err = max(err, abs(float(no[i]) - ro), float((ns[i].double() - rs).abs().max()))
    if err > 2e-4 or abs(float(no[0]) - 0.3) > 1e-6:
        raise AssertionError(f"relocation formula mismatch ({err})")
    # densification step on a fixture
    n = 40
    g = torch.Generator().manual_seed(0)
    params = {"means": torch.randn(n, 3, generator=g), "scales": torch.randn(n, 3, generator=g) - 2,
              "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
              "opacities": torch.randn(n, generator=g), "sh0": torch.randn(n, 1, 3, generator=g),
              "na_features": torch.randn(n, 8, generator=g)}
    params["opacities"][:5] = -10.0  # dead
    params = {k: torch.nn.Parameter(v) for k, v in params.items()}
    opts = {k: torch.optim.Adam([p], lr=0.01) for k, p in params.items()}
    sum(p.square().sum() for p in params.values()).backward()
    for o_ in opts.values():
        o_.step()
    appended = []

    def append(rows):
        appended.append(len(rows["means"]))
        for k in params:
            params[k] = torch.nn.Parameter(torch.cat([params[k].detach(), rows[k]]))

    before = {k: v.detach().clone() for k, v in params.items()}
    ev = mcmc_step(params, opts, append, cap_max=41, generator=torch.Generator().manual_seed(1))
    if ev["relocated"] != 5 or appended != [1] or len(params["means"]) != 41:
        raise AssertionError(f"MCMC relocation/add counts wrong: {ev}, {appended}")
    revived = params["opacities"][:5].detach()
    if (revived <= math.log(0.005 / 0.995)).any():
        raise AssertionError("dead Gaussians were not relocated")
    moved = [(params["means"][i].detach() - before["means"]).abs().sum(-1).min().item() for i in range(5)]
    if max(moved) > 0:
        raise AssertionError("relocated Gaussians must copy an alive Gaussian's mean")
    zeroed = (opts["means"].state[opts["means"].param_groups[0]["params"][0]]["exp_avg"].abs().sum(-1) == 0).sum()
    if zeroed == 0:
        raise AssertionError("Adam moments of the sampled Gaussians must be reset")
    # noise: opacity gate sigmoid(-(100 o - 0.5)) -> opaque Gaussians do not move
    sc = torch.full((2, 3), math.log(0.1))
    q = torch.tensor([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])
    d = noise_delta(sc, q, torch.tensor([10.0, -10.0]), 1.0, torch.Generator().manual_seed(0))
    if d[0].abs().max() > 1e-30 or d[1].abs().max() == 0:
        raise AssertionError("noise must be gated by opacity")
    return {"relocation_err": err, "relocated": ev["relocated"], "added": ev["added"]}


def schedules(method, device):
    from splatstudio.specs import ComponentSpec

    from .density import create_density

    checks = {
        "cos_7000": cosine_decay(7000, 1.0, 7000, 37000), "cos_22000": cosine_decay(22000, 1.0, 7000, 37000),
        "cos_37000": cosine_decay(37000, 1.0, 7000, 37000), "cos_30000": cosine_decay(30000, 1.0, 7000, 37000),
        "mlp_0": lr_decay(0, 3e-3, 3e-4, 30000, 3000, 1e-8),
        "mlp_3000": lr_decay(3000, 3e-3, 3e-4, 30000, 3000, 1e-8),
        "mlp_30000": lr_decay(30000, 3e-3, 3e-4, 30000, 3000, 1e-8),
    }
    exp = {"cos_7000": 1.0, "cos_22000": 0.5, "cos_37000": 0.0,
           "cos_30000": 0.5 * (1 + math.cos(math.pi * 23 / 30)),
           "mlp_0": 3e-11, "mlp_3000": 3e-3 * 0.1 ** 0.1, "mlp_30000": 3e-4}
    bad = {k: (checks[k], exp[k]) for k in exp if abs(checks[k] - exp[k]) > 1e-9 * max(1, exp[k] * 1e9)}
    if bad:
        raise AssertionError(f"schedule values wrong: {bad}")
    dens = create_density(ComponentSpec("d", "density", "x", (), (), {"appearance": "neural"}))
    dens.extent = 2.0
    lrs = dens.lrs(1)
    if abs(lrs["sh0"] - 0.00025 / A.SH_C0) > 1e-12 or abs(lrs["na_features"] - 0.008) > 1e-12 \
            or abs(lrs["means"] - 0.00016 * 2 * (0.01 ** (1 / 30000))) > 1e-12:
        raise AssertionError(f"density learning rates wrong: {lrs}")
    sh = create_density(ComponentSpec("d", "density", "x", (), (), {
        "appearance": "sh", "lr_base": 0.0025 * A.SH_C0, "lr_residual": 0.000125, "cosine_decay": False}))
    l2 = sh.lrs(25000)
    if abs(l2["sh0"] - 0.0025) > 1e-12 or abs(l2["shN"] - 0.000125) > 1e-12:
        raise AssertionError(f"SH baseline learning rates wrong: {l2}")
    return {**checks, "neural_lrs_step1": lrs}


def ppisp(method, device):
    from .ppisp_vendor import ppisp_apply
    from .ppisp_vendor.torch_reference import ppisp_apply_torch

    g = torch.Generator(device="cpu").manual_seed(0)
    W, H = 40, 30
    rgb = torch.rand(H, W, 3, generator=g).to(device)
    exp_p = (torch.randn(4, generator=g) * 0.3).to(device)
    vig = (torch.randn(1, 3, 5, generator=g) * 0.1).to(device)
    col = (torch.randn(4, 8, generator=g) * 0.05).to(device)
    crf = (torch.randn(1, 3, 4, generator=g) * 0.2).to(device)
    out = ppisp_apply(exp_p, vig, col, crf, rgb, None, W, H, 0, 2)
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    coords = torch.stack([xs + 0.5, ys + 0.5], -1).float().reshape(-1, 2)
    ref = ppisp_apply_torch(exp_p, vig, col, crf, rgb.reshape(-1, 3), coords, W, H, 0, 2).reshape(H, W, 3)
    err = float((out - ref).abs().max())
    if err > 1e-4:
        raise AssertionError(f"vendored CUDA PPISP != torch reference ({err})")
    # Bank round trip through na_blob and controller use for novel views
    params, camera = _scene(device, n=70000 // 4 * 4)
    cams = [{"viewmat": camera["viewmat"]}, {"viewmat": torch.eye(4, device=device) * 2}]
    b = _bank(device, ppisp=True, cameras=cams)
    previous, bank_mod.ACTIVE = bank_mod.ACTIVE, b
    try:
        with torch.no_grad():
            b.ppisp.exposure_params.fill_(0.3)
        r = _renderer()
        train_view = r.render(params, camera, 3)["rgb"].detach()
        with torch.no_grad():
            novel = r.render(params, camera, 3)["rgb"]
            flat = b.pack()
            params["na_blob"].data.view(-1)[:flat.numel()] = flat
        bank_mod.ACTIVE = None
        r2 = _renderer()
        with torch.no_grad():
            restored = r2.render(params, camera, 3)["rgb"]
        rt_err = float((restored - novel).abs().max())
        b2 = r2._loaded[1]
        live, back = b.tensors(), b2.tensors()
        state_err = max(float((live[k] - back[k]).abs().max()) for k in live)
        if set(live) != set(back) or state_err > 0:
            raise AssertionError(f"na_blob round trip does not restore the MLP/PPISP state ({state_err})")
        exposure_effect = float((train_view - novel).abs().max())
    finally:
        bank_mod.ACTIVE = previous
    # the Faster-GS forward is not bit-deterministic for 70k overlapping splats (~1.6e-4)
    if rt_err > 1e-3:
        raise AssertionError(f"na_blob round trip changes the evaluation render ({rt_err})")
    if exposure_effect < 1e-3:
        raise AssertionError("training views must use their per-frame PPISP parameters")
    return {"cuda_vs_torch_err": err, "blob_roundtrip_err": rt_err, "blob_values": int(flat.numel()),
            "frame_exposure_effect": exposure_effect}
