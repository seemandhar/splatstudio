"""Mechanism probes for SWAG. Each one compares against an independently written formula or an
ablation and raises if the mechanism is missing or wrong."""

import math

import torch
from splatstudio.components import build
from splatstudio.state import sample_scene

from . import bank as bank_mod
from .bank import SwagBank
from .hashgrid import HashGrid


def _reference_hash(grid, x):
    """Independent per-point / per-level loop (plain Python ints)."""
    out = []
    for p in x.tolist():
        feats = []
        for res, table in zip(grid.resolutions, grid.tables):
            size = table.shape[0]
            dense = (res + 1) ** 3 <= grid.table_size
            pos = [min(max(v, 0.0), 1.0) * res for v in p]
            base = [min(math.floor(v), res - 1) for v in pos]
            fr = [v - b for v, b in zip(pos, base)]
            acc = torch.zeros(table.shape[1], dtype=torch.float64)
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        c = (base[0] + dx, base[1] + dy, base[2] + dz)
                        if dense:
                            i = c[0] + (res + 1) * (c[1] + (res + 1) * c[2])
                        else:
                            i = ((c[0] * 1) ^ (c[1] * 2654435761) ^ (c[2] * 805459861)) % size
                        w = ((fr[0] if dx else 1 - fr[0]) * (fr[1] if dy else 1 - fr[1])
                             * (fr[2] if dz else 1 - fr[2]))
                        acc += w * table[i].detach().double().cpu()
            feats.append(acc)
        out.append(torch.cat(feats))
    return torch.stack(out)


def hash_grid(method, device):
    grid = HashGrid(levels=12, features=2, log2_table=19, n_min=16, n_max=2048, seed=3)
    with torch.no_grad():
        for t in grid.tables:
            t.normal_()  # larger values so mistakes are visible
    if grid.resolutions[0] != 16 or grid.resolutions[-1] not in (2047, 2048):
        raise AssertionError(f"level resolutions {grid.resolutions}")
    dense = [(r + 1) ** 3 <= grid.table_size for r in grid.resolutions]
    if not (dense[0] and not dense[-1]):
        raise AssertionError("expected both dense and hashed levels")
    g = torch.Generator().manual_seed(0)
    x = torch.rand(16, 3, generator=g)
    x[0] = torch.tensor([0.0, 1.0, 0.5])  # boundary
    got = grid(x).double()
    ref = _reference_hash(grid, x)
    err = float((got - ref).abs().max())
    if err > 1e-3:  # fp32 x*res at the 2048 level; a wrong index gives O(1) errors
        raise AssertionError(f"hash grid differs from reference: {err}")
    spread = float(got.std(0).mean())
    if spread < 1e-3:
        raise AssertionError("encoding is constant in x")
    if grid.out_dim + 3 + 24 != 51:
        raise AssertionError("MLP input dimension must be 51 (supp. Tab. 7)")
    return {"max_abs_error": err, "levels": grid.levels, "dense_levels": sum(dense),
            "resolutions": grid.resolutions, "feature_std": spread}


def concrete_opacity(method, device):
    from .renderer import concrete, image_opacity

    T, eps = 0.1, 1e-10
    d = torch.tensor([-2.0, -0.5, -0.05, 0.0, 0.05, 0.5, 0.9, 1.1, 3.0], dtype=torch.float64)
    u = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.5, 0.5, 0.99], dtype=torch.float64)
    got = concrete(d, u, T, eps)
    ref = torch.tensor([1 / (1 + math.exp(-(math.log(abs(a) + eps) + math.log(b)
                                            - math.log(1 - b)) / T))
                        for a, b in zip(d.tolist(), u.tolist())], dtype=torch.float64)
    err = float((got - ref).abs().max())
    half = concrete(d, torch.full_like(d, 0.5), T, eps)
    a = d.abs() + eps
    closed = a ** (1 / T) / (1 + a ** (1 / T))
    err_half = float((half - closed).abs().max())
    alpha = torch.tensor([0.9, 0.2, 0.5], dtype=torch.float64)
    dt = torch.tensor([0.3, 0.6, 0.0], dtype=torch.float64)
    op = image_opacity(alpha, dt)
    if not torch.allclose(op, torch.tensor([0.6, 0.0, 0.5], dtype=torch.float64)):
        raise AssertionError(f"Eq. 11 wrong: {op}")
    # Near-binary at T = 0.1 for |d| = 1: P(0.05 < s < 0.95) = 2 sigmoid(0.1 logit(0.95)) - 1
    # = 0.1455 (vs 0.9 for T = 1).
    gen = torch.Generator().manual_seed(0)
    uu = torch.rand(100000, generator=gen, dtype=torch.float64).clamp(1e-6, 1 - 1e-6)
    s = concrete(torch.ones_like(uu), uu, T, eps)
    mid = float(((s > 0.05) & (s < 0.95)).double().mean())
    mean = float(s.mean())
    if err > 1e-9 or err_half > 1e-9 or abs(mid - 0.1455) > 0.01 or abs(mean - 0.5) > 0.02:
        raise AssertionError(f"concrete: err {err}, err_half {err_half}, mid {mid}, mean {mean}")
    # The renderer must apply it (transient on) / not apply it (SWAG-A).
    renderer, _, _ = build(method)
    return {"max_abs_error": err, "u_half_error": err_half, "fraction_in_0.05_0.95": mid,
            "mean_at_location_1": mean, "renderer_transient": renderer.cfg["transient"]}


def _toy_bank(device, n_views=3, seed=0, points=None):
    vms = []
    for i in range(n_views):
        v = torch.eye(4, device=device)
        v[0, 3] = 0.05 * i
        vms.append(v)
    pts = points if points is not None else torch.rand(64, 3, device=device) - 0.5
    bank = SwagBank(vms, pts, {"seed": seed, "hash_log2_table": 14}, device)
    with torch.no_grad():  # put the concrete location near |d| = 1 so l_I visibly moves alpha^I
        bank.mlp.net[-1].bias[3] = 1.0
    return bank, vms


def _toy_camera(viewmat, device, w=48, h=40):
    K = torch.tensor([[60.0, 0, w / 2], [0, 60.0, h / 2], [0, 0, 1]], device=device)
    return {"viewmat": viewmat, "K": K, "width": w, "height": h}


def transient_variance_probe(method, device):
    from .density import transient_variance

    renderer, _, _ = build(method)
    params, _ = sample_scene(method, device)
    params = {k: v.detach() for k, v in params.items()}
    bank, vms = _toy_bank(device, 4)
    with torch.no_grad():
        bank.image_embeddings[:, 0] = torch.tensor([0.5, 1.5, 0.8, 1.2], device=device)
    n = len(params["means"])
    varying = torch.arange(n, device=device) % 2 == 1

    def fake_outputs(colors, means, embedding):  # Gaussian-wise known delta alpha
        out = torch.zeros(len(means), 4, device=means.device)
        out[:, 3] = torch.where(varying, embedding[0], torch.zeros_like(out[:, 3]))
        return out

    bank.mlp_outputs = fake_outputs
    cams = [_toy_camera(v, device) for v in vms]
    bank.eval()
    var = transient_variance(renderer, params, bank, cams)
    d = bank.image_embeddings[:, 0].double()
    ref = (d.abs() ** 10 / (1 + d.abs() ** 10)).var(unbiased=False)
    v0, v1 = var[~varying], var[varying]
    if float(v0.max()) > 1e-12 or float((v1 - ref).abs().max()) > 1e-5:
        raise AssertionError(f"variance static {float(v0.max())} varying {v1} vs {float(ref)}")
    return {"static_max_var": float(v0.max()), "varying_var": float(v1.mean()),
            "reference_var": float(ref)}


def appearance_mlp(method, device):
    from .renderer import concrete, sh_colors

    renderer, _, _ = build(method)
    params, _ = sample_scene(method, device)
    params = {k: v.detach() for k, v in params.items()}
    with torch.no_grad():
        params["shN"].normal_(0, 0.1)
        params["opacities"].fill_(3.0)
        params["scales"].fill_(math.log(0.2))
    bank, vms = _toy_bank(device, 3, points=params["means"])
    cam = _toy_camera(vms[1], device)
    prev = bank_mod.ACTIVE
    bank_mod.ACTIVE = bank
    try:
        bank.eval()
        with torch.no_grad():
            u = torch.full((len(params["means"]),), 0.5, device=device)
            emb = bank.image_embeddings[1].detach()
            bank.set_override(emb)
            col, op, _ = renderer.conditioned(params, cam, 3, bank)
            # independent computation
            c = sh_colors(params, cam, 3)
            lo, hi = bank.aabb
            h = bank.grid(((params["means"] - lo) / (hi - lo)).clamp(0, 1))
            x = torch.cat([c, h, emb[None].expand(len(c), -1)], -1)
            for layer in bank.mlp.net:
                x = layer(x)
            exp_col = torch.sigmoid(x[:, :3]) if renderer.cfg["appearance"] else c
            exp_op = torch.sigmoid(params["opacities"])
            if renderer.cfg["transient"]:
                exp_op = (exp_op - concrete(x[:, 3], u, 0.1, 1e-10)).clamp_min(0)
            e1 = float((col - exp_col).abs().max())
            e2 = float((op - exp_op).abs().max())
            img_a = renderer.render(params, cam, 3)["rgb"]
            bank.set_override(bank.image_embeddings[0].detach() + 3.0)
            img_b = renderer.render(params, cam, 3)["rgb"]
            bank.set_override(None)
            img_none_bank = None
        bank_mod.ACTIVE = None
        with torch.no_grad():
            img_none_bank = renderer.render(params, cam, 3)["rgb"]
        bank_mod.ACTIVE = bank
        diff = float((img_a - img_b).abs().max())
        if e1 > 1e-5 or e2 > 1e-5 or diff < 1e-4:
            raise AssertionError(f"colour err {e1}, opacity err {e2}, embedding effect {diff}")
        # gradient reaches the bank and the screen-space means through a training render
        bank.train()
        params2 = {k: v.clone().requires_grad_(True) for k, v in params.items()}
        out = renderer.render(params2, _toy_camera(vms[2], device), 3)
        out["info"]["means2d"].retain_grad()
        out["rgb"].sum().backward()
        if out["view_index"] != 2:
            raise AssertionError("training view not matched to its embedding")
        g_emb = float(bank.image_embeddings.grad[2].abs().sum())
        g_other = float(bank.image_embeddings.grad[0].abs().sum())
        g_mlp = float(sum(p.grad.abs().sum() for p in bank.mlp.parameters()))
        g_m2d = float(out["info"]["means2d"].grad.abs().sum())
        if g_emb <= 0 or g_other != 0 or g_mlp <= 0 or g_m2d <= 0:
            raise AssertionError(f"grads emb {g_emb} other {g_other} mlp {g_mlp} m2d {g_m2d}")
    finally:
        bank_mod.ACTIVE = prev
    return {"colour_error": e1, "opacity_error": e2, "embedding_effect": diff,
            "grad_embedding": g_emb, "grad_mlp": g_mlp, "grad_means2d": g_m2d,
            "no_bank_render_finite": bool(torch.isfinite(img_none_bank).all())}


def bank_training(method, device):
    from .density import position_lr

    _, density, _ = build(method)
    if density.cfg["refine_every"] != 2000 or density.cfg["refine_stop_iter"] != 30000:
        raise AssertionError("densification schedule differs from supp. Tab. 7")
    if density.inner.strategy.refine_every != 2000:
        raise AssertionError("schedule not forwarded to the strategy")
    lr0 = 1.0
    checks = {s: position_lr(lr0, s, 30000, 0.01) for s in (0, 15000, 30000, 60000)}
    if not (abs(checks[15000] - 0.1) < 1e-9 and abs(checks[60000] - 0.01) < 1e-12):
        raise AssertionError(f"position lr schedule {checks}")
    bank, _ = _toy_bank(device, 2)
    before = [p.detach().clone() for p in (bank.image_embeddings, bank.grid.table,
                                           bank.mlp.net[0].weight)]
    pts = (torch.rand(256, 3, device=device) - 0.5) * 0.8
    out = bank.mlp_outputs(torch.rand(256, 3, device=device), pts, bank.image_embeddings[0])
    out.square().sum().backward()
    bank.after_step(1)
    after = [bank.image_embeddings, bank.grid.table, bank.mlp.net[0].weight]
    moved = [float((a - b).abs().max()) for a, b in zip(after, before)]
    if min(moved) <= 0:
        raise AssertionError(f"bank parameters not updated: {moved}")
    return {"moved": moved, "lr": checks}


def nerfw_protocol(method, device):
    from .evaluate import metrics, optimize_embedding

    renderer, _, _ = build(method)
    params, _ = sample_scene(method, device, n=12)
    params = {k: v.detach() for k, v in params.items()}
    with torch.no_grad():
        params["scales"].fill_(math.log(0.3))
        params["opacities"].fill_(2.0)
    bank, vms = _toy_bank(device, 2, points=params["means"])
    for p in bank.parameters():
        p.requires_grad_(False)
    cam = _toy_camera(vms[0], device, w=96, h=80)
    prev = bank_mod.ACTIVE
    bank_mod.ACTIVE = bank
    try:
        bank.eval()
        with torch.no_grad():
            bank.set_override(bank.image_embeddings[1] + 2.0)
            target = renderer.render(params, cam, 3)["rgb"]
            bank.set_override(None)
        frame = {**cam, "image": target}
        _, hist = optimize_embedding(renderer, params, bank, frame, 3, {"iters": 60, "lr": 0.05})
        if not hist[-1] < 0.7 * hist[0]:
            raise AssertionError(f"embedding optimisation did not reduce the loss: {hist[0]}"
                                 f" -> {hist[-1]}")
        w = cam["width"]
        pred = target.clone()
        noisy = target.clone()
        noisy[:, : w // 2] = torch.rand_like(noisy[:, : w // 2])
        m1 = metrics(pred[:, w // 2:], target[:, w // 2:], device)
        m2 = metrics(pred[:, w // 2:], noisy[:, w // 2:], device)
        if abs(m1["psnr"] - m2["psnr"]) > 1e-6:
            raise AssertionError("right-half metric depends on the left half")
    finally:
        bank_mod.ACTIVE = prev
    return {"loss_first": hist[0], "loss_last": hist[-1]}


def hash_cuda(method, device):
    """CUDA hash kernel == PyTorch reference (forward, table gradient, centre gradient)."""
    grid = HashGrid(levels=12, features=2, log2_table=19, n_min=16, n_max=2048, seed=5).to(device)
    with torch.no_grad():
        grid.table.normal_()
    g = torch.Generator(device=device).manual_seed(1)
    x = torch.rand(5000, 3, generator=g, device=device) * 1.1 - 0.05  # some outside the box
    go = torch.randn(5000, 24, generator=g, device=device)
    res = {}
    for name, use in (("cuda", True), ("torch", False)):
        grid.use_cuda = use
        xx = x.clone().requires_grad_(True)
        grid.table.grad = None
        y = grid(xx)
        (y * go).sum().backward()
        res[name] = (y.detach(), grid.table.grad.clone(), xx.grad.clone())
    grid.use_cuda = True
    errs = [float((a - b).abs().max()) for a, b in zip(res["cuda"], res["torch"])]
    scale = [float(b.abs().max()) for b in res["torch"]]
    rel = [e / max(s, 1e-12) for e, s in zip(errs, scale)]
    if max(rel) > 1e-3:
        raise AssertionError(f"CUDA hash kernel differs: rel errors {rel}")
    return {"rel_error_out_table_x": rel}
