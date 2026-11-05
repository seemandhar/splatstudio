"""Mechanism probes for the Compact-3DGS port. Each raises if the mechanism is absent or wrong."""

import io
import math
from heapq import heapify, heappop, heappush

import numpy as np
import torch
from gsplat import rasterization
from splatstudio.components import build
from splatstudio.state import initialize
from splatstudio.verification import ProbeUnavailable

from . import codec, hashgrid, model
from .density import schedule_events
from .renderer import BLOB, FEATURE, MASK

C0 = 0.28209479177387814


def _cuda(device):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ProbeUnavailable("CUDA is required")


def _camera(device, size=64, f=60.0):
    return {"viewmat": torch.eye(4, device=device),
            "K": torch.tensor([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]], device=device),
            "width": size, "height": size}


def _scene(method, device, n=1200, seed=0):
    """Random Gaussians in front of an identity camera; components built and density initialised."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    points = torch.rand(n, 3, generator=g) * 2 - 1
    points[:, 2] = points[:, 2] * 0.5 + 3.0
    colors = torch.rand(n, 3, generator=g)
    params, opts = initialize(method, points.to(device), colors.to(device))
    with torch.no_grad():
        params["scales"].copy_((0.03 * (0.5 + 1.5 * torch.rand(n, 3, generator=g))).log()
                               .to(device))
        params["opacities"].fill_(0.0)
        params["quats"].copy_(torch.nn.functional.normalize(
            torch.randn(n, 4, generator=g), dim=-1).to(device))
    renderer, density, losses = build(method)
    density.initialize(params, opts, 1.0)
    renderer.bank = density.bank
    with torch.no_grad():  # a non-trivial (non-initial) colour field
        density.bank.hash.params.normal_(0, 0.5)
        for w in density.bank.mlp:
            w.mul_(3.0)
    return params, opts, renderer, density, dict(losses)


def _plain(means, quats, scales, opacities, sh0, camera):
    img, alpha, _ = rasterization(means=means, quats=quats, scales=scales, opacities=opacities,
                                  colors=sh0[:, None, :], viewmats=camera["viewmat"][None],
                                  Ks=camera["K"][None], width=camera["width"],
                                  height=camera["height"], sh_degree=0, packed=False)
    return img[0] + (1 - alpha[0])


def _release():
    model.ACTIVE["bank"] = None


# ------------------------------------------------------------------------------ hash grid
EXPECTED_LEVEL_SIZES = [4096, 13824, 39304, 117656, 357912] + [524288] * 11


def hash_grid(method, device):
    _cuda(device)
    scales, res, offsets = hashgrid.level_table()
    sizes = (offsets[1:] - offsets[:-1]).tolist()
    if sizes != EXPECTED_LEVEL_SIZES or int(offsets[-1]) * 2 != 12_599_920:
        raise AssertionError(f"level table {sizes} differs from tcnn's offset table")
    if res.tolist()[:6] != [16, 24, 34, 49, 71, 102] or res.tolist()[-1] != 4085:
        raise AssertionError(f"resolutions {res.tolist()} differ from tcnn grid_resolution")
    grid = hashgrid.HashGrid().to(device)
    with torch.no_grad():
        grid.params.normal_()
    g = torch.Generator(device="cpu").manual_seed(1)
    x = torch.rand(4000, 3, generator=g)
    x[:8] = torch.tensor([[0, 0, 0], [1, 1, 1], [0, 1, 0.5], [0.999, 0.001, 0.5],
                          [0.5, 0.5, 0.5], [1, 0, 1], [0.25, 0.75, 1], [0, 0.3, 1]])
    x = x.to(device)
    out = grid(x)
    ref = grid.reference(x)
    err = float((out - ref).abs().max())
    if err > 1e-4:
        raise AssertionError(f"CUDA hash grid differs from the reference by {err}")
    w = torch.randn_like(out)
    (out * w).sum().backward()
    p = grid.params.detach().clone().requires_grad_(True)
    (grid.reference(x, p) * w).sum().backward()
    gerr = float((grid.params.grad - p.grad).abs().max() / p.grad.abs().max())
    if gerr > 1e-4:
        raise AssertionError(f"hash-grid parameter gradient differs by {gerr} (relative)")
    # a hashed level really hashes (prime hash) and a dense level really is dense
    probe = torch.tensor([[0.3, 0.6, 0.9]], device=device)
    with torch.no_grad():
        grid.params.zero_()
        lvl = 10
        pos = probe[0] * float(scales[lvl]) + 0.5
        cell = torch.floor(pos).long().tolist()
        h = ((cell[0] * 1) ^ ((cell[1] * 2654435761) & 0xFFFFFFFF)
             ^ ((cell[2] * 805459861) & 0xFFFFFFFF)) % sizes[lvl]
        grid.params[(int(offsets[lvl]) + h) * 2] = 1.0
        frac = (pos - torch.floor(pos)).tolist()
        expected = (1 - frac[0]) * (1 - frac[1]) * (1 - frac[2])
        got = float(grid(probe)[0, 2 * lvl])
    if abs(got - expected) > 1e-5:
        raise AssertionError(f"hashed-level corner weight {got} != {expected}")
    return {"levels": len(sizes), "entries": int(offsets[-1]), "params": int(offsets[-1]) * 2,
            "max_abs_error": err, "max_rel_grad_error": gerr, "hashed_corner_weight": got}


# ---------------------------------------------------------------------------- neural colour
def _contract_ref(p):
    out = []
    for v in p.detach().cpu().double().numpy():
        r = np.linalg.norm(v)
        out.append(v if r <= 1 else (2 - 1 / r) * v / r)  # paper Eq. 8
    return torch.tensor(np.array(out)) / 4 + 0.5


def _sh9_ref(v):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    pi = math.pi
    return torch.stack([
        torch.full_like(x, 1 / (2 * math.sqrt(pi))),
        -math.sqrt(3) * y / (2 * math.sqrt(pi)), math.sqrt(3) * z / (2 * math.sqrt(pi)),
        -math.sqrt(3) * x / (2 * math.sqrt(pi)),
        math.sqrt(15) * x * y / (2 * math.sqrt(pi)), -math.sqrt(15) * y * z / (2 * math.sqrt(pi)),
        math.sqrt(5) * (3 * z * z - 1) / (4 * math.sqrt(pi)),
        -math.sqrt(15) * x * z / (2 * math.sqrt(pi)),
        math.sqrt(15) * (x * x - y * y) / (4 * math.sqrt(pi))], dim=-1)


def neural_color(method, device):
    _cuda(device)
    try:
        params, _, renderer, density, _ = _scene(method, device)
        bank = density.bank
        p = params["means"].detach().clone()
        p[:5] = torch.tensor([[0.2, 0.1, 0.3], [3.0, 0.0, 0.0], [0, -5, 2], [0.5, 0.5, 0.7],
                              [10, 10, 10]], device=device)
        cref = _contract_ref(p).float().to(device)
        if float((model.contract_to_unisphere(p) - cref).abs().max()) > 1e-6:
            raise AssertionError("contraction differs from paper Eq. 8")
        cam = _camera(device)
        center = model.camera_center(cam["viewmat"])
        d = p - center
        d = d / d.norm(dim=1, keepdim=True)
        enc = _sh9_ref((2 * d - 1).double()).float()  # official quirk: tcnn maps x -> 2x - 1
        feats = bank.hash.reference(cref)
        inp = torch.cat([feats, enc, torch.ones(len(p), 7, device=device)], 1)
        w = [t.detach() for t in bank.mlp]
        h = torch.relu(inp @ w[0].T)
        h = torch.relu(h @ w[1].T)
        sh0_ref = (h @ w[2].T)[:, :3]
        sh0 = bank.color(p, center)
        err = float((sh0 - sh0_ref).abs().max())
        if err > 1e-3 * max(1.0, float(sh0_ref.abs().max())):
            raise AssertionError(f"neural colour differs from the independent formula by {err}")
        bank.step = 1
        with torch.no_grad():
            img = renderer.render(params, cam)["rgb"]
            m = (torch.sigmoid(params[MASK]) > 0.01).float()  # Eq. 1 forward value
            ref_img = _plain(params["means"], params["quats"], params["scales"].exp() * m[:, None],
                             torch.sigmoid(params["opacities"]) * m,
                             bank.color(params["means"], center), cam)
            ierr = float((img - ref_img).abs().max())
            if ierr > 1e-5:
                raise AssertionError(f"renderer image differs from SH0 = MLP colours by {ierr}")
            cam2 = dict(cam)
            v = torch.eye(4, device=device)
            v[0, 3] = 0.3
            cam2["viewmat"] = v
            c1 = bank.color(params["means"], model.camera_center(cam["viewmat"]))
            c2 = bank.color(params["means"], model.camera_center(v))
            view_delta = float((c1 - c2).abs().max())
        if view_delta < 1e-4:
            raise AssertionError("colour is not view dependent")
        return {"max_color_error": err, "max_image_error": ierr, "view_dependence": view_delta,
                "mlp_params_tcnn": bank.mlp_param_count(), "mlp_input": bank.n_in}
    finally:
        _release()


# ------------------------------------------------------------------------------ volume mask
def volume_mask(method, device):
    _cuda(device)
    try:
        params, opts, renderer, density, losses = _scene(method, device)
        bank = density.bank
        n = len(params["means"])
        with torch.no_grad():
            params[MASK].copy_(torch.linspace(-8, 3, n, device=device))
        keep = torch.sigmoid(params[MASK]) > 0.01
        cam = _camera(device)
        center = model.camera_center(cam["viewmat"])
        bank.step = 1
        with torch.no_grad():
            img = renderer.render(params, cam)["rgb"]
            col = bank.color(params["means"], center)
            ref = _plain(params["means"][keep], params["quats"][keep],
                         params["scales"].exp()[keep], torch.sigmoid(params["opacities"])[keep],
                         col[keep], cam)
        err = float((img - ref).abs().max())
        if err > 1e-5:
            raise AssertionError(f"masked render differs from the render of unmasked Gaussians: {err}")
        # STE gradient: dL/dm = (dL/ds_hat . s + dL/do_hat o) sigmoid'(m) on every Gaussian
        target = torch.rand(64, 64, 3, generator=torch.Generator().manual_seed(3)).to(device)
        out = renderer.render(params, cam)
        loss = (out["rgb"] - target).square().sum()
        (gm,) = torch.autograd.grad(loss, params[MASK])
        s = params["scales"].detach().exp()
        o = torch.sigmoid(params["opacities"].detach())
        hard = keep.float()
        s_hat = (s * hard[:, None]).requires_grad_(True)
        o_hat = (o * hard).requires_grad_(True)
        img2 = _plain(params["means"].detach(), params["quats"].detach(), s_hat, o_hat,
                      col.detach(), cam)
        gs, go = torch.autograd.grad((img2 - target).square().sum(), (s_hat, o_hat))
        sig = torch.sigmoid(params[MASK].detach())
        analytic = ((gs * s).sum(1) + go * o) * sig * (1 - sig)
        scale = float(analytic.abs().max())
        gerr = float((gm - analytic).abs().max())
        if scale == 0 or gerr > 1e-3 * scale:
            raise AssertionError(f"mask STE gradient differs from the analytic one: {gerr}/{scale}")
        lam = 5e-4
        lv = losses["mask-loss"](params, {}, None)
        (glv,) = torch.autograd.grad(lv, params[MASK])
        if abs(float(lv) - lam * float(sig.mean())) > 1e-9:
            raise AssertionError("mask loss != lambda mean sigmoid(m)")
        if float((glv - lam / n * sig * (1 - sig)).abs().max()) > 1e-9:
            raise AssertionError("mask loss gradient wrong")
        cfg = density.cfg
        expected = {500: [], 600: ["mask_prune"], 650: [], 14900: ["mask_prune"], 15000:
                    ["mask_prune"], 15100: [], 16000: ["mask_prune"], 29999: [],
                    30000: ["mask_prune", "encode"]}
        got = {k: schedule_events(k, cfg) for k in expected}
        if got != expected:
            raise AssertionError(f"mask prune schedule {got} != {expected}")
        before = n
        density._mask_prune(params, opts)
        if len(params["means"]) != int(keep.sum()) or len(params["means"]) == before:
            raise AssertionError("mask pruning did not remove exactly the masked Gaussians")
        return {"masked": int((~keep).sum()), "image_error": err, "mask_grad_error": gerr,
                "mask_grad_scale": scale, "mask_loss": float(lv)}
    finally:
        _release()


# ------------------------------------------------------------------------------------ R-VQ
def rvq(method, device):
    _cuda(device)
    torch.manual_seed(0)
    # K-means init: one Lloyd iteration from the sampled means, empty clusters keep the sample
    x = torch.randn(500, 3, device=device)
    g = torch.Generator(device=device).manual_seed(5)
    km = model.kmeans(x, 16, 1, g)
    g = torch.Generator(device=device).manual_seed(5)
    init = x[torch.randperm(500, device=device, generator=g)[:16]]
    d = ((x[:, None] - init[None]) ** 2).sum(-1)
    a = d.argmin(1)
    ref = torch.stack([x[a == k].mean(0) if (a == k).any() else init[k] for k in range(16)])
    if float((km - ref).abs().max()) > 1e-5:
        raise AssertionError("kmeans init differs from one Lloyd iteration")
    # residual cascade with fixed codebooks (training=False path)
    vq = model.ResidualVQ(3, 8, 3, 1, 1e-4).to(device)
    books = [torch.randn(8, 3, device=device) * (0.5**k) for k in range(3)]
    for layer, b in zip(vq.layers, books):
        layer.init_codebook(b)  # sets initted; codebook replaced below
        with torch.no_grad():
            layer.embed.copy_(b)
    q, idx = vq(x, train=False)
    res, out, ref_idx = x.clone(), torch.zeros_like(x), []
    for b in books:
        i = ((res[:, None] - b[None]) ** 2).sum(-1).argmin(1)
        ref_idx.append(i)
        out = out + b[i]
        res = res - b[i]
    if not torch.equal(idx, torch.stack(ref_idx, -1)) or float((q - out).abs().max()) > 1e-6:
        raise AssertionError("R-VQ cascade differs from the independent loop (Eq. 4-5)")
    if float((vq.decode(idx) - out).abs().max()) > 1e-6:
        raise AssertionError("R-VQ decode differs")
    # training path: in-place Adam(lr) step on mse(Z[i], x), re-quantise, straight-through
    xs = x.clone().requires_grad_(True)
    first = vq.layers[0]
    e0 = first.embed.detach().clone()
    i0 = ((x[:, None] - e0[None]) ** 2).sum(-1).argmin(1)
    grad = torch.zeros_like(e0).index_add_(0, i0, 2 * (e0[i0] - x)) / x.numel()
    expected = e0 - 1e-4 * grad / (grad.abs() + 1e-8)  # first Adam step (bias-corrected)
    qt, _ = vq(xs, train=True)
    if float((first.embed.detach() - expected).abs().max()) > 1e-7:
        raise AssertionError("in-place codebook update is not one Adam(1e-4) step on the mse")
    (gx,) = torch.autograd.grad(qt.sum(), xs)
    if float((gx - 3.0).abs().max()) > 1e-6:
        raise AssertionError("straight-through gradient is not L x upstream (vector-quantize)")
    # renderer: R-VQ only for steps > rvq_iter
    try:
        params, _, renderer, density, _ = _scene(method, device, seed=2)
        bank = density.bank
        cfg = density.cfg
        if cfg["total_steps"] - cfg["rvq_iter"] != 1000 or bank.rvq_iter != cfg["rvq_iter"]:
            raise AssertionError("R-VQ must cover exactly the last 1000 iterations (§3.4)")
        if (cfg["rvq_size"], cfg["rvq_num"], cfg["kmeans_iters"]) != (64, 6, 1):
            raise AssertionError("R-VQ configuration differs from the paper (C=64, L=6)")
        cam = _camera(device)
        bank.step = cfg["rvq_iter"]
        with torch.no_grad():
            a = renderer.render(params, cam)["rgb"]
        if bank.vq_scale.initted:
            raise AssertionError("R-VQ ran before rvq_iter")
        bank.step = cfg["rvq_iter"] + 1
        with torch.no_grad():
            b = renderer.render(params, cam)["rgb"]
            s_q = bank.vq_scale(params["scales"].exp(), train=False)[0]
            r_q = bank.vq_rot(torch.nn.functional.normalize(params["quats"], dim=-1), False)[0]
            col = bank.color(params["means"], model.camera_center(cam["viewmat"]))
            m = (torch.sigmoid(params[MASK]) > 0.01).float()  # Eq. 1 forward value
            c = _plain(params["means"], r_q, s_q * m[:, None],
                       torch.sigmoid(params["opacities"]) * m, col, cam)
        if not bank.vq_scale.initted or float((b - c).abs().max()) > 1e-5:
            raise AssertionError("render after rvq_iter does not use the R-VQ geometry")
        if float((a - b).abs().max()) < 1e-5:
            raise AssertionError("R-VQ has no effect on the render")
    finally:
        _release()
    return {"kmeans_error": float((km - ref).abs().max()), "ste_grad": float(gx.mean()),
            "rvq_render_delta": float((a - b).abs().max())}


# --------------------------------------------------------------------------------- Huffman
def _textbook_cost(freqs):
    heap = list(freqs)
    heapify(heap)
    cost = 0
    while len(heap) > 1:
        a, b = heappop(heap), heappop(heap)
        cost += a + b
        heappush(heap, a + b)
    return cost


def huffman_codec(method, device):
    _cuda(device)
    rng = np.random.default_rng(0)
    sym = np.minimum(rng.geometric(0.15, size=200_000), 255) - 1
    lengths, counts, total = codec.huffman_lengths(sym)
    table = codec.dahuffman_table(counts)
    eof_len = table[codec.EOF][0]
    opt = _textbook_cost(list(counts.values()) + [1])
    if total + eof_len != opt:
        raise AssertionError(f"Huffman cost {total}+{eof_len} != optimal {opt}")
    kraft = sum(2.0 ** -lengths[s] for s in lengths) + 2.0 ** -eof_len
    if abs(kraft - 1) > 1e-9:
        raise AssertionError("code lengths violate Kraft equality")
    enc = codec.huffman_encode(sym)
    dec = codec.huffman_decode(enc).numpy()
    if not np.array_equal(dec, sym):
        raise AssertionError("canonical Huffman decode is not lossless")
    if enc["bits"] != total or len(enc["payload"]) != (total + 7) // 8:
        raise AssertionError("payload size differs from the Huffman cost")
    # post_quant: min/max of the *unpruned* values; pruned values map to round(-min/(max-min)*255)
    v = torch.tensor([-0.5, -0.05, 0.0, 0.08, 0.2, 1.5])
    out, q, mm = codec.post_quant(v, prune=True, prune_threshold=0.1)
    exp_q = [round((x * (abs(x) > 0.1) + 0.5) / 2.0 * 255) for x in v.tolist()]
    if q.tolist() != exp_q or mm.tolist() != [-0.5, 1.5]:
        raise AssertionError(f"post_quant {q.tolist()} != {exp_q}")
    if float((codec.dequant(q, mm) - out).abs().max()) > 0:
        raise AssertionError("dequantisation differs from post_quant's decoded value")
    idx = rng.integers(0, 64, size=999)
    if not np.array_equal(codec.unpack_indices(codec.pack_indices(idx, 6), 999, 6), idx):
        raise AssertionError("6-bit index packing is not lossless")
    return {"symbols": int(sym.size), "bits": total, "optimal_with_eof": opt, "eof_len": eof_len,
            "bits_per_symbol": total / sym.size}


# ------------------------------------------------------------------------------ schedules
def net_schedule(method, device):
    from .density import DEFAULTS

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p], lr=0.01, eps=1e-15)
    sched = torch.optim.lr_scheduler.ChainedScheduler([
        torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=100),
        torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[5000, 15000, 25000], gamma=0.33)])
    worst = 0.0
    for epoch in range(30001):
        if epoch in (0, 1, 50, 99, 100, 101, 4999, 5000, 5001, 15000, 25000, 29999):
            ref = opt.param_groups[0]["lr"]
            got = model.net_lr(0.01, epoch)
            worst = max(worst, abs(got - ref) / ref)
        opt.step()
        sched.step()
    if worst > 1e-6:
        raise AssertionError(f"closed-form net lr differs from ChainedScheduler by {worst}")
    cfg = dict(DEFAULTS)
    # official loop: densify branch (it < 15000: densify_and_prune if it > 500 and it % 100 == 0)
    # else mask_prune if it % 1000 == 0; final_prune at 30000
    for it in range(1, 30001):
        official = []
        if it < 15000:
            if it > 500 and it % 100 == 0:
                official.append("mask_prune")
        elif it % 1000 == 0:
            official.append("mask_prune")
        if it == 30000:
            official.append("encode")
        if schedule_events(it, cfg) != official:
            raise AssertionError(f"step {it}: {schedule_events(it, cfg)} != {official}")
    return {"max_rel_lr_error": worst, "steps_checked": 30000}


# ------------------------------------------------------------------------------ bake + eval
def bake_eval(method, device):
    _cuda(device)
    try:
        params, opts, renderer, density, _ = _scene(method, device, seed=4)
        bank = density.bank
        cam = _camera(device)
        with torch.no_grad():
            params[MASK][:50] = -10.0  # these must be removed by the final prune
        # unbaked, bank-free render must refuse
        fresh = type(renderer)(next(c for c in method.components if c.kind == "renderer"))
        model.ACTIVE["bank"] = None
        try:
            fresh.render(params, cam)
            raise AssertionError("an un-baked state rendered without the bank")
        except ValueError:
            pass
        model.ACTIVE["bank"] = bank
        bank.step = density.cfg["rvq_iter"] + 1
        out = renderer.render(params, cam)  # trains the codebooks once (in-place step)
        out["rgb"].sum().backward()
        density._mask_prune(params, opts)
        n = len(params["means"])
        report = density._encode(params, opts)
        enc = density.encoded
        if n != 1150 or not bank.baked:
            raise AssertionError(f"final prune/bake wrong (n={n})")
        with torch.no_grad():
            baked = renderer.render(params, cam)["rgb"]
        buf = io.BytesIO()
        torch.save({"params": dict(params)}, buf)
        buf.seek(0)
        loaded = torch.load(buf, map_location=device, weights_only=True)["params"]
        model.ACTIVE["bank"] = None
        fresh = type(renderer)(next(c for c in method.components if c.kind == "renderer"))
        with torch.no_grad():
            again = fresh.render(loaded, cam)["rgb"]
        if not torch.equal(baked, again):
            raise AssertionError("fresh bank-free render differs from the baked render")
        # decoded state == official in-memory decoded model
        dec = codec.decode(density.stream, device)
        with torch.no_grad():
            feats = bank.features(dec["means"], dec["hash"])
            if not torch.equal(loaded[FEATURE], feats):
                raise AssertionError("baked features differ from hash(contract(fp16 xyz))")
            if not torch.allclose(loaded["scales"].exp(), dec["scales"].abs().clamp_min(1e-12)):
                raise AssertionError("baked scales differ from the R-VQ decode")
        # sizes: official final_prune formulas
        n_hash = 12_599_920
        sizes = enc["paper_size_mb"]
        exp_ours = (n * 3 * 2 + n * 2 + 2 * (n * 36 / 8) + 64 * 6 * 3 * 4 + 64 * 6 * 4 * 4
                    + n_hash * 2 + 8192 * 2) / 1e6
        if abs(sizes["ours"]["total"] - exp_ours) > 1e-9:
            raise AssertionError(f"'Ours' size {sizes['ours']['total']} != {exp_ours}")
        if density.cfg["comp"]:
            hb = enc["huffman_bits"]
            exp_pp = (n * 6 + hb["scale"] / 8 + hb["rotation"] / 8 + 64 * 6 * 7 * 4
                      + hb["opacity"] / 8 + hb["hash"] / 8 + 8192 * 2) / 1e6
            if abs(sizes["ours_pp"]["total"] - exp_pp) > 1e-9:
                raise AssertionError("'Ours+PP' size differs from the official formula")
        blob_rows_needed = math.ceil((8 + 8192) / 8)
        return {"gaussians": n, "paper_size": report, "actual_bytes": enc["actual_bytes"]["total"],
                "blob_rows_needed": blob_rows_needed, "fresh_equals_baked": True}
    finally:
        _release()


def train_render_gradient(method, device):
    _cuda(device)
    try:
        params, _, renderer, density, losses = _scene(method, device, seed=6)
        cam = _camera(device)
        density.bank.step = 1
        out = renderer.render(params, cam)
        density.before_backward(params, None, 1, out)
        target = (torch.arange(64, device=device)[None, :, None] % 2).expand(64, 64, 3).float()
        loss = (out["rgb"] - target).square().mean() + losses["mask-loss"](params, out, target)
        loss.backward()
        checks = {"means2d": out["info"]["means2d"].grad, "hash": density.bank.hash.params.grad,
                  "mlp0": density.bank.mlp[0].grad, "mask": params[MASK].grad,
                  "scales": params["scales"].grad, "opacities": params["opacities"].grad,
                  "means": params["means"].grad}
        ev = {}
        for k, g in checks.items():
            if g is None or not torch.isfinite(g).all() or not torch.count_nonzero(g):
                raise AssertionError(f"no finite non-zero gradient for {k}")
            ev[k] = float(g.norm())
        if params[FEATURE].grad is not None or params[BLOB].grad is not None:
            raise AssertionError("eval-only fields received a gradient")
        return ev
    finally:
        _release()
