"""Mechanism probes for WE-GS. Each compares against an independently written formula or an
ablation and raises if the mechanism is missing or wrong."""

import math

import torch
from gsplat import rasterization
from splatstudio.components import build
from splatstudio.state import sample_scene

from . import bank as bank_mod
from .bank import BANK_DEFAULTS, WEGSBank
from .losses import projected_mask_values

SMALL = {"encoder_mid": 8, "unet_channels": 8, "transnet_width": 32}


def _camera(device, w=48, h=40, image=None):
    K = torch.tensor([[60.0, 0, w / 2], [0, 60.0, h / 2], [0, 0, 1]], device=device)
    cam = {"viewmat": torch.eye(4, device=device), "K": K, "width": w, "height": h}
    cam["image"] = image if image is not None else torch.rand(h, w, 3, device=device)
    return cam


def _scene(method, device, n=12):
    params, _ = sample_scene(method, device, n=n)
    params = {k: v.detach() for k, v in params.items()}
    with torch.no_grad():
        params["shN"].normal_(0, 0.1)
        params["opacities"].fill_(2.0)
        params["scales"].fill_(math.log(0.15))
    return params


def _with_bank(bank):
    class _Ctx:
        def __enter__(self):
            self.prev = bank_mod.ACTIVE
            bank_mod.ACTIVE = bank

        def __exit__(self, *a):
            bank_mod.ACTIVE = self.prev

    return _Ctx()


def _pe(x, n):
    out = [x]
    for i in range(n):
        f = 2.0**i
        out += [torch.sin(f * x), torch.cos(f * x)]
    return torch.cat(out, -1)


def sh_transfer(method, device):
    renderer, _, _ = build(method)
    params = _scene(method, device)
    bank = WEGSBank(SMALL, device)
    cam = _camera(device)
    with _with_bank(bank), torch.no_grad():
        out = renderer.render(params, cam, 3)
        lk, _ = bank.encode(cam["image"])
        sh = torch.cat([params["sh0"], params["shN"]], 1)
        n = len(sh)
        x = torch.cat([sh.reshape(n, -1), lk[None].expand(n, -1), _pe(params["means"], 4)], -1)
        for layer in bank.transnet.net:
            x = layer(x)
        ref_sh = sh + torch.tanh(x).view(sh.shape)
        ref, _, _ = rasterization(params["means"], params["quats"], params["scales"].exp(),
                                  torch.sigmoid(params["opacities"]), ref_sh,
                                  cam["viewmat"][None], cam["K"][None], cam["width"],
                                  cam["height"], sh_degree=3)
        err = float((out["rgb"] - ref[0]).abs().max())
        base, _, _ = rasterization(params["means"], params["quats"], params["scales"].exp(),
                                   torch.sigmoid(params["opacities"]), sh,
                                   cam["viewmat"][None], cam["K"][None], cam["width"],
                                   cam["height"], sh_degree=3)
        residual_effect = float((out["rgb"] - base[0]).abs().max())
        other = renderer.render(params, cam, 3, embedding=lk + 2.0)["rgb"]
        emb_effect = float((out["rgb"] - other).abs().max())
    if err > 1e-5 or residual_effect < 1e-4 or emb_effect < 1e-4:
        raise AssertionError(f"render err {err}, residual effect {residual_effect}, "
                             f"embedding effect {emb_effect}")
    p2 = {k: v.clone().requires_grad_(True) for k, v in params.items()}
    with _with_bank(bank):
        renderer.render(p2, cam, 3)["rgb"].sum().backward()
    g_t = float(sum(p.grad.abs().sum() for p in bank.transnet.parameters()))
    g_e = float(sum(p.grad.abs().sum() for p in bank.encoder.feature_parameters()))
    g_sh = float(p2["sh0"].grad.abs().sum())
    if g_t <= 0 or g_e <= 0 or g_sh <= 0:
        raise AssertionError(f"grads transnet {g_t} encoder {g_e} sh {g_sh}")
    return {"render_error": err, "residual_effect": residual_effect,
            "embedding_effect": emb_effect, "grad_transnet": g_t, "grad_encoder": g_e}


def attention_encoder(method, device):
    torch.manual_seed(0)
    bank = WEGSBank(SMALL, device)
    enc = bank.encoder
    img = torch.rand(40, 56, 3, device=device)
    with torch.no_grad():
        lk, m, mid = enc(img)
        x = enc.prepare(img)
        f = enc.enc1(x)
        w1, b1 = enc.cam[0].weight, enc.cam[0].bias
        w2, b2 = enc.cam[2].weight, enc.cam[2].bias

        def mlp(v):
            return torch.relu(v @ w1.T + b1) @ w2.T + b2

        avg = f.sum((2, 3)) / (f.shape[2] * f.shape[3])
        mx = f.flatten(2).max(-1).values
        att = 1 / (1 + torch.exp(-(mlp(avg) + mlp(mx))))           # Eq. 15
        f1 = att[:, :, None, None] * f                              # Eq. 16
        e15 = float((att - mid["Fc"]).abs().max())
        e16 = float((f1 - mid["F1"]).abs().max())
        f2 = m * f1
        ref_l = enc.enc2(f2 + f).flatten()                          # Eq. 18
        e18 = float((ref_l - lk).abs().max())
        no_res = enc.enc2(f2).flatten()
        res_effect = float((no_res - lk).abs().max())
        l0, _, _ = enc(img, mask_override=torch.zeros_like(m))
        l1, _, _ = enc(img, mask_override=torch.ones_like(m))
        mask_effect = float((l0 - l1).abs().max())
    shape_ok = tuple(m.shape) == (1, 1, 48 // 4, 64 // 4)  # 40x56 -> 48x64 (x16) -> H/4, W/4
    if (e15 > 1e-5 or e16 > 1e-5 or e18 > 1e-5 or not shape_ok or float(m.min()) < 0
            or float(m.max()) > 1 or mask_effect < 1e-6 or res_effect < 1e-6):
        raise AssertionError(f"eq15 {e15} eq16 {e16} eq18 {e18} shape {tuple(m.shape)} "
                             f"mask effect {mask_effect} residual effect {res_effect}")
    if BANK_DEFAULTS["appearance_dim"] != 16 or BANK_DEFAULTS["input_size"] != 0:
        raise AssertionError("defaults differ from §5.1 (dim 16, encoder on the image, W' = W/4)")
    full = WEGSBank({}, device).encoder
    with torch.no_grad():
        lf, mf, _ = full(torch.rand(64, 80, 3, device=device))
    if lf.shape != (16,) or tuple(mf.shape) != (1, 1, 16, 20):
        raise AssertionError(f"default encoder shapes {lf.shape} {mf.shape}")
    return {"eq15_err": e15, "eq16_err": e16, "eq18_err": e18, "mask_effect": mask_effect,
            "residual_connection_effect": res_effect}


def _loss(method, name):
    _, _, losses = build(method)
    return dict(losses)[name]


def masked_loss(method, device):
    h, w = 24, 32
    rgb = torch.rand(h, w, 3, device=device, requires_grad=True)
    gt = torch.rand(h, w, 3, device=device)
    m = torch.ones(1, 1, h // 4, w // 4, device=device)
    m[..., : w // 8] = 0.0
    m.requires_grad_(True)
    from .renderer import full_mask

    mf = full_mask(m, h, w)
    val = _loss(method, "wegs-masked-photometric")({}, {"rgb": rgb, "static_mask_full": mf}, gt)
    g_rgb, g_m = torch.autograd.grad(val, (rgb, m))
    left = float(g_rgb[:, : w // 2 - 4].abs().max())   # fully masked columns (bilinear edge)
    right = float(g_rgb[:, w // 2 + 4:].abs().sum())
    if left != 0.0 or right <= 0 or float(g_m.abs().sum()) <= 0:
        raise AssertionError(f"masked grad left {left}, right {right}, mask grad "
                             f"{float(g_m.abs().sum())}")
    from .losses import masked_photometric

    v_det = masked_photometric(rgb, gt, mf, 0.0, 1.0, True)
    v_raw = masked_photometric(rgb, gt, mf, 0.0, 1.0, False)
    g_ssim = torch.autograd.grad(v_det, m)[0]
    if abs(float(v_det) - float(v_raw)) > 1e-7 or float(g_ssim.abs().max()) != 0.0:
        raise AssertionError("D-SSIM term must have the Eq. 19 value and no gradient on M")
    mm = torch.rand(1, 1, 6, 8, device=device, requires_grad=True)
    mmf = full_mask(mm, h, w)
    reg = _loss(method, "wegs-mask-reg")({}, {"static_mask": mm, "static_mask_full": mmf}, gt)
    ref = 1e-5 * float(((1 - mm.detach()) ** 2).sum())  # Eq. 20 on M_k at W' x H' 
    if abs(float(reg) - ref) > 1e-9 * max(1, ref):
        raise AssertionError(f"L_regM {float(reg)} vs {ref}")
    return {"left_grad_max": left, "right_grad_sum": right, "regM": float(reg)}


def sh_reg(method, device):
    d = torch.randn(10, 16, 3, device=device, requires_grad=True)
    val = _loss(method, "wegs-sh-reg")({}, {"sh_residual": d}, None)
    ref = 0.1 * sum(math.sqrt(float((r.detach() ** 2).sum())) for r in d) / len(d)
    if abs(float(val) - ref) > 1e-5 or not val.requires_grad:
        raise AssertionError(f"L_regSH {float(val)} vs {ref}")
    return {"value": float(val), "reference": ref}


def transient_opacity_reg(method, device):
    w, h = 64, 48
    xs = torch.tensor([-0.6, -0.3, 0.3, 0.6, 5.0], device=device)  # last one off-screen
    means = torch.stack([xs, torch.zeros_like(xs), torch.full_like(xs, 3.0)], -1)
    n = len(means)
    logits = torch.zeros(n, device=device, requires_grad=True)
    op = torch.sigmoid(logits)
    K = torch.tensor([[60.0, 0, w / 2], [0, 60.0, h / 2], [0, 0, 1]], device=device)
    _, _, info = rasterization(means, torch.tensor([[1.0, 0, 0, 0]] * n, device=device),
                               torch.full((n, 3), 0.05, device=device), op,
                               torch.rand(n, 3, device=device), torch.eye(4, device=device)[None],
                               K[None], w, h, packed=False)
    mask = torch.ones(1, 1, h // 4, w // 4, device=device)
    mask[..., : w // 8] = 0.0  # left half transient
    fn = _loss(method, "wegs-transient-opacity-reg")
    outputs = {"opacity": op, "static_mask": mask, "info": info}
    val = fn({}, outputs, torch.zeros(h, w, 3, device=device))
    g = torch.autograd.grad(val, logits)[0]
    expected = torch.tensor([True, True, False, False, False], device=device)
    got = g.abs() > 0
    mv = projected_mask_values(mask, info["means2d"].reshape(-1, n, 2)[0], w, h)
    ref = 1e-5 * float((op.detach() * expected).sum())
    if not torch.equal(got, expected) or abs(float(val) - ref) > 1e-12:
        raise AssertionError(f"grad pattern {got.tolist()} vs {expected.tolist()}, "
                             f"value {float(val)} vs {ref}, mask at centres {mv.tolist()}")
    return {"value": float(val), "transient_gaussians": outputs["transient_gaussians"]}


def bank_training(method, device):
    from .density import position_lr

    _, density, _ = build(method)
    s = density.inner.strategy
    if (s.refine_every, s.refine_start_iter, s.refine_stop_iter, s.reset_every) != (
            100, 500, 15000, 3000):
        raise AssertionError("density schedule differs from 3DGS")
    if abs(position_lr(1.0, 15000, 30000, 0.01) - 0.1) > 1e-9 or \
            abs(position_lr(1.0, 150000, 30000, 0.01) - 0.01) > 1e-12:
        raise AssertionError("position lr schedule")
    bank = WEGSBank(SMALL, device)
    lrs = {g["name"]: g["lr"] for g in bank.optimizer.param_groups}
    if lrs != {"transnet": 1e-4, "encoder": 1e-4, "unet": 1e-5}:
        raise AssertionError(f"lrs {lrs}")
    groups = {"transnet": bank.transnet.net[0].weight, "enc1": bank.encoder.enc1[1].weight,
              "cam": bank.encoder.cam[0].weight, "unet": bank.encoder.up2.weight,
              "enc2": bank.encoder.enc2[-1].weight}
    before = {k: v.detach().clone() for k, v in groups.items()}
    img = torch.rand(40, 48, 3, device=device)
    lk, m = bank.encode(img)
    sh = torch.randn(32, 16, 3, device=device)
    d = bank.residual(sh, torch.randn(32, 3, device=device), lk)
    (d.square().sum() + (1 - m).square().sum()).backward()
    bank.after_step(1)
    moved = {k: float((groups[k] - before[k]).abs().max()) for k in groups}
    if min(moved.values()) <= 0:
        raise AssertionError(f"bank parameters not updated: {moved}")
    return {"moved": moved, "lrs": lrs}


def feedforward_eval(method, device):
    from .evaluate import render_feedforward

    renderer, _, _ = build(method)
    params = _scene(method, device)
    bank = WEGSBank(SMALL, device)
    bank.eval()
    a = _camera(device, image=torch.zeros(40, 48, 3, device=device))
    b = _camera(device, image=torch.ones(40, 48, 3, device=device))
    with _with_bank(bank):
        ra, oa = render_feedforward(renderer, params, a, 3)
        rb, ob = render_feedforward(renderer, params, b, 3)
    diff = float((ra - rb).abs().max())
    emb = float((oa["embedding"] - ob["embedding"]).abs().max())
    if diff < 1e-5 or emb < 1e-6 or "static_mask" not in oa:
        raise AssertionError(f"test image does not drive the render: diff {diff}, emb {emb}")
    raised = False
    prev = bank_mod.ACTIVE
    bank_mod.ACTIVE = None
    try:
        renderer.render(params, a, 3)
    except RuntimeError:
        raised = True
    finally:
        bank_mod.ACTIVE = prev
    if not raised:
        raise AssertionError("renderer silently rendered without the bank")
    return {"render_diff": diff, "embedding_diff": emb}
