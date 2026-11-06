"""DeblurGS evaluation protocol (paper Sec. 4.2 "post-optimization"; official test.py):
the trained scene is frozen and every test pose is aligned by iNeRF-style optimisation before
the metrics are computed.

official optimize_test_pose: per test view a quaternion of R_c2w (Adam lr 5e-5) and the w2c
translation T (lr 5e-4), eps 1e-15; 2000 epochs, each visiting every test view once in shuffled
order; StepLR(step 100 epochs, gamma 0.9); loss L1(clamp(gamma(render over white), 0, 1), GT).
Metrics on the optimised poses: PSNR, SSIM, LPIPS-Alex (paper), plus LPIPS-VGG and the
runtime's valid-convolution SSIM; "unaligned" metrics (= ss-eval) are reported alongside.

    python -m ss_deblurgs.evaluate --run RUN [--split test] [--epochs 2000] [--device cuda:0]
writes RUN/<split>_tpo.json.
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from splatstudio.components import build
from splatstudio.data import load_dataset, on_device
from splatstudio.losses import ssim as ssim_valid
from splatstudio.registry import load
from splatstudio.runtime import specification_id, write_json

from . import bank as bank_mod

POSE_DEFAULTS = {"epochs": 2000, "rot_lr": 5e-5, "trans_lr": 5e-4, "step_size": 100,
                 "gamma": 0.9}


def quat_wxyz_from_rotmat(R):
    """Unit quaternion (w, x, y, z) of a rotation matrix (Shepperd)."""
    m = R.double()
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    cands = torch.stack([tr, m[0, 0], m[1, 1], m[2, 2]])
    i = int(torch.argmax(cands))
    if i == 0:
        s = torch.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif i == 1:
        s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif i == 2:
        s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = torch.stack(q).float()
    return q / q.norm()


def rotmat_from_quat_wxyz(q):
    w, x, y, z = q
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


class PoseModel(torch.nn.Module):
    """test.py OptimPoseModel: rot = unit quaternion of R_c2w, trans = w2c translation."""

    def __init__(self, viewmats):
        super().__init__()
        rots, trans = [], []
        for vm in viewmats:
            rots.append(quat_wxyz_from_rotmat(vm[:3, :3].T))  # R_c2w = R_w2c^T
            trans.append(vm[:3, 3].float())
        self.rot = torch.nn.Parameter(torch.stack(rots).to(viewmats[0].device))
        self.trans = torch.nn.Parameter(torch.stack(trans))

    def forward(self, i):
        q = self.rot[i] + 1e-8
        r_c2w = rotmat_from_quat_wxyz(q / q.norm())
        top = torch.cat([r_c2w.T, self.trans[i][:, None]], dim=1)
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=top.device)
        return torch.cat([top, bottom], dim=0)


def optimize_test_poses(renderer, params, frames, sh_degree, cfg=None, seed=0, log=None):
    """Returns the optimised viewmats (list) and the per-epoch mean L1."""
    cfg = {**POSE_DEFAULTS, **(cfg or {})}
    model = PoseModel([f["viewmat"] for f in frames])
    opt = torch.optim.Adam([{"params": [model.rot], "lr": cfg["rot_lr"]},
                            {"params": [model.trans], "lr": cfg["trans_lr"]}], lr=5e-4, eps=1e-15)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg["step_size"], gamma=cfg["gamma"])
    rng = random.Random(seed)
    frozen = {k: v.detach() for k, v in params.items()}
    history = []
    for epoch in range(int(cfg["epochs"])):
        order = list(range(len(frames)))
        rng.shuffle(order)
        total = 0.0
        while order:
            i = order.pop()
            out = renderer.render_sharp(frozen, frames[i], sh_degree, viewmat=model(i))
            loss = F.l1_loss(out["rgb"].clamp(0.0, 1.0), frames[i]["image"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        sched.step()
        history.append(total / len(frames))
        if log is not None and epoch % 200 == 0:
            log(f"epoch {epoch} L1 {history[-1]:.5f}")
    with torch.no_grad():
        return [model(i).detach() for i in range(len(frames))], history


def ssim_same(image, target):
    """utils/loss_utils.ssim (3DGS): 11x11 Gaussian window sigma 1.5, same (zero) padding."""
    coords = torch.arange(11, device=image.device, dtype=image.dtype) - 5
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None].expand(3, 1, 11, 11)
    x, y = (t.permute(2, 0, 1)[None] for t in (image, target))

    def conv(t):
        return F.conv2d(t, w, padding=5, groups=3)

    mx, my = conv(x), conv(y)
    vx, vy, cov = conv(x * x) - mx * mx, conv(y * y) - my * my, conv(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * cov + c2))
            / ((mx * mx + my * my + c1) * (vx + vy + c2))).mean()


_LP = {}


def lpips_net(net, device):
    key = (net, device)
    if key not in _LP:
        import lpips

        _LP[key] = lpips.LPIPS(net=net, verbose=False).to(device).eval()
    return _LP[key]


@torch.no_grad()
def metrics(rgb, gt, device):
    mse = (rgb - gt).square().mean().clamp_min(1e-12)
    x, y = (t.permute(2, 0, 1)[None] * 2 - 1 for t in (rgb, gt))
    return {"psnr": float(-10 * torch.log10(mse)), "ssim": float(ssim_same(rgb, gt)),
            "ssim_valid": float(ssim_valid(rgb, gt)),
            "lpips_alex": float(lpips_net("alex", device)(x, y)),
            "lpips_vgg": float(lpips_net("vgg", device)(x, y))}


def metrics_official(rgb, gt, device):
    """Official test.py evaluate() conventions (review-1 major): no clamp; PSNR = mean of
    per-channel PSNRs over the [3, H*W] view; 3DGS same-padded SSIM; LPIPS-Alex fed [0, 1]
    images directly (lpipsPyTorch copy whose z-score layer expects [-1, 1], i.e. the 3DGS
    convention), reproduced with the `lpips` package by passing [0, 1] tensors unchanged."""
    a, b = rgb.permute(2, 0, 1), gt.permute(2, 0, 1)
    mse_c = (a - b).reshape(3, -1).square().mean(dim=1).clamp_min(1e-12)
    return {"psnr": float((-10 * torch.log10(mse_c)).mean()),
            "ssim": float(ssim_same(rgb, gt)),
            "lpips_alex": float(lpips_net("alex", device)(a[None], b[None]))}


def evaluate(run, split="test", device="cuda:0", epochs=None, seed=0):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    method = load(config["method"])
    ckpt = torch.load(run / "checkpoint.pt", map_location=device, weights_only=True)
    if specification_id(method) != ckpt["specification_id"]:
        raise ValueError(f"{run}: installed method specification differs from the checkpoint")
    dataset = load_dataset(config["data"], config["downscale"])
    if dataset["protocol_id"] != ckpt["protocol_id"]:
        raise ValueError("Dataset or evaluation protocol changed since training")
    renderer, _, _ = build(method)
    if bank_mod.ACTIVE is not None:
        raise RuntimeError("evaluation must not run with a live training bank")
    names = dataset["splits"][split]
    frames = [on_device(dataset["frames"][n], device) for n in names]
    params, degree = ckpt["params"], ckpt["sh_degree"]
    torch.manual_seed(seed)
    poses, history = optimize_test_poses(renderer, params, frames, degree,
                                         {"epochs": epochs} if epochs else None, seed,
                                         log=lambda m: print(m, flush=True))
    rows = []
    with torch.no_grad():
        for name, frame, vm in zip(names, frames, poses):
            before = renderer.render_sharp(params, frame, degree)["rgb"]
            after = renderer.render_sharp(params, frame, degree, viewmat=vm)["rgb"]
            rows.append({"frame": name,
                         "unaligned": metrics(before.clamp(0, 1), frame["image"], device),
                         "aligned": metrics(after.clamp(0, 1), frame["image"], device),
                         "aligned_official": metrics_official(after, frame["image"], device),
                         "aligned_unclamped_psnr": float(-10 * torch.log10(
                             (after - frame["image"]).square().mean().clamp_min(1e-12))),
                         "pose_delta_translation": float((vm[:3, 3] - frame["viewmat"][:3, 3])
                                                         .norm()),
                         "viewmat": vm.cpu().tolist()})
    summary = {k: {m: sum(r[k][m] for r in rows) / len(rows) for m in rows[0][k]}
               for k in ("unaligned", "aligned", "aligned_official")}
    result = {"run": str(run.resolve()), "split": split, "metrics": summary,
              "pose_optimisation": {**POSE_DEFAULTS, **({"epochs": epochs} if epochs else {}),
                                    "final_l1": history[-1], "first_l1": history[0]},
              "per_view": rows,
              "note": "HEADLINE = aligned_official (official test.py evaluate() conventions: unclamped, "
                      "per-channel PSNR, same-padded SSIM, LPIPS-Alex on [0,1] input). "
                      "aligned = same poses, clamped, runtime conventions: psnr, ssim (3DGS "
                      "same-padded), lpips_alex; unaligned = ss-eval rendering"}
    write_json(run / f"{split}_tpo.json", result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    r = evaluate(a.run, a.split, a.device, a.epochs, a.seed)
    print(json.dumps(r["metrics"], indent=1))


if __name__ == "__main__":
    main()
