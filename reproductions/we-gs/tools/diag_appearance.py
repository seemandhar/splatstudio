"""Gap-analysis diagnostic (train views; test views only inspected, no optimisation):
embedding spread, SH-residual magnitude, effect of appearance on PSNR, mask mean.
    python tools/diag_appearance.py RUN OUT_JSON [n_train] [save_dir]"""
import json
import random
import sys
from pathlib import Path

import torch
from splatstudio.components import build
from splatstudio.data import load_dataset, on_device
from splatstudio.registry import load

from ss_wegs import bank as bank_mod
from ss_wegs.bank import WEGSBank


def psnr(a, b):
    return float(-10 * torch.log10((a - b).square().mean().clamp_min(1e-12)))


@torch.no_grad()
def main(run, out, n_train=60, save_dir=None):
    run = Path(run)
    cfg = json.loads((run / "config.json").read_text())
    method = load(cfg["method"])
    dev = "cuda:0"
    ck = torch.load(run / "checkpoint.pt", map_location=dev, weights_only=True)
    ds = load_dataset(cfg["data"], cfg["downscale"])
    renderer, _, _ = build(method)
    bank = WEGSBank.from_export(ck["density_state"]["bank"], dev)
    bank_mod.ACTIVE = bank
    params = {k: v.detach() for k, v in ck["params"].items()}
    deg = ck["sh_degree"]
    random.seed(0)
    train = random.sample(ds["splits"]["train"], min(n_train, len(ds["splits"]["train"])))
    emb, rows = [], {"train": [], "test": []}
    for split, names in (("train", train), ("test", ds["splits"]["test"])):
        for n in names:
            f = on_device(ds["frames"][n], dev)
            out_ = renderer.render(params, f, deg)
            lk = out_["embedding"]
            if split == "train":
                emb.append(lk)
            rows[split].append({"frame": n, "lk": lk, "rgb": out_["rgb"].clamp(0, 1),
                                "gt": f["image"], "frame_data": f,
                                "res_norm": float(out_["sh_residual"].flatten(1).norm(dim=1).mean()),
                                "res_dc": float(out_["sh_residual"][:, 0].abs().mean()),
                                "mask": float(out_["static_mask"].mean())})
    E = torch.stack(emb)
    mean_l = E.mean(0)
    res = {"embedding_std_per_dim_train": E.std(0).tolist(),
           "embedding_norm_mean": float(E.norm(dim=1).mean()),
           "sh0_abs_mean": float(params["sh0"].abs().mean()),
           "shN_abs_mean": float(params["shN"].abs().mean())}
    for split in ("train", "test"):
        out_rows = []
        for r in rows[split]:
            f = r["frame_data"]
            mean_rgb = renderer.render(params, f, deg, embedding=mean_l)["rgb"].clamp(0, 1)
            d = (E - r["lk"][None]).norm(dim=1)
            gt, pr = r["gt"], r["rgb"]
            # global per-channel affine colour fit (diagnostic of residual colour shift)
            X = torch.cat([pr.reshape(-1, 3), torch.ones_like(pr.reshape(-1, 3)[:, :1])], 1)
            A = torch.linalg.lstsq(X, gt.reshape(-1, 3)).solution
            fit = (X @ A).reshape(gt.shape).clamp(0, 1)
            row = {"frame": r["frame"], "psnr": psnr(pr, gt), "psnr_mean_embedding":
                   psnr(mean_rgb, gt), "psnr_after_affine_colour_fit": psnr(fit, gt),
                   "res_norm": r["res_norm"], "res_dc_abs": r["res_dc"], "mask": r["mask"],
                   "nn_train_embedding_dist": float(d.min()) if split == "test" else
                   float(d.sort().values[1]), "embedding_norm": float(r["lk"].norm())}
            out_rows.append(row)
            if save_dir and split == "test":
                from PIL import Image
                p = Path(save_dir)
                p.mkdir(parents=True, exist_ok=True)
                im = torch.cat([gt, pr, mean_rgb], 1)
                Image.fromarray((im.cpu().numpy() * 255).astype("uint8")).save(
                    p / f"{len(out_rows):02d}.jpg")
        res[split] = {"per_view": out_rows,
                      **{k: sum(r[k] for r in out_rows) / len(out_rows)
                         for k in out_rows[0] if k != "frame"}}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({s: {k: v for k, v in res[s].items() if k != "per_view"}
                      for s in ("train", "test")}, indent=1))
    print("emb std", [round(x, 3) for x in res["embedding_std_per_dim_train"]])


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]) if len(a) > 3 else 60, a[4] if len(a) > 4 else None)
