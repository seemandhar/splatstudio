"""A small native trainer; reproduction claims require separate author-code experiments."""

import hashlib
import importlib.metadata
import json
import random
import time
from pathlib import Path

import torch

from .components import build
from .contracts import require_valid
from .data import initial_points, load_dataset, on_device
from .losses import check_loss, ssim
from .provenance import source_manifest
from .state import check_state, initialize


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def specification_id(method):
    return hashlib.sha256(json.dumps(method.to_dict(), sort_keys=True).encode()).hexdigest()


def environment():
    result = {}
    for package in ("splatstudio", "torch", "gsplat", "numpy", "Pillow"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unavailable"
    result["cuda_runtime"] = torch.version.cuda
    return result


@torch.no_grad()
def measure(renderer, params, dataset, split, degree, device, image_directory=None):
    values = []
    for name in dataset["splits"][split]:
        frame = on_device(dataset["frames"][name], device)
        rgb = renderer.render(params, frame, sh_degree=degree)["rgb"].clamp(0, 1)
        if image_directory is not None:
            from PIL import Image

            folder = Path(image_directory)
            folder.mkdir(parents=True, exist_ok=True)
            index = len(values)
            for label, value in (("render", rgb), ("target", frame["image"])):
                pixels = (value.detach().cpu().numpy() * 255).round().astype("uint8")
                Image.fromarray(pixels).save(folder / f"{index:04d}-{label}.png")
        mse = (rgb - frame["image"]).square().mean().clamp_min(1e-12)
        values.append(
            {
                "frame": name,
                "psnr": (-10 * torch.log10(mse)).item(),
                "ssim": ssim(rgb, frame["image"]).item(),
            }
        )
    metrics = {k: sum(v[k] for v in values) / len(values) for k in ("psnr", "ssim")}
    return metrics, values


def train(
    method,
    data,
    output,
    *,
    steps=30000,
    seed=0,
    downscale=1,
    num_gaussians=10000,
    max_gaussians=1000000,
    device="cuda",
    log_every=100,
):
    require_valid(method)
    if steps <= 0 or num_gaussians < 2 or max_gaussians < num_gaussians or log_every < 1:
        raise ValueError("Invalid step, point, resource-budget, or logging arguments")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("The native gsplat trainer requires CUDA")
    if torch.device(device).index is None:
        device = f"cuda:{torch.cuda.current_device()}"
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Output directory must be empty; choose a new run directory")
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.set_device(device)
    dataset = load_dataset(data, downscale)
    renderer, density, losses = build(method)
    points, colors = initial_points(dataset, num_gaussians, seed, device)
    params, optimizers = initialize(method, points, colors)
    scene_scale = max(
        float(torch.linalg.vector_norm(points - points.mean(dim=0), dim=1).quantile(0.9)), 0.01
    )
    optimizers["means"].param_groups[0]["lr"] *= scene_scale
    density.initialize(params, optimizers, scene_scale)
    config = {
        "method": method.name,
        "specification": method.to_dict(),
        "specification_id": specification_id(method),
        "data": str(Path(data).resolve()),
        "steps": steps,
        "seed": seed,
        "downscale": downscale,
        "num_gaussians": num_gaussians,
        "max_gaussians": max_gaussians,
        "device": device,
        "environment": environment(),
        "protocol": dataset["protocol"],
        "protocol_id": dataset["protocol_id"],
        "gpu": torch.cuda.get_device_name(device),
        "paper_reproduction": "unverified",
        "source_manifest": source_manifest(method),
        "initialization_sha256": hashlib.sha256(
            points.detach().cpu().numpy().tobytes() + colors.detach().cpu().numpy().tobytes()
        ).hexdigest(),
    }
    write_json(output / "config.json", config)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    names = dataset["splits"]["train"]
    initial_lr = optimizers["means"].param_groups[0]["lr"]
    try:
        with (output / "training.jsonl").open("w") as log:
            for step in range(1, steps + 1):
                frame = on_device(dataset["frames"][random.choice(names)], device)
                degree = min((step - 1) // 1000, 3)
                outputs = renderer.render(params, frame, sh_degree=degree)
                outputs["step"] = step
                density.before_backward(params, optimizers, step, outputs)
                values = {name: loss(params, outputs, frame["image"]) for name, loss in losses}
                for name, value in values.items():
                    check_loss(value, name)
                total = sum(values.values())
                total.backward()
                for opt in optimizers.values():
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                optimizers["means"].param_groups[0]["lr"] = initial_lr * (0.01 ** (step / steps))
                old_count = len(params["means"])
                density.after_backward(params, optimizers, step, outputs)
                if len(params["means"]) > max_gaussians:
                    raise RuntimeError(
                        "Gaussian budget exceeded; run stopped without silently changing the method"
                    )
                if (
                    step == 1
                    or step % log_every == 0
                    or step == steps
                    or len(params["means"]) != old_count
                ):
                    check_state(method, params, optimizers)
                    row = {
                        "step": step,
                        "loss": total.item(),
                        "gaussians": len(params["means"]),
                        "losses": {k: v.item() for k, v in values.items()},
                    }
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(device)
        checkpoint = output / "checkpoint.pt"
        torch.save(
            {
                "params": dict(params),
                "optimizers": {k: o.state_dict() for k, o in optimizers.items()},
                "density_state": density.state,
                "step": steps,
                "sh_degree": degree,
                "specification_id": specification_id(method),
                "protocol_id": dataset["protocol_id"],
                "seed": seed,
            },
            checkpoint,
        )
        metrics, views = measure(renderer, params, dataset, "validation", degree, device)
        result = {
            "schema_version": 1,
            "method": method.name,
            "scene": dataset["root"].name,
            "seed": seed,
            "split": "validation",
            "protocol_id": dataset["protocol_id"],
            "metrics": metrics,
            "per_view": views,
            "paper_reproduction": "unverified",
            "resources": {
                "gaussians": len(params["means"]),
                "peak_memory_bytes": peak,
                "training_seconds": elapsed,
                "checkpoint_bytes": checkpoint.stat().st_size,
            },
        }
        write_json(output / "validation.json", result)
        write_json(output / "status.json", {"status": "completed", "steps": steps})
        return result
    except Exception as exc:
        write_json(output / "status.json", {"status": "failed", "reason": str(exc)})
        raise


def evaluate(method, run, *, split="test", device="cuda", save_images=False):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    checkpoint = torch.load(run / "checkpoint.pt", map_location=device, weights_only=True)
    if specification_id(method) != checkpoint["specification_id"]:
        raise ValueError("Installed method specification differs from the checkpoint")
    if source_manifest(method) != config["source_manifest"]:
        raise ValueError("SDK, plugin, or renderer source changed since training")
    dataset = load_dataset(config["data"], config["downscale"])
    if dataset["protocol_id"] != checkpoint["protocol_id"]:
        raise ValueError("Dataset or evaluation protocol changed since training")
    renderer, _, _ = build(method)
    metrics, views = measure(
        renderer,
        checkpoint["params"],
        dataset,
        split,
        checkpoint["sh_degree"],
        device,
        run / "images" / split if save_images else None,
    )
    result = json.loads((run / "validation.json").read_text())
    result.update(split=split, metrics=metrics, per_view=views)
    write_json(run / f"{split}.json", result)
    return result
