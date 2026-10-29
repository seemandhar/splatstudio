"""Create portable records and figures from measured local runs; never invent scores."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export_suite(source, name):
    summary = json.loads((source / "summary.json").read_text())
    records = []
    for run in summary["runs"]:
        if run["status"] != "passed":
            records.append(run)
            continue
        method, seed = run["method"], run["seed"]
        folder = source / f"{method}-seed{seed}"
        output = ROOT / "results/native" / name / f"{method}-seed{seed}"
        output.mkdir(parents=True, exist_ok=True)
        raw = (folder / "config.json").read_bytes()
        config = json.loads(raw)
        config["data"] = f"prepared-dataset:{name}"
        config["local_record_sha256"] = hashlib.sha256(raw).hexdigest()
        write(output / "config.json", config)
        for filename in ("test.json", "validation.json", "training.jsonl", "status.json"):
            shutil.copy2(folder / filename, output / filename)
        shutil.copy2(source / f"{method}-verification.json", output / "verification.json")
        result = run["result"]
        records.append(
            {
                "status": "passed",
                "method": method,
                "seed": seed,
                "metrics": result["metrics"],
                "resources": result["resources"],
                "protocol_id": result["protocol_id"],
                "paper_reproduction": "unverified",
                "record": f"{method}-seed{seed}/test.json",
            }
        )
    recipe = {k: v for k, v in summary["recipe"].items() if k not in {"data", "output"}}
    write(
        ROOT / f"results/native/{name}/summary.json",
        {"scene": name, "recipe": recipe, "runs": records},
    )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poster", default="artifacts/release-poster")
    parser.add_argument("--spheres", default="artifacts/release-native")
    parser.add_argument("--zoo", default="artifacts/zoo-release/summary.json")
    args = parser.parse_args()
    poster_source, sphere_source = ROOT / args.poster, ROOT / args.spheres
    poster, spheres = export_suite(poster_source, "poster"), export_suite(sphere_source, "spheres")
    audit = json.loads((ROOT / args.zoo).read_text())
    write(ROOT / "results/validation/zoo.json", audit)
    with (ROOT / "results/native/metrics.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            lineterminator="\n",
            fieldnames=[
                "scene",
                "method",
                "seed",
                "psnr",
                "ssim",
                "gaussians",
                "training_seconds",
                "peak_memory_bytes",
            ],
        )
        writer.writeheader()
        for name, records in (("poster", poster), ("spheres", spheres)):
            for r in records:
                if r["status"] == "passed":
                    writer.writerow(
                        {
                            k: {"scene": name, **r, **r["metrics"], **r["resources"]}[k]
                            for k in writer.fieldnames
                        }
                    )
    names = [r["method"] for r in poster]
    colors = ["#6c7cdb", "#4ba58f", "#dd9662", "#a37dd0", "#659bb3"]
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, metric, label in (
        (axes[0], "psnr", "Held-out PSNR (dB)"),
        (axes[1], "gaussians", "Final primitives (thousands)"),
    ):
        values = [
            r["metrics"][metric] if metric == "psnr" else r["resources"][metric] / 1000
            for r in poster
        ]
        bars = ax.barh(names, values, color=colors, height=0.58)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_xlim(0, max(values) * 1.2)
        ax.bar_label(bars, fmt="%.2f", padding=6, fontsize=10)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Poster · 1,200 steps · one seed · short execution check", fontsize=16, weight="bold"
    )
    fig.savefig(ROOT / "docs/assets/results.png", dpi=160, facecolor="white")
    plt.close(fig)
    fig, axes = plt.subplots(2, 6, figsize=(13.5, 8.8), layout="constrained")
    for row, frame in enumerate((0, 14)):
        for col, method in enumerate([None, *names]):
            folder = poster_source / f"{method or names[0]}-seed0/images/test"
            image = folder / f"{frame:04d}-{'render' if method else 'target'}.png"
            axes[row, col].imshow(Image.open(image))
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(method or "Target", fontsize=11, weight="bold", pad=12)
    fig.suptitle(
        "Held-out views · fixed first and midpoint test cameras", fontsize=17, weight="bold"
    )
    fig.savefig(ROOT / "docs/assets/qualitative.png", dpi=150, facecolor="white")
    plt.close(fig)
    passed = sum(m["training_check"]["status"] == "passed" for m in audit["methods"])
    config = json.loads((poster_source / "gsplat-seed0/config.json").read_text())
    splits = {k: len(v) for k, v in config["protocol"]["splits"].items()}
    page = f"""# Results and validation

This release publishes **{passed}/33 short Nerfstudio training checks**, all 33 configuration imports, six native fixture runs (including the external scale-prior plugin), and five native runs on the real Poster scene. These establish installation and limited execution coverage. **No complete paper reproduction is certified by this release.**

## Fresh native runs

The real scene is [Nerfstudio's Poster capture](https://docs.nerf.studio/quickstart/custom_dataset.html). OpenCV undistortion with alpha=0 and 8× downscaling produces 135×240 RGB images. The deterministic partition has {splits["train"]} train, {splits["validation"]} validation, and {splits["test"]} test images. All five configurations use seed 0, 2,000 initial SfM points, 1,200 training steps, a 100,000-primitive stop limit, and the same training/evaluation protocol. Test images do not participate in optimization. This short recipe is fixed across configurations but **does not fix final primitive counts**.

| Native configuration | Test PSNR ↑ | Test SSIM ↑ | Final primitives | Peak allocated memory, MiB | Train seconds |
| :--- | ---: | ---: | ---: | ---: | ---: |
"""
    for r in poster:
        m, resources = r["metrics"], r["resources"]
        page += f"| `{r['method']}` | {m['psnr']:.3f} | {m['ssim']:.4f} | {resources['gaussians']:,} | {resources['peak_memory_bytes'] / 2**20:.1f} | {resources['training_seconds']:.1f} |\n"
    page += """
![Measured quality and primitive counts](assets/results.png)

![Target and native outputs](assets/qualitative.png)

The qualitative grid uses the first and midpoint held-out cameras, selected by index. Metrics average per-view values, not selected images. This is one seed on one real scene, with short schedules, no uncertainty estimate, and different final populations. Training times were measured on a shared multi-GPU host; they are not controlled throughput comparisons. Numbers must not be presented as a paper leaderboard or method superiority claim. The 2D recipe uses the actual 2D rasterizer, while its normal/distortion warmups occur after these runs. Neither geometry reconstruction quality nor the late loss schedule is established here.

The sphere fixture is independently ray-cast by NumPy: 16 images at 32×32, with 8 train/4 validation/4 test views. Each native recipe and the external plugin runs 700 steps from 192 points, crossing the standard density-control start. It is a software fixture with simple geometry and initialization from its known surfaces.

| Fixture configuration | Test PSNR | Test SSIM | Final primitives |
| :--- | ---: | ---: | ---: |
"""
    for r in spheres:
        page += f"| `{r['method']}` | {r['metrics']['psnr']:.3f} | {r['metrics']['ssim']:.4f} | {r['resources']['gaussians']} |\n"
    page += """
[Portable native records](../results/native) contain per-view metrics, exact specifications, initialization and data hashes, source manifests, dependency versions, GPU, training traces, and verification reports. [CSV export](../results/native/metrics.csv). Local filesystem paths are replaced with dataset descriptors. Checkpoints and source datasets are regenerated by the commands below and are not committed.

```bash
# Native environment; see installation.md for CUDA setup.
pip install -e '.[runtime,data]'
pip install -e examples/scale_prior
splatstudio demo-data --output artifacts/release-spheres --size 32
python scripts/run_native_suite.py --data artifacts/release-spheres \\
  --output artifacts/release-native --steps 700 --num-gaussians 192 \\
  --methods gsplat gsplat-absgrad gsplat-aa gsplat-mcmc gsplat-surfel scale-prior

# Obtain the Poster dataset in a Nerfstudio environment:
# ns-download-data nerfstudio --capture-name poster
splatstudio prepare-data --data /path/to/poster \\
  --output artifacts/poster-undistorted --downscale 8
python scripts/run_native_suite.py --data artifacts/poster-undistorted \\
  --output artifacts/release-poster --steps 1200 --num-gaussians 2000
```

## Collection audit

All 33 entries import against **stock Nerfstudio 1.1.5**. Twenty-nine pass ten training steps with real forward/backward, optimizer updates, callbacks, finite-state checks, and checkpoint writing. DepthRegGS, FSGS, StableGS, and FeatureGS are not run because the fixture does not supply their teacher weights/feature targets. Their absence is not counted as success.

The audit found and fixed CoR-GS initialization before Nerfstudio's device property exists, uint8 ground-truth handling in CoR-GS/Scaffold-GS, and two shadowed normal-loss functions affecting the legacy 2DGS port. [Final audit](../results/validation/zoo.json) includes every result and source hashes. [Development findings](../results/validation/development-findings.json) retain earlier failures and the preliminary native 2D regularization issue. The audit leaves late densification, sparsity schedules, compression exports, geometry accuracy, and original-author fidelity untested.

```bash
# In the native environment, convert the analytic fixture for Nerfstudio:
python scripts/prepare_zoo_fixture.py --source artifacts/release-spheres \\
  --output artifacts/zoo-spheres
# Then activate the Nerfstudio collection environment.
python scripts/validate_zoo.py --data artifacts/zoo-spheres \\
  --output artifacts/zoo-release --steps 10 --gpus 0
```

## Historical results

The [historical JSON](../results/historical/records.json) and [CSV](../results/historical/metrics.csv) preserve 58 existing scene evaluations across 10 catalog configurations. They include PSNR/SSIM/LPIPS values as recorded by earlier SPLATIFY experiments, plus hashes of the source records. Source exports include per-image dispersion; this is not uncertainty across training seeds.

Original checkpoints, commits, seed values, split definitions, and resource budgets were not recovered. These exports are labeled `historical_unverified`, are not fresh SplatStudio results, and cannot be used as matched reference measurements. Missing results for other methods are left missing.

## What a reproduction comparison needs

Run original-author code and the candidate with an identical dataset, split, metric implementation, and documented training/resource budget. Match scenes and seeds, then use `ss-compare`. The report gives mean signed, mean absolute, and worst absolute gaps, all per-pair gaps, and resource ratios. Equal scene weighting prevents repeated seeds from giving a scene extra influence.

Source and protocol hashes detect changes; they do not prove the underlying method. PSNR tolerance alone cannot distinguish a faithful implementation from a strong baseline. Independent tests for the paper's actual mechanisms, repeated seeds, held-out scene selection, and relevant baselines remain required. See [the research roadmap](roadmap.md).
"""
    (ROOT / "docs/results.md").write_text(page)
    print(
        f"Published {len(poster) + len(spheres)} native runs and {passed} collection training checks"
    )


if __name__ == "__main__":
    main()
