"""Export metric values from an existing archive without paths or invented provenance."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="results/historical")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = {
        m["name"]
        for m in json.loads((root / "src/splatstudio/catalog.json").read_text())["methods"]
    }
    aliases = {"dwtgs_12v": "dwtgs", "fsgs_24v": "fsgs"}
    records = []
    for path in sorted(Path(args.source).glob("*/*.json")):
        name = aliases.get(path.parent.name, path.parent.name)
        if name not in names:
            continue
        raw = path.read_bytes()
        source = json.loads(raw)
        metrics = {
            k: v
            for k, v in source.get("results", {}).items()
            if k in {"psnr", "ssim", "lpips", "psnr_std", "ssim_std", "lpips_std"}
        }
        if not metrics:
            continue
        records.append(
            {
                "catalog_method": name,
                "recorded_method": source["method_name"],
                "scene": source["experiment_name"],
                "metrics": metrics,
                "status": "historical_unverified",
                "seed": None,
                "protocol_id": None,
                "source_record": f"{path.parent.name}/{path.name}",
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "paper_reproduction": "unverified",
            }
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "source": "Existing SPLATIFY evaluation archive",
        "limitations": [
            "Not rerun with this repository",
            "Original recipe/commit unavailable",
            "No verified seed, split, image resolution, or resource budget",
            "Reported standard deviations are from source exports, not across seeds",
            "Not eligible for paired comparison or a SplatStudio leaderboard",
        ],
        "records": records,
    }
    (output / "records.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (output / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(
            stream,
            lineterminator="\n",
            fieldnames=[
                "catalog_method",
                "recorded_method",
                "scene",
                "psnr",
                "ssim",
                "lpips",
                "status",
                "source_sha256",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({k: record.get(k, record["metrics"].get(k)) for k in writer.fieldnames})
    print(
        json.dumps(
            {"records": len(records), "methods": len({r["catalog_method"] for r in records})}
        )
    )


if __name__ == "__main__":
    main()
