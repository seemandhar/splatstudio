"""Paired reporting without cancellation of positive and negative reproduction gaps."""

import math
from statistics import mean


def compare(reference, candidate, psnr_tolerance=0.5):
    if not math.isfinite(psnr_tolerance) or psnr_tolerance < 0:
        raise ValueError("PSNR tolerance must be finite and nonnegative")

    def indexed(records):
        result = {}
        for r in records:
            key = (r["scene"], r["seed"], r["split"])
            if key in result:
                raise ValueError(f"Duplicate result for {key}")
            if r["split"] not in {"validation", "test"}:
                raise ValueError(
                    "Only explicitly identified validation or test results can be paired"
                )
            if not r.get("protocol_id"):
                raise ValueError("A protocol_id is required")
            if "psnr" not in r["metrics"] or any(
                not math.isfinite(v) for v in r["metrics"].values()
            ):
                raise ValueError("Metrics must include PSNR and all must be finite")
            result[key] = r
        return result

    refs, cands = indexed(reference), indexed(candidate)
    if not refs or refs.keys() != cands.keys():
        raise ValueError(
            f"Paired coverage differs: missing={sorted(refs.keys() - cands.keys())}, "
            f"extra={sorted(cands.keys() - refs.keys())}"
        )
    rows = []
    for key in sorted(refs):
        a, b = refs[key], cands[key]
        if a["protocol_id"] != b["protocol_id"]:
            raise ValueError(f"Evaluation protocol mismatch for {key}")
        if a["metrics"].keys() != b["metrics"].keys():
            raise ValueError(f"Metric coverage mismatch for {key}")
        gaps = {m: b["metrics"][m] - a["metrics"][m] for m in a["metrics"]}
        ratios = {}
        for k in ("gaussians", "peak_memory_bytes", "training_seconds", "checkpoint_bytes"):
            av, bv = a.get("resources", {}).get(k), b.get("resources", {}).get(k)
            if av is not None and bv is not None:
                if not math.isfinite(av) or not math.isfinite(bv) or av < 0 or bv < 0:
                    raise ValueError(f"Invalid resource measurement {k}")
                ratios[k] = bv / av if av else None
        rows.append(
            {
                "scene": key[0],
                "seed": key[1],
                "split": key[2],
                "gaps": gaps,
                "within_psnr_tolerance": abs(gaps["psnr"]) <= psnr_tolerance,
                "resource_ratios_candidate_over_reference": ratios,
            }
        )
    summary = {}
    # Equal scene weighting: repeated seeds do not give a scene extra influence.
    for metric in rows[0]["gaps"]:
        scenes = sorted({r["scene"] for r in rows})
        summary[metric] = {
            "mean_signed_gap": mean(
                mean(r["gaps"][metric] for r in rows if r["scene"] == s) for s in scenes
            ),
            "mean_absolute_gap": mean(
                mean(abs(r["gaps"][metric]) for r in rows if r["scene"] == s) for s in scenes
            ),
            "worst_absolute_gap": max(abs(r["gaps"][metric]) for r in rows),
        }
    return {
        "schema_version": 1,
        "pairs": rows,
        "summary": summary,
        "psnr_tolerance_db": psnr_tolerance,
        "pairs_within_psnr_tolerance": sum(r["within_psnr_tolerance"] for r in rows),
        "pair_count": len(rows),
        "paper_reproduction": "not_established_by_metrics_alone",
    }
