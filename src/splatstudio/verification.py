"""Evidence reports distinguish checked behavior from paper reproduction."""

from dataclasses import asdict
from datetime import datetime, timezone

from .contracts import validate
from .registry import resolve


class ProbeUnavailable(RuntimeError):
    pass


def verify(method, device="cpu", static_only=False):
    issues = validate(method)
    results = []
    if not issues and not static_only:
        for probe in method.probes:
            try:
                if probe.requires_cuda and not device.startswith("cuda"):
                    raise ProbeUnavailable("CUDA is required; this probe was not executed")
                evidence = resolve(probe.function)(method, device)
                if not isinstance(evidence, dict) or not evidence:
                    raise ValueError("A probe must return a nonempty evidence dictionary")
                results.append({"name": probe.name, "status": "passed", "evidence": evidence})
            except ProbeUnavailable as exc:
                results.append({"name": probe.name, "status": "unverified", "reason": str(exc)})
            except Exception as exc:
                results.append(
                    {
                        "name": probe.name,
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    statuses = {r["name"]: r["status"] for r in results}
    coverage = []
    for c in method.contributions:
        states = [statuses.get(p, "unverified") for p in c.probes]
        status = (
            "failed"
            if "failed" in states
            else "passed"
            if states and all(s == "passed" for s in states)
            else "unverified"
        )
        coverage.append({"name": c.name, "status": status, "probes": list(c.probes)})
    if issues or any(r["status"] == "failed" for r in results):
        status = "failed"
    elif static_only:
        status = "static_checks_passed"
    elif not results or not coverage or any(r["status"] != "passed" for r in results + coverage):
        status = "unverified"
    else:
        status = "declared_checks_passed"
    return {
        "schema_version": 1,
        "method": method.name,
        "status": status,
        "paper_reproduction": "unverified",
        "device": device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "contract_issues": [asdict(i) for i in issues],
        "probes": results,
        "contributions": coverage,
        "specification": method.to_dict(),
    }
