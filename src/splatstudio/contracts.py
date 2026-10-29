"""Check component dependencies before allocating training resources."""

import math
from dataclasses import dataclass

from .specs import MethodSpec

BASE_CAPABILITIES = frozenset({"state.gaussian3d", "camera.pinhole"})
BASE_FIELDS = frozenset({"means", "scales", "quats", "opacities", "sh0", "shN"})


@dataclass(frozen=True)
class Issue:
    code: str
    message: str


def validate(method: MethodSpec) -> list[Issue]:
    issues = []
    if method.api_version != 1:
        issues.append(Issue("api_version", f"Unsupported API version {method.api_version}"))
    if method.representation not in {"gaussian3d", "surfel2d"}:
        issues.append(Issue("representation", "The native runner supports gaussian3d and surfel2d"))
    names = [c.name for c in method.components]
    if len(names) != len(set(names)):
        issues.append(Issue("duplicate_component", "Component names must be unique"))
    for kind in ("renderer", "density"):
        count = sum(c.kind == kind for c in method.components)
        if count != 1:
            issues.append(Issue("exclusive_component", f"Expected one {kind}; found {count}"))
    for c in method.components:
        if c.kind not in {"renderer", "density", "loss"}:
            issues.append(Issue("component_kind", f"Unsupported component kind: {c.kind}"))
    # Dependency resolution rejects circular promises: a provider cannot satisfy its own input.
    available = {f"state.{method.representation}", "camera.pinhole"}
    pending = list(method.components)
    while pending:
        ready = [c for c in pending if set(c.requires) <= available]
        if not ready:
            for c in pending:
                missing = sorted(set(c.requires) - available)
                issues.append(Issue("missing_capability", f"{c.name} requires {missing}"))
            break
        for c in ready:
            available.update(c.provides)
            pending.remove(c)
    field_names = [f.name for f in method.fields]
    if len(set(field_names)) != len(field_names) or set(field_names) & BASE_FIELDS:
        issues.append(
            Issue("field_collision", "Custom fields must be unique and not replace base fields")
        )
    for f in method.fields:
        if f.domain != "gaussian":
            issues.append(Issue("field_domain", f"{f.name}: unsupported domain {f.domain}"))
        if not f.name or "." in f.name or any(not isinstance(n, int) or n <= 0 for n in f.shape):
            issues.append(Issue("field_shape", f"Invalid field declaration {f.name}"))
        if not math.isfinite(f.learning_rate) or f.learning_rate <= 0:
            issues.append(Issue("field_lr", f"Invalid learning rate for {f.name}"))
        if not math.isfinite(f.initial_value):
            issues.append(Issue("field_initial_value", f"Nonfinite initial value for {f.name}"))
    probes = [p.name for p in method.probes]
    contributions = [c.name for c in method.contributions]
    if len(probes) != len(set(probes)) or len(contributions) != len(set(contributions)):
        issues.append(Issue("duplicate_evidence", "Probe and contribution names must be unique"))
    for c in method.contributions:
        missing = set(c.probes) - set(probes)
        if missing:
            issues.append(
                Issue("unknown_probe", f"{c.name} references unknown probes {sorted(missing)}")
            )
    return issues


def require_valid(method: MethodSpec):
    issues = validate(method)
    if issues:
        raise ValueError("Invalid method:\n" + "\n".join(f"{i.code}: {i.message}" for i in issues))
