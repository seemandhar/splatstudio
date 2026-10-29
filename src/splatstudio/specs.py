"""Serializable declarations; these describe requirements, not proofs of correctness."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    kind: str  # renderer, density, or loss in the native Gaussian runtime
    factory: str  # importable module:function taking this ComponentSpec
    requires: tuple[str, ...] = ()
    provides: tuple[str, ...] = ()
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FieldSpec:
    name: str
    shape: tuple[int, ...]
    learning_rate: float = 0.001
    initial_value: float = 0.0
    domain: str = "gaussian"


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    function: str  # module:function(method, device) -> dict of measured evidence
    description: str
    requires_cuda: bool = False


@dataclass(frozen=True)
class ContributionSpec:
    name: str
    description: str
    probes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MethodSpec:
    name: str
    description: str
    components: tuple[ComponentSpec, ...]
    probes: tuple[ProbeSpec, ...] = ()
    contributions: tuple[ContributionSpec, ...] = ()
    fields: tuple[FieldSpec, ...] = ()
    citations: tuple[str, ...] = ()
    representation: str = "gaussian3d"
    api_version: int = 1

    def to_dict(self):
        return asdict(self)
