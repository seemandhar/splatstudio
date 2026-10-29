"""The SDK imports without torch, gsplat, CUDA, or an LLM dependency."""

from .specs import ComponentSpec, ContributionSpec, FieldSpec, MethodSpec, ProbeSpec

__version__ = "0.1.0"
__all__ = ["ComponentSpec", "ContributionSpec", "FieldSpec", "MethodSpec", "ProbeSpec"]
