"""Hash the SDK, plugins, and renderer source rather than trusting a version label."""

import hashlib
import importlib
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest(method):
    modules = {"splatstudio", "gsplat"}
    modules.update(c.factory.split(":")[0].split(".")[0] for c in method.components)
    modules.update(p.function.split(":")[0].split(".")[0] for p in method.probes)
    files = {}
    for name in sorted(modules):
        module = importlib.import_module(name)
        if not getattr(module, "__file__", None):
            raise ValueError(f"Cannot record source provenance for namespace package {name}")
        entry = Path(module.__file__).resolve()
        root = entry.parent
        sources = root.rglob("*") if hasattr(module, "__path__") else [entry]
        for path in sorted(sources):
            if path.is_file() and path.suffix in {
                ".py",
                ".cu",
                ".cpp",
                ".cuh",
                ".h",
                ".hpp",
                ".so",
            }:
                files[f"{name}/{path.relative_to(root)}"] = file_hash(path)
    return files
