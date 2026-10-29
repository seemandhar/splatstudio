"""Check shipped links, catalog consistency, and source/evidence alignment without CUDA."""

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    methods = json.loads((ROOT / "src/splatstudio/catalog.json").read_text())["methods"]
    entries = set(
        re.findall(
            r"^ss-([\w-]+)\s*=",
            (ROOT / "packages/nerfstudio-methods/pyproject.toml").read_text(),
            re.M,
        )
    )
    names = [m["name"] for m in methods]
    if len(names) != len(set(names)) or set(names) != entries:
        errors.append("Catalog and package registrations differ")
    for method in methods:
        for name in (method["source"], f"docs/methods/{method['name']}.md"):
            if not (ROOT / name).is_file():
                errors.append(f"Missing method source/card: {name}")
        if method["paper_reproduction"] != "unverified":
            errors.append(f"Unexpected reproduction certification: {method['name']}")
    documents = [
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        *ROOT.glob("docs/**/*.md"),
        ROOT / "packages/nerfstudio-methods/README.md",
    ]
    for document in documents:
        if document.name in {"research-plan.md", "validation.md"}:
            continue
        source = document.read_text()
        links = re.findall(r"\]\(([^)]+)\)", source) + re.findall(r'(?:src|href)="([^"]+)"', source)
        for link in links:
            if "://" in link or link.startswith("#"):
                continue
            target = unquote(link.split("#")[0])
            if not (document.parent / target).exists():
                errors.append(f"Broken link: {document.relative_to(ROOT)} -> {link}")
    for path in ROOT.glob("docs/assets/*.svg"):
        ElementTree.parse(path)
    for path in ROOT.glob("results/native/*/*/config.json"):
        record = json.loads(path.read_text())
        for name, digest in record["source_manifest"].items():
            if name.startswith("splatstudio/"):
                local = ROOT / "src" / name
                if not local.exists() or hashlib.sha256(local.read_bytes()).hexdigest() != digest:
                    errors.append(f"Native evidence/source drift: {path.parent.name} / {name}")
    audit = json.loads((ROOT / "results/validation/zoo.json").read_text())
    for name, digest in audit["source_hashes"].items():
        local = ROOT / "packages/nerfstudio-methods/splatstudio_zoo" / name
        if hashlib.sha256(local.read_bytes()).hexdigest() != digest:
            errors.append(f"Collection evidence/source drift: {name}")
    for error in errors:
        print(error)
    if errors:
        return 1
    print(
        f"Validated {len(methods)} catalog entries, public links, SVGs, and measured source hashes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
