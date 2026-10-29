"""Create an ordinary Python package that registers without modifying SplatStudio."""

import re
from pathlib import Path


def create(name, output=None):
    if not re.fullmatch(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*", name):
        raise ValueError("Use a lowercase method name such as my-method")
    from .registry import discover

    if name in discover():
        raise ValueError(f"Method {name!r} is already registered")
    root = Path(output or name)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Scaffold output directory must be empty")
    module = "ss_" + name.replace("-", "_")
    (root / module).mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(f'''[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "splatstudio-method-{name}"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["splatstudio>=0.1,<0.2"]

[project.entry-points."splatstudio.methods"]
{name} = "{module}:method"

[tool.setuptools.packages.find]
include = ["{module}*"]
''')
    (root / module / "__init__.py").write_text(f'''from dataclasses import replace
from splatstudio.methods import make_method


def method():
    # Replace or add ComponentSpec objects to implement your method.
    # Add ContributionSpec and independent ProbeSpec entries for new mechanisms.
    # Inherited probes check infrastructure; they do not validate a new paper.
    return replace(make_method("{name}"),
                   description="Unmodified scaffold; no new paper contribution yet.")
''')
    (root / "README.md").write_text(f"""# {name}

Install into the same Python environment as SplatStudio:

```bash
pip install -e .
ss-inspect {name}
ss-verify {name} --device cuda
ss-train {name} --data /path/to/dataset --output runs/{name}
```

This scaffold initially runs the experimental gsplat baseline. It is not a new
algorithm or a verified paper reproduction. Add executable probes for every
new contribution; custom renderers are ordinary component factories.
""")
    return {"path": str(root.resolve()), "method": name, "module": module}
