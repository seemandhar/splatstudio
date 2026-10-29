"""Build method documentation from the packaged catalog and measured audit."""

import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="results/validation/zoo.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    catalog_path = root / "src/splatstudio/catalog.json"
    catalog = json.loads(catalog_path.read_text())
    audit = {
        r["name"].removeprefix("ss-"): r
        for r in json.loads((root / args.audit).read_text())["methods"]
    }
    historical = json.loads((root / "results/historical/records.json").read_text())["records"]
    counts = Counter(r["catalog_method"] for r in historical)
    for method in catalog["methods"]:
        evidence = audit[method["name"]]
        method["configuration_import"] = evidence["configuration_import"]
        method["training_check"] = evidence["training_check"]["status"]
        method["training_check_scope"] = evidence["training_check"].get(
            "scope", evidence["training_check"].get("reason", "")
        )
        method["historical_records"] = counts[method["name"]]
        name = method["name"]
        references = "\n".join(
            f"- [Reference {i + 1}]({url})" for i, url in enumerate(method["references"])
        )
        if not references:
            references = "No standalone paper reference is asserted for this configuration. Consult the source lineage."
        card = f"""# {method["title"]}

`{name}` · {method["category"]} · Nerfstudio research collection

```bash
pip install -e packages/nerfstudio-methods
{method["command"]}
```

| Check | Status |
| :--- | :--- |
| Configuration import | {method["configuration_import"]} |
| Short training check | {method["training_check"]} |
| Complete paper reproduction | **Unverified** |
| Historical scene records | {counts[name]} (not rerun with this package) |

**Training scope:** {method["training_check_scope"]}

**Implementation notes:** {method["notes"]}

Ten-step checks use a 32×32 analytic sphere fixture, full-resolution training, a fixed seed, and stock Nerfstudio 1.1.5. They exercise model construction, forward/backward, optimizer steps, callbacks, finite parameters, and checkpoint writing. They do not exercise all schedules, prove convergence, or establish the paper's mechanisms. See [the complete validation record](../../results/validation/zoo.json).

[Implementation](../../{method["source"]}) · [Configurations](../../packages/nerfstudio-methods/splatstudio_zoo/configs.py) · [Source lineage](../../packages/nerfstudio-methods/ORIGIN.json) · [All methods](../methods.md)

## References

{references}

References identify the intended technique; they do not certify this port's fidelity. Cite the original authors when using their work.
"""
        (root / f"docs/methods/{name}.md").write_text(card)
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    page = """# Method collection

The collection contains **33 research configurations**, including baseline and capacity variants and exploratory extensions. These are imported implementations from [SPLATIFY](https://github.com/seemandhar/splatify), packaged for standard Nerfstudio discovery. They are not 33 independent, verified paper reproductions.

Install with `pip install -e packages/nerfstudio-methods` in a Nerfstudio environment, then run `ns-train ss-METHOD --data /path/to/scene`. Native configurations use `ss-train`; see the [root quickstart](../README.md).

All 33 configuration imports passed on stock Nerfstudio 1.1.5. Short training checks and their exact scope are listed below. **Passed** means ten steps on a 32×32 analytic fixture with finite state and checkpoint writing. **Not run** identifies missing auxiliary inputs. These checks leave paper reproduction unverified.

| Configuration | Area | Training check | Historical records |
| :--- | :--- | :--- | ---: |
"""
    for method in sorted(catalog["methods"], key=lambda m: (m["category"], m["name"])):
        name = method["name"]
        page += f"| [{name}](methods/{name}.md) | {method['category']} | {method['training_check'].replace('_', ' ')} | {counts[name]} |\n"
    page += """
## Known boundaries

The imported `twodgs` port uses the shared 3D rasterizer with geometric regularizers. It does not implement the original paper's 2D ray-splat intersection. The native `gsplat-surfel` recipe does call the actual gsplat 2D rasterizer, with its separate L1 distortion convention.

GaussianShader uses a PyTorch approximation of its shading path. CompGS includes approximated pruning cadence. DepthRegGS/FSGS require depth models, StableGS requires DUSt3R priors, and FeatureGS needs feature targets. See individual cards for details. Nebula, bubble, and diamond configurations are exploratory appearance models; no physical fidelity or scientific priority is asserted.

Historical records are inherited exports with unverified protocols, not results from this release. The [results report](results.md) explains their scope and the new native experiments. Machine-readable metadata ships in [catalog.json](../src/splatstudio/catalog.json).
"""
    (root / "docs/methods.md").write_text(page)
    print(f"Generated {len(catalog['methods'])} method cards and index")


if __name__ == "__main__":
    main()
