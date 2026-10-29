# Build a method

An independently installed package is the unit of extension. Human-written methods and generated methods use the same interface.

```bash
ss-create my-method --output ../my-method
pip install -e ../my-method
ss-inspect my-method
```

The scaffold initially runs the baseline. Edit its `method()` factory to return a `MethodSpec` with your components. For example, add a custom loss to the existing rendering and density recipe:

```python
from dataclasses import replace
from splatstudio import ComponentSpec
from splatstudio.methods import make_method


def method():
    base = make_method("my-method")
    prior = ComponentSpec(
        name="scale-prior",
        kind="loss",
        factory="ss_my_method:make_prior",
        requires=("state.gaussian3d",),
        config={"weight": 0.001},
    )
    return replace(base, components=base.components + (prior,))


def make_prior(spec):
    def loss(params, outputs, target):
        return spec.config["weight"] * params["scales"].exp().square().mean()
    return loss
```

This example defines a regularizer, not a paper contribution. To check its mechanism, add a `ProbeSpec` that compares derivatives against an independently derived formula, then attach that probe to a `ContributionSpec`. The complete [scale-prior package](../examples/scale_prior/ss_scale_prior/__init__.py) demonstrates both and declares an extra per-Gaussian field.

```bash
pip install -e examples/scale_prior
ss-verify scale-prior --device cuda
ss-train scale-prior --data /path/to/scene --output runs/scale-prior
```

## Replace rendering or density control

The factory is an import path, not a fixed callback slot. A renderer package can own its Python/CUDA code and declare outputs such as absolute screen gradients or surface normals. A density controller declares the statistics it consumes and manages its own state. Incompatible requirements fail before training. The native `gsplat-surfel` configuration demonstrates replacing the 3D rasterizer and selecting a corresponding gradient adapter.

See the exact [runtime interface](interfaces.md) before implementing a component. The native state currently supports 3D Gaussian and 2D surfel primitives, SH color, and per-Gaussian extra fields. An anchor representation or dynamic field requires a runtime extension; changing a declaration does not make it supported.

## Test the claimed mechanism

A useful probe can fail when a method-specific mechanism is removed or replaced. Examples include gradient cancellation cases, a known optical depth ordering, analytic derivatives, topology transitions with optimizer moments, or independently rendered geometric fixtures. A test that checks only whether your function returned its expected shape establishes interface compatibility.

Use `ss-verify METHOD --output evidence.json --device cuda` to retain measured evidence. Empty contribution checks and unavailable dependencies remain unverified. Then run a small scene to exercise actual optimization, followed by a complete, preregistered evaluation protocol to study reproduction. See [the results policy](results.md).

## Share

Commit the plugin source, `pyproject.toml`, paper references, supported representation, data prerequisites, configuration, and independent tests to your repository. Others install it with `pip install -e /path/to/your/package` and use `ss-train my-method`. A remote package can also be installed with standard pip Git URLs. SplatStudio discovers it without modifying the central registry.
