from .contracts import require_valid
from .registry import resolve


def build(method):
    require_valid(method)
    instances = [(c, resolve(c.factory)(c)) for c in method.components]
    renderer = next(obj for c, obj in instances if c.kind == "renderer")
    density = next(obj for c, obj in instances if c.kind == "density")
    rspec = next(c for c, _ in instances if c.kind == "renderer")
    if set(rspec.provides) != set(renderer.capabilities):
        raise ValueError("Renderer's actual capabilities disagree with its declared specification")
    dspec = next(c for c, _ in instances if c.kind == "density")
    if set(dspec.requires) != set(density.requires):
        raise ValueError("Density controller's actual requirements disagree with its specification")
    if not set(density.requires) <= set(renderer.capabilities):
        raise ValueError("Renderer does not supply the density controller's required statistics")
    losses = [(c.name, obj) for c, obj in instances if c.kind == "loss"]
    if not losses:
        raise ValueError("At least one loss is required for training")
    return renderer, density, losses
