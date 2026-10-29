"""Python entry points are the public plugin registration mechanism."""

from importlib import import_module, metadata

from .specs import MethodSpec


def resolve(path):
    module, sep, name = path.partition(":")
    if not sep or not module or not name:
        raise ValueError(f"Expected 'module:object', received {path!r}")
    obj = import_module(module)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def discover():
    from .methods import BUILTINS

    found = dict(BUILTINS)
    points = metadata.entry_points(group="splatstudio.methods")
    for ep in sorted(points, key=lambda e: (e.name, e.value)):
        if ep.name in found:
            raise ValueError(
                f"Duplicate method {ep.name!r}; uninstall or rename the conflicting plugin"
            )
        found[ep.name] = ep.value
    return found


def load(name):
    found = discover()
    if name not in found:
        raise ValueError(f"Unknown method {name!r}. Installed methods: {', '.join(sorted(found))}")
    obj = resolve(found[name])
    method = obj() if callable(obj) else obj
    if not isinstance(method, MethodSpec):
        raise TypeError(f"{name} does not export a MethodSpec")
    if method.name != name:
        raise ValueError(f"Entry point {name!r} exports method {method.name!r}")
    return method
