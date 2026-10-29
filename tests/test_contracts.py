from dataclasses import replace

import pytest

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec
from splatstudio.components import build
from splatstudio.contracts import validate
from splatstudio.methods import absolute, vanilla
from splatstudio.verification import verify


def test_3d_renderer_cannot_satisfy_surfel_representation():
    from splatstudio.methods import surfel

    method = surfel()
    assert not validate(method)
    broken = replace(method, components=(vanilla().components[0], *method.components[1:]))
    assert any(i.code == "missing_capability" for i in validate(broken))


def test_missing_absolute_gradient_is_rejected():
    method = absolute()
    renderer = replace(method.components[0], provides=("render.rgb", "gradient.screen.signed"))
    broken = replace(method, components=(renderer, *method.components[1:]))
    assert any(i.code == "missing_capability" for i in validate(broken))


def test_advertising_an_unimplemented_capability_fails_at_runtime():
    method = absolute()
    renderer = replace(method.components[0], config={"absgrad": False})
    broken = replace(method, components=(renderer, *method.components[1:]))
    assert validate(broken) == []  # A declaration alone is insufficient evidence.
    with pytest.raises(ValueError, match="actual capabilities"):
        build(broken)


def test_competing_density_controllers_are_rejected():
    method = vanilla()
    second = replace(method.components[1], name="another-density")
    assert any(
        i.code == "exclusive_component"
        for i in validate(replace(method, components=method.components + (second,)))
    )


def test_circular_capability_promises_do_not_validate():
    method = vanilla()
    a = ComponentSpec("a", "loss", "unused:factory", ("b",), ("a",))
    b = ComponentSpec("b", "loss", "unused:factory", ("a",), ("b",))
    assert any(
        i.code == "missing_capability"
        for i in validate(replace(method, components=method.components + (a, b)))
    )


@pytest.mark.parametrize(
    "field",
    [
        FieldSpec("means", (3,)),
        FieldSpec("x", (-1,)),
        FieldSpec("x", (2,), domain="anchor"),
        FieldSpec("x", (2,), learning_rate=float("nan")),
    ],
)
def test_invalid_fields_are_rejected(field):
    assert validate(replace(vanilla(), fields=(field,)))


def test_contribution_without_probes_is_unverified():
    method = replace(
        vanilla(), probes=(), contributions=(ContributionSpec("new", "New mechanism"),)
    )
    report = verify(method)
    assert report["status"] == "unverified"
    assert report["contributions"][0]["status"] == "unverified"
    assert report["paper_reproduction"] == "unverified"


def test_cpu_skip_is_not_a_success():
    method = vanilla()
    method = replace(method, probes=(method.probes[-1],), contributions=(method.contributions[-1],))
    report = verify(method, "cpu")
    assert report["status"] == "unverified"
    assert report["probes"][0]["status"] == "unverified"
