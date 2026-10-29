"""A tiny external loss plugin with an analytic oracle; not a new research method."""

from dataclasses import replace

from splatstudio import ComponentSpec, ContributionSpec, FieldSpec, ProbeSpec
from splatstudio.methods import make_method


def method():
    base = make_method("scale-prior")
    loss = ComponentSpec(
        "scale-prior",
        "loss",
        "ss_scale_prior:create_loss",
        ("state.gaussian3d",),
        (),
        {"weight": 0.001},
    )
    probe = ProbeSpec(
        "scale-prior-gradient",
        "ss_scale_prior:gradient_probe",
        "Compare the implemented log-scale derivative to an independent formula",
    )
    contribution = ContributionSpec(
        "scale-penalty", "Mean squared activated scale regularization", (probe.name,)
    )
    return replace(
        base,
        description="Example external plugin; no paper reproduction claim.",
        components=base.components + (loss,),
        probes=base.probes + (probe,),
        contributions=base.contributions + (contribution,),
        fields=(FieldSpec("example_features", (4,), initial_value=0.5),),
    )


def create_loss(spec):
    weight = float(spec.config["weight"])

    def loss(params, outputs, target):
        return weight * params["scales"].exp().square().mean()

    return loss


def gradient_probe(method, device):
    import torch

    from splatstudio.registry import resolve

    spec = next(c for c in method.components if c.name == "scale-prior")
    fn = resolve(spec.factory)(spec)
    x = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -1.5, -2.5]], device=device, requires_grad=True)
    result = fn({"scales": x}, {}, None)
    gradient = torch.autograd.grad(result, x)[0]
    # Independently derived d [w * mean(exp(2*x))] / dx.
    expected = (2 * spec.config["weight"] / x.numel()) * torch.exp(2 * x.detach())
    torch.testing.assert_close(gradient, expected)
    if not torch.count_nonzero(gradient):
        raise AssertionError("Scale regularizer is inactive")
    return {
        "gradient_norm": gradient.norm().item(),
        "maximum_absolute_error": (gradient - expected).abs().max().item(),
    }
