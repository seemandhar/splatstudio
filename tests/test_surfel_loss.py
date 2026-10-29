import pytest
import torch

from splatstudio.losses import create_surfel_geometry
from splatstudio.methods import surfel


@pytest.mark.parametrize("step, expected", [(1, 0.0), (3001, 0.002), (7001, 0.052)])
def test_surfel_geometry_warmup_has_known_value_and_gradients(step, expected):
    loss = create_surfel_geometry(surfel().components[-1])
    normal = torch.tensor([[[1.0, 0, 0]]], requires_grad=True)
    distortion = torch.full((1, 1, 1), 0.2, requires_grad=True)
    outputs = {
        "normal": normal,
        "surface_normal": torch.tensor([[[0.0, 1, 0]]]),
        "alpha": torch.ones(1, 1, 1),
        "distortion": distortion,
        "step": step,
    }
    value = loss({}, outputs, None)
    assert value.item() == pytest.approx(expected)
    value.backward()
    assert distortion.grad.item() == pytest.approx(0.01 if step > 3000 else 0.0)
    assert normal.grad[0, 0, 1].item() == pytest.approx(-0.05 if step > 7000 else 0.0)
