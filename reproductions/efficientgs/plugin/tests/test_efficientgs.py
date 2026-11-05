import torch

from ss_efficientgs import core, method
from ss_efficientgs.density import DEFAULTS


def test_band_and_mask():
    assert core.band_of_rest_coeff().tolist() == [1] * 3 + [2] * 5 + [3] * 7
    m = core.rest_mask(torch.tensor([0.0, 1.0, 2.0, 3.0]), 3)
    assert m.sum(dim=1).tolist() == [0, 3, 8, 15]
    assert core.rest_mask(torch.tensor([3.0]), 1).shape == (1, 3)


def test_sparse_selection_matches_official():
    lw = torch.tensor([0.1, 5.0, 3.0, 0.0, 2.0, 7.0, 1.0, 0.5, 0.2, 4.0])
    mask = core.sparse_increment_mask(lw, 0.2)  # k = 2 -> lw >= third largest (4.0)
    assert mask.nonzero().flatten().tolist() == [1, 5, 9]
    order = core.increment_orders(torch.tensor([3.0, 0, 1, 0, 0, 2, 0, 0, 0, 0]), mask)
    assert order.tolist() == [3, 1, 1, 0, 0, 3, 0, 0, 0, 1]


def test_sparse_steps():
    assert core.sparse_steps(15500, 1000, 3) == (16000, 17000, 18000)
    assert core.sparse_steps(16000, 1000, 2) == (16000, 17000)


def test_densify_select():
    stat = torch.tensor([0.0014, 0.0014, 0.0001, 0.0])
    count = torch.tensor([2.0, 2.0, 1.0, 0.0])
    scale = torch.tensor([0.001, 1.0, 0.001, 0.001])
    grads, clone, split = core.densify_select(stat, count, scale, 0.0007, 0.01, 1.0)
    assert grads[3] == 0
    assert clone.tolist() == [True, False, False, False]
    assert split.tolist() == [False, True, False, False]


def test_blend_reference_topk_conventions():
    # two Gaussians on one pixel: front alpha 0.9, back alpha 0.9*... ranks differ by convention
    means = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64)
    conics = torch.tensor([[1.0, 0.0, 1.0]] * 2, dtype=torch.float64)
    opac = torch.tensor([0.9, 0.6], dtype=torch.float64)
    colors = torch.zeros(2, 3, dtype=torch.float64)
    depths = torch.tensor([1.0, 2.0])
    paper = core.blend_reference(means, conics, opac, colors, depths, 1, 1, weight_after_T=False)
    code = core.blend_reference(means, conics, opac, colors, depths, 1, 1, weight_after_T=True)
    # paper weights: 0.9 vs 0.06 -> front; code weights: 0.09 vs 0.024 -> front
    assert paper["dominant"].tolist() == [1, 0] and code["dominant"].tolist() == [1, 0]
    opac2 = torch.tensor([0.99, 0.9], dtype=torch.float64)
    code2 = core.blend_reference(means, conics, opac2, colors, depths, 1, 1, weight_after_T=True)
    # code: 0.99*0.01=0.0099 vs 0.9*0.1*0.01=0.0009 -> front still
    assert code2["dominant"].tolist() == [1, 0]


def test_method_spec():
    m = method()
    assert [f.name for f in m.fields] == ["sh_order"]
    density = next(c for c in m.components if c.kind == "density")
    assert density.config["grad_threshold"] == DEFAULTS["grad_threshold"] == 0.0007
    assert density.config["top_k"] == 1 and density.config["sparse_rate"] == 0.2
    assert density.config["prune_steps"] == [15000, 18000, 21000]
