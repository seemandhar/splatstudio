import torch

from ss_fatesgs import losses, method, method_2dgs, probes


def test_rank_formula():
    probes.rank_formula(None)


def test_smooth_formula():
    probes.smooth_formula(None)


def test_feature_consistency():
    probes.feature_consistency(None)


def test_reg_formula():
    probes.surfel_reg_formula(None)


def test_center_crop_matches_torchvision():
    from torchvision.transforms import CenterCrop
    x = torch.rand(1, 581, 777)
    for h, w in losses.RANK_DEFAULTS["crops"]:
        assert torch.equal(losses.center_crop(x, h, w), CenterCrop((h, w))(x))


def test_specs():
    names = [c.name for c in method().components]
    assert {"depth-rank", "depth-smooth", "feature"} <= set(names)
    assert not {"depth-rank", "depth-smooth", "feature"} & {c.name for c in method_2dgs().components}
