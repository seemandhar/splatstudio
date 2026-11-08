import math

import torch

from ss_wegs import method
from ss_wegs.density import position_lr
from ss_wegs.losses import mask_regulariser, projected_mask_values, sh_regulariser
from ss_wegs.networks import PositionalEncoding, TransNet
from ss_wegs.probes import attention_encoder, masked_loss, sh_reg


def test_pe_matches_code_convention():
    pe = PositionalEncoding(4)
    assert torch.allclose(pe.freqs, torch.tensor([1.0, 2.0, 4.0, 8.0]))
    assert pe.out_dim(3) == 27
    assert PositionalEncoding(0).out_dim(48) == 48


def test_transnet_shape_and_range():
    t = TransNet(16, 16, 32, 4, 0, 4)
    d = t(torch.randn(7, 16, 3), torch.randn(7, 3), torch.randn(16))
    assert d.shape == (7, 16, 3) and float(d.abs().max()) <= 1
    assert sum(isinstance(m, torch.nn.Linear) for m in t.net) == 4


def test_regularisers():
    m = torch.full((1, 1, 2, 2), 0.5)
    assert math.isclose(float(mask_regulariser(m)), 1.0)
    d = torch.zeros(3, 16, 3)
    d[0, 0, 0] = 3.0
    d[0, 0, 1] = 4.0
    assert math.isclose(float(sh_regulariser(d)), 5.0 / 3, rel_tol=1e-6)


def test_projected_mask_lookup():
    m = torch.zeros(1, 1, 4, 8)
    m[..., 4:] = 1.0
    v = projected_mask_values(m, torch.tensor([[2.0, 4.0], [28.0, 4.0]]), 32, 16)
    assert v.tolist() == [0.0, 1.0]


def test_position_lr():
    assert math.isclose(position_lr(2.0, 30000, 30000, 0.01), 0.02)
    assert math.isclose(position_lr(2.0, 150000, 30000, 0.01), 0.02)


def test_cpu_probes():
    attention_encoder(method(), "cpu")
    masked_loss(method(), "cpu")
    sh_reg(method(), "cpu")
