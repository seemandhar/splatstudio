import torch

from ss_deblurring_3dgs import method_defocus, method_motion, probes
from ss_deblurring_3dgs.bank import clamp_delta, embed
from ss_deblurring_3dgs.density import depth_prune_threshold, expon_lr
from ss_deblurring_3dgs.renderer import covariance, quat_to_rotmat_unnormalized


def test_mlp_architecture_both_modes():
    probes.mlp_architecture(method_defocus(), "cpu")
    probes.mlp_architecture(method_motion(), "cpu")


def test_extra_points_cpu():
    probes.extra_points(method_defocus(), "cpu")


def test_depth_pruning_cpu():
    probes.depth_pruning(method_defocus(), "cpu")
    probes.depth_pruning(method_motion(), "cpu")


def test_schedule_cpu():
    probes.schedule(method_defocus(), "cpu")
    probes.schedule(method_motion(), "cpu")


def test_photometric_cpu():
    probes.photometric(method_defocus(), "cpu")


def test_pure_functions():
    assert embed(torch.zeros(1, 3), 3).shape == (1, 21)
    assert float(clamp_delta(torch.tensor(-5.0), 0.01, 1.1)) == 1.0
    assert abs(float(clamp_delta(torch.tensor(20.0), 0.01, 1.1)) - 1.1) < 1e-6
    q = torch.nn.functional.normalize(torch.randn(10, 4), dim=-1)
    R = quat_to_rotmat_unnormalized(q)
    assert torch.allclose(R @ R.transpose(1, 2), torch.eye(3).expand(10, 3, 3), atol=1e-5)
    R2 = quat_to_rotmat_unnormalized(q * 1.05)
    assert not torch.allclose(R2 @ R2.transpose(1, 2), torch.eye(3).expand(10, 3, 3), atol=1e-3)
    s = torch.rand(10, 3)
    assert torch.allclose(covariance(s, q), probes.ref_cov(s, q), atol=1e-6)
    z = torch.tensor([0.0, 5.0, 10.0])
    assert torch.allclose(depth_prune_threshold(z, 0.006, 3), torch.tensor([0.006, 0.003, 0.002]))
    assert abs(expon_lr(15000, 1.0, 0.01, 30000) - 0.1) < 1e-9
