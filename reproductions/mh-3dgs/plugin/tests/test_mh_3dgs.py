import torch

from ss_mh_3dgs import core, mcmc_control, method, probes


def test_importance_formula():
    assert probes.importance_formula(None)["max_abs_error"] < 1e-9


def test_voxel_acceptance():
    probes.voxel_acceptance(None)


def test_birth_proposals():
    probes.birth_proposals(None)


def test_schedules():
    out = probes.schedules(None)
    assert out["events"] == 144


def test_regularizers():
    probes.regularizers(None)


def test_voxel_counts_negative_coords():
    pts = torch.tensor([[-0.001, 0.0, 0.0], [-0.019, 0.0, 0.0], [0.001, 0.0, 0.0]])
    q = torch.tensor([[-0.01, 0.0, 0.0], [0.01, 0.0, 0.0], [5.0, 5.0, 5.0]])
    assert core.voxel_counts_at(pts, q, 0.02).tolist() == [2, 1, 0]


def test_lookup_truncates_and_clamps():
    smap = torch.arange(12.0).reshape(1, 3, 4).expand(3, 3, 4)
    means = torch.tensor([[-0.2, -0.5, 0.0], [0.99, 0.6, 0.0], [5.0, 5.0, 0.0]])
    s, _ = core.lookup_world_xy(means, smap, smap)
    # x=int(-0.6)=0, y=int(-1.0)=-1->0 ; x=int(2.97)=2, y=int(1.2)=1 ; clamp to (3,2)
    assert s.tolist() == [0.0, 6.0, 11.0]


def test_specs():
    assert method().name == "mh-3dgs"
    assert mcmc_control().name == "mh-3dgs-mcmc-control"
