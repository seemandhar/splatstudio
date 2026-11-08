"""CPU unit tests (CUDA probes run in ss-verify)."""

from ss_loss_aware_dc import method, probes


def test_joint_split_matrix():
    out = probes.joint_split_matrix(method(), "cpu")
    assert out["chain_rule_rel_err"] < 1e-8


def test_decisions():
    out = probes.decisions(method(), "cpu")
    assert out["split"] == [0, 7] and out["clone"] == [3] and out["decay"] == [5]


def test_schedule():
    probes.schedule(method(), "cpu")


def test_proxy_curvature():
    probes.proxy_curvature(method(), "cpu")
