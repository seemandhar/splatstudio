import pytest

from splatstudio.reporting import compare


def row(scene, psnr, *, seed=0, protocol="same", gaussians=100):
    return {
        "scene": scene,
        "seed": seed,
        "split": "test",
        "protocol_id": protocol,
        "metrics": {"psnr": psnr, "ssim": 0.9, "lpips": 0.1},
        "resources": {"gaussians": gaussians},
    }


def test_errors_do_not_cancel_and_resources_are_visible():
    report = compare([row("a", 30), row("b", 30)], [row("a", 31, gaussians=350), row("b", 29)])
    assert report["summary"]["psnr"]["mean_signed_gap"] == 0
    assert report["summary"]["psnr"]["mean_absolute_gap"] == 1
    assert report["summary"]["psnr"]["worst_absolute_gap"] == 1
    assert report["pairs_within_psnr_tolerance"] == 0
    assert report["pairs"][0]["resource_ratios_candidate_over_reference"]["gaussians"] == 3.5


@pytest.mark.parametrize(
    "candidate",
    [
        [row("b", 30)],
        [row("a", 30, protocol="different")],
        [row("a", float("nan"))],
        [row("a", 30), row("a", 30)],
    ],
)
def test_mismatched_or_invalid_results_are_rejected(candidate):
    with pytest.raises(ValueError):
        compare([row("a", 30)], candidate)


def test_scenes_are_equally_weighted_across_seeds():
    ref = [row("a", 30), row("a", 30, seed=1), row("b", 30)]
    cand = [row("a", 31), row("a", 31, seed=1), row("b", 30)]
    assert compare(ref, cand)["summary"]["psnr"]["mean_absolute_gap"] == 0.5


def test_missing_lpips_is_not_silently_ignored():
    candidate = row("a", 30)
    del candidate["metrics"]["lpips"]
    with pytest.raises(ValueError, match="Metric coverage"):
        compare([row("a", 30)], [candidate])
