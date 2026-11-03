"""Noise-only validation; deliberately no FDMT/DDMT signal-sensitivity claims."""

import runpy
from pathlib import Path

import numpy as np
import pytest

from dmtlib import DDMT, FDMT, FDMTFFT

ANALYSIS = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "bench/scripts/fdmt_noise.py")
)
WHITE = ANALYSIS["WHITE_ENGINES"]


@pytest.mark.parametrize(
    ("nchans", "nsamps", "delays", "blocks"),
    [(16, 512, [0, 64, 128], 2), (17, 256, [0, 97, 768], 4)],
)
def test_fdmt_and_ddmt_preserve_white_noise(
    nchans: int,
    nsamps: int,
    delays: list[int],
    blocks: int,
) -> None:
    streams, plan, _ = ANALYSIS["collect_noise"](
        nchans=nchans,
        nsamps=nsamps,
        realizations=256,
        dt_grid=delays,
        blocks=blocks,
        seed=45872,
    )
    report = ANALYSIS["analyze"](streams, plan)
    for name in WHITE:
        for trial, row in enumerate(report[name]):
            assert row["passed"], (name, delays[trial], row)
    # Positive control: these geometries have multi-sample channel windows.
    # A regression that accidentally enables averaging must fail the null check.
    assert not report["fdmt_box"][-1]["passed"]
    observed = report["fdmt_box"][-1]["acf"]["1"]
    assert observed > 0.5 * report["fdmt_box"][-1]["energy"]


def test_gpu_fdmt_and_ddmt_preserve_white_noise(gpu_backend: str) -> None:
    # Device construction/execution errors propagate after backend discovery.
    streams, plan, _ = ANALYSIS["collect_noise"](
        nchans=17,
        nsamps=512,
        realizations=256,
        dt_grid=[0, 64, 256],
        backend=gpu_backend,
        seed=45872,
    )
    report = ANALYSIS["analyze"](streams, plan)
    for name in WHITE:
        assert all(row["passed"] for row in report[name]), (name, report[name])


def test_cold_start_is_not_complete_support_and_reset_restarts_it() -> None:
    f = FDMT(
        400.0,
        800.0,
        17,
        256,
        0.001,
        dt_grid=np.array([128]),
        use_box_smearing=False,
        mode="valid",
    )
    ones = np.ones((17, 256), dtype=np.float32)
    cold = f.execute(ones)
    warm = f.execute(ones)
    assert f.startup_samples == f.plan.startup_samples == 128
    assert cold[0, 127] < 17
    np.testing.assert_array_equal(cold[0, 128:], 17)
    np.testing.assert_array_equal(warm, 17)
    f.reset_history()
    np.testing.assert_array_equal(f.execute(ones), cold)
    # DDMT instead withholds unsupported samples on its first call.
    d = DDMT(400.0, 800.0, 17, 0.001, np.asarray(f.dm_grid_final, dtype=np.float32))
    first = d.execute(ones)
    assert first.shape[-1] < 256
    np.testing.assert_array_equal(first, 17)
    assert d.execute(ones).shape[-1] == 256


def test_signed_unfiltered_noise_across_block_boundaries() -> None:
    rng = np.random.default_rng(82941)
    beams, chans, block = 256, 7, 256
    delays = np.array([-384, -47, 0, 47, 384], dtype=np.int64)
    x = rng.standard_normal((beams, chans, block * 6), dtype=np.float32)
    kwargs = {
        "dt_grid": delays,
        "use_box_smearing": False,
        "mode": "valid",
        "nbeams": beams,
    }
    whole = FDMT(400.0, 800.0, chans, block * 6, 0.001, **kwargs)
    stream = FDMT(400.0, 800.0, chans, block, 0.001, **kwargs)
    expected = whole.execute(x)
    actual = np.concatenate(
        [
            stream.execute(np.ascontiguousarray(x[..., start : start + block]))
            for start in range(0, x.shape[-1], block)
        ],
        axis=-1,
    )
    np.testing.assert_array_equal(actual, expected)
    for trial in range(delays.size):
        row = ANALYSIS["white_noise_metrics"](
            actual[:, trial, block * 2 :], chans, comparisons=delays.size
        )
        assert row["passed"], (delays[trial], row)


@pytest.mark.parametrize(
    "grid",
    [[256], [-384, 0, 47], [0, 97, 768], list(range(0, 600, 7)), [-500, -3]],
)
@pytest.mark.parametrize("box", [False, True])
@pytest.mark.parametrize("int_tree", [True, False])
def test_startup_samples_is_the_cold_start_ramp(
    grid: list[int], box: bool, int_tree: bool
) -> None:
    chans, nsamps = 17, 2048
    f = FDMT(
        400.0,
        800.0,
        chans,
        nsamps,
        0.001,
        dt_grid=np.array(grid),
        use_box_smearing=box,
        int_tree=int_tree,
    )
    ones = np.ones((chans, nsamps), dtype=np.float32)
    cold = np.asarray(f.execute(ones)).reshape(-1, nsamps)
    warm = np.asarray(f.execute(ones)).reshape(-1, nsamps)
    s = f.startup_samples
    # Exact: the last incomplete output is s - 1, and every later one is warm.
    np.testing.assert_allclose(cold[:, s:], warm[:, s:], rtol=0, atol=1e-5)
    assert np.any(np.abs(cold[:, s - 1] - warm[:, s - 1]) > 1e-3)


@pytest.mark.parametrize("box", [False, True])
def test_fdmt_fft_startup_samples_bounds_the_ramp(box: bool) -> None:
    chans, nsamps = 17, 2048
    f = FDMTFFT(
        400.0,
        800.0,
        chans,
        nsamps,
        0.001,
        dt_grid=np.array([0, 97, 768]),
        use_box_smearing=box,
    )
    ones = np.ones((chans, nsamps), dtype=np.float32)
    cold = np.asarray(f.execute(ones)).reshape(-1, nsamps)
    warm = np.asarray(f.execute(ones)).reshape(-1, nsamps)
    s = f.startup_samples
    assert s >= f.plan.startup_samples + f.output_latency
    np.testing.assert_allclose(cold[:, s:], warm[:, s:], rtol=0, atol=1e-2)


def test_cold_start_mean_ramp_and_its_removal() -> None:
    # Low-frequency excess from nonzero-mean input and a cold start. Dropping startup_samples restores white statistics.
    result = ANALYSIS["cold_start_noise"](
        mean=1.0, nchans=16, nsamps=4096, realizations=128, delay=1024, seed=7
    )
    assert result["startup_samples"] == 1024
    assert not result["fdmt_cold"]["passed"]
    assert result["fdmt_cold"]["spectral_mean"][1] > 100
    assert result["fdmt_trimmed"]["passed"], result["fdmt_trimmed"]
    assert result["ddmt"]["passed"], result["ddmt"]
