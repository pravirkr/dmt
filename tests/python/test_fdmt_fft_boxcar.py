"""FDMTFFTBoxcar detection, calibration, streaming and binding contracts."""

import numpy as np
import pytest

import dmtlib
from dmtlib import (
    BoxcarConfig,
    FDMTBoxcar,
    FDMTFFTBoxcar,
    FDMTPlan,
    PeakFindingMode,
    add_frb_track,
)

NCHANS = 64
NSAMPS = 2048
GRID = list(range(0, 120, 3))


def backends() -> list[str]:
    # The GPU backend only with a device for it (conftest's probe), so these
    # cases skip, rather than fail, on a GPU build without a GPU.
    try:
        from .conftest import _usable
    except ImportError:
        from conftest import _usable
    names = ["cpu"]
    names += [b for b in ("cuda", "hip") if _usable(b)]
    return names


def best(candidates: np.ndarray) -> np.void:
    return candidates[np.argmax(candidates["snr"])]


@pytest.mark.parametrize("backend", backends())
def test_detects_pulse_like_fdmt_boxcar(backend: str) -> None:
    plan = FDMTPlan(1100, 1500, NCHANS, NSAMPS, 0.001, dt_grid=GRID)
    rng = np.random.default_rng(7)
    data = rng.standard_normal((NCHANS, 2 * NSAMPS), dtype=np.float32)
    pulse = np.zeros((NCHANS, NSAMPS), dtype=np.float32)
    add_frb_track(pulse, plan, 25, 1.5, 900, 4)
    data[:, NSAMPS:] += pulse
    config = BoxcarConfig(64, wtsp=1.5, detect_thresh=7.0)
    fft = FDMTFFTBoxcar(plan, config, backend=backend)
    fdmt = FDMTBoxcar(plan, config, backend=backend)
    found = {}
    for name, search in (("fft", fft), ("fdmt", fdmt)):
        parts = [
            search.execute(data[:, k * NSAMPS : (k + 1) * NSAMPS].copy())
            for k in range(2)
        ]
        parts.append(search.flush())
        found[name] = np.concatenate([p.candidates for p in parts])
    a, b = best(found["fft"]), best(found["fdmt"])
    # Same trial and arrival (input-stream samples), comparable S/N: the
    # Fourier tree aligns channels with fractional, not rounded, shifts.
    assert abs(int(a["dm_idx"]) - 25) <= 1
    assert abs(int(a["peak_sample"]) - int(b["peak_sample"])) <= 2
    assert a["snr"] == pytest.approx(b["snr"], rel=0.15)
    assert fft.output_latency == 64


def test_noise_follows_the_analytical_scale() -> None:
    # All points above 1 sigma: the tail fraction of a unit normal.
    plan = FDMTPlan(1100, 1500, NCHANS, 8192, 0.001, dt_grid=[0, 17, 51])
    config = BoxcarConfig(
        1,
        detect_thresh=1.0,
        peak_mode=PeakFindingMode.ALL_ABOVE_THRESHOLD,
        max_candidates=1 << 20,
        max_cands_per_dm=0,
    )
    search = FDMTFFTBoxcar(plan, config)
    rng = np.random.default_rng(3)
    for _ in range(2):
        result = search.execute(rng.standard_normal((NCHANS, 8192), dtype=np.float32))
    assert not result.overflowed
    fraction = len(result.candidates) / (3 * 8192)
    assert fraction == pytest.approx(0.1587, abs=0.01)


def match_runs(
    got: np.ndarray, ref: np.ndarray, thresh: float, rtol: float, atol: float = 0.0
) -> None:
    """Per row, runs of both sets whose extents overlap form a cluster (a
    near-threshold point may split or join runs); each cluster's peak S/N
    agrees within rtol (plus atol S/N units), and a cluster on one side only
    is near threshold."""
    row = ("beam_idx", "dm_idx", "filter_idx")
    tagged = [(c, side) for side, cands in ((0, got), (1, ref)) for c in cands]
    tagged.sort(key=lambda x: (*(int(x[0][f]) for f in row), int(x[0]["begin_sample"])))
    clusters: list[list] = []
    for c, side in tagged:
        last = clusters[-1] if clusters else None
        key = tuple(int(c[f]) for f in row)
        if last is not None and last[0] == key and int(c["begin_sample"]) <= last[1]:
            last[1] = max(last[1], int(c["end_sample"]))
            last[2][side] = max(last[2][side], float(c["snr"]))
        else:
            peaks = [0.0, 0.0]
            peaks[side] = float(c["snr"])
            clusters.append([key, int(c["end_sample"]), peaks])
    for key, end, (a, b) in clusters:
        if a and b:
            assert abs(a - b) <= rtol * abs(b) + atol, (key, end, a, b)
        else:
            assert max(a, b) < thresh * (1 + rtol) + atol, (key, end, a, b)


@pytest.mark.parametrize(("frac", "rtol"), [(False, 1e-4), (True, 0.02)])
def test_streamed_blocks_match_one_long_block(frac: bool, rtol: float) -> None:
    # Integer delays stream exactly; fractional ones differ from one long
    # call by the interpolation guard's truncation (FDMTFFT itself: < 0.5%).
    rng = np.random.default_rng(11)
    data = rng.standard_normal((2, NCHANS, 3 * NSAMPS), dtype=np.float32)
    data[:, :, NSAMPS - 40 : NSAMPS + 40] += 1.2
    small = FDMTPlan(1100, 1500, NCHANS, NSAMPS, 0.001, dt_grid=GRID)
    large = FDMTPlan(1100, 1500, NCHANS, 3 * NSAMPS, 0.001, dt_grid=GRID)
    config = BoxcarConfig(32, wtsp=2.0, detect_thresh=8.0, max_cands_per_dm=0)
    streamed = FDMTFFTBoxcar(
        small, config, nthreads=2, nbeams=2, fractional_delays=frac
    )
    whole = FDMTFFTBoxcar(large, config, nbeams=2, fractional_delays=frac)
    parts = []
    for k in range(3):
        result = streamed.execute(data[:, :, k * NSAMPS : (k + 1) * NSAMPS].copy())
        assert result.input_sample_begin == k * NSAMPS
        parts.append(result.candidates)
    parts.append(streamed.flush().candidates)
    got = np.concatenate(parts)
    ref = np.concatenate([whole.execute(data).candidates, whole.flush().candidates])
    assert len(ref) > 100
    match_runs(got, ref, config.detect_thresh, rtol)


def test_contract_and_validation() -> None:
    plan = FDMTPlan(1100, 1500, 8, 256, 0.001, dt_grid=[0, 5, 9])
    subbands = FDMTPlan(1100, 1500, 8, 256, 0.001, dt_grid=[0, 5, 9], subband_depth=1)
    with pytest.raises(ValueError, match="subband"):
        FDMTFFTBoxcar(subbands, BoxcarConfig(8))
    with pytest.raises(ValueError, match="packed"):
        FDMTFFTBoxcar(plan, BoxcarConfig(8, input_nbits=8, input_mean=1))
    search = FDMTFFTBoxcar(plan, BoxcarConfig(8), kill_mask=[1, 1, 0, 1, 1, 1, 1, 1])
    as_array = FDMTFFTBoxcar(
        plan, BoxcarConfig(8), kill_mask=np.array([1, 1, 0, 1, 1, 1, 1, 1])
    )
    assert as_array.search_plans[0].warmup_samples == (
        search.search_plans[0].warmup_samples
    )
    with pytest.raises(TypeError, match="bool or integer"):
        FDMTFFTBoxcar(plan, BoxcarConfig(8), kill_mask=np.ones(8, dtype=np.float32))
    with pytest.raises(ValueError, match="float32"):
        search.execute(np.zeros((8, 256), dtype=np.float64))
    assert search.memory_usage.total > 0
    assert search.chunk_nsamps > 0
    assert search.search_plans[0].widths == [1, 2, 4, 8]
    assert "FDMTFFTBoxcar" in repr(search)
    integer = FDMTFFTBoxcar(plan, BoxcarConfig(8), fractional_delays=False)
    assert integer.output_latency == 0


@pytest.mark.parametrize("backend", backends())
def test_drain_finds_a_pulse_in_the_last_latency_samples(backend: str) -> None:
    # A pulse ending in the last output_latency samples: flush() never
    # searches it, drain() does; the stream then refuses more blocks.
    plan = FDMTPlan(1100, 1500, NCHANS, NSAMPS, 0.001, dt_grid=GRID)
    rng = np.random.default_rng(3)
    data = rng.standard_normal((NCHANS, NSAMPS), dtype=np.float32)
    add_frb_track(data, plan, 25, 3.0, NSAMPS - 20, 4)
    config = BoxcarConfig(64, wtsp=1.5, detect_thresh=8.0)
    found = {}
    for end in ("flush", "drain"):
        search = FDMTFFTBoxcar(plan, config, backend=backend)
        first = search.execute(data)
        assert search.searched_nsamps == NSAMPS - search.output_latency
        last = getattr(search, end)()
        cands = np.concatenate([first.candidates, last.candidates])
        found[end] = cands[cands["dm_idx"] == 25]
        if end == "drain":
            assert search.searched_nsamps == NSAMPS
            assert (cands["end_sample"] < NSAMPS).all()
            with pytest.raises(RuntimeError, match="drained"):
                search.execute(data)
            search.reset_history()
            assert search.searched_nsamps == 0
    assert len(found["flush"]) == 0
    assert len(found["drain"]) > 0
    assert int(best(found["drain"])["peak_sample"]) >= NSAMPS - 20


def test_fdmt_boxcar_drain_reports_its_latency() -> None:
    plan = FDMTPlan(1100, 1500, NCHANS, NSAMPS, 0.001, dt_grid=GRID)
    search = FDMTBoxcar(plan, BoxcarConfig(64))
    assert search.search_latency == 0
    search.execute(np.zeros((NCHANS, NSAMPS), dtype=np.float32))
    assert search.searched_nsamps == NSAMPS
    assert len(search.drain().candidates) == 0


@pytest.mark.parametrize("backend", backends())
def test_memory_budget_shortens_segments(backend: str) -> None:
    # 256 channels: the transform's buffers dominate the fixed ones.
    nchans = 256
    plan = FDMTPlan(1100, 1500, nchans, 4 * NSAMPS, 0.001, dt_grid=GRID)
    rng = np.random.default_rng(11)
    data = np.round(rng.standard_normal((nchans, 8 * NSAMPS), dtype=np.float32) * 16)
    data /= 16
    config = BoxcarConfig(16, wtsp=1.5, detect_thresh=3.5, max_cands_per_dm=0)
    config.max_candidates = 20000
    unbounded = FDMTFFTBoxcar(plan, config, backend=backend)
    budget = unbounded.memory_usage.total * 6 // 10
    capped = FDMTFFTBoxcar(plan, config, backend=backend, max_memory_bytes=budget)
    assert capped.memory_usage.total <= budget
    assert capped.chunk_nsamps < unbounded.chunk_nsamps
    found = []
    for search in (unbounded, capped):
        parts = [
            search.execute(data[:, k * 4 * NSAMPS : (k + 1) * 4 * NSAMPS].copy())
            for k in range(2)
        ]
        parts.append(search.drain())
        found.append(np.concatenate([p.candidates for p in parts]))
    assert capped.memory_usage.total <= budget  # nothing allocated lazily
    assert len(found[0]) > 100
    # Fractional delays: segment edges truncate the interpolation kernel as
    # block edges do, so the bound of the streaming test applies (2%, plus a
    # small absolute term for noise peaks).
    match_runs(found[1], found[0], config.detect_thresh, 0.02, 0.05)
    with pytest.raises(ValueError, match="max_memory_bytes"):
        FDMTFFTBoxcar(plan, config, backend=backend, max_memory_bytes=1024)


@pytest.mark.parametrize("backend", backends())
def test_candidate_frontier(backend: str) -> None:
    plan = FDMTPlan(1100, 1500, NCHANS, NSAMPS, 0.001, dt_grid=GRID)
    rng = np.random.default_rng(3)
    search = FDMTFFTBoxcar(
        plan, BoxcarConfig(16, wtsp=1.5, detect_thresh=3.0), backend=backend
    )
    frontier = search.get_candidate_frontier()
    assert frontier == 0
    for _ in range(3):
        data = rng.standard_normal((NCHANS, NSAMPS), dtype=np.float32)
        result = search.execute(data)
        assert np.all(result.candidates["begin_sample"] >= frontier)
        nxt = search.get_candidate_frontier()
        assert frontier <= nxt <= search.searched_nsamps
        frontier = nxt
    result = search.drain()
    assert np.all(result.candidates["begin_sample"] >= frontier)
    assert search.get_candidate_frontier() == search.searched_nsamps
