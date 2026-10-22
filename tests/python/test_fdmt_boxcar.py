"""FDMTBoxcar ownership, streaming, metadata and binding contracts."""

import numpy as np
import pytest

from dmtlib import BoxcarConfig, FDMTBoxcar, FDMTPlan, PeakFindingMode


def config(
    mode: PeakFindingMode = PeakFindingMode.CONTIGUOUS_PEAK_PER_WIDTH,
) -> BoxcarConfig:
    return BoxcarConfig(
        max_boxcar=17,
        wtsp=1.5,
        detect_thresh=2,
        subband_levels=2,
        max_cands_per_dm=0,
        peak_mode=mode,
    )


def sorted_candidates(parts: list[np.ndarray]) -> np.ndarray:
    data = np.concatenate(parts)
    fields = [
        "beam_idx",
        "subband_level",
        "subband_idx",
        "dm_idx",
        "filter_idx",
        "begin_sample",
        "end_sample",
        "peak_sample",
    ]
    return np.sort(data, order=fields)


@pytest.mark.parametrize("mode", list(PeakFindingMode.__members__.values()))
def test_streamed_matches_monolithic(mode: PeakFindingMode) -> None:
    rng = np.random.default_rng(42)
    samples = 733
    blocks = 3
    data = rng.standard_normal((2, 7, samples * blocks), dtype=np.float32)
    data[:, :3, samples - 5 : samples + 9] += 5
    grid = [-73, -12, 0, 4, 97]
    small = FDMTPlan(700, 1200, 7, samples, 0.001, dt_grid=grid)
    large = FDMTPlan(700, 1200, 7, samples * blocks, 0.001, dt_grid=grid)
    streamed = FDMTBoxcar(small, config(mode), nthreads=2, nbeams=2, chunk_nsamps=64)
    whole = FDMTBoxcar(large, config(mode), nbeams=2)
    parts = []
    for k in range(blocks):
        result = streamed.execute(data[:, :, k * samples : (k + 1) * samples].copy())
        assert result.input_sample_begin == k * samples
        assert result.input_nsamps == samples
        assert not result.overflowed
        assert result.emitted_count == len(result.candidates)
        parts.append(result.candidates)
    parts.append(streamed.flush().candidates)
    actual = sorted_candidates(parts)
    expected = sorted_candidates(
        [whole.execute(data).candidates, whole.flush().candidates]
    )
    assert len(actual) > 0
    for name in actual.dtype.names:
        if name == "snr":
            np.testing.assert_allclose(actual[name], expected[name], rtol=2e-5)
        else:
            np.testing.assert_array_equal(actual[name], expected[name])
    saved = actual.copy()
    streamed.reset_history()
    streamed.execute(data[:, :, :samples].copy())
    np.testing.assert_array_equal(actual, saved)
    assert streamed.flush().input_nsamps == 0
    assert streamed.flush().emitted_count == 0
    assert streamed.backend == "cpu"
    assert streamed.nbeams == 2
    assert streamed.chunk_nsamps == 64
    assert streamed.memory_usage.total > 0
    assert "no root" in streamed.summary()
    assert len(streamed.search_plans) == 7


def test_validation_and_owned_results() -> None:
    plan = FDMTPlan(700, 1200, 4, 32, 0.001, 10)
    cfg = config(PeakFindingMode.ALL_ABOVE_THRESHOLD)
    cfg.max_candidates = 2
    cfg.max_cands_per_dm = 3
    search = FDMTBoxcar(plan, cfg)
    data = np.full((4, 32), 10, dtype=np.float32)
    result = search.execute(data)
    saved = result.candidates.copy()
    assert result.overflowed
    assert result.emitted_count == 2
    assert (
        result.total_detected
        == result.emitted_count + result.dropped_global + result.dropped_per_dm
    )
    assert result.candidates.dtype["peak_sample"] == np.dtype("uint64")
    search.reset_history()
    search.execute(np.zeros_like(data))
    np.testing.assert_array_equal(result.candidates, saved)
    for invalid in [
        data.astype(np.float64),
        data[:, ::-1],
        data.reshape(128),
        data[:, :-1],
    ]:
        with pytest.raises(ValueError, match="float32"):
            search.execute(invalid)
    with pytest.raises(ValueError):
        FDMTBoxcar(plan, cfg, nbeams=0)
    cfg.noise_sigma = np.nan
    with pytest.raises(ValueError):
        FDMTBoxcar(plan, cfg)


@pytest.mark.parametrize("mode", list(PeakFindingMode.__members__.values()))
def test_gpu_matches_cpu(gpu_backend: str, mode: PeakFindingMode) -> None:
    rng = np.random.default_rng(81)
    plan = FDMTPlan(700, 1200, 7, 733, 0.001, dt_grid=[-73, -12, 0, 4, 97])
    cpu = FDMTBoxcar(plan, config(mode), chunk_nsamps=64)
    gpu = FDMTBoxcar(plan, config(mode), chunk_nsamps=64, backend=gpu_backend)
    wanted = []
    actual = []
    for _ in range(3):
        data = np.round(rng.standard_normal((7, 733), dtype=np.float32) * 16) / 16
        data[:3, 320:340] += 4
        wanted.append(cpu.execute(data).candidates)
        actual.append(gpu.execute(data).candidates)
    wanted.append(cpu.flush().candidates)
    actual.append(gpu.flush().candidates)
    expected = sorted_candidates(wanted)
    result = sorted_candidates(actual)
    for name in expected.dtype.names:
        if name == "snr":
            np.testing.assert_allclose(result[name], expected[name], rtol=2e-5)
        else:
            np.testing.assert_array_equal(result[name], expected[name])


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Independently padded rows, with least significant sample bits first."""
    lanes = ((codes[..., None] >> np.arange(bits)) & 1).astype(np.uint8)
    return np.packbits(lanes.reshape(*codes.shape[:-1], -1), axis=-1, bitorder="little")


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("mode", list(PeakFindingMode.__members__.values()))
def test_packed_streaming_and_centered_reference(
    bits: int, mode: PeakFindingMode
) -> None:
    rng = np.random.default_rng(13)
    samples = 139
    blocks = 4
    maximum = (1 << bits) - 1
    codes = rng.integers(0, maximum + 1, (2, 7, blocks * samples), dtype=np.uint32)
    mean = maximum / 2
    cfg = config(mode)
    cfg.input_nbits = bits
    cfg.input_mean = mean
    cfg.noise_sigma = np.sqrt(maximum * (maximum + 2) / 12)
    cfg.detect_thresh = 2.713
    grid = [-73, -12, 0, 4, 97]
    plan = FDMTPlan(700, 1200, 7, samples, 0.001, dt_grid=grid)
    packed = FDMTBoxcar(plan, cfg, nbeams=2, nthreads=2, chunk_nsamps=64)
    cfg.input_nbits = 32
    cfg.input_mean = 0
    floating = FDMTBoxcar(plan, cfg, nbeams=2)
    actual, expected = [], []
    for k in range(blocks):
        block = codes[:, :, k * samples : (k + 1) * samples]
        actual.append(packed.execute(pack_codes(block, bits)).candidates)
        expected.append(
            floating.execute((block.astype(np.float32) - mean).copy()).candidates
        )
    actual.append(packed.flush().candidates)
    expected.append(floating.flush().candidates)
    a, b = sorted_candidates(actual), sorted_candidates(expected)
    assert len(a) > 0
    for name in a.dtype.names:
        if name == "snr":
            np.testing.assert_allclose(a[name], b[name], rtol=2e-5)
        else:
            np.testing.assert_array_equal(a[name], b[name])
    with pytest.raises(ValueError):
        packed.execute(np.zeros((2, 7, samples), np.float32))
    with pytest.raises(ValueError):
        packed.execute(pack_codes(codes[:, :, :samples], bits)[:, :, :-1].copy())
    assert len(packed.search_plans[-1].integer_thresh_lut) > 0


def test_constructor_calibrates_unsigned_codes() -> None:
    plan = FDMTPlan(700, 1200, 4, 64, 0.001, dt_grid=[0, 1])
    cfg = BoxcarConfig(
        8,
        detect_thresh=2,
        noise_sigma=2,
        input_nbits=8,
        input_mean=127,
        peak_mode=PeakFindingMode.ALL_ABOVE_THRESHOLD,
        max_cands_per_dm=0,
    )
    search = FDMTBoxcar(plan, cfg)
    result = search.execute(np.full((4, 64), 129, dtype=np.uint8))
    candidates = result.candidates
    width4 = candidates[(candidates["dm_idx"] == 0) & (candidates["width"] == 4)]
    assert len(width4) == 61
    np.testing.assert_allclose(width4["snr"], 4, rtol=1e-6)
