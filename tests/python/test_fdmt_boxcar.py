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
    small = FDMTPlan(700, 1200, 7, samples, 0.001, dt_grid=grid, subband_depth=2)
    large = FDMTPlan(
        700, 1200, 7, samples * blocks, 0.001, dt_grid=grid, subband_depth=2
    )
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
    # Depth 2 over 7 channels: 4 quarters + 3 staggered, 2 halves + 1, root.
    plans = streamed.search_plans
    assert len(plans) == 11
    assert [p.staggered for p in plans].count(True) == 4
    assert np.any(actual["staggered"] == 1)


def test_validation_and_owned_results() -> None:
    plan = FDMTPlan(700, 1200, 4, 32, 0.001, 10, subband_depth=1)
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
    plan = FDMTPlan(
        700, 1200, 7, 733, 0.001, dt_grid=[-73, -12, 0, 4, 97], subband_depth=2
    )
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
    plan = FDMTPlan(700, 1200, 7, samples, 0.001, dt_grid=grid, subband_depth=2)
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


def run(search: FDMTBoxcar, blocks: list[np.ndarray]) -> np.ndarray:
    parts = [search.execute(block).candidates for block in blocks]
    parts.append(search.flush().candidates)
    return sorted_candidates(parts)


def assert_same(a: np.ndarray, b: np.ndarray) -> None:
    assert len(a) == len(b)
    for name in a.dtype.names:
        np.testing.assert_array_equal(a[name], b[name])


def test_kill_mask_zeroes_channels_and_masks_the_noise() -> None:
    rng = np.random.default_rng(5)
    samples = 256
    plan = FDMTPlan(700, 1200, 8, samples, 0.001, dt_grid=[0, 5, 17], subband_depth=1)
    kill = [1, 0, 1, 1, 0, 0, 0, 0]
    keep = np.array(kill, dtype=bool)
    clean = [
        np.round(rng.standard_normal((8, samples), dtype=np.float32) * 16) / 16
        for _ in range(3)
    ]
    for block in clean:
        block[~keep] = 0
    dirty = [block.copy() for block in clean]
    for block in dirty:
        block[~keep] = 30
    masked = FDMTBoxcar(plan, config(), kill_mask=kill)
    as_array = FDMTBoxcar(plan, config(), kill_mask=keep)
    got = run(masked, dirty)
    assert len(got) > 0
    # Killed channels are never seen: zeroing them gives the same search.
    assert_same(got, run(as_array, clean))
    # Rows of the killed upper half have no noise and never detect.
    assert np.all(got["chan_begin"] < 4)
    for p in masked.search_plans:
        for f, width in enumerate(p.widths):
            var = plan.get_subband_variance_grid(
                p.level, p.subband_idx, width, kill_mask=kill
            )
            inv = np.asarray(p.inv_sigma_lut).reshape(-1, len(p.widths))[:, f]
            live = var > 0
            np.testing.assert_allclose(1 / inv[live] ** 2, var[live], rtol=1e-5)
            assert np.all(inv[~live] == 0)
    assert masked.memory_usage.staging == 8 * samples * 4
    assert FDMTBoxcar(plan, config()).memory_usage.staging == 0
    # The noise model differs from the unmasked search's.
    unmasked = FDMTBoxcar(plan, config())
    assert (
        masked.search_plans[-1].inv_sigma_lut != unmasked.search_plans[-1].inv_sigma_lut
    )
    with pytest.raises(ValueError, match="kill mask"):
        FDMTBoxcar(plan, config(), kill_mask=[1] * 7)
    with pytest.raises(TypeError, match="bool or integer"):
        FDMTBoxcar(plan, config(), kill_mask=np.ones(8, dtype=np.float32))


def test_kill_mask_pulse_snr() -> None:
    # A pulse of amplitude a in every channel at DM 0 scores a sqrt(N - k)
    # with k channels killed, a sqrt(N) unmasked.
    plan = FDMTPlan(700, 1200, 8, 512, 0.001, dt_grid=[0, 3, 9])
    cfg = BoxcarConfig(max_boxcar=1, detect_thresh=2)
    data = np.zeros((8, 512), np.float32)
    data[:, 300] = 4
    kill = [1, 1, 0, 1, 1, 1, 0, 1]

    def root_snr(search: FDMTBoxcar) -> float:
        c = search.execute(data).candidates
        c = c[(c["dm_idx"] == 0) & (c["subband_level"] == 0)]
        assert np.all(c["peak_sample"] == 300)
        return float(c["snr"].max())

    np.testing.assert_allclose(
        root_snr(FDMTBoxcar(plan, cfg, kill_mask=kill)), 4 * np.sqrt(6), rtol=1e-6
    )
    np.testing.assert_allclose(
        root_snr(FDMTBoxcar(plan, cfg)), 4 * np.sqrt(8), rtol=1e-6
    )


@pytest.mark.parametrize("bits", [8, 32])
def test_kill_mask_gpu_matches_cpu(gpu_backend: str, bits: int) -> None:
    rng = np.random.default_rng(9)
    samples = 733
    plan = FDMTPlan(
        700, 1200, 7, samples, 0.001, dt_grid=[-73, -12, 0, 4, 97], subband_depth=2
    )
    kill = np.array([0, 1, 1, 0, 1, 1, 1], dtype=np.uint8)
    cfg = config()
    blocks = []
    if bits == 8:
        cfg.input_nbits = 8
        cfg.input_mean = 127.5
        cfg.noise_sigma = np.sqrt(255 * 257 / 12)
        for _ in range(3):
            codes = rng.integers(0, 256, (7, samples), dtype=np.uint8)
            codes[kill == 0] = 255
            blocks.append(codes)
    else:
        for _ in range(3):
            data = np.round(rng.standard_normal((7, samples), dtype=np.float32) * 16)
            data /= 16
            data[kill == 0] = 30
            data[1:3, 320:340] += 4
            blocks.append(data)
    cpu = run(FDMTBoxcar(plan, cfg, chunk_nsamps=64, kill_mask=kill), blocks)
    gpu = run(
        FDMTBoxcar(plan, cfg, chunk_nsamps=64, kill_mask=kill, backend=gpu_backend),
        blocks,
    )
    assert len(cpu) > 0
    if bits == 8:
        assert_same(gpu, cpu)
        return
    for name in cpu.dtype.names:
        if name == "snr":
            np.testing.assert_allclose(gpu[name], cpu[name], rtol=2e-5)
        else:
            np.testing.assert_array_equal(gpu[name], cpu[name])


def test_drain_searches_the_skewed_tail() -> None:
    # On the skewed-window path (forced by a budget below the tree path's)
    # subband rows lag the input by up to search_latency samples: flush()
    # leaves the stream's tail unsearched, drain() searches it and then
    # agrees with the tree path on every candidate; the stream then refuses
    # blocks until reset_history().
    rng = np.random.default_rng(5)
    nchans, nsamps, nblocks = 1152, 4096, 2
    n = nsamps * nblocks
    plan = FDMTPlan(1100.0, 1714.0, nchans, nsamps, 455.6e-6, 2700, subband_depth=1)
    codes = rng.integers(0, 216, (nchans, n), dtype=np.uint8)
    codes[600:1100, n - 12 : n - 6] += 40
    blocks = [
        np.ascontiguousarray(codes[:, k * nsamps : (k + 1) * nsamps])
        for k in range(nblocks)
    ]
    cfg = BoxcarConfig(
        17,
        wtsp=1.5,
        detect_thresh=5.5,
        noise_sigma=float(np.sqrt((216**2 - 1) / 12)),
        input_nbits=8,
        input_mean=107.5,
    )
    found = {}
    for end in ("flush", "drain"):
        search = FDMTBoxcar(plan, cfg, nthreads=4, max_memory_bytes=75 << 20)
        latency = search.search_latency
        assert latency > 0
        parts = [search.execute(block).candidates for block in blocks]
        assert search.searched_nsamps == n - latency
        parts.append(getattr(search, end)().candidates)
        found[end] = sorted_candidates(parts)
        if end == "drain":
            assert search.searched_nsamps == n
            assert (found[end]["end_sample"] < n).all()
            with pytest.raises(RuntimeError, match="drained"):
                search.execute(blocks[0])
            search.reset_history()
            assert search.searched_nsamps == 0
    tree = FDMTBoxcar(plan, cfg, nthreads=4)
    assert tree.search_latency == 0
    expected = run(tree, blocks)
    assert_same(found["drain"], expected)
    assert len(found["flush"]) < len(found["drain"])


def test_drain_contract_on_the_gpu(gpu_backend: str) -> None:
    # The GPU searches this plan on its direct path (search_latency 0; its
    # windowed top, which lags, is covered by the C++ [skew] tests): drain()
    # ends the stream with the CPU tree path's candidates, bit for bit on
    # packed input, and the stream then refuses blocks.
    rng = np.random.default_rng(6)
    nchans, nsamps = 1152, 4096
    plan = FDMTPlan(1100.0, 1714.0, nchans, nsamps, 455.6e-6, 2700, subband_depth=1)
    codes = rng.integers(0, 216, (nchans, 2 * nsamps), dtype=np.uint8)
    codes[600:1100, -12:-6] += 40
    blocks = [
        np.ascontiguousarray(codes[:, k * nsamps : (k + 1) * nsamps]) for k in range(2)
    ]
    cfg = BoxcarConfig(
        17,
        wtsp=1.5,
        detect_thresh=5.5,
        noise_sigma=float(np.sqrt((216**2 - 1) / 12)),
        input_nbits=8,
        input_mean=107.5,
    )
    gpu = FDMTBoxcar(plan, cfg, backend=gpu_backend)
    parts = [gpu.execute(block).candidates for block in blocks]
    parts.append(gpu.drain().candidates)
    assert gpu.searched_nsamps == 2 * nsamps
    with pytest.raises(RuntimeError, match="drained"):
        gpu.execute(blocks[0])
    gpu.reset_history()
    assert gpu.searched_nsamps == 0
    expected = run(FDMTBoxcar(plan, cfg, nthreads=4), blocks)
    assert len(expected) > 0
    assert_same(sorted_candidates(parts), expected)


def masked_stream(search: FDMTBoxcar, blocks: list, masks: list) -> np.ndarray:
    parts = [
        search.execute(block, kill_mask=mask).candidates
        for block, mask in zip(blocks, masks, strict=True)
    ]
    parts.append(search.flush().candidates)
    return sorted_candidates(parts)


def mask_blocks(seed: int = 5) -> tuple[FDMTPlan, list, list]:
    # 16 channels, 300 samples per block: room for the largest support plus
    # max_boxcar - 1 (a per-call mask requires it).
    rng = np.random.default_rng(seed)
    plan = FDMTPlan(700, 1200, 16, 300, 0.001, dt_grid=[0, 4, 20, 60], subband_depth=1)
    blocks = []
    for _ in range(4):
        data = np.round(rng.standard_normal((16, 300), dtype=np.float32) * 16) / 16
        data[:, 290:] += 2
        blocks.append(data)
    masks = [
        (rng.random(16) > 0.3).astype(np.uint8),
        (rng.random(16) > 0.5).astype(bool),
        None,
        np.ones(16, dtype=np.uint8),
    ]
    return plan, blocks, masks


def test_per_call_kill_mask() -> None:
    plan, blocks, masks = mask_blocks()
    cfg = config()
    cfg.skip_cold_prefix = False
    # uint8 and bool masks; an all-ones or omitted mask is the old call.
    plain = masked_stream(FDMTBoxcar(plan, cfg), blocks, [None] * 4)
    ones = masked_stream(FDMTBoxcar(plan, cfg), blocks, [np.ones(16, dtype=bool)] * 4)
    np.testing.assert_array_equal(ones, plain)
    # A constant per-call mask is the constructor mask, bit for bit.
    fixed = masks[0]
    by_call = masked_stream(FDMTBoxcar(plan, cfg), blocks, [fixed] * 4)
    by_ctor = masked_stream(FDMTBoxcar(plan, cfg, kill_mask=fixed), blocks, [None] * 4)
    np.testing.assert_array_equal(by_call, by_ctor)
    # Changing masks: the constructor mask ANDs in; nonzero means active.
    changing = masked_stream(FDMTBoxcar(plan, cfg), blocks, masks)
    assert not np.array_equal(changing, plain)
    scaled = [None if m is None else m.astype(np.uint8) * 7 for m in masks]
    np.testing.assert_array_equal(
        masked_stream(FDMTBoxcar(plan, cfg), blocks, scaled), changing
    )
    ones16 = np.ones(16, dtype=np.uint8)
    np.testing.assert_array_equal(
        masked_stream(FDMTBoxcar(plan, cfg, kill_mask=ones16), blocks, masks),
        changing,
    )


def test_per_call_kill_mask_validation() -> None:
    plan, blocks, _ = mask_blocks()
    search = FDMTBoxcar(plan, config())
    reference = FDMTBoxcar(plan, config())
    with pytest.raises(ValueError, match="kill_mask"):
        search.execute(blocks[0], kill_mask=np.ones(15, dtype=np.uint8))
    with pytest.raises(TypeError):
        search.execute(blocks[0], kill_mask=np.ones(16, dtype=np.float32))
    # The rejected calls left the stream untouched.
    for block in blocks:
        np.testing.assert_array_equal(
            search.execute(block).candidates, reference.execute(block).candidates
        )
    # Blocks shorter than the largest support plus max_boxcar - 1.
    short = FDMTPlan(700, 1200, 16, 60, 0.001, dt_grid=[0, 4, 20, 60])
    with pytest.raises(ValueError, match="at least"):
        FDMTBoxcar(short, config()).execute(
            np.zeros((16, 60), dtype=np.float32), kill_mask=np.ones(16, np.uint8)
        )
    # Packed input takes no per-call mask.
    cfg = config()
    cfg.input_nbits = 8
    cfg.input_mean = 1.0
    packed = FDMTBoxcar(plan, cfg)
    with pytest.raises(ValueError, match="float32"):
        packed.execute(
            np.zeros((16, 300), dtype=np.uint8), kill_mask=np.ones(16, np.uint8)
        )


def test_per_call_kill_mask_gpu_matches_cpu(gpu_backend: str) -> None:
    plan, blocks, masks = mask_blocks(9)
    cpu = masked_stream(FDMTBoxcar(plan, config()), blocks, masks)
    gpu = masked_stream(FDMTBoxcar(plan, config(), backend=gpu_backend), blocks, masks)
    for name in cpu.dtype.names:
        if name == "snr":
            np.testing.assert_allclose(gpu[name], cpu[name], rtol=2e-5)
        else:
            np.testing.assert_array_equal(gpu[name], cpu[name])


@pytest.mark.parametrize("mode", list(PeakFindingMode.__members__.values()))
def test_candidate_frontier(mode: PeakFindingMode) -> None:
    # Every candidate of a later call begins at or after the frontier, which
    # never decreases; a long excursion holds it at its begin.
    plan, blocks, masks = mask_blocks(13)
    for block in blocks[1:3]:
        block += 2.5  # one excursion over two blocks in most rows
    search = FDMTBoxcar(plan, config(mode))
    assert search.get_candidate_frontier() == 0
    frontier = 0
    held = []
    for block, mask in zip(blocks, masks, strict=True):
        result = search.execute(block, kill_mask=mask)
        assert np.all(result.candidates["begin_sample"] >= frontier)
        nxt = search.get_candidate_frontier()
        assert frontier <= nxt <= search.searched_nsamps
        frontier = nxt
        held.append(nxt)
    if mode != PeakFindingMode.ALL_ABOVE_THRESHOLD:
        assert held[1] < 300  # the excursion opened in block 1 is still open
    result = search.flush()
    assert np.all(result.candidates["begin_sample"] >= frontier)
    assert search.get_candidate_frontier() == search.searched_nsamps
    search.reset_history()
    assert search.get_candidate_frontier() == 0
