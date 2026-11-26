"""FDMT level-0 boxes of coherently dedispersed data (FDMTPlan chan_dm)."""

import numpy as np
import pytest

from dmtlib import (
    FDMT,
    FDMTFFT,
    BoxcarConfig,
    FDMTBoxcar,
    FDMTFFTBoxcar,
    FDMTPlan,
    PeakFindingMode,
)

K_DM = 1.0 / 2.41e-4
AMP = 48.0 / np.sqrt(128 * 4)  # burst amplitude: matched S/N 48


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    # `backend`: the CPU and every usable GPU backend of this build.
    if "backend" in metafunc.fixturenames:
        try:
            from .conftest import gpu_backends  # noqa: PLC0415
        except ImportError:
            from conftest import gpu_backends  # noqa: PLC0415
        metafunc.parametrize("backend", ["cpu", *gpu_backends()])


Case = tuple[float, float, int, float, np.ndarray, float]

# (f_min, f_max, nchans, tsamp, dt grid, chan_dm): a wide band with mixed-sign
# trials, and a narrow band with a negative chan_dm.
CASES = [
    (704.0, 1216.0, 128, 256e-6, np.arange(-100, 400, 3), 413.0),
    (1100.0, 1714.0, 48, 455.6e-6, np.arange(-40, 200, 3), -150.0),
]


def reference_boxes(plan: FDMTPlan, chan_dm: float) -> tuple[np.ndarray, int]:
    """(lag, s, point) per level-0 coordinate, from the channel geometry."""
    grids = plan.container.dt_grids[0]
    rows = []
    leads = []
    for g in grids:
        f_lo, f_hi = float(g.f_start), float(g.f_end)
        g_c = (0.5 * (f_lo + f_hi)) ** -2
        l_lo = K_DM / plan.tsamp * chan_dm * (f_lo**-2 - g_c)
        l_hi = K_DM / plan.tsamp * chan_dm * (g_c - f_hi**-2)
        leads.append(-np.rint(min(l_lo, -l_hi) - 0.25))
        for dt in g.dt_grid:
            e1, e2 = (l_lo, dt - l_hi) if dt >= 0 else (-l_hi, -dt + l_lo)
            span = np.rint(abs(e2 - e1))
            mid = 0.5 * (e1 + e2)
            rows.append((np.rint(mid - 0.5 * span), span, np.rint(mid)))
    causal = int(max(0, *leads))
    out = np.array(rows, dtype=np.int64)
    out[:, 0] += causal
    out[:, 2] += causal
    return out, causal


def box_outputs(
    plan: FDMTPlan, x: np.ndarray, t0: int, n: int, smear: bool
) -> np.ndarray:
    """Output rows from trace_dm and the level-0 boxes (float64)."""
    ndm = len(plan.dt_grid_final)
    smearing = np.asarray(plan.smearing_grid_final).reshape(ndm, -1).astype(int)
    weight = np.asarray(plan.box_weights, np.float64) * 2.0**-plan.box_fraction_bits
    out = np.zeros((ndm, n))
    t = np.arange(t0, t0 + n)
    for d in range(ndm):
        start = np.asarray(plan.trace_dm(d))
        for c in range(x.shape[0]):
            s = smearing[d, c]
            if smear:
                idx = t[:, None] + start[c] + np.arange(s + 1)[None, :]
                out[d] += x[c, idx].sum(axis=1) * weight[s]
            else:
                # trace_dm gives the box's far end; the sample sits at
                # point = far end - (lag + s - point) of that channel's row.
                out[d] += x[c, t + start[c] + _point_offset(plan, d, c)]
    return out


def _level0_rows(plan: FDMTPlan, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Level-0 coordinate and merge shift of every channel of trial d."""
    coords = plan.container.coordinates
    grids = plan.container.dt_grids
    rows = np.zeros(plan.nchans, dtype=np.int64)
    shift = np.zeros(plan.nchans, dtype=np.int64)
    copy = np.iinfo(np.uint64).max

    def walk(level: int, idx: int, target: int) -> None:
        c = coords[level][idx]
        if level == 0:
            rows[int(c["i_sub"])] = idx
            shift[int(c["i_sub"])] = target
            return
        if int(c["i_coord_head"]) == copy:
            walk(level - 1, int(c["i_coord_tail"]), target)
            return
        dt = grids[level][int(c["i_sub"])].dt_grid[int(c["i_dt"])]
        delay = int(c["delay"])
        walk(level - 1, int(c["i_coord_tail"]), target - (0 if dt >= 0 else delay))
        walk(level - 1, int(c["i_coord_head"]), target - (delay if dt >= 0 else 0))

    walk(len(coords) - 1, d, 0)
    return rows, shift


def _point_offset(plan: FDMTPlan, d: int, c: int) -> int:
    # The sample the row reads sits lag + s - point after the box's far end.
    rows, _ = _level0_rows(plan, d)
    lag, s, point = plan.level0_boxes[rows[c]]
    return int(lag + s - point)


def stream(engine: FDMT | FDMTFFT, blocks: list[np.ndarray], nsamps: int) -> np.ndarray:
    return np.concatenate(
        [np.asarray(engine.execute(b)).reshape(-1, nsamps).copy() for b in blocks],
        axis=1,
    )


@pytest.mark.parametrize("case", CASES)
def test_plan_boxes_match_reference(case: Case) -> None:
    f_min, f_max, nchans, tsamp, grid, chan_dm = case
    plan = FDMTPlan(f_min, f_max, nchans, 512, tsamp, dt_grid=grid, chan_dm=chan_dm)
    expected, causal = reference_boxes(plan, chan_dm)
    np.testing.assert_array_equal(plan.level0_boxes, expected)
    assert plan.level0_causal_delay == causal
    assert plan.chan_dm == chan_dm
    assert (plan.level0_boxes[:, 0] >= 0).all()
    assert (plan.level0_boxes[:, 2] >= 0).all()


def test_plan_without_chan_dm_is_unchanged() -> None:
    plan = FDMTPlan(704.0, 4032.0, 64, 512, 64e-6, dt_grid=np.arange(-30, 90))
    assert plan.level0_boxes.shape == (0, 3)
    assert plan.level0_causal_delay == 0
    assert plan.chan_dm == 0.0


def test_plan_rejects_bad_chan_dm() -> None:
    with pytest.raises(ValueError, match="valid"):
        FDMTPlan(704.0, 4032.0, 64, 512, 64e-6, 100, mode="full", chan_dm=413.0)
    with pytest.raises(ValueError, match="finite"):
        FDMTPlan(704.0, 4032.0, 64, 512, 64e-6, 100, chan_dm=float("nan"))


_BOX_REFS: dict[tuple, np.ndarray] = {}


@pytest.mark.parametrize("fuse", [None, 0, 1, 3])
@pytest.mark.parametrize("smear", [True, False])
@pytest.mark.parametrize("case", CASES)
def test_fdmt_reads_the_boxes(
    case: Case, smear: bool, fuse: int | None, backend: str
) -> None:
    # fuse 0: the GPU's unfused level-0 kernel; 1 and 3 (and None, the
    # automatic depth) its fused tile.
    f_min, f_max, nchans, tsamp, grid, chan_dm = case
    nsamps = 1024
    plan = FDMTPlan(f_min, f_max, nchans, nsamps, tsamp, dt_grid=grid, chan_dm=chan_dm)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((nchans, 4 * nsamps)).astype(np.float32)
    blocks = [
        np.ascontiguousarray(x[:, k * nsamps : (k + 1) * nsamps]) for k in range(4)
    ]
    kw = {} if fuse is None else {"fuse_levels": fuse}
    eng = FDMT(plan, use_box_smearing=smear, backend=backend, nthreads=1, **kw)
    out = stream(eng, blocks, nsamps)
    t0 = plan.startup_samples + 1
    n = 4 * nsamps - t0
    # The reference depends on neither the fusion depth nor the backend.
    key = (f_min, f_max, nchans, tsamp, chan_dm, smear)
    if key not in _BOX_REFS:
        _BOX_REFS[key] = box_outputs(plan, x.astype(np.float64), t0, n, smear)
    ref = _BOX_REFS[key]
    np.testing.assert_allclose(out[:, t0:], ref, rtol=0, atol=2e-5 * np.abs(ref).max())


@pytest.mark.parametrize("fuse", [None, 0, 1, 3])
@pytest.mark.parametrize("case", CASES)
def test_packed_gpu_matches_cpu(gpu_backend: str, case: Case, fuse: int | None) -> None:
    f_min, f_max, nchans, tsamp, grid, chan_dm = case
    nsamps = 512
    plan = FDMTPlan(f_min, f_max, nchans, nsamps, tsamp, dt_grid=grid, chan_dm=chan_dm)
    rng = np.random.default_rng(11)
    codes = [rng.integers(0, 256, (nchans, nsamps), dtype=np.uint8) for _ in range(3)]
    kw = {} if fuse is None else {"fuse_levels": fuse}
    outs = []
    for backend in ("cpu", gpu_backend):
        for int_tree in (True, False):
            eng = FDMT(plan, backend=backend, int_tree=int_tree, nthreads=1, **kw)
            outs.append(
                np.concatenate(
                    [np.asarray(eng.execute(c, nbits=8)).copy() for c in codes], axis=-1
                )
            )
    for o in outs[1:]:
        np.testing.assert_array_equal(o, outs[0])


@pytest.mark.parametrize("smear", [True, False])
@pytest.mark.parametrize("case", CASES)
def test_fdmt_fft_integer_delays_match_fdmt(
    case: Case, smear: bool, backend: str
) -> None:
    f_min, f_max, nchans, tsamp, grid, chan_dm = case
    nsamps = 1024
    plan = FDMTPlan(f_min, f_max, nchans, nsamps, tsamp, dt_grid=grid, chan_dm=chan_dm)
    rng = np.random.default_rng(5)
    blocks = [
        rng.standard_normal((nchans, nsamps)).astype(np.float32) for _ in range(3)
    ]
    ref = stream(FDMT(plan, use_box_smearing=smear), blocks, nsamps)
    fft = FDMTFFT(
        plan, use_box_smearing=smear, backend=backend, fractional_delays=False
    )
    out = stream(fft, blocks, nsamps)
    keep = slice(max(plan.startup_samples, nsamps), None)
    np.testing.assert_allclose(
        out[:, keep], ref[:, keep], rtol=0, atol=2e-5 * np.abs(ref[:, keep]).max()
    )


@pytest.mark.parametrize("smear", [True, False])
def test_fdmt_fft_fractional_gpu_matches_cpu(gpu_backend: str, smear: bool) -> None:
    f_min, f_max, nchans, tsamp, grid, chan_dm = CASES[0]
    nsamps = 1024
    plan = FDMTPlan(f_min, f_max, nchans, nsamps, tsamp, dt_grid=grid, chan_dm=chan_dm)
    rng = np.random.default_rng(13)
    blocks = [
        rng.standard_normal((nchans, nsamps)).astype(np.float32) for _ in range(3)
    ]
    outs = [
        stream(FDMTFFT(plan, use_box_smearing=smear, backend=be), blocks, nsamps)
        for be in ("cpu", gpu_backend)
    ]
    keep = slice(max(plan.startup_samples, nsamps), None)
    ref = outs[0][:, keep]
    np.testing.assert_allclose(
        outs[1][:, keep], ref, rtol=0, atol=2e-5 * np.abs(ref).max()
    )


@pytest.mark.parametrize("engine", ["fdmt", "fft"])
def test_two_beams_match_single_beams(engine: str, backend: str) -> None:
    f_min, f_max, nchans, tsamp, grid, chan_dm = CASES[0]
    nsamps = 512
    plan = FDMTPlan(f_min, f_max, nchans, nsamps, tsamp, dt_grid=grid, chan_dm=chan_dm)
    rng = np.random.default_rng(17)
    beams = [
        [rng.standard_normal((nchans, nsamps)).astype(np.float32) for _ in range(3)]
        for _ in range(2)
    ]
    cls = FDMT if engine == "fdmt" else FDMTFFT
    one = [stream(cls(plan, backend=backend), b, nsamps) for b in beams]
    two = cls(plan, backend=backend, nbeams=2)
    both = np.concatenate(
        [
            np.asarray(two.execute(np.stack([beams[0][k], beams[1][k]]))).copy()
            for k in range(3)
        ],
        axis=-1,
    )
    for i in range(2):
        np.testing.assert_array_equal(both[i], one[i])
    if engine == "fdmt":
        codes = [
            [rng.integers(0, 256, (nchans, nsamps), dtype=np.uint8) for _ in range(3)]
            for _ in range(2)
        ]
        single = []
        for b in codes:
            eng = FDMT(plan, backend=backend)
            single.append(
                np.concatenate(
                    [np.asarray(eng.execute(c, nbits=8)).copy() for c in b], -1
                )
            )
        two = FDMT(plan, backend=backend, nbeams=2)
        both = np.concatenate(
            [
                np.asarray(
                    two.execute(np.stack([codes[0][k], codes[1][k]]), nbits=8)
                ).copy()
                for k in range(3)
            ],
            axis=-1,
        )
        for i in range(2):
            np.testing.assert_array_equal(both[i], single[i])


@pytest.mark.parametrize("smear", [True, False])
def test_skewed_path_matches_the_tree(smear: bool) -> None:
    # A small budget forces the CPU skewed-window path (leaf tables from the
    # same level-0 boxes); packed input keeps every sum exact.
    nsamps, nchans = 2048, 1152
    plan = FDMTPlan(1100.0, 1714.0, nchans, nsamps, 455.6e-6, 2700, chan_dm=413.0)
    rng = np.random.default_rng(17)
    codes = [rng.integers(0, 256, (nchans, nsamps), dtype=np.uint8) for _ in range(3)]
    tree = FDMT(plan, use_box_smearing=smear, nthreads=1)
    budget = (
        FDMT.estimate_memory(plan, use_box_smearing=smear, nthreads=1).usage.total // 2
    )
    skew = FDMT(plan, use_box_smearing=smear, nthreads=1, max_memory_bytes=budget)
    assert tree.history_state_size() != skew.history_state_size()  # skewed path
    a = np.concatenate(
        [np.asarray(tree.execute(c, nbits=8)).copy() for c in codes], axis=-1
    )
    b = np.concatenate(
        [np.asarray(skew.execute(c, nbits=8)).copy() for c in codes], axis=-1
    )
    np.testing.assert_array_equal(b, a)


def test_white_noise_matches_the_plan_variance() -> None:
    nsamps = 8192
    plan = FDMTPlan(
        704.0,
        1216.0,
        128,
        nsamps,
        256e-6,
        dt_grid=np.arange(0, 1200, 30),
        chan_dm=413.0,
    )
    rng = np.random.default_rng(7)
    blocks = [rng.standard_normal((128, nsamps)).astype(np.float32) for _ in range(6)]
    out = stream(FDMT(plan, nthreads=1), blocks, nsamps)[:, plan.startup_samples + 1 :]
    model = np.array(
        [plan.get_effective_sigma(i, 1, True) for i in range(out.shape[0])]
    )
    ratio = out.std(axis=1) / model
    assert abs(ratio.mean() - 1) < 0.01
    assert np.abs(ratio - 1).max() < 0.03


def boxcar_config(
    mode: PeakFindingMode = PeakFindingMode.CONTIGUOUS_PEAK_PER_WIDTH,
) -> BoxcarConfig:
    return BoxcarConfig(
        max_boxcar=17, wtsp=1.5, detect_thresh=4, max_cands_per_dm=0, peak_mode=mode
    )


def sorted_candidates(parts: list[np.ndarray]) -> np.ndarray:
    data = np.concatenate(parts)
    order = np.lexsort(
        [
            data[k]
            for k in (
                "filter_idx",
                "begin_sample",
                "peak_sample",
                "dm_idx",
                "subband_idx",
                "subband_level",
            )
        ]
    )
    return data[order]


def burst(
    plan: FDMTPlan, nsamps: int, nblocks: int, dm: float, amp: float, seed: int
) -> list[np.ndarray]:
    """Noise plus a 4-sample burst at dm, coherently dedispersed at chan_dm."""
    rng = np.random.default_rng(seed)
    nch = plan.nchans
    x = rng.standard_normal((nch, nblocks * nsamps)).astype(np.float32)
    df = (plan.f_max - plan.f_min) / nch
    sub = (np.arange(16) + 0.5) / 16
    for c in range(nch):
        f = plan.f_min + (c + sub) * df
        f_c = plan.f_min + (c + 0.5) * df
        t = (
            600
            + K_DM
            * (dm * (f**-2 - plan.f_max**-2) - plan.chan_dm * (f**-2 - f_c**-2))
            / plan.tsamp
        )
        for tt in t:
            k = int(np.floor(tt))
            x[c, k : k + 4] += amp / 16
    return [
        np.ascontiguousarray(x[:, k * nsamps : (k + 1) * nsamps])
        for k in range(nblocks)
    ]


@pytest.mark.parametrize("subband_depth", [0, 2])
def test_boxcar_gpu_matches_cpu(gpu_backend: str, subband_depth: int) -> None:
    nsamps = 4096
    plan = FDMTPlan(
        704.0,
        1216.0,
        128,
        nsamps,
        256e-6,
        dt_grid=np.arange(0, 10000, 40),
        subband_depth=subband_depth,
        chan_dm=413.0,
    )
    blocks = [np.round(b * 16) / 16 for b in burst(plan, nsamps, 5, 413.0, 3.0, 21)]
    cpu = FDMTBoxcar(plan, boxcar_config(), chunk_nsamps=512)
    gpu = FDMTBoxcar(plan, boxcar_config(), chunk_nsamps=512, backend=gpu_backend)
    want = [cpu.execute(b).candidates for b in blocks] + [cpu.flush().candidates]
    got = [gpu.execute(b).candidates for b in blocks] + [gpu.flush().candidates]
    a, b = sorted_candidates(want), sorted_candidates(got)
    assert len(a) > 0
    for name in a.dtype.names:
        if name == "snr":
            np.testing.assert_allclose(b[name], a[name], rtol=2e-5)
        else:
            np.testing.assert_array_equal(b[name], a[name])


def test_boxcar_finds_burst_at_chan_dm(backend: str) -> None:
    nsamps = 4096
    plan = FDMTPlan(
        704.0,
        1216.0,
        128,
        nsamps,
        256e-6,
        dt_grid=np.arange(0, 10000, 8),
        chan_dm=413.0,
    )
    blocks = burst(plan, nsamps, 5, 413.0, AMP, 5)
    cfg = boxcar_config(PeakFindingMode.CONTIGUOUS_PEAK_ACROSS_WIDTHS)
    cfg.detect_thresh = 8
    search = FDMTBoxcar(plan, cfg, backend=backend)
    cands = sorted_candidates(
        [search.execute(b).candidates for b in blocks] + [search.flush().candidates]
    )
    assert len(cands) > 0
    best = cands[np.argmax(cands["snr"])]
    dms = plan.dm_grid_final
    assert abs(dms[best["dm_idx"]] - 413.0) < 20.0
    # Matched S/N of the burst: AMP * sqrt(nchans * width) = 48.
    assert 36.0 < best["snr"] < 52.0


def test_fdmt_fft_boxcar_accepts_chan_dm() -> None:
    nsamps = 4096
    plan = FDMTPlan(
        704.0,
        1216.0,
        128,
        nsamps,
        256e-6,
        dt_grid=np.arange(0, 10000, 8),
        chan_dm=413.0,
    )
    blocks = burst(plan, nsamps, 5, 413.0, AMP, 5)
    cfg = boxcar_config(PeakFindingMode.CONTIGUOUS_PEAK_ACROSS_WIDTHS)
    cfg.detect_thresh = 8
    search = FDMTFFTBoxcar(plan, cfg)
    cands = sorted_candidates(
        [search.execute(b).candidates for b in blocks] + [search.flush().candidates]
    )
    assert len(cands) > 0
    best = cands[np.argmax(cands["snr"])]
    assert abs(plan.dm_grid_final[best["dm_idx"]] - 413.0) < 20.0
    assert 36.0 < best["snr"] < 55.0


def test_kill_masks_gpu_match_cpu(gpu_backend: str) -> None:
    # Constructor mask plus a per-call mask that changes between blocks.
    nsamps = 16384  # per-call masks: blocks cover the largest row support
    plan = FDMTPlan(
        704.0,
        1216.0,
        128,
        nsamps,
        256e-6,
        dt_grid=np.arange(0, 10000, 40),
        subband_depth=1,
        chan_dm=413.0,
    )
    # Sixteenths keep the float sums exact on both backends.
    blocks = [np.round(b * 16) / 16 for b in burst(plan, nsamps, 3, 413.0, 3.0, 23)]
    fixed = np.ones(128, np.uint8)
    fixed[10:14] = 0
    masks = []
    for k in range(3):
        m = np.ones(128, np.uint8)
        m[(20 * k) % 128 : (20 * k) % 128 + 9] = 0
        masks.append(m)
    res = []
    for backend in ("cpu", gpu_backend):
        search = FDMTBoxcar(plan, boxcar_config(), kill_mask=fixed, backend=backend)
        parts = [
            search.execute(b, kill_mask=m).candidates
            for b, m in zip(blocks, masks, strict=True)
        ]
        res.append(sorted_candidates([*parts, search.flush().candidates]))
    a, b = res
    assert len(a) > 0
    for name in a.dtype.names:
        if name == "snr":
            np.testing.assert_allclose(b[name], a[name], rtol=2e-5)
        else:
            np.testing.assert_array_equal(b[name], a[name])


@pytest.mark.parametrize("fuse", [0, 1, 3])
def test_chunked_gpu_matches_cpu(gpu_backend: str, fuse: int) -> None:
    # Chunks shorter than the block (the relaid-out device plan).
    nsamps = 1024
    plan = FDMTPlan(
        704.0,
        1216.0,
        128,
        nsamps,
        256e-6,
        dt_grid=np.arange(-100, 400, 3),
        chan_dm=413.0,
    )
    rng = np.random.default_rng(29)
    codes = [rng.integers(0, 256, (128, nsamps), dtype=np.uint8) for _ in range(3)]
    outs = []
    for backend, chunk in (("cpu", None), (gpu_backend, 256)):
        kw = {} if chunk is None else {"chunk_nsamps": chunk}
        eng = FDMT(plan, backend=backend, fuse_levels=fuse, **kw)
        outs.append(
            np.concatenate(
                [np.asarray(eng.execute(c, nbits=8)).copy() for c in codes], axis=-1
            )
        )
    np.testing.assert_array_equal(outs[1], outs[0])


def _history_round_trip(kw: dict, *, skewed: bool = False) -> None:
    nsamps, nchans = 2048, 1152
    plan = FDMTPlan(1100.0, 1714.0, nchans, nsamps, 455.6e-6, 2700, chan_dm=413.0)
    if skewed:
        budget = FDMT.estimate_memory(plan, nthreads=1, **kw).usage.total // 2
        kw = {**kw, "max_memory_bytes": budget}
    rng = np.random.default_rng(37)
    codes = [rng.integers(0, 256, (nchans, nsamps), dtype=np.uint8) for _ in range(3)]
    a = FDMT(plan, nthreads=1, **kw)
    for c in codes[:2]:
        a.execute(c, nbits=8)
    saved = np.asarray(a.save_history())
    want = np.asarray(a.execute(codes[2], nbits=8)).copy()
    b = FDMT(plan, nthreads=1, **kw)
    b.load_history(saved)
    np.testing.assert_array_equal(np.asarray(b.execute(codes[2], nbits=8)), want)
    # A history saved without chan_dm (other level-0 boxes) is refused.
    plain = FDMTPlan(1100.0, 1714.0, nchans, nsamps, 455.6e-6, 2700)
    c = FDMT(plain, nthreads=1, **kw)
    other = np.asarray(c.save_history())
    with pytest.raises(ValueError):
        b.load_history(other)


@pytest.mark.parametrize("path", ["tree", "skewed"])
def test_history_round_trip(path: str) -> None:
    _history_round_trip({}, skewed=(path == "skewed"))


def test_history_round_trip_gpu(gpu_backend: str) -> None:
    _history_round_trip({"backend": gpu_backend})


@pytest.mark.parametrize("subband_depth", [0, 2])
def test_boxcar_packed_matches_float(subband_depth: int, backend: str) -> None:
    nsamps = 4096
    plan = FDMTPlan(
        704.0,
        1216.0,
        128,
        nsamps,
        256e-6,
        dt_grid=np.arange(0, 10000, 40),
        subband_depth=subband_depth,
        chan_dm=413.0,
    )
    rng = np.random.default_rng(41)
    codes = [rng.integers(0, 256, (128, nsamps), dtype=np.uint8) for _ in range(4)]
    for c in codes[1:3]:
        c[:, 1500:1504] = np.minimum(c[:, 1500:1504].astype(int) + 60, 255)
    cfg = boxcar_config()
    cfg.input_nbits = 8
    cfg.input_mean = 127.5
    cfg.noise_sigma = float(np.sqrt((256**2 - 1) / 12))
    packed = FDMTBoxcar(plan, cfg, backend=backend)
    cfg.input_nbits = 32
    cfg.input_mean = 0.0
    floating = FDMTBoxcar(plan, cfg, backend=backend)
    # drain(): the packed CPU search takes the skewed path, whose subband
    # rows lag (search_latency); the float one the tree path.
    a = sorted_candidates(
        [packed.execute(c).candidates for c in codes] + [packed.drain().candidates]
    )
    b = sorted_candidates(
        [floating.execute(c.astype(np.float32) - 127.5).candidates for c in codes]
        + [floating.drain().candidates]
    )
    assert len(a) > 0
    for name in a.dtype.names:
        if name == "snr":
            np.testing.assert_allclose(a[name], b[name], rtol=2e-3)
        else:
            np.testing.assert_array_equal(a[name], b[name])


@pytest.mark.parametrize("engine", ["fdmt", "fft"])
@pytest.mark.parametrize("dm", [413.0, 250.0])
def test_candidate_times_are_arrival_times(
    dm: float, engine: str, backend: str
) -> None:
    # Rows lag the bottom-edge arrival by level0_causal_delay; candidates
    # report the arrival (the trailing endpoint of the 4-sample burst).
    nsamps = 4096
    plan = FDMTPlan(
        704.0,
        1216.0,
        128,
        nsamps,
        256e-6,
        dt_grid=np.arange(0, 10000, 4),
        chan_dm=413.0,
    )
    assert plan.level0_causal_delay > 0
    blocks = burst(plan, nsamps, 5, dm, 2 * AMP, 9)
    cfg = boxcar_config(PeakFindingMode.CONTIGUOUS_PEAK_ACROSS_WIDTHS)
    cfg.detect_thresh = 8
    if engine == "fft":
        search = FDMTFFTBoxcar(plan, cfg, backend=backend)
    else:
        search = FDMTBoxcar(plan, cfg, backend=backend)
    cands = sorted_candidates(
        [search.execute(b).candidates for b in blocks] + [search.flush().candidates]
    )
    best = cands[np.argmax(cands["snr"])]
    arrival = 600 + K_DM * dm * (704.0**-2 - 1216.0**-2) / 256e-6
    assert abs(int(best["peak_sample"]) - (arrival + 3)) <= 3
