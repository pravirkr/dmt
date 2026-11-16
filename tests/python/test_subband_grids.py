"""Subband DM grids of the subband search (FDMTPlan subband_max_snr_loss)."""

import numpy as np
import pytest

import dmtlib
from dmtlib import (
    BoxcarConfig,
    FDMTBoxcar,
    FDMTPlan,
    OverflowPolicy,
    PeakFindingMode,
)

F_MIN, F_MAX, NCHANS, TSAMP, NSAMPS = 704.0, 1216.0, 128, 256e-6, 2048
DMS = dmtlib.generate_dm_grid(F_MIN, F_MAX, NCHANS, TSAMP, 0.0, 300.0, max_snr_loss=0.1)


def _gpu_backends() -> list[str]:
    usable = []
    for b in ("cuda", "hip"):
        if b not in dmtlib.available_backends():
            continue
        try:
            dmtlib.FDMT(1000.0, 1500.0, 4, 8, 0.001, 2, backend=b)
        except Exception:  # noqa: BLE001, S112 - no device: CPU only
            continue
        usable.append(b)
    return usable


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "backend" in metafunc.fixturenames:
        metafunc.parametrize("backend", ["cpu", *_gpu_backends()])


def make_plan(loss: float, depth: int = 3, nsamps: int = NSAMPS) -> FDMTPlan:
    return FDMTPlan(
        F_MIN,
        F_MAX,
        NCHANS,
        nsamps,
        TSAMP,
        dm_grid=DMS,
        subband_depth=depth,
        subband_max_snr_loss=loss,
    )


def config(
    nbits: int = 32, policy: OverflowPolicy = OverflowPolicy.DROP_NEW
) -> BoxcarConfig:
    keep = policy == OverflowPolicy.KEEP_BRIGHTEST
    return BoxcarConfig(
        17,
        wtsp=1.5,
        detect_thresh=4.0,
        noise_sigma=12.0 if nbits == 8 else 1.0,
        peak_mode=PeakFindingMode.CONTIGUOUS_PEAK_PER_WIDTH,
        max_candidates=1 << 20,
        max_cands_per_dm=64 if keep else 0,
        input_nbits=nbits,
        input_mean=127.0 if nbits == 8 else 0.0,
        overflow_policy=policy,
    )


def blocks(
    nbits: int = 32, n: int = 4, seed: int = 3, nsamps: int = NSAMPS
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        x = rng.standard_normal((NCHANS, nsamps)).astype(np.float32)
        x = np.round(x * 16) / 16  # exact float sums
        if nbits == 8:
            x = np.clip(np.rint(127 + 12 * x), 0, 255).astype(np.uint8)
        out.append(x)
    return out


def run(
    search: FDMTBoxcar, data: list[np.ndarray], masks: list[np.ndarray] | None = None
) -> np.ndarray:
    parts = []
    for k, b in enumerate(data):
        kw = {} if masks is None else {"kill_mask": masks[k]}
        parts.append(np.asarray(search.execute(b, **kw).candidates).copy())
    parts.append(np.asarray(search.flush().candidates).copy())
    return np.concatenate(parts)


def keyed(plan: FDMTPlan, search: FDMTBoxcar, cands: np.ndarray) -> dict:
    """Candidates by (level, subband, dt, width, peak, begin) -> snr."""
    planes = {(p.subband_level, p.subband_idx): p for p in search.search_plans}
    out = {}
    for c in cands:
        p = planes[(int(c["subband_level"]), int(c["subband_idx"]))]
        grid = plan.container.dt_grids[p.level][p.subband_idx]
        dt = int(grid.dt_grid[p.coord_indices[int(c["dm_idx"])]])
        key = (
            p.level,
            p.subband_idx,
            dt,
            int(c["filter_idx"]),
            int(c["peak_sample"]),
            int(c["begin_sample"]),
        )
        out[key] = float(c["snr"])
    return out


def sort(c: np.ndarray) -> np.ndarray:
    keys = [
        "filter_idx",
        "begin_sample",
        "peak_sample",
        "dm_idx",
        "subband_idx",
        "subband_level",
    ]
    return c[np.lexsort([c[k] for k in keys])]


def test_off_by_default() -> None:
    plan = make_plan(0.0)
    assert plan.subband_max_snr_loss == 0.0
    search = FDMTBoxcar(plan, config())
    for p in search.search_plans:
        grid = plan.container.dt_grids[p.level][p.subband_idx]
        assert grid.search_rows == []
        np.testing.assert_array_equal(p.coord_indices, np.arange(grid.ndt))


@pytest.mark.parametrize("bad", [-0.1, 1.0, float("nan")])
def test_rejects_bad_loss(bad: float) -> None:
    with pytest.raises(ValueError):
        make_plan(bad)


def test_selection() -> None:
    plain, pruned = make_plan(0.0), make_plan(0.2)
    rows_plain = sum(len(p.dm_grid) for p in FDMTBoxcar(plain, config()).search_plans)
    planes = FDMTBoxcar(pruned, config()).search_plans
    assert sum(len(p.dm_grid) for p in planes) < rows_plain
    root = plain.niters
    for p in planes:
        grid = pruned.container.dt_grids[p.level][p.subband_idx]
        idx = np.asarray(p.coord_indices)
        assert np.all(np.diff(idx) > 0)
        assert idx[-1] < grid.ndt
        if p.level == root:  # the full band is never pruned
            np.testing.assert_array_equal(p.dm_grid, pruned.dm_grid_final)
            assert idx.size == grid.ndt
        if p.staggered:  # its own grid, every row searched
            assert idx.size == grid.ndt
    # The full band's trials are unchanged.
    np.testing.assert_array_equal(pruned.dt_grid_final, plain.dt_grid_final)


@pytest.mark.parametrize("nbits", [32, 8])
def test_rows_match_the_unpruned_plan(nbits: int, backend: str) -> None:
    # Row (level, subband, dt) of a pruned plan is the same FDMT node as in
    # the unpruned plan: same samples, same noise, same candidates.
    plain, pruned = make_plan(0.0), make_plan(0.2)
    data = blocks(nbits)
    s0 = FDMTBoxcar(plain, config(nbits), backend=backend)
    s1 = FDMTBoxcar(pruned, config(nbits), backend=backend)
    a = keyed(plain, s0, run(s0, data))
    b = keyed(pruned, s1, run(s1, data))
    assert len(b) > 100
    root = plain.niters
    staggered = {(p.level, p.subband_idx) for p in s1.search_plans if p.staggered}
    del staggered
    # Nodes of the unpruned plan (the pruned plan's subband grids may hold
    # other trials: its staggered grids push their own down the tree).
    nodes = {
        (p.level, p.subband_idx): set(
            plain.container.dt_grids[p.level][p.subband_idx].dt_grid
        )
        for p in s0.search_plans
    }
    shared = 0
    for key, snr in b.items():
        if key[2] not in nodes[(key[0], key[1])]:
            continue
        assert key in a, key
        assert snr == a[key]
        shared += 1
    assert shared > 100
    # Every candidate of a scored tree row is kept.
    scored = {}
    for p in s1.search_plans:
        if not p.staggered:
            grid = pruned.container.dt_grids[p.level][p.subband_idx]
            scored[(p.level, p.subband_idx)] = {
                int(grid.dt_grid[i]) for i in p.coord_indices
            }
    for key in a:
        rows = scored.get((key[0], key[1]))
        if rows is not None and key[2] in rows:
            assert key in b, key
    assert any(k[0] == root for k in b)


def test_gpu_matches_cpu(gpu_backend: str) -> None:
    plan = make_plan(0.2)
    for nbits in (32, 8):
        for policy in (OverflowPolicy.DROP_NEW, OverflowPolicy.KEEP_BRIGHTEST):
            data = blocks(nbits)
            cfg = config(nbits, policy)
            cpu = sort(run(FDMTBoxcar(plan, cfg), data))
            gpu = sort(run(FDMTBoxcar(plan, cfg, backend=gpu_backend), data))
            assert len(cpu) > 0
            for name in cpu.dtype.names:
                if name == "snr":
                    np.testing.assert_allclose(gpu[name], cpu[name], rtol=2e-5)
                else:
                    np.testing.assert_array_equal(gpu[name], cpu[name])


def test_per_call_mask_matches_a_masked_engine(backend: str) -> None:
    # The per-call noise tables cover exactly the scored rows.
    # (per-call masks need blocks longer than the largest row support)
    plan = make_plan(0.2, nsamps=8192)
    mask = np.ones(NCHANS, dtype=np.uint8)
    mask[10:30] = 0
    mask[90] = 0
    data = blocks(nsamps=8192, n=3)
    for b in data:
        b[mask == 0] = 0
    fixed = run(FDMTBoxcar(plan, config(), backend=backend, kill_mask=mask), data)
    per_call = run(
        FDMTBoxcar(plan, config(), backend=backend), data, [mask] * len(data)
    )
    a, b = sort(fixed), sort(per_call)
    assert len(a) > 0
    for name in a.dtype.names:
        if name == "snr":
            np.testing.assert_allclose(b[name], a[name], rtol=1e-6)
        else:
            np.testing.assert_array_equal(b[name], a[name])


def test_history_of_another_plan_is_refused() -> None:
    plain, pruned = make_plan(0.0), make_plan(0.2)
    data = blocks(n=2)
    a = dmtlib.FDMT(plain)
    for b in data:
        a.execute(b)
    saved = np.asarray(a.save_history())
    c = dmtlib.FDMT(pruned)
    with pytest.raises(ValueError):
        c.load_history(saved)
