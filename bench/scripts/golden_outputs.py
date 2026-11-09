"""Capture and compare dmtlib outputs bit for bit (A/B checks of refactors).

    python bench/scripts/golden_outputs.py capture out.npz [--backend cpu]
    python bench/scripts/golden_outputs.py compare base.npz new.npz [--tol 0]
        [--only PATTERN] [--exclude PATTERN ...]

`capture` runs a fixed set of FDMT, DDMT, FDMTFFT, CohFDMT and FDMTBoxcar
configurations on one backend and stores every output (FDMTBoxcar candidate
streams field by field, sorted; the CPU skewed-window path through a memory
budget). Two data
sets are used: integer-valued inputs, whose sums are exact on every backend
(so a GPU capture must equal the CPU one bit for bit), and normal-distributed
float inputs (bitwise only between builds of the same backend). `compare`
reports each differing key and exits non-zero on any mismatch beyond `--tol`
(absolute), or when the key sets differ. Across backends, exclude the keys
that differ by design: the skewed-window state (`--exclude skew/`, the GPU
runs its direct path and the saved state is backend-specific) and the
float-input FDMTBoxcar `snr` fields, which differ by rounding.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import dmtlib

F_MIN, F_MAX, TSAMP = 1100.0, 1500.0, 0.001


def _pack(values: np.ndarray, nbits: int) -> np.ndarray:
    """Packs (rows, nsamps) unsigned samples LSB-first, as the engines read them."""
    rows, nsamps = values.shape
    if nbits >= 8:
        dtype = np.uint8 if nbits == 8 else "<u2"
        return values.astype(dtype).view(np.uint8).reshape(rows, -1)
    per_byte = 8 // nbits
    row_bytes = (nsamps * nbits + 7) // 8
    padded = np.zeros((rows, row_bytes * per_byte), dtype=np.uint32)
    padded[:, :nsamps] = values
    shifts = (np.arange(per_byte, dtype=np.uint32) * nbits)[None, None, :]
    return ((padded.reshape(rows, row_bytes, per_byte) << shifts).sum(axis=2)).astype(
        np.uint8
    )


def _fdmt(out: dict, backend: str, kind: str, data: np.ndarray) -> None:
    nchans, nsamps = data.shape[-2:]
    for mode in ("full", "roll", "valid"):
        for nbeams in (1, 3):
            wf = np.ascontiguousarray(np.broadcast_to(data, (nbeams, nchans, nsamps)))
            for fuse in (0, None):
                key = f"fdmt/{kind}/{mode}/b{nbeams}/f{fuse}"
                f = dmtlib.FDMT(
                    F_MIN,
                    F_MAX,
                    nchans,
                    nsamps,
                    TSAMP,
                    128,
                    mode=mode,
                    nbeams=nbeams,
                    fuse_levels=fuse,
                    backend=backend,
                )
                out[key] = f.execute(wf)
                if mode == "valid":
                    out[key + "/block2"] = f.execute(wf[..., ::-1].copy())
                    out[key + "/history"] = f.save_history()


def _fdmt_packed(out: dict, backend: str, rng: np.random.Generator) -> None:
    nchans, nsamps = 128, 1000
    for nbits in (1, 2, 4, 8, 16):
        values = rng.integers(0, 2**nbits, size=(nchans, nsamps), dtype=np.uint32)
        packed = _pack(values, nbits)
        for int_tree in (False, True):
            f = dmtlib.FDMT(
                F_MIN,
                F_MAX,
                nchans,
                nsamps,
                TSAMP,
                128,
                int_tree=int_tree,
                backend=backend,
            )
            out[f"fdmt/packed{nbits}/i{int(int_tree)}"] = f.execute(packed, nbits)


def _fdmt_stepper(out: dict, backend: str, data: np.ndarray) -> None:
    nchans, nsamps = data.shape
    f = dmtlib.FDMT(
        F_MIN, F_MAX, nchans, nsamps, TSAMP, 128, int_tree=False, backend=backend
    )
    f.reset(data)
    f.advance(2)
    out["fdmt/stepper/sub0"] = np.array(f.view_subband_data(0))
    out["fdmt/stepper/final"] = f.finalize()


def _ddmt(out: dict, backend: str, kind: str, data: np.ndarray) -> None:
    nchans = data.shape[0]
    d = dmtlib.DDMT(F_MIN, F_MAX, nchans, TSAMP, 100.0, 1.0, backend=backend)
    out[f"ddmt/{kind}/f32"] = d.execute(data)
    out[f"ddmt/{kind}/f32/block2"] = d.execute(data[:, ::-1].copy())
    out[f"ddmt/{kind}/f32/history"] = d.save_history()


def _ddmt_packed(out: dict, backend: str, rng: np.random.Generator) -> None:
    nchans, nsamps = 128, 3000
    for nbits in (1, 2, 4, 8, 16):
        values = rng.integers(0, 2**nbits, size=(nchans, nsamps), dtype=np.uint32)
        d = dmtlib.DDMT(
            F_MIN, F_MAX, nchans, TSAMP, 100.0, 1.0, nbits=nbits, backend=backend
        )
        out[f"ddmt/packed{nbits}"] = d.execute(_pack(values, nbits), nsamps)


def _fft(out: dict, backend: str, data: np.ndarray) -> None:
    nchans, nsamps = data.shape
    for mode in ("full", "roll", "valid"):
        for frac, tag in ((True, ""), (False, "_int")):
            f = dmtlib.FDMTFFT(
                F_MIN,
                F_MAX,
                nchans,
                nsamps,
                TSAMP,
                128,
                mode=mode,
                backend=backend,
                fractional_delays=frac,
            )
            out[f"fdmt_fft{tag}/{mode}"] = f.execute(data)
    coh = dmtlib.CohFDMT(
        dmtlib.CohFDMTConfig(400.0, 1.0, 16, 4.0e-6, 10.0, 11.0), backend=backend
    )
    rng = np.random.default_rng(3)
    raw = rng.integers(0, 256, size=coh.input_size(), dtype=np.uint8)
    out["cfdmt/u8"] = coh.execute(raw)
    out["cfdmt/u8/block2"] = coh.execute(raw[::-1].copy())


_CAND_ORDER = (
    "beam_idx",
    "subband_level",
    "subband_idx",
    "dm_idx",
    "filter_idx",
    "begin_sample",
    "end_sample",
    "peak_sample",
)


def _store_candidates(out: dict, key: str, parts: list, counts: list) -> None:
    """Store a candidate stream field by field, in a fixed order.

    The engines' storage order is unspecified (parallel), so the candidates
    are sorted first. Counts are (emitted, detected, dropped global, dropped
    per DM) per call.
    """
    cands = np.sort(np.concatenate(parts), order=list(_CAND_ORDER))
    for name in cands.dtype.names:
        out[f"{key}/{name}"] = np.ascontiguousarray(cands[name])
    out[f"{key}/counts"] = np.asarray(counts, dtype=np.uint64)


def _run_search(search: dmtlib.FDMTBoxcar, blocks: list[np.ndarray], end: str) -> tuple:
    parts, counts = [], []
    for block in [*blocks, None]:
        if block is None:
            result = search.drain() if end == "drain" else search.flush()
        else:
            result = search.execute(block)
        parts.append(result.candidates)
        counts.append(
            (
                result.emitted_count,
                result.total_detected,
                result.dropped_global,
                result.dropped_per_dm,
            )
        )
    return parts, counts


def _fdmt_boxcar(out: dict, backend: str) -> None:
    """FDMTBoxcar candidate streams on integer input (exact sums).

    Several width banks, including two whose geometric widths stop below
    `max_boxcar` (9 at wtsp 2, 17 at wtsp 1.5) and one that ends on it (32),
    subband depths 0 and 2, every peak mode, trapezoids, two beams, float
    input holding integers and packed 8-bit codes, streamed over three
    blocks and ended by flush() or drain().
    """
    rng = np.random.default_rng(2026)
    nchans, nsamps, nblocks = 64, 512, 3
    # Centred integers: mean 0, variance 2 (uniform on -2..2).
    codes = rng.integers(0, 5, size=(2, nchans, nsamps * nblocks)).astype(np.int32)
    codes[:, 10:40, 700:712] += 2  # a broad, undispersed burst
    codes[1, :, 1100:1103] += 1
    ints = (codes - 2).astype(np.float32)
    sigma = float(np.sqrt(2.0))
    modes = {
        "pw": dmtlib.PeakFindingMode.CONTIGUOUS_PEAK_PER_WIDTH,
        "aw": dmtlib.PeakFindingMode.CONTIGUOUS_PEAK_ACROSS_WIDTHS,
        "all": dmtlib.PeakFindingMode.ALL_ABOVE_THRESHOLD,
    }
    cases = [
        # Fields: max_boxcar, wtsp, depth, peak mode, trapezoid, beams, end.
        (9, 2.0, 0, "pw", False, 1, "flush"),
        (17, 1.5, 2, "pw", False, 2, "drain"),
        (17, 1.5, 2, "aw", False, 1, "flush"),
        (32, 2.0, 1, "all", False, 1, "drain"),
        (17, 1.5, 1, "pw", True, 1, "drain"),
    ]
    for max_boxcar, wtsp, depth, mode, trap, nbeams, end in cases:
        plan = dmtlib.FDMTPlan(
            F_MIN, F_MAX, nchans, nsamps, TSAMP, 96, subband_depth=depth
        )
        tag = f"w{max_boxcar}_s{wtsp}/d{depth}/{mode}/t{int(trap)}/b{nbeams}/{end}"
        for kind in ("float", "u8"):
            cfg = dmtlib.BoxcarConfig(
                max_boxcar,
                wtsp=wtsp,
                detect_thresh=4.0,
                noise_sigma=sigma,
                peak_mode=modes[mode],
                max_cands_per_dm=0,
                use_trapezoid=trap,
            )
            if kind == "u8":
                cfg.input_nbits = 8
                cfg.input_mean = 2.0
            search = dmtlib.FDMTBoxcar(
                plan, cfg, nbeams=nbeams, chunk_nsamps=128, backend=backend
            )
            blocks = []
            for k in range(nblocks):
                sl = slice(k * nsamps, (k + 1) * nsamps)
                if kind == "u8":
                    block = codes[:nbeams, :, sl].astype(np.uint8)
                else:
                    block = ints[:nbeams, :, sl]
                blocks.append(np.ascontiguousarray(block))
            parts, counts = _run_search(search, blocks, end)
            _store_candidates(out, f"fdmt_boxcar/{kind}/{tag}", parts, counts)
            out[f"fdmt_boxcar/{kind}/{tag}/widths"] = np.asarray(
                search.search_plans[0].widths, dtype=np.int64
            )
    # Overflow counters: small buffers, first-come order.
    plan = dmtlib.FDMTPlan(F_MIN, F_MAX, nchans, nsamps, TSAMP, 96, subband_depth=1)
    cfg = dmtlib.BoxcarConfig(
        17,
        wtsp=1.5,
        detect_thresh=3.0,
        noise_sigma=sigma,
        peak_mode=modes["all"],
        max_candidates=4096,
        max_cands_per_dm=16,
    )
    search = dmtlib.FDMTBoxcar(plan, cfg, chunk_nsamps=128, backend=backend)
    blocks = [
        np.ascontiguousarray(ints[0, :, k * nsamps : (k + 1) * nsamps])
        for k in range(nblocks)
    ]
    _, counts = _run_search(search, blocks, "flush")
    out["fdmt_boxcar/overflow/counts"] = np.asarray(counts, dtype=np.uint64)


def _skewed(out: dict, backend: str) -> None:
    """Capture the CPU skewed-window path, forced by a memory budget.

    A 1152-channel plan to 2700 samples of delay on 4096-sample blocks, with
    a budget below the tree path's. The GPU takes its direct path at this
    size (its skewed levels are covered by the C++ [skew] tests); the same
    configuration is captured there too.
    """
    rng = np.random.default_rng(77)
    nchans, nsamps, nblocks = 1152, 4096, 3
    # Codes 0..215 plus a burst of 40: every value fits in 8 bits.
    codes = rng.integers(0, 216, size=(nchans, nsamps * nblocks), dtype=np.int32)
    codes[200:900, 5000:5006] += 40
    plan = dmtlib.FDMTPlan(1100.0, 1714.0, nchans, nsamps, 455.6e-6, 2700)
    budget = 30 << 20
    f = dmtlib.FDMT(plan, max_memory_bytes=budget, backend=backend)
    for k in range(2):
        block = codes[:, k * nsamps : (k + 1) * nsamps].astype(np.float32)
        out[f"skew/fdmt/f32/block{k}"] = f.execute(block)
    out["skew/fdmt/f32/history"] = f.save_history()
    f = dmtlib.FDMT(plan, max_memory_bytes=budget, backend=backend)
    for k in range(2):
        block = codes[:, k * nsamps : (k + 1) * nsamps].astype(np.uint8)
        out[f"skew/fdmt/u8/block{k}"] = f.execute(block, 8)
    out["skew/fdmt/u8/history"] = f.save_history()

    sub = dmtlib.FDMTPlan(
        1100.0, 1714.0, nchans, nsamps, 455.6e-6, 2700, subband_depth=2
    )
    # Uniform codes on 0..215: mean 107.5, variance (216**2 - 1) / 12. Float
    # input is the centred codes times 2 (integers, exact in float32).
    sigma = float(np.sqrt((216**2 - 1) / 12))
    for kind in ("float", "u8"):
        if kind == "u8":
            cfg = dmtlib.BoxcarConfig(
                17,
                wtsp=1.5,
                detect_thresh=5.5,
                noise_sigma=sigma,
                input_nbits=8,
                input_mean=107.5,
            )
        else:
            cfg = dmtlib.BoxcarConfig(
                17, wtsp=1.5, detect_thresh=5.5, noise_sigma=2 * sigma
            )
        search = dmtlib.FDMTBoxcar(sub, cfg, max_memory_bytes=75 << 20, backend=backend)
        if backend == "cpu" and search.search_latency == 0:
            msg = "the skewed golden did not take the skewed path"
            raise RuntimeError(msg)
        blocks = []
        for k in range(nblocks):
            block = codes[:, k * nsamps : (k + 1) * nsamps]
            if kind == "u8":
                blocks.append(np.ascontiguousarray(block.astype(np.uint8)))
            else:
                blocks.append((2 * block - 215).astype(np.float32))
        parts, counts = _run_search(search, blocks, "drain")
        _store_candidates(out, f"skew/fdmt_boxcar/{kind}", parts, counts)


def capture(path: str, backend: str) -> None:
    out: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(12345)
    ints = rng.integers(0, 8, size=(256, 1024)).astype(np.float32)
    floats = rng.standard_normal((256, 1024), dtype=np.float32)
    for kind, data in (("int", ints), ("float", floats)):
        _fdmt(out, backend, kind, data)
        _ddmt(out, backend, kind, data)
    _fdmt_packed(out, backend, rng)
    _fdmt_stepper(out, backend, ints)
    _ddmt_packed(out, backend, rng)
    _fft(out, backend, floats)
    _fdmt_boxcar(out, backend)
    _skewed(out, backend)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in out.items()})
    print(f"captured {len(out)} arrays on {backend} -> {path}")


def compare(
    base: str, new: str, tol: float, only: str | None, exclude: list[str]
) -> int:
    a, b = np.load(base), np.load(new)

    def wanted(k: str) -> bool:
        if only is not None and not k.startswith(only):
            return False
        return not any(e in k for e in exclude)

    keys_a = {k for k in a.files if wanted(k)}
    keys_b = {k for k in b.files if wanted(k)}
    status = 0
    for k in sorted(keys_a ^ keys_b):
        print(f"only in {'base' if k in keys_a else 'new'}: {k}")
        status = 1
    same = 0
    for k in sorted(keys_a & keys_b):
        x, y = a[k], b[k]
        if x.shape != y.shape:
            print(f"DIFF {k}: shape {x.shape} vs {y.shape}")
            status = 1
            continue
        if np.array_equal(x, y):
            same += 1
            continue
        diff = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
        ndiff = int(np.count_nonzero(x != y))
        verdict = "ok " if diff <= tol else "DIFF"
        print(f"{verdict} {k}: {ndiff}/{x.size} differ, max |diff| {diff:.3g}")
        if diff > tol:
            status = 1
    print(f"{same}/{len(keys_a & keys_b)} arrays bitwise identical")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("out")
    cap.add_argument("--backend", default="cpu")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("base")
    cmp_.add_argument("new")
    cmp_.add_argument("--tol", type=float, default=0.0)
    cmp_.add_argument("--only", help="compare keys with this prefix only")
    cmp_.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="skip keys containing this text (repeatable)",
    )
    args = parser.parse_args()
    if args.cmd == "capture":
        capture(args.out, args.backend)
        return 0
    return compare(args.base, args.new, args.tol, args.only, args.exclude)


if __name__ == "__main__":
    sys.exit(main())
