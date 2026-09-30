"""Capture and compare dmtlib outputs bit for bit (A/B checks of refactors).

    python bench/scripts/golden_outputs.py capture out.npz [--backend cpu]
    python bench/scripts/golden_outputs.py compare base.npz new.npz [--tol 0]

`capture` runs a fixed set of FDMT, DDMT, FDMTFFT and CohFDMT configurations
on one backend and stores every output. Two data
sets are used: integer-valued inputs, whose sums are exact on every backend
(so a GPU capture must equal the CPU one bit for bit), and normal-distributed
float inputs (bitwise only between builds of the same backend). `compare`
reports each differing key and exits non-zero on any mismatch beyond `--tol`
(absolute), or when the key sets differ.
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
        1250.0, 25.0, 4, 1.0e-6, 1 << 10, 2, 4.0e-6, 5.0, 0.0, 32, backend=backend
    )
    rng = np.random.default_rng(3)
    raw = rng.integers(0, 256, size=2 * 2 * coh.plan.nsamp * 4, dtype=np.uint8)
    out["cfdmt/u8"] = coh.execute(raw)
    out["cfdmt/u8/block2"] = coh.execute(raw[::-1].copy())


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
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in out.items()})
    print(f"captured {len(out)} arrays on {backend} -> {path}")


def compare(base: str, new: str, tol: float, only: str | None) -> int:
    a, b = np.load(base), np.load(new)
    keys_a = {k for k in a.files if only is None or k.startswith(only)}
    keys_b = {k for k in b.files if only is None or k.startswith(only)}
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
    args = parser.parse_args()
    if args.cmd == "capture":
        capture(args.out, args.backend)
        return 0
    return compare(args.base, args.new, args.tol, args.only)


if __name__ == "__main__":
    sys.exit(main())
