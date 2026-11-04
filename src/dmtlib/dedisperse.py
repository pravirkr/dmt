"""One-shot dedispersion of a whole observation into plain time series.

The streaming engines (FDMT, DDMT, ...) are built for block-by-block
pipelines: fixed block sizes, history kept across calls, and each engine's own
time reference. :func:`dedisperse` wraps them for the common offline case: one
waterfall in, only fully supported output samples out, labelled with the
effective DMs and an explicit time reference.

By default every engine refers output times to ``f_min``, the band's bottom
edge (:attr:`Dedispersed.f_ref`): ``data[:, i]`` is the sum for a signal
arriving at ``f_ref`` at input sample ``start + i``. ``start`` differs between
engines (FDMT's skipped startup, the DDMT family's ``plan.time_offset``), so
align results from different engines by ``start``, not by column. The direct
engines also accept ``f_ref=f_max`` (the top band edge), which gives
``start == 0`` so sample 0 is the file's first sample at ``f_max`` (the
convention of SIGPROC and PRESTO time series).
:meth:`Dedispersed.arrival_times` converts the times to any frequency
(infinite frequency by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .grid import DM_CONSTANT
from .libdmt import DDMT, DDMTFFT, FDMT, FDMTFFT, SDMT

Engine = Literal["fdmt", "fdmt_fft", "ddmt", "sdmt", "ddmt_fft"]

_DIRECT = {"ddmt": DDMT, "sdmt": SDMT, "ddmt_fft": DDMTFFT}
_TREE = {"fdmt": FDMT, "fdmt_fft": FDMTFFT}


def dispersion_delay(
    dm: float | np.ndarray,
    f_lo: float | np.ndarray,
    f_hi: float | np.ndarray,
) -> float | np.ndarray:
    """Dispersion delay in seconds of ``f_lo`` behind ``f_hi`` (MHz).

    Uses the library's constant (``1 / 2.41e-4`` s MHz^2 cm^3 / pc, the
    TEMPO2 convention). ``f_hi = np.inf`` gives the delay from infinite
    frequency. Arguments broadcast.
    """
    f_lo = np.asarray(f_lo, dtype=np.float64)
    f_hi = np.asarray(f_hi, dtype=np.float64)
    inv_hi = np.where(
        np.isinf(f_hi), 0.0, 1.0 / np.where(np.isinf(f_hi), 1.0, f_hi) ** 2
    )
    out = DM_CONSTANT * np.asarray(dm, dtype=np.float64) * (1.0 / f_lo**2 - inv_hi)
    return out if out.ndim else float(out)


@dataclass(frozen=True)
class Dedispersed:
    """Fully supported dedispersed time series of one waterfall.

    Attributes
    ----------
    data : numpy.ndarray
        ``(ndms, nout)`` float32. ``data[d, i]`` is the sum over channels for a
        signal arriving at :attr:`f_ref` at input sample ``start + i``.
    dms : numpy.ndarray
        Effective DM of each row (pc cm^-3). FDMT engines snap requested DMs
        to their integer delay grid; use these values, not the requested ones.
    f_ref : float
        Reference frequency of the output times (MHz).
    start : int
        Input sample index of ``data[:, 0]``.
    tsamp : float
        Sampling interval (s).
    engine : str
        Engine that produced the data.
    """

    data: np.ndarray
    dms: np.ndarray
    f_ref: float
    start: int
    tsamp: float
    engine: str

    def arrival_times(self, f_mhz: float = np.inf) -> np.ndarray:
        """Arrival time at ``f_mhz`` of every output sample, ``(ndms, nout)``.

        Seconds from the first input sample. The default, infinite frequency,
        removes the engine's reference convention; may be negative.
        """
        t_ref = (self.start + np.arange(self.data.shape[-1])) * self.tsamp
        if f_mhz == self.f_ref:
            shift = np.zeros_like(self.dms, dtype=np.float64)
        elif f_mhz > self.f_ref:
            shift = -dispersion_delay(self.dms, self.f_ref, f_mhz)
        else:
            shift = dispersion_delay(self.dms, f_mhz, self.f_ref)
        return t_ref[None, :] + np.asarray(shift)[:, None]


def dedisperse(
    waterfall: np.ndarray,
    f_min: float,
    f_max: float,
    tsamp: float,
    dms: np.ndarray,
    *,
    engine: Engine = "fdmt",
    f_ref: float | None = None,
    use_box_smearing: bool = False,
    nthreads: int = 1,
    backend: str = "cpu",
) -> Dedispersed:
    """Dedisperse a whole waterfall, returning only fully supported samples.

    Parameters
    ----------
    waterfall : numpy.ndarray
        ``(nchans, nsamps)`` float32, ascending frequency (row 0 lowest),
        band edges ``f_min``/``f_max``. For a SIGPROC header use
        :func:`dmtlib.sigproc_band` and :func:`dmtlib.to_ascending`.
    f_min, f_max : float
        Band edges (MHz), not channel centres.
    tsamp : float
        Sampling interval (s).
    dms : array_like
        Requested DM trials (pc cm^-3).
    engine : {'fdmt', 'fdmt_fft', 'ddmt', 'sdmt', 'ddmt_fft'}
        Transform. ``'fdmt'`` is fastest; ``'fdmt_fft'`` (fractional tree
        delays) and ``'ddmt_fft'`` (exact fractional channel delays) keep
        more signal at high Fourier frequencies.
    f_ref : float, optional
        Reference frequency of the output times (MHz), default ``f_min``.
        DDMT, SDMT and DDMT-FFT accept ``f_min`` or ``f_max``; the FDMT
        engines only ``f_min``.
    use_box_smearing : bool
        FDMT engines only. ``False`` (the default here, unlike the engine
        classes) keeps white input white, for periodicity searches. ``True``
        applies the single-pulse box filter, which colours the noise.
    nthreads, backend
        Passed to the engine.

    Returns
    -------
    Dedispersed
        Data with the incomplete cold-start samples removed, plus effective
        DMs and the time reference.
    """
    wf = np.ascontiguousarray(waterfall, dtype=np.float32)
    if wf.ndim != 2:
        msg = f"waterfall must be (nchans, nsamps), got shape {wf.shape}"
        raise ValueError(msg)
    nchans, nsamps = wf.shape
    dm_req = np.atleast_1d(np.asarray(dms, dtype=np.float32))
    kw = {"nthreads": nthreads, "backend": backend}
    if engine in _TREE:
        if f_ref is not None and f_ref != f_min:
            msg = (
                f"{engine} refers times to f_min; for another f_ref use a direct "
                "engine ('ddmt', 'sdmt', 'ddmt_fft') or Dedispersed.arrival_times()"
            )
            raise ValueError(msg)
        eng = _TREE[engine](
            f_min,
            f_max,
            nchans,
            nsamps,
            tsamp,
            dm_grid=dm_req,
            use_box_smearing=use_box_smearing,
            mode="valid",
            **kw,
        )
        skip = int(eng.startup_samples)
        if skip >= nsamps:
            msg = (
                f"{engine}: {nsamps} samples do not cover the {skip}-sample "
                "startup; pass a longer waterfall or fewer/lower DMs"
            )
            raise ValueError(msg)
        latency = int(getattr(eng, "output_latency", 0))
        out = np.asarray(eng.execute(wf)).reshape(-1, nsamps)[:, skip:]
        return Dedispersed(
            data=np.ascontiguousarray(out),
            dms=np.asarray(eng.dm_grid_final, dtype=np.float64),
            f_ref=float(eng.plan.f_ref),
            start=skip - latency,
            tsamp=tsamp,
            engine=engine,
        )
    if engine in _DIRECT:
        if use_box_smearing:
            msg = f"{engine} has no box smearing; use use_box_smearing=False"
            raise ValueError(msg)
        eng = _DIRECT[engine](f_min, f_max, nchans, tsamp, dm_req, f_ref=f_ref, **kw)
        out = np.asarray(eng.execute(wf)).reshape(dm_req.size, -1)
        if out.shape[-1] == 0:
            msg = f"{engine}: {nsamps} samples do not cover the maximum delay"
            raise ValueError(msg)
        return Dedispersed(
            data=out,
            dms=dm_req.astype(np.float64),
            f_ref=float(eng.plan.f_ref),
            start=int(eng.plan.time_offset),
            tsamp=tsamp,
            engine=engine,
        )
    msg = f"unknown engine {engine!r}; choose from {sorted({**_TREE, **_DIRECT})}"
    raise ValueError(msg)
