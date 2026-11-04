"""Channel-order helpers for SIGPROC-style filterbank headers.

Every dmt engine takes waterfalls in **ascending** frequency order: row 0 is
the lowest channel, whose lower edge is ``f_min``, and band edges (not
channel centres) define the band. SIGPROC headers instead give ``fch1``, the
centre of the first channel, and the signed channel offset ``foff`` (usually
negative: the first channel is the highest). :func:`sigproc_band` converts a
header to dmt's band edges and :func:`to_ascending` flips the data when
needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Band:
    """Band edges in dmt's convention.

    Attributes
    ----------
    f_min, f_max : float
        Lower and upper band edges (MHz).
    nchans : int
        Number of channels.
    flip : bool
        True when the source rows are descending in frequency and must be
        reversed (:func:`to_ascending`) before they reach a dmt engine.
    """

    f_min: float
    f_max: float
    nchans: int
    flip: bool

    @property
    def df(self) -> float:
        """Channel width (MHz), positive."""
        return (self.f_max - self.f_min) / self.nchans


def sigproc_band(fch1: float, foff: float, nchans: int) -> Band:
    """Band edges of a SIGPROC header (``fch1`` a channel centre, MHz).

    ``foff < 0`` (the usual SIGPROC order, highest channel first) gives
    ``flip=True``.

    Examples
    --------
    >>> band = sigproc_band(fch1=1500.0, foff=-1.0, nchans=4)
    >>> band.f_min, band.f_max, band.flip
    (1496.5, 1500.5, True)
    """
    if nchans <= 0:
        msg = f"nchans must be positive, got {nchans}"
        raise ValueError(msg)
    if foff == 0 or not np.isfinite(foff) or not np.isfinite(fch1):
        msg = f"fch1 and foff must be finite and foff nonzero, got {fch1}, {foff}"
        raise ValueError(msg)
    f_last = fch1 + (nchans - 1) * foff
    lo_centre = min(fch1, f_last)
    width = abs(foff)
    f_min = lo_centre - width / 2
    if f_min <= 0:
        msg = f"band reaches non-positive frequency (f_min={f_min} MHz)"
        raise ValueError(msg)
    return Band(
        f_min=f_min,
        f_max=f_min + nchans * width,
        nchans=nchans,
        flip=foff < 0,
    )


def to_ascending(waterfall: np.ndarray, band: Band, axis: int = 0) -> np.ndarray:
    """Return ``waterfall`` with its channel ``axis`` in ascending order.

    A view (no copy) when ``band.flip`` is False; otherwise a reversed view.
    Engines need C-contiguous input, so pass the result through
    :func:`numpy.ascontiguousarray` when it was flipped.
    """
    if waterfall.shape[axis] != band.nchans:
        msg = (
            f"waterfall has {waterfall.shape[axis]} channels on axis {axis}, "
            f"band has {band.nchans}"
        )
        raise ValueError(msg)
    return np.flip(waterfall, axis=axis) if band.flip else waterfall
