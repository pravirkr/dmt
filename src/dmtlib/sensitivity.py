"""S/N recovery of the incoherent dedispersion engines at the exact trial DM.

The question this module answers: given an ideal dispersed pulse at a DM
that is one of the search trials, how much of its S/N does each incoherent
engine (FDMT, FDMT-FFT, DDMT, SDMT, DDMT-FFT) recover when it integrates
exactly that trial? The choice of DM grid (trial spacing) is a separate
question and is not tested here.

The signal model is deliberately ideal, so that only the integration of the
dispersion curve is tested:

- white Gaussian noise of unit variance per raw sample;
- a flat spectrum: every channel carries the same fluence;
- an intrinsic boxcar profile of ``W`` samples (``W = 0`` is a delta
  function), the kernel every search pipeline matches it with;
- cold-plasma dispersion only (no scattering or scintillation), including
  the exact intra-channel smearing.

Every pulse sits exactly on a trial of the search grid (the piecewise-uniform
Levin grid of the configuration), and every engine is built with only that
trial. For FDMT this is
bit-identical to the same trial of the full grid (the plan computes exactly
the tree nodes each trial needs); DDMT, SDMT and DDMT-FFT compute each trial
independently anyway; FDMT-FFT's segmentation depends on the plan, which
changes its output by ~1 per cent of the noise.

The engines are linear, so the noise-free response divided by the analytic
noise of the trial gives the expected S/N exactly. Two references normalise
it:

``snr_injected``
    ``F * sqrt(nchans / max(W, 1))``: the S/N of the injected pulse without
    any dispersion, caught by a matched boxcar.
``snr_matched_filter``
    ``sqrt(sum(s**2))`` of the injected (smeared, sampled) waterfall: no
    linear detector working on detected intensities can exceed it, so it is
    the incoherent limit. ``snr_matched_filter / snr_injected`` is the loss
    to intra-channel smearing and sampling, which only coherent
    dedispersion recovers.

A noise realisation, the same for every engine, calibrates each engine's
detection threshold at a fixed false-alarm rate and gives the completeness
of the search.

With box smearing the FDMT averages each channel's level-0 box of ``L_c``
samples, with the box-mean weight ``round(2**b / L_c) * 2**-b``, ``b =
ceil(log2(L_c)) + 2`` (the plan's ``box_weights``): within 1/8 of
``1 / L_c``, the matched filter for intra-channel smearing. The FDMT without
box smearing (:data:`VARIANTS`) is measured as well.

:data:`CRYOPAF` is the 512-channel CryoPAF benchmark over 704-1216 MHz at
512 us, DM 0-2000, with Levin tolerance 1.1. :data:`CRYOPAF_LBAND` is the
1152-channel L-band streaming target. :data:`PARKES_UWL` covers 704-4032 MHz
with 6656 channels at 64 us and Levin tolerance 1.6. :data:`MEERKAT_LBAND`
is the 4096-channel MeerTRAP filterbank mode over 856-1712 MHz at 612.48 us
(306.24 us downsampled by two).
:data:`CONFIGS` holds these presets by name.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from . import libdmt
from .simulate import DM_CONSTANT

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

ENGINES: tuple[str, ...] = ("FDMT", "FDMT-FFT", "DDMT", "SDMT", "DDMT-FFT")
"""The five engines compared (FDMT-FFT with fractional delays)."""

VARIANTS: tuple[str, ...] = ("FDMT-nobox",)
"""FDMT variants: the FDMT without box smearing (``use_box_smearing=False``)."""

WIDTHS: tuple[float, ...] = (1.0, 8.0, 64.0)
"""Default intrinsic pulse widths in samples."""

_ALL = frozenset(ENGINES + VARIANTS)


# --------------------------------------------------------------------------
# Dispersion and the exact pulse model
# --------------------------------------------------------------------------
def _channel_edges(
    f_min: float, f_max: float, nchans: int
) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper edges of each channel, MHz (float64)."""
    df = (f_max - f_min) / nchans
    lo = f_min + df * np.arange(nchans, dtype=np.float64)
    return lo, lo + df


def band_delay(dm: float, f_min: float, f_max: float, tsamp: float) -> float:
    """Dispersion delay between ``f_max`` and ``f_min`` in samples."""
    return dm * DM_CONSTANT * (f_min**-2 - f_max**-2) / tsamp


def channel_sweep(
    f_min: float, f_max: float, nchans: int, tsamp: float, dm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel arrival offset and intra-channel sweep of a dispersed pulse.

    Parameters
    ----------
    f_min, f_max : float
        Band edges in MHz; channel ``c`` spans
        ``[f_min + c * df, f_min + (c + 1) * df]``.
    nchans : int
        Number of channels.
    tsamp : float
        Sampling interval in seconds.
    dm : float
        Dispersion measure (pc cm^-3, >= 0).

    Returns
    -------
    t_hi : numpy.ndarray
        Arrival time of each channel's upper edge after the arrival at
        ``f_max``, in samples.
    smear : numpy.ndarray
        Time the pulse takes to sweep through each channel, in samples.
    """
    if dm < 0:
        msg = f"dm must be >= 0, got {dm}"
        raise ValueError(msg)
    f_lo, f_hi = _channel_edges(f_min, f_max, nchans)
    scale = dm * DM_CONSTANT / tsamp
    t_hi = scale * (f_hi**-2 - f_max**-2)
    t_lo = scale * (f_lo**-2 - f_max**-2)
    return t_hi, t_lo - t_hi


def _sweep_primitive(
    s: np.ndarray, smear: np.ndarray, f_lo: np.ndarray, f_hi: np.ndarray
) -> np.ndarray:
    """Integral of the channel's arrival CDF from -inf to ``s``.

    ``s`` is time after the channel's upper-edge arrival. Inside the sweep the
    frequency arriving at ``s`` is ``u**-0.5`` with ``u`` linear in ``s``; the
    closed form avoids the cancellation of ``2 (sqrt(u) - 1/f_hi) / kappa``.
    """
    sc = np.clip(s, 0.0, smear)
    safe = np.where(smear > 0.0, smear, 1.0)
    u = f_hi**-2 + (sc / safe) * (f_lo**-2 - f_hi**-2)
    inside = sc / (f_hi - f_lo) * (f_hi - 2.0 / (np.sqrt(u) + 1.0 / f_hi))
    return inside + np.maximum(s - smear, 0.0)


def _sweep_cdf(
    s: np.ndarray, smear: np.ndarray, f_lo: np.ndarray, f_hi: np.ndarray
) -> np.ndarray:
    """Fraction of a delta pulse's channel energy that arrived strictly before s."""
    sc = np.clip(s, 0.0, smear)
    safe = np.where(smear > 0.0, smear, 1.0)
    u = f_hi**-2 + (sc / safe) * (f_lo**-2 - f_hi**-2)
    inside = (f_hi - u**-0.5) / (f_hi - f_lo)
    return np.where(s <= 0.0, 0.0, np.where(s >= smear, 1.0, inside))


def _profile_cdf(
    s: np.ndarray,
    smear: np.ndarray,
    f_lo: np.ndarray,
    f_hi: np.ndarray,
    width: float,
) -> np.ndarray:
    """CDF of the intrinsic boxcar of ``width`` samples convolved with the sweep."""
    if width <= 0.0:
        return _sweep_cdf(s, smear, f_lo, f_hi)
    return (
        _sweep_primitive(s, smear, f_lo, f_hi)
        - _sweep_primitive(s - width, smear, f_lo, f_hi)
    ) / width


def pulse_samples(
    f_min: float,
    f_max: float,
    nchans: int,
    tsamp: float,
    dm: float,
    t_arr: float,
    width: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact fluence fraction per sample of a dispersed pulse, per channel.

    Returns ``(values, first)``: ``values[c, k]`` is the fraction of channel
    ``c``'s fluence in sample ``first[c] + k`` (sample ``i`` integrates
    ``[i, i + 1)``). Each row sums to 1.
    """
    f_lo, f_hi = _channel_edges(f_min, f_max, nchans)
    t_hi, smear = channel_sweep(f_min, f_max, nchans, tsamp, dm)
    start = t_arr + t_hi
    first = np.floor(start).astype(np.int64)
    nk = math.ceil(float(np.max(smear, initial=0.0)) + width) + 2
    bounds = first[:, None] + np.arange(nk + 1, dtype=np.float64)[None, :]
    cdf = _profile_cdf(
        bounds - start[:, None], smear[:, None], f_lo[:, None], f_hi[:, None], width
    )
    return np.diff(cdf, axis=1), first


def inject_pulse(
    waterfall: np.ndarray,
    f_min: float,
    f_max: float,
    tsamp: float,
    dm: float,
    t_arr: float,
    width: float = 0.0,
    fluence: float = 1.0,
) -> float:
    """Add a noise-free dispersed pulse with exact intra-channel smearing.

    The pulse has a flat spectrum and an intrinsic boxcar profile of ``width``
    samples (``0``: a delta function). Its leading edge reaches ``f_max`` at
    ``t_arr`` (fractional samples) and frequency ``f`` at
    ``t_arr + DM_CONSTANT * dm * (f**-2 - f_max**-2) / tsamp``. Each channel's
    energy, uniform in frequency, is integrated exactly over every sample: the
    arrival-time CDF of a channel has a closed form, so no oversampling is
    involved.

    Parameters
    ----------
    waterfall : numpy.ndarray
        ``(nchans, nsamps)`` array the pulse is added to, in place.
    f_min, f_max : float
        Band edges in MHz.
    tsamp : float
        Sampling interval in seconds.
    dm : float
        Dispersion measure (pc cm^-3).
    t_arr : float
        Arrival time of the leading edge at ``f_max``, in samples.
    width : float, optional
        Intrinsic boxcar width in samples (default 0).
    fluence : float, optional
        Integrated signal per channel, in units of the per-sample noise sigma
        times one sample (default 1).

    Returns
    -------
    float
        ``sum(s**2)`` of the added pulse: the square of its matched-filter
        S/N for unit-variance noise.

    Raises
    ------
    ValueError
        If the pulse does not fit inside the waterfall.
    """
    nchans, nsamps = waterfall.shape
    values, first = pulse_samples(f_min, f_max, nchans, tsamp, dm, t_arr, width)
    nk = values.shape[1]
    if first.min() < 0 or first.max() + nk > nsamps:
        msg = (
            f"pulse spans samples [{first.min()}, {first.max() + nk}), which does "
            f"not fit in nsamps={nsamps}"
        )
        raise ValueError(msg)
    cols = first[:, None] + np.arange(nk)[None, :]
    rows = np.arange(nchans)[:, None]
    waterfall[rows, cols] += (fluence * values).astype(waterfall.dtype)
    return float(np.sum(np.square(fluence * values)))


# --------------------------------------------------------------------------
# Boxcar search and noise models
# --------------------------------------------------------------------------
def boxcar_widths(w_max: float) -> np.ndarray:
    """Search widths: every integer to 16, then a ``2**(1/4)`` ladder.

    The ladder keeps the width-mismatch loss of a boxcar below about 0.4 per
    cent; it stops at the first width >= ``w_max``.
    """
    widths = list(range(1, min(16, math.ceil(w_max)) + 1))
    k = 1
    while widths[-1] < w_max:
        w = round(16 * 2 ** (k / 4))
        if w > widths[-1]:
            widths.append(w)
        k += 1
    return np.asarray(widths, dtype=np.int64)


def _trapezoid_ssq(length: np.ndarray, width: float) -> np.ndarray:
    """Sum of squares of a box of ``length`` convolved with a boxcar."""
    h, lo = np.maximum(length, width), np.minimum(length, width)
    return (h - lo + 1.0) * lo * lo + (2.0 * lo - 1.0) * lo * (lo - 1.0) / 3.0


def box_smearing_variance(
    smearing: np.ndarray,
    widths: Sequence[int] | np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """FDMT output variance after a boxcar, per trial and width (float64).

    Each channel enters a trial through a box of ``L = |smearing[c]| + 1``
    samples (FDMT's level-0 box smearing), scaled by ``weights`` (default 1,
    the box sum). Convolved with a boxcar of ``w`` samples the box is a
    trapezoid whose sum of squares is ``(h - l + 1) l**2 + l (l - 1)(2 l - 1)
    / 3`` with ``h, l`` the longer and shorter box; the trial variance is the
    sum over channels of ``weight**2`` times that. With the plan's box-mean
    weights (:func:`plan_box_weights`) this is the float64 version of
    ``FDMTPlan.get_effective_variance_grid``.

    Parameters
    ----------
    smearing : numpy.ndarray
        ``(ndm, nchans)`` per-channel level-0 smearing (``smearing_grid_final``).
    widths : sequence of int
        Boxcar widths in samples.
    weights : numpy.ndarray, optional
        ``(ndm, nchans)`` weight of each channel's box.

    Returns
    -------
    numpy.ndarray
        Variance table of shape ``(ndm, len(widths))``.
    """
    length = np.abs(np.asarray(smearing, dtype=np.float64)) + 1.0
    wsq = np.ones_like(length) if weights is None else np.square(weights)
    table = np.empty((length.shape[0], len(widths)), dtype=np.float64)
    for j, w in enumerate(np.asarray(widths, dtype=np.float64)):
        table[:, j] = np.sum(wsq * _trapezoid_ssq(length, w), axis=1)
    return table


@dataclass(frozen=True)
class Peak:
    """Best boxcar response of one dedispersed series."""

    snr: float
    t: int
    width: int


def boxcar_snr(
    series: np.ndarray, sigma: np.ndarray, widths: Sequence[int] | np.ndarray
) -> np.ndarray:
    """S/N of every boxcar, ``(len(widths), len(series))``.

    Row ``j`` holds ``sum(series[t:t+w]) / sigma[j]`` for ``w = widths[j]``
    at every start ``t``; starts past the end are ``-inf``.
    """
    series = np.asarray(series, dtype=np.float64)
    csum = np.concatenate(([0.0], np.cumsum(series)))
    out = np.full((len(widths), series.size), -np.inf)
    for j, w in enumerate(widths):
        w = int(w)  # noqa: PLW2901
        if w <= series.size:
            out[j, : series.size - w + 1] = (csum[w:] - csum[:-w]) / sigma[j]
    return out


def boxcar_peak(
    series: np.ndarray, sigma: np.ndarray, widths: Sequence[int] | np.ndarray
) -> Peak:
    """Best ``sum(series[t:t+w]) / sigma[j]`` over start ``t`` and width."""
    snr = boxcar_snr(series, sigma, widths)
    j, t = np.unravel_index(int(np.argmax(snr)), snr.shape)
    return Peak(float(snr[j, t]), int(t), int(widths[j]))


# --------------------------------------------------------------------------
# Configuration and engines
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SearchConfig:
    """One search configuration shared by every engine.

    The trial grid is the piecewise-uniform Levin grid
    (:meth:`DDMTPlan.generate_levin_dm_grid_piecewise`) from ``dm_min`` to
    ``dm_max``, the same grid for every engine (DDMT-FFT runs one NUFFT per
    uniform segment).

    Parameters
    ----------
    f_min, f_max : float
        Band edges in MHz.
    nchans : int
        Number of channels.
    tsamp : float
        Sampling interval in seconds.
    dm_min, dm_max : float, optional
        DM range in pc cm^-3 (default 0 to 2000).
    levin_tol : float, optional
        Levin S/N-loss tolerance (default 1.1).
    levin_pulse_width : float, optional
        Levin pulse width in seconds (default 0: ``tsamp``).
    nthreads : int, optional
        Threads per engine (default 1).
    name : str, optional
        Label stored with the results.
    nsamps_ref : int, optional
        Reference streaming block length (default 32768). Sensitivity
        measurements may choose their own injection window length.
    subband_depth : int, optional
        Subband search depth of the FDMT plan (default 0: full band only);
        depth ``k`` searches ``2**k`` equal subbands and the ``2**k - 1``
        staggered ones between them.
    """

    f_min: float
    f_max: float
    nchans: int
    tsamp: float
    dm_min: float = 0.0
    dm_max: float = 2000.0
    levin_tol: float = 1.1
    levin_pulse_width: float = 0.0
    nthreads: int = 1
    name: str = "custom"
    nsamps_ref: int = 32768
    subband_depth: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.dm_min < self.dm_max:
            msg = f"need 0 <= dm_min < dm_max, got {self.dm_min}, {self.dm_max}"
            raise ValueError(msg)

    @cached_property
    def dm_trials(self) -> np.ndarray:
        """The trial DMs of the search (float32, ascending)."""
        dms = libdmt.DDMTPlan.generate_levin_dm_grid_piecewise(
            self.dm_min,
            self.dm_max,
            self.tsamp,
            self.levin_pulse_width or self.tsamp,
            self.f_min,
            self.f_max,
            self.nchans,
            self.levin_tol,
        )
        return np.asarray(dms, dtype=np.float32)

    @cached_property
    def dm_per_dt(self) -> float:
        """DM per sample of delay across the band, from the library (FDMT)."""
        plan = libdmt.FDMTPlan(self.f_min, self.f_max, 2, 8, self.tsamp, 1)
        return float(np.asarray(plan.dm_grid_final, dtype=np.float64)[1])

    def band_delay(self, dm: float) -> float:
        """Delay between ``f_max`` and ``f_min`` at ``dm``, in samples."""
        return band_delay(dm, self.f_min, self.f_max, self.tsamp)

    def bottom_smear(self, dm: float) -> float:
        """Sweep time through the lowest channel at ``dm``, in samples."""
        sweep = channel_sweep(self.f_min, self.f_max, self.nchans, self.tsamp, dm)
        return float(sweep[1][0])

    def as_dict(self) -> dict[str, Any]:
        """Return the configuration as a JSON-serialisable dict."""
        return asdict(self)


CRYOPAF = SearchConfig(
    704.0, 1216.0, 512, 512e-6, dm_min=0.0, dm_max=2000.0, levin_tol=1.1,
    name="cryopaf",
)  # fmt: skip
"""CryoPAF benchmark: 512 channels over 704-1216 MHz at 512 us, DM 0-2000,
Levin tolerance 1.1.
Use ``dataclasses.replace(CRYOPAF, nthreads=8)`` to change the thread count."""

CRYOPAF_LBAND = SearchConfig(
    1100.0, 1714.0, 1152, 455.6e-6, dm_min=0.0, dm_max=2000.0, levin_tol=1.1,
    name="cryopaf_lband", nsamps_ref=32768, subband_depth=1,
)  # fmt: skip
"""L-band real-time search target: 14.9291-second blocks, searching the full
band, its two 576-channel halves and the staggered central half. The DM-2000
band sweep is 4.03365 seconds (8853.49 samples); streaming valid mode also
permits shorter blocks."""

PARKES_UWL = SearchConfig(
    704.0, 4032.0, 6656, 64e-6, dm_min=0.0, dm_max=2000.0, levin_tol=1.6,
    name="parkes_uwl", nsamps_ref=32768,
)  # fmt: skip
"""Ultra-wideband benchmark with 2.097152-second streaming blocks. Levin
tolerance 1.6 keeps the grid under 5000 trials (4749; tolerance 1.1 would
need 12941). At DM 2000 the dispersion sweep spans several blocks; histories
must be warmed before steady-state timing. Full-band search is the default."""

MEERKAT_LBAND = SearchConfig(
    856.0, 1712.0, 4096, 612.48e-6, dm_min=0.0, dm_max=2000.0, levin_tol=1.1,
    name="meerkat_lband", nsamps_ref=32768,
)  # fmt: skip
"""MeerKAT L-band benchmark: the 4096-channel MeerTRAP / TRAPUM filterbank
mode over 856-1712 MHz, downsampled by two to 612.48 us, DM 0-2000, Levin
tolerance 1.1, in 20.0697446-second blocks. The grid has 4057 trials in two
uniform segments (3.28 delay samples per trial to DM 1836, 6.56 beyond), and
the 12-level tree needs 101x fewer operations than brute force. The DM-2000
band sweep is 8.49428 seconds (13868.67 samples)."""

CONFIGS = {cfg.name: cfg for cfg in (CRYOPAF, CRYOPAF_LBAND, PARKES_UWL, MEERKAT_LBAND)}
"""The named configurations, by ``SearchConfig.name``."""


def box_weights(length: np.ndarray) -> np.ndarray:
    """Box-mean weight of each level-0 box of length ``length``.

    The FDMT's dyadic weight ``round(2**b / L) * 2**-b`` with ``b =
    ceil(log2(L)) + 2`` (within 1/8 of ``1 / L``), the same as
    :func:`plan_box_weights`.
    """
    length = np.asarray(length, dtype=np.float64)
    b = np.ceil(np.log2(length)) + 2
    # (2**b + L // 2) // L: the library's integer round-half-up.
    n = np.floor((2.0**b + np.floor(length / 2)) / length)
    return n * 2.0**-b


def plan_box_weights(plan: Any, length: np.ndarray) -> np.ndarray:  # noqa: ANN401
    """Return the FDMT's own box-mean weights for boxes of ``length`` samples.

    Exactly the factors ``plan.box_weights[L - 1] * 2**-plan.box_fraction_bits``
    the engines apply (``use_box_smearing=True``); equal to
    ``box_weights(length)``.
    """
    table = np.asarray(plan.box_weights, dtype=np.float64)
    idx = np.asarray(length, dtype=np.int64) - 1
    return table[idx] * 2.0 ** -int(plan.box_fraction_bits)


def fdmt_rows(
    plan: Any,  # noqa: ANN401
    waterfall: np.ndarray,
    *,
    row: int = 0,
    chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """One FDMT trial computed directly from its plan.

    Trial ``row`` of an FDMT plan averages, for output time ``T``, the samples
    ``[T + s_c, T + s_c + L_c)`` of channel ``c`` (``s_c`` from
    ``plan.trace_dm(row)``, ``L_c`` its level-0 box length) with the plan's
    box-mean weights (:func:`plan_box_weights`) and sums the channels: the
    FDMT's output, computed without the tree (a test of the engine).

    Parameters
    ----------
    plan : FDMTPlan
        Plan of an FDMT with box smearing and non-negative delays.
    waterfall : numpy.ndarray
        ``(nchans, nsamps)`` input.
    row : int, optional
        Trial index into the plan's final grid.
    chunk : int, optional
        Output samples computed at a time (memory).

    Returns
    -------
    series : numpy.ndarray
        The trial's output (float64) at every ``T`` whose window lies inside
        the input; sample 0 is ``T = max(0, -min(s_c))``.
    channel_weights : numpy.ndarray
        Weight of each channel's box.
    """
    nchans, nsamps = waterfall.shape
    shifts = np.asarray(plan.trace_dm(row)).astype(np.int64)
    smearing = np.asarray(plan.smearing_grid_final).reshape(-1, nchans)[row]
    length = (np.abs(smearing) + 1).astype(np.int64)
    wts = plan_box_weights(plan, length)
    t0 = max(0, -int(shifts.min()))
    t_end = nsamps - int((shifts + length).max()) + 1
    csum = np.zeros((nchans, nsamps + 1))
    np.cumsum(waterfall, axis=1, dtype=np.float64, out=csum[:, 1:])
    out = np.zeros(max(t_end - t0, 0))
    chans = np.arange(nchans)[None, :]
    for a in range(t0, t_end, chunk):
        times = np.arange(a, min(a + chunk, t_end))[:, None]
        start = times + shifts[None, :]
        boxes = csum[chans, start + length[None, :]] - csum[chans, start]
        out[a - t0 : a - t0 + times.size] = boxes @ wts
    return out, wts


def fdmt_alignment_error(plan: Any, row: int) -> np.ndarray:  # noqa: ANN401
    """Per-channel timing error of one FDMT trial, in samples.

    The centre of channel ``c``'s window ``[T + s_c, T + s_c + L_c)`` minus
    the centroid of a pulse's sweep through the channel at the trial's own
    delay, after removing the median over channels (a common offset only
    moves the output time). Rounding every channel delay once would give
    errors uniform in ``[-0.5, 0.5]`` (standard deviation ``1/sqrt(12)``).

    Parameters
    ----------
    plan : FDMTPlan
        The plan of an FDMT engine (``fdmt.plan``).
    row : int
        Index into the plan's final trial grid (non-negative delay).

    Returns
    -------
    numpy.ndarray
        Error of every channel, in samples.
    """
    nchans = int(plan.nchans)
    f_min, f_max = float(plan.f_min), float(plan.f_max)
    delay = float(np.asarray(plan.dt_grid_final)[row])
    shifts = np.asarray(plan.trace_dm(row), dtype=np.float64)
    smearing = np.asarray(plan.smearing_grid_final).reshape(-1, nchans)[row]
    f_lo, f_hi = _channel_edges(f_min, f_max, nchans)
    scale = delay / (f_min**-2 - f_max**-2)
    t_hi = scale * (f_hi**-2 - f_max**-2)
    sweep = scale * (f_lo**-2 - f_hi**-2)
    centre = t_hi + sweep * f_lo / (f_lo + f_hi)
    error = shifts + (np.abs(smearing) + 1.0) / 2 - centre
    return error - np.median(error)


class TrialRunner:
    """One engine computing a single trial DM.

    Attributes
    ----------
    name : str
        Engine label (:data:`ENGINES` or :data:`VARIANTS`).
    dm : float
        The DM the engine integrates (FDMT and FDMT-FFT: their nearest integer
        delay, converted back by the library).
    nchans, nsamps : int
        Input waterfall shape.
    """

    def __init__(self, name: str, cfg: SearchConfig, dm: float, nsamps: int) -> None:
        if name not in _ALL:
            msg = f"unknown engine {name!r}; choose from {sorted(_ALL)}"
            raise ValueError(msg)
        self.name = name
        self.nchans = cfg.nchans
        self.nsamps = nsamps
        band = (cfg.f_min, cfg.f_max, cfg.nchans)
        # The trial and its neighbour on the grid: FDMT needs a non-zero delay,
        # and every trial of an FDMT plan is computed independently of the
        # others, so the extra trial changes nothing.
        trials = cfg.dm_trials.astype(np.float64)
        above = trials[trials > dm + 0.75 * cfg.dm_per_dt]
        below = trials[trials < dm - 0.75 * cfg.dm_per_dt]
        other = float(above[0]) if above.size else float(below[-1])
        pair = sorted([float(dm), other])
        self.row = pair.index(float(dm))
        grid = np.asarray(pair, dtype=np.float32)
        kw = {"nthreads": cfg.nthreads}
        self.engine: Any = None
        if name in ("FDMT", "FDMT-nobox"):
            box = name == "FDMT"
            self.engine = libdmt.FDMT(
                *band, nsamps, cfg.tsamp, dm_grid=grid, use_box_smearing=box, **kw
            )
            self.plan = self.engine.plan
        elif name == "FDMT-FFT":
            self.engine = libdmt.FDMTFFT(*band, nsamps, cfg.tsamp, dm_grid=grid, **kw)
            self.plan = self.engine.plan
        else:
            cls = {"DDMT": libdmt.DDMT, "SDMT": libdmt.SDMT, "DDMT-FFT": libdmt.DDMTFFT}
            self.engine = cls[name](*band, cfg.tsamp, dm_arr=grid, **kw)
            self.plan = self.engine.plan
        if hasattr(self.plan, "smearing_grid_final"):
            dms = np.asarray(self.plan.dm_grid_final)
            if len(dms) != len(pair):
                msg = f"FDMT merged the trials {pair} into one delay"
                raise ValueError(msg)
            smearing = np.asarray(self.plan.smearing_grid_final)
            self._smearing = smearing.reshape(len(pair), -1)[self.row]
            length = np.abs(self._smearing) + 1.0
            self._channel_weights = plan_box_weights(self.plan, length)
        else:
            dms = np.asarray(self.plan.dm_arr)
        self.dm = float(dms[self.row])

    def run(self, waterfall: np.ndarray) -> np.ndarray:
        """Dedisperse one block (history reset first); a float64 series."""
        self.engine.reset_history()
        out = np.asarray(self.engine.execute(waterfall))
        return out[self.row].astype(np.float64)

    def sigma(self, widths: Sequence[int] | np.ndarray) -> np.ndarray:
        """Analytic noise sigma of the trial after each boxcar width."""
        w = np.asarray(widths, dtype=np.float64)
        if self.name in ("FDMT", "FDMT-FFT"):
            smearing = self._smearing[None, :]
            weights = self._channel_weights[None, :]
            return np.sqrt(box_smearing_variance(smearing, w, weights))[0]
        if self.name == "FDMT-nobox":
            return np.sqrt(self.nchans * w)
        return float(self.plan.effective_sigma) * np.sqrt(w)


# --------------------------------------------------------------------------
# The experiment
# --------------------------------------------------------------------------
def select_trials(cfg: SearchConfig, n_dm: int, seed: int = 0) -> np.ndarray:
    """``n_dm`` DMs spread evenly over the grid, at integer band delays.

    One uniformly random DM is drawn in each of ``n_dm`` equal DM strata and
    moved to the nearest trial of the grid, then to the nearest integer
    delay across the band, so that every engine (FDMT included, whose trials
    are integer delays) integrates exactly the injected DM. The lowest trial
    is always included.
    """
    rng = np.random.default_rng(seed)
    grid = cfg.dm_trials.astype(np.float64)
    edges = np.linspace(cfg.dm_min, cfg.dm_max, n_dm + 1)
    draws = rng.uniform(edges[:-1], edges[1:])
    idx = np.abs(grid[None, :] - draws[:, None]).argmin(axis=1)
    dms = np.concatenate(([grid[0]], grid[idx]))
    return np.unique(np.rint(dms / cfg.dm_per_dt)) * cfg.dm_per_dt


@dataclass
class Layout:
    """Where the pulses of one DM sit in the stream (all at the same DM)."""

    dm: float
    widths: np.ndarray  # per pulse
    phases: np.ndarray  # per pulse
    t_arr: np.ndarray  # leading-edge arrival at f_max, per pulse
    spacing: int
    nsamps: int


def make_layout(
    cfg: SearchConfig,
    dm: float,
    widths: Sequence[float],
    n_phase: int,
    rng: np.random.Generator,
) -> Layout:
    """Place ``len(widths) * n_phase`` pulses at one DM, ``spacing`` apart.

    The spacing keeps every dedispersed pulse, its smearing and the engines'
    guards inside its own window, so the pulses do not overlap in any
    engine's output at that DM.
    """
    w_max = float(max(widths))
    spacing = 2 * math.ceil(w_max + cfg.bottom_smear(dm) + 160) + 256
    pulse_w = np.repeat(np.asarray(widths, dtype=np.float64), n_phase)
    phases = np.tile(np.arange(n_phase), len(widths)) + rng.uniform(size=pulse_w.size)
    phases /= n_phase
    t_arr = spacing * (1 + np.arange(pulse_w.size)) + phases
    nsamps = int(t_arr[-1]) + math.ceil(cfg.band_delay(dm)) + 2 * spacing
    return Layout(dm, pulse_w, phases, t_arr, spacing, 1024 * math.ceil(nsamps / 1024))


def pulse_windows(series: np.ndarray, layout: Layout) -> np.ndarray:
    """Start of each pulse's output window (length ``layout.spacing``).

    Every engine maps arrival time to output time with its own constant
    offset (reference frequency, latency, cold-start trimming). The offset is
    the shift that maximises the noise-free output summed over the pulse
    positions, smoothed over a quarter of the spacing.
    """
    spacing = layout.spacing
    k = max(spacing // 4, 1)
    smooth = np.convolve(np.abs(series), np.ones(k) / k, mode="same")
    base = np.floor(layout.t_arr).astype(np.int64)
    offsets = np.arange(-spacing, len(series) - int(base[-1]))
    total = np.zeros(offsets.size)
    for b in base:
        idx = b + offsets
        ok = (idx >= 0) & (idx < len(smooth))
        total[ok] += smooth[idx[ok]]
        total[~ok] = -np.inf
    offset = int(offsets[int(np.argmax(total))])
    return base + offset - spacing // 2


@dataclass
class RecoveryResult:
    """Per-pulse S/N recovery of every engine, and its detection statistics.

    Per-pulse arrays: ``dm``, ``width``, ``phase``, ``snr_injected``,
    ``snr_matched_filter``. Per engine (``engines[name]``), per pulse:
    ``snr`` (expected S/N at the exact trial), ``t`` and ``best_width`` of
    its best boxcar, ``snr_noisy`` (best S/N in the pulse's window with the
    noise added at unit fluence); and per DM (``dms``): ``dm_engine`` (the DM
    the engine integrated), ``threshold`` (S/N for ``far_per_hour`` false
    candidates per hour of the whole search, from the fitted up-crossing
    rate; ``tail_beta`` is the fit's exponent), ``n_eff`` and
    ``threshold_median`` (the effective independent tests per window fitted
    at the median of the window maxima, and the threshold they imply:
    optimistic for correlated output) and ``sigma_ratio`` (measured over
    analytic noise).
    ``snr_detect`` (per pulse) is the injected S/N above which the pulse,
    scaled, plus the noise, exceeds the threshold; ``snr_grid`` is the
    default axis of :meth:`completeness`.
    """

    config: dict[str, Any]
    dms: np.ndarray
    dm: np.ndarray
    width: np.ndarray
    phase: np.ndarray
    snr_injected: np.ndarray
    snr_matched_filter: np.ndarray
    snr_grid: np.ndarray
    engines: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    _ARRAYS = (
        "dms",
        "dm",
        "width",
        "phase",
        "snr_injected",
        "snr_matched_filter",
        "snr_grid",
    )

    def eta(self, name: str) -> np.ndarray:
        """Recovered over injected S/N."""
        return self.engines[name]["snr"] / self.snr_injected

    def eta_mf(self, name: str) -> np.ndarray:
        """Recovered over matched-filter S/N (the incoherent limit)."""
        return self.engines[name]["snr"] / self.snr_matched_filter

    def completeness(self, name: str, select: np.ndarray) -> np.ndarray:
        """Return the detected fraction of the selected pulses at ``snr_grid``.

        A pulse injected at S/N ``x`` is detected when ``x`` reaches its
        ``snr_detect``.
        """
        detect = self.engines[name]["snr_detect"][select]
        return (self.snr_grid[:, None] >= detect[None, :]).mean(axis=1)

    def save(self, path: str | Path) -> Path:
        """Write ``<path>.npz`` (arrays) and ``<path>.json`` (config, metadata)."""
        path = Path(path).with_suffix("")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: getattr(self, k) for k in self._ARRAYS}
        for name, cols in self.engines.items():
            for key, arr in cols.items():
                arrays[f"engine:{name}:{key}"] = arr
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        info = {"config": self.config, "engines": list(self.engines), **self.meta}
        path.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
        return path.with_suffix(".npz")

    @classmethod
    def load(cls, path: str | Path) -> RecoveryResult:
        """Read a result written by :meth:`save`."""
        path = Path(path).with_suffix("")
        info = json.loads(path.with_suffix(".json").read_text())
        with np.load(path.with_suffix(".npz")) as data:
            arrays = {k: data[k] for k in cls._ARRAYS}
            engines: dict[str, dict[str, np.ndarray]] = {}
            for key in data.files:
                if key.startswith("engine:"):
                    _, name, col = key.split(":", 2)
                    engines.setdefault(name, {})[col] = data[key]
        config = info.pop("config")
        info.pop("engines", None)
        return cls(config=config, engines=engines, meta=info, **arrays)


def effective_tests(maxima: np.ndarray) -> float:
    """Effective number of independent unit-Gaussian tests behind maxima.

    Fits ``P(max <= x) = Phi(x)**n`` at the median of the observed maxima:
    ``n = ln(1/2) / ln(Phi(median))``.
    """
    med = float(np.median(maxima))
    phi = 0.5 * math.erfc(-med / math.sqrt(2.0))
    return math.log(0.5) / math.log(phi)


TAIL_LEVELS: tuple[float, ...] = tuple(np.arange(2.0, 4.51, 0.25))
"""Levels (in sigma) at which :func:`tail_fit` counts up-crossings."""


def upcrossings(
    series: Iterable[np.ndarray], levels: Sequence[float]
) -> tuple[np.ndarray, int]:
    """Up-crossings of each level by the series, and the samples counted.

    An up-crossing of ``u`` is a sample above ``u`` whose predecessor is not:
    the start of one excursion, i.e. one false candidate of a search that
    merges neighbouring samples above threshold. Non-finite samples (boxcars
    past a series' end) never cross.
    """
    levels = np.asarray(levels, dtype=np.float64)
    counts = np.zeros(levels.size)
    total = 0
    for row in series:
        x = np.asarray(row, dtype=np.float64)
        prev, cur = x[:-1, None], x[1:, None]
        counts += np.count_nonzero((prev <= levels) & (cur > levels), axis=0)
        total += max(x.size - 1, 0)
    return counts, total


def tail_fit(
    levels: Sequence[float], counts: np.ndarray, nsamples: int, min_count: float = 5.0
) -> tuple[float, float]:
    """Fit ``rate(u) = exp(c) * u**beta * phi(u)`` to up-crossing counts.

    The form is Rice's: a smooth stationary Gaussian process crosses ``u`` at
    a rate proportional to ``exp(-u**2 / 2)`` (``beta = 0``); independent
    samples cross at about ``Q(u) ~ phi(u) / u`` (``beta = -1``), and a
    correlated discrete sequence lies in between. The best boxcar over a
    ladder of widths is the maximum of a field in time and log-width, whose
    excursions grow as ``u * phi(u)`` (``beta = +1``, Siegmund & Worsley
    1995); long simulations of the search statistic give ``beta`` = 1.05
    for white noise and 0.66 for the FDMT's at DM 2000, and the fit over
    2-4.5 sigma extrapolates their thresholds to within 0.04 sigma.
    Weighted least squares on ``log`` counts (weights: the counts, i.e. Poisson); levels
    with fewer than ``min_count`` crossings are left out. Returns ``(c, beta)`` with
    the rate per sample; ``phi`` includes its ``1 / sqrt(2 pi)``.
    """
    levels = np.asarray(levels, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    ok = counts >= min_count
    if np.count_nonzero(ok) < 2:
        msg = "tail_fit: fewer than two levels with enough up-crossings"
        raise ValueError(msg)
    u = levels[ok]
    y = np.log(counts[ok] / nsamples) + 0.5 * u * u + 0.5 * math.log(2 * math.pi)
    a = np.vstack([np.ones(u.size), np.log(u)]).T
    w = np.sqrt(counts[ok])
    c, beta = np.linalg.lstsq(a * w[:, None], y * w, rcond=None)[0]
    return float(c), float(beta)


def tail_threshold(c: float, beta: float, rate: float) -> float:
    """Level ``u`` (> 1) at which :func:`tail_fit`'s rate equals ``rate``."""
    target = math.log(rate) + 0.5 * math.log(2 * math.pi)
    lo, hi = 1.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if c + beta * math.log(mid) - 0.5 * mid * mid > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gaussian_isf(p: float) -> float:
    """Inverse survival function of the unit Gaussian: ``x`` with tail ``p``."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * math.erfc(mid / math.sqrt(2.0)) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class _Accumulator:
    """Per-engine results while :func:`measure_recovery` runs."""

    pulse: dict[str, list[Any]] = field(
        default_factory=lambda: {
            k: [] for k in ("snr", "t", "best_width", "snr_noisy", "snr_detect")
        }
    )
    per_dm: dict[str, list[float]] = field(
        default_factory=lambda: {
            k: []
            for k in (
                "dm_engine",
                "n_eff",
                "threshold",
                "threshold_median",
                "tail_beta",
                "sigma_ratio",
            )
        }
    )


def _measure_engine(
    acc: _Accumulator,
    runner: TrialRunner,
    signal: np.ndarray,
    noise: np.ndarray,
    layout: Layout,
    widths: np.ndarray,
    snr_inj: np.ndarray,
    tests_per_hour: float,
    far_per_hour: float,
    crossing_rate: float,
) -> None:
    """Record the expected, noisy and detection S/N of the pulses of one DM.

    The threshold comes from the up-crossings of the search statistic (the
    best boxcar S/N at each sample) in the noise run, fitted with
    :func:`tail_fit` and extrapolated to ``crossing_rate`` false candidates
    per sample and trial. The median-based :func:`effective_tests` estimate
    is kept as ``threshold_median``: it assumes independent tests, which is
    optimistic for correlated output.
    """
    clean = runner.run(signal)
    noisy = runner.run(noise)
    sigma = runner.sigma(widths)
    starts = pulse_windows(clean, layout)
    span = layout.spacing
    fields = []
    for a in starts:
        sig = boxcar_snr(clean[a : a + span], sigma, widths)
        nse = boxcar_snr(noisy[a : a + span], sigma, widths)
        fields.append((sig, nse))
    n_eff = effective_tests(np.asarray([nse.max() for _, nse in fields]))
    threshold_median = gaussian_isf(far_per_hour / (tests_per_hour * n_eff))
    # The search statistic of each window: best boxcar at every start.
    best = [np.max(nse, axis=0) for _, nse in fields]
    counts, nsamples = upcrossings(best, TAIL_LEVELS)
    c, beta = tail_fit(TAIL_LEVELS, counts, nsamples)
    threshold = tail_threshold(c, beta, crossing_rate)
    # Past every engine's cold start: the pulse windows themselves.
    body = np.concatenate([noisy[a : a + span] for a in starts])
    acc.per_dm["dm_engine"].append(runner.dm)
    acc.per_dm["n_eff"].append(n_eff)
    acc.per_dm["threshold"].append(threshold)
    acc.per_dm["threshold_median"].append(threshold_median)
    acc.per_dm["tail_beta"].append(beta)
    acc.per_dm["sigma_ratio"].append(float(body.std() / sigma[0]))
    for k, (sig, nse) in enumerate(fields):
        j, t = np.unravel_index(int(np.argmax(sig)), sig.shape)
        acc.pulse["snr"].append(float(sig[j, t]))
        acc.pulse["t"].append(int(starts[k] + t))
        acc.pulse["best_width"].append(int(widths[j]))
        acc.pulse["snr_noisy"].append(float(np.max(sig + nse)))
        ok = np.isfinite(sig)
        fluence = detection_fluence(sig[ok], nse[ok], threshold)
        acc.pulse["snr_detect"].append(fluence * float(snr_inj[k]))


def detection_fluence(signal: np.ndarray, noise: np.ndarray, threshold: float) -> float:
    """Smallest fluence ``f`` with ``max(f * signal + noise) > threshold``.

    ``signal`` and ``noise`` are the S/N of the same boxcars for the pulse at
    unit fluence and for the noise alone; the maximum is convex and
    non-decreasing in ``f``, so bisection finds the crossing. Returns 0 if
    the noise alone exceeds the threshold.
    """
    if np.max(noise) > threshold:
        return 0.0
    top = float(np.max(signal))
    if top <= 0.0:
        return math.inf
    lo, hi = 0.0, (threshold - float(np.min(noise))) / top
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if np.max(mid * signal + noise) > threshold:
            hi = mid
        else:
            lo = mid
    return hi


def measure_recovery(
    cfg: SearchConfig,
    dms: Sequence[float],
    widths: Sequence[float] = WIDTHS,
    engines: Iterable[str] = ENGINES + VARIANTS,
    *,
    n_phase: int = 8,
    seed: int = 0,
    far_per_hour: float = 1.0,
    snr_grid: Sequence[float] | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> RecoveryResult:
    """Inject pulses at trial DMs and record every engine's S/N at that trial.

    For every DM of ``dms`` (trials of ``cfg.dm_trials``), ``len(widths) *
    n_phase`` pulses with random sub-sample phases are placed in one stream,
    and every engine, built with that single trial, dedisperses it twice:
    noise-free (the expected S/N, by linearity) and unit-variance Gaussian
    noise alone (the same noise for every engine). The noise run gives each
    engine's S/N threshold for ``far_per_hour`` false candidates per hour of
    a search over all ``len(cfg.dm_trials)`` trials (trials counted as
    independent): the up-crossing rate of its search statistic, fitted
    (:func:`tail_fit`) and extrapolated to the target rate. It also gives
    for every pulse the injected S/N above which ``F * signal + noise``
    exceeds that threshold (``snr_detect``, :func:`detection_fluence`).
    ``snr_grid`` is the default axis of :meth:`RecoveryResult.completeness`.

    Returns
    -------
    RecoveryResult
        Per-pulse references and per-engine results.
    """
    engines = list(engines)
    grid = np.geomspace(3, 1000, 97) if snr_grid is None else snr_grid
    snr_grid = np.asarray(grid, dtype=np.float64)
    rng = np.random.default_rng(seed)
    w_search = boxcar_widths(2.0 * (max(widths) + cfg.bottom_smear(max(dms))) + 4.0)
    rows: dict[str, list[float]] = {k: [] for k in ("dm", "width", "phase", "mf")}
    inj_all: list[np.ndarray] = []
    accs = {name: _Accumulator() for name in engines}
    for i, dm in enumerate(dms):
        layout = make_layout(cfg, float(dm), widths, n_phase, rng)
        signal = np.zeros((cfg.nchans, layout.nsamps), dtype=np.float32)
        for t, w in zip(layout.t_arr, layout.widths, strict=True):
            ssq = inject_pulse(signal, cfg.f_min, cfg.f_max, cfg.tsamp, dm, t, w)
            rows["mf"].append(math.sqrt(ssq))
        noise = rng.standard_normal(signal.shape, dtype=np.float32)
        snr_inj = np.sqrt(cfg.nchans / np.maximum(layout.widths, 1.0))
        inj_all.append(snr_inj)
        rows["dm"].extend([float(dm)] * layout.widths.size)
        rows["width"].extend(layout.widths.tolist())
        rows["phase"].extend(layout.phases.tolist())
        tests_per_hour = len(cfg.dm_trials) * 3600.0 / (layout.spacing * cfg.tsamp)
        # False candidates per output sample and trial at the target rate.
        crossing_rate = far_per_hour * cfg.tsamp / (3600.0 * len(cfg.dm_trials))
        for name in engines:
            runner = TrialRunner(name, cfg, float(dm), layout.nsamps)
            _measure_engine(
                accs[name], runner, signal, noise, layout, w_search,
                snr_inj, tests_per_hour, far_per_hour, crossing_rate,
            )  # fmt: skip
        if progress is not None:
            progress(i + 1, len(dms), float(dm))

    result = RecoveryResult(
        config=cfg.as_dict() | {"n_trials": len(cfg.dm_trials)},
        dms=np.asarray(dms, dtype=np.float64),
        dm=np.asarray(rows["dm"]),
        width=np.asarray(rows["width"]),
        phase=np.asarray(rows["phase"]),
        snr_injected=np.concatenate(inj_all),
        snr_matched_filter=np.asarray(rows["mf"]),
        snr_grid=snr_grid,
        meta={
            "widths": [float(w) for w in widths],
            "n_phase": n_phase,
            "seed": seed,
            "far_per_hour": far_per_hour,
            "search_widths": w_search.tolist(),
        },
    )
    for name, acc in accs.items():
        cols = {k: np.asarray(v) for k, v in acc.pulse.items()}
        cols |= {k: np.asarray(v) for k, v in acc.per_dm.items()}
        result.engines[name] = cols
    return result
