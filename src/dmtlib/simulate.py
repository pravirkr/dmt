"""Astrophysical signal simulation utilities for radio dispersion testing.

Provides functions to generate synthetic fast radio bursts (FRBs) and periodic
dispersed pulsar signals with accurate channel dispersion delays and intra-channel
smearing across arbitrary frequency bands.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

# Dispersion constant (MHz^2 * s / (pc cm^-3)) - Lorimer & Kramer (2005)
DM_CONSTANT = 4.148808e3


def get_dmdelays(
    dm: float,
    f_min: float,
    f_max: float,
    tsamp: float,
    nchans: int,
    *,
    in_samples: bool = True,
) -> np.ndarray:
    """Calculate dispersive time delays across frequency channels relative to f_min.

    Calculates the cold-plasma group delay for each channel frequency :math:`\\nu_i`:

    .. math::
        \\Delta t_i = k_{\\text{DM}} \\cdot \\text{DM} \\cdot \\left( f_{\\text{min}}^{-2} - \\nu_i^{-2} \\right)

    Parameters
    ----------
    dm : float
        Dispersion measure in :math:`\\text{pc} \\, \\text{cm}^{-3}`.
    f_min : float
        Bottom edge frequency of the band in MHz.
    f_max : float
        Top edge frequency of the band in MHz.
    tsamp : float
        Sampling interval in seconds.
    nchans : int
        Number of frequency channels.
    in_samples : bool, default=True
        If True, returns integer delays rounded to the nearest sample bin.
        If False, returns continuous floating-point delays in seconds.

    Returns
    -------
    np.ndarray
        Array of length `nchans` containing relative delays per channel.
    """
    foff = (f_max - f_min) / nchans
    chan_freqs = np.arange(nchans, dtype=np.float64) * foff + f_min
    delays = dm * DM_CONSTANT * (f_min**-2 - chan_freqs**-2)
    if in_samples:
        return (delays / tsamp).round().astype(np.int32)
    return delays


def generate_frb(
    f_min: float,
    f_max: float,
    nchans: int,
    nsamps: int,
    tsamp: float,
    dm: float,
    amp: float = 1.0,
    offset: int = 0,
    width: int = 1,
    noise_rms: float = 0.0,
) -> np.ndarray:
    """Generate a dispersed Fast Radio Burst (FRB) pulse in a frequency-time waterfall.

    Injects a top-hat pulse of specified width and amplitude with quadratic cold-plasma
    delay trajectory, embedded in optional additive white Gaussian noise.

    Parameters
    ----------
    f_min : float
        Bottom edge frequency in MHz.
    f_max : float
        Top edge frequency in MHz.
    nchans : int
        Number of frequency channels.
    nsamps : int
        Total number of time samples.
    tsamp : float
        Sampling interval in seconds.
    dm : float
        Dispersion measure in :math:`\\text{pc} \\, \\text{cm}^{-3}`.
    amp : float, default=1.0
        Peak signal amplitude added per channel.
    offset : int, default=0
        Time sample index at the reference frequency where the pulse begins.
    width : int, default=1
        Intrinsic pulse duration in time samples.
    noise_rms : float, default=0.0
        Standard deviation of zero-mean Gaussian background noise.

    Returns
    -------
    np.ndarray
        2D waterfall array of shape `(nchans, nsamps)` with float32 values.
    """
    rng = np.random.default_rng()
    arr = rng.standard_normal((nchans, nsamps)) * noise_rms
    arr[:, offset : offset + width] += amp
    delays = get_dmdelays(dm, f_min, f_max, tsamp, nchans, in_samples=True)
    new_ar = np.zeros_like(arr, dtype=np.float32)
    for ichan in range(nchans):
        new_ar[ichan] = np.roll(arr[ichan], -delays[ichan])
    return new_ar


@njit(cache=True, fastmath=True)
def cff(f1_start: float, f1_end: float, f2_start: float, f2_end: float) -> float:
    """Compute the fractional dispersion delay ratio between two frequency intervals."""
    return (f1_start**-2 - f1_end**-2) / (f2_start**-2 - f2_end**-2)


@njit(cache=True, fastmath=True)
def generate_pure_frb(
    nchans: int,
    nsamps: int,
    f_min: float,
    f_max: float,
    dt: int,
    pulse_toa: float,
    amplitude: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Generate a noise-free dispersed pulse with exact channel-edge integration.

    Parameters
    ----------
    nchans : int
        Number of channels.
    nsamps : int
        Number of time samples.
    f_min, f_max : float
        Band edges in MHz.
    dt : int
        Dispersive delay across the full band in samples.
    pulse_toa : float
        Arrival time at lowest frequency in samples.
    amplitude : float, default=1.0
        Pulse peak amplitude.

    Returns
    -------
    tuple[np.ndarray, float]
        A tuple of (waterfall array of shape `(nchans, nsamps)`, number of dispersed samples).
    """
    arr = np.zeros((nchans, nsamps), dtype=np.float32)
    foff = (f_max - f_min) / nchans
    chan_freqs = np.arange(nchans, dtype=np.float32) * foff + f_min + foff / 2
    # Use channel edges for physical accuracy
    freqs_bot = chan_freqs - foff / 2
    freqs_top = chan_freqs + foff / 2

    dt_start = dt * cff(f_min, freqs_bot, f_min, f_max)
    tstart = pulse_toa - dt_start
    tstart_int = tstart.astype(np.int32)
    tstart_frac = tstart - tstart_int

    dt_sub = dt * cff(freqs_bot, freqs_top, f_min, f_max)
    tend = tstart - dt_sub
    tend_int = tend.astype(np.int32)
    tend_frac = 1 - (tend - tend_int)

    nsamps_dispersed = 0
    for ichan in range(nchans):
        tstart_i = tstart_int[ichan]
        tend_i = tend_int[ichan]

        if 0 <= tend_i <= tstart_i < nsamps:
            if tend_i == tstart_i:
                arr[ichan, tend_i] = amplitude
                nsamps_dispersed += 1
            else:
                arr[ichan, tend_i : tstart_i + 1] = amplitude / dt_sub[ichan]
                arr[ichan, tend_i] *= tend_frac[ichan]
                arr[ichan, tstart_i] *= tstart_frac[ichan]
                nsamps_dispersed += tstart_i - tend_i + 1
        elif tend_i < 0 <= tstart_i < nsamps:
            arr[ichan, : tstart_i + 1] = amplitude / dt_sub[ichan]
            arr[ichan, tstart_i] *= tstart_frac[ichan]
            nsamps_dispersed += tstart_i + 1
        elif 0 <= tend_i < nsamps <= tstart_i:
            arr[ichan, tend_i:] = amplitude / dt_sub[ichan]
            arr[ichan, tend_i] *= tend_frac[ichan]
            nsamps_dispersed += nsamps - tend_i

    return arr, float(nsamps_dispersed)


@njit(cache=True, fastmath=True)
def generate_dispersed_periodic_signal(
    nchans: int,
    nsamps: int,
    f_min: float,
    f_max: float,
    dm: float,
    tsamp: float,
    spin_freqs: np.ndarray,
    amplitude: float = 1.0,
    os_factor: int = 32,
) -> np.ndarray:
    """Generate a dispersed periodic signal with accurate intra-channel smearing.

    Synthesizes the signal on an oversampled time grid, evaluates exact arrival
    times across each channel's frequency boundaries, smears via moving average
    boxcar convolution, and bins down to the target sampling resolution.

    Parameters
    ----------
    nchans : int
        Number of frequency channels.
    nsamps : int
        Number of time samples.
    f_min, f_max : float
        Frequency boundaries in MHz.
    dm : float
        Dispersion measure in :math:`\\text{pc} \\, \\text{cm}^{-3}`.
    tsamp : float
        Target time sampling interval in seconds.
    spin_freqs : np.ndarray
        Array of fundamental and harmonic spin frequencies in Hz.
    amplitude : float, default=1.0
        Peak signal amplitude.
    os_factor : int, default=32
        Oversampling factor for numerical intra-channel smearing integration.

    Returns
    -------
    np.ndarray
        Simulated filterbank waterfall of shape `(nchans, nsamps)` with float32 values.
    """
    tsamp_fine = tsamp / os_factor
    nsamps_fine = nsamps * os_factor
    t_fine = np.arange(nsamps_fine) * tsamp_fine

    signal_fine = np.zeros(nsamps_fine, dtype=np.float32)
    for freq in spin_freqs:
        signal_fine += amplitude * np.sin(2 * np.pi * freq * t_fine)

    waterfall = np.zeros((nchans, nsamps), dtype=np.float32)
    foff = (f_max - f_min) / nchans
    chan_freqs = np.arange(nchans, dtype=np.float32) * foff + f_min + foff / 2
    freqs_bot = chan_freqs - foff / 2
    freqs_top = chan_freqs + foff / 2
    dt_total = int(np.ceil(DM_CONSTANT * (f_min**-2 - f_max**-2) * dm / tsamp_fine))
    dt_bot = dt_total * cff(f_min, freqs_bot, f_min, f_max)
    dt_sub = dt_total * cff(freqs_bot, freqs_top, f_min, f_max)

    smeared_signal = np.zeros(nsamps_fine, dtype=np.float32)
    for i in range(nchans):
        smear_width = max(1, int(dt_sub[i]))
        start_idx = int(dt_bot[i])
        shifted_signal = np.roll(signal_fine, -start_idx)
        if smear_width > 1:
            cumsum = 0.0
            for j in range(smear_width):
                cumsum += shifted_signal[j]
                smeared_signal[j] = cumsum / (j + 1)
            for j in range(smear_width, len(shifted_signal)):
                cumsum += shifted_signal[j] - shifted_signal[j - smear_width]
                smeared_signal[j] = cumsum / smear_width
        else:
            smeared_signal[:] = shifted_signal
        # Bin down to the desired time resolution
        for j in range(nsamps):
            start = j * os_factor
            end = start + os_factor
            total = np.sum(smeared_signal[start:end])
            waterfall[i, j] = total / os_factor
    return waterfall
