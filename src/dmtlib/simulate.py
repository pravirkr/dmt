import numpy as np
from numba import njit


def get_dmdelays(
    dm: float,
    f_min: float,
    f_max: float,
    tsamp: float,
    nchans: int,
    *,
    in_samples: bool = True,
) -> np.ndarray:
    """Calculate the DM delays for a given frequency range."""
    foff = (f_max - f_min) / nchans
    chan_freqs = np.arange(nchans, dtype=np.float64) * foff + f_min
    delays = dm * 4.148808e3 * (f_min**-2 - chan_freqs**-2)
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
    amp: float = 1,
    offset: int = 0,
    width: int = 1,
    noise_rms: float = 0,
) -> np.ndarray:
    """Generate a simple frb signal."""
    rng = np.random.default_rng()
    arr = rng.standard_normal((nchans, nsamps)) * noise_rms
    arr[:, offset : offset + width] += amp
    delays = get_dmdelays(dm, f_min, f_max, tsamp, nchans, in_samples=True)
    new_ar = np.zeros_like(arr)
    for ichan in range(nchans):
        new_ar[ichan] = np.roll(arr[ichan], -delays[ichan])
    return new_ar


@njit(cache=True)
def cff(f1_start: float, f1_end: float, f2_start: float, f2_end: float) -> float:
    return (f1_start**-2 - f1_end**-2) / (f2_start**-2 - f2_end**-2)


@njit(cache=True)
def generate_pure_frb(
    nchans: int,
    nsamps: int,
    f_min: float,
    f_max: float,
    dt: int,
    pulse_toa: float,
    amplitude: float = 1,
) -> tuple[np.ndarray, float]:
    arr = np.zeros((nchans, nsamps), dtype=np.float32)
    foff = (f_max - f_min) / nchans
    freqs = np.arange(nchans, dtype=np.float32) * foff + f_min + foff / 2
    freqs_min_sub = freqs - foff / 2
    freqs_max_sub = freqs + foff / 2

    dt_start = dt * cff(f_min, freqs_min_sub, f_min, f_max)
    tstart = pulse_toa - dt_start
    tstart_int = tstart.astype(np.int32)
    tstart_frac = tstart - tstart_int

    dt_sub = dt * cff(freqs_min_sub, freqs_max_sub, f_min, f_max)
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
                arr[ichan, tend_i:tstart_i+1] = amplitude / dt_sub[ichan]
                arr[ichan, tend_i] *= tend_frac[ichan]
                arr[ichan, tstart_i] *= tstart_frac[ichan]
                nsamps_dispersed += tstart_i - tend_i + 1
        elif tend_i < 0 <= tstart_i < nsamps:
            arr[ichan, :tstart_i + 1] = amplitude / dt_sub[ichan]
            arr[ichan, tstart_i] *= tstart_frac[ichan]
            nsamps_dispersed += tstart_i + 1
        elif 0 <= tend_i < nsamps <= tstart_i:
            arr[ichan, tend_i:] = amplitude / dt_sub[ichan]
            arr[ichan, tend_i] *= tend_frac[ichan]
            nsamps_dispersed += nsamps - tend_i

    return arr, nsamps_dispersed
