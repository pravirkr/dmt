import numpy as np


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
