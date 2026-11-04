"""SIGPROC header to dmt band conversion."""

import numpy as np
import pytest

from dmtlib import DDMT, dispersion_delay, sigproc_band, to_ascending


def test_descending_header_flips() -> None:
    band = sigproc_band(fch1=1500.0, foff=-1.0, nchans=4)
    assert (band.f_min, band.f_max, band.flip) == (1496.5, 1500.5, True)
    assert band.df == 1.0
    wf = np.arange(8, dtype=np.float32).reshape(4, 2)
    np.testing.assert_array_equal(to_ascending(wf, band), wf[::-1])


def test_ascending_header_is_unchanged() -> None:
    band = sigproc_band(fch1=400.25, foff=0.5, nchans=8)
    assert (band.f_min, band.f_max, band.flip) == (400.0, 404.0, False)
    wf = np.zeros((8, 3), dtype=np.float32)
    assert to_ascending(wf, band) is wf


def test_flipped_sigproc_data_dedisperses_like_ascending() -> None:
    # The same dispersed pulse written in SIGPROC (descending) order.
    nchans, nsamps, tsamp, dm = 32, 2048, 0.001, 50.0
    band = sigproc_band(fch1=1500.0 - 2.5, foff=-5.0, nchans=nchans)
    centres = band.f_min + band.df * (np.arange(nchans) + 0.5)
    lag = np.rint(dispersion_delay(dm, band.f_min, centres) / tsamp).astype(int)
    asc = np.zeros((nchans, nsamps), dtype=np.float32)
    asc[np.arange(nchans), 1500 - lag] = 1.0
    sigproc = asc[::-1].copy()
    engine = DDMT(band.f_min, band.f_max, nchans, tsamp, np.array([dm], np.float32))
    y = np.asarray(engine.execute(np.ascontiguousarray(to_ascending(sigproc, band))))
    assert y.ravel()[1500 - engine.plan.time_offset] == nchans


def test_rejects_bad_headers() -> None:
    with pytest.raises(ValueError, match="foff"):
        sigproc_band(1400.0, 0.0, 16)
    with pytest.raises(ValueError, match="nchans"):
        sigproc_band(1400.0, -1.0, 0)
    with pytest.raises(ValueError, match="channels"):
        to_ascending(np.zeros((3, 4)), sigproc_band(1400.0, -1.0, 4))
