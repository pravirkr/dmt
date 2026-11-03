"""Channel-frequency and time-reference conventions, and dmtlib.dedisperse."""

import numpy as np
import pytest

from dmtlib import (
    DDMT,
    DDMTFFT,
    FDMT,
    SDMT,
    DDMTPlan,
    dedisperse,
    dispersion_delay,
)
from dmtlib.simulate import generate_pure_frb

F_MIN, F_MAX, NCHANS, TSAMP = 400.0, 800.0, 64, 0.001
DF = (F_MAX - F_MIN) / NCHANS


def centre_delays(dm: float) -> np.ndarray:
    """Independent channel-centre delays (samples) behind the top channel."""
    centres = F_MIN + DF * (np.arange(NCHANS) + 0.5)
    return dispersion_delay(dm, centres, F_MAX - DF / 2) / TSAMP


@pytest.mark.parametrize("cls", [DDMT, SDMT])
@pytest.mark.parametrize("dm", [37.3, 112.0])
def test_direct_engines_use_channel_centres(cls: type, dm: float) -> None:
    delays = np.rint(centre_delays(dm)).astype(int)
    t0, nsamps = 300, 4096
    x = np.zeros((NCHANS, nsamps), dtype=np.float32)
    x[np.arange(NCHANS), t0 + delays] = 1
    engine = cls(F_MIN, F_MAX, NCHANS, TSAMP, np.array([dm], dtype=np.float32))
    y = np.asarray(engine.execute(x)).ravel()
    assert engine.plan.f_ref == pytest.approx(F_MAX - DF / 2)
    assert y.argmax() == t0
    assert y[t0] == NCHANS


def test_ddmt_fft_fractional_delays_use_channel_centres() -> None:
    dm = 37.3
    plan = DDMTPlan(F_MIN, F_MAX, NCHANS, TSAMP, np.array([dm], dtype=np.float32))
    rate = np.asarray(plan.fractional_delay_table)
    np.testing.assert_allclose(rate * dm, centre_delays(dm), rtol=1e-5, atol=1e-3)
    delays = centre_delays(dm)
    t0, nsamps = 300, 4096
    t = np.arange(nsamps)
    # A band-limited (sinc) impulse at the exact fractional delay.
    x = np.sinc(t[None, :] - (t0 + delays[:, None])).astype(np.float32)
    engine = DDMTFFT(F_MIN, F_MAX, NCHANS, TSAMP, np.array([dm], dtype=np.float32))
    y = np.asarray(engine.execute(x)).ravel()
    assert y.argmax() == t0
    assert y[t0] == pytest.approx(NCHANS, rel=2e-2)


def test_fdmt_references_band_bottom_edge() -> None:
    f = FDMT(F_MIN, F_MAX, NCHANS, 1024, TSAMP, dt_grid=np.array([256]))
    assert f.plan.f_ref == F_MIN
    trace = f.plan.trace_dm(0)
    # The top channel's upper edge (f_max) leads f_min by the whole delay.
    assert trace[-1] == -256
    assert f.startup_samples == 256


def test_dedisperse_engines_agree_on_infinite_frequency_toa() -> None:
    # 256 channels keep intra-channel smearing to a few samples: raw FDMT
    # times each channel's upper-edge (leading) sample, so its TOA runs late
    # by about half a channel's smear, which grows at coarser channelisation.
    nchans, nsamps, dm, toa = 256, 8192, 60.0, 5000.0
    dt = float(dispersion_delay(dm, F_MIN, F_MAX)) / TSAMP
    wf, _ = generate_pure_frb(nchans, nsamps, F_MIN, F_MAX, dt=dt, pulse_toa=toa)
    truth = toa * TSAMP - float(dispersion_delay(dm, F_MIN, np.inf))
    for engine in ("fdmt", "fdmt_fft", "ddmt", "sdmt", "ddmt_fft"):
        r = dedisperse(wf, F_MIN, F_MAX, TSAMP, [dm], engine=engine)
        assert r.dms[0] == pytest.approx(dm, abs=0.05)
        # A smeared pulse peaks on a plateau: time its centre, not argmax.
        y = r.data[0]
        top = np.flatnonzero(y >= y.max() - 1e-3 * abs(y.max()))
        t_inf = np.interp(top.mean(), np.arange(y.size), r.arrival_times()[0])
        assert abs(t_inf - truth) <= 1.5 * TSAMP, engine
        np.testing.assert_allclose(
            r.arrival_times(r.f_ref)[0], (r.start + np.arange(r.data.shape[1])) * TSAMP
        )


def test_dedisperse_drops_fdmt_startup_and_keeps_white_noise() -> None:
    rng = np.random.default_rng(3)
    wf = rng.standard_normal((NCHANS, 8192), dtype=np.float32) + 5.0
    r = dedisperse(wf, F_MIN, F_MAX, TSAMP, [0.0, 50.0], engine="fdmt")
    f = FDMT(F_MIN, F_MAX, NCHANS, 8192, TSAMP, dm_grid=np.array([0.0, 50.0]))
    assert r.start == f.startup_samples
    assert r.data.shape == (2, 8192 - f.startup_samples)
    # Every retained sample sums all channels: mean 5 * nchans, no ramp.
    assert abs(r.data.mean() / (5.0 * NCHANS) - 1) < 1e-3
    np.testing.assert_allclose(r.data.std(axis=1), np.sqrt(NCHANS), rtol=0.05)


def test_dedisperse_rejects_bad_input() -> None:
    wf = np.zeros((NCHANS, 512), dtype=np.float32)
    with pytest.raises(ValueError, match="startup"):
        dedisperse(wf, F_MIN, F_MAX, TSAMP, [500.0])
    with pytest.raises(ValueError, match="box smearing"):
        dedisperse(wf, F_MIN, F_MAX, TSAMP, [1.0], engine="ddmt", use_box_smearing=True)
    with pytest.raises(ValueError, match="unknown engine"):
        dedisperse(wf, F_MIN, F_MAX, TSAMP, [1.0], engine="nope")
    with pytest.raises(ValueError, match="nchans, nsamps"):
        dedisperse(wf[0], F_MIN, F_MAX, TSAMP, [1.0])
