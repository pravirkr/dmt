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
from dmtlib.grid import DM_CONSTANT
from dmtlib.simulate import generate_pure_frb

F_MIN, F_MAX, NCHANS, TSAMP = 400.0, 800.0, 64, 0.001
DF = (F_MAX - F_MIN) / NCHANS


def centre_lags(dm: float) -> np.ndarray:
    """Independent channel-centre lags (samples) behind the bottom edge f_min."""
    centres = F_MIN + DF * (np.arange(NCHANS) + 0.5)
    return dispersion_delay(dm, F_MIN, centres) / TSAMP


@pytest.mark.parametrize("cls", [DDMT, SDMT])
@pytest.mark.parametrize("dm", [37.3, 112.0])
def test_direct_engines_use_channel_centres(cls: type, dm: float) -> None:
    lags = np.rint(centre_lags(dm)).astype(int)
    s, nsamps = 3000, 4096  # f_min arrival sample
    x = np.zeros((NCHANS, nsamps), dtype=np.float32)
    x[np.arange(NCHANS), s - lags] = 1
    engine = cls(F_MIN, F_MAX, NCHANS, TSAMP, np.array([dm], dtype=np.float32))
    y = np.asarray(engine.execute(x)).ravel()
    assert engine.plan.f_ref == F_MIN
    offset = engine.plan.time_offset
    assert offset == int(np.ceil(centre_lags(dm)[-1]))
    assert y.argmax() == s - offset
    assert y[s - offset] == NCHANS


def test_ddmt_fft_fractional_delays_use_channel_centres() -> None:
    dm = 37.3
    plan = DDMTPlan(F_MIN, F_MAX, NCHANS, TSAMP, np.array([dm], dtype=np.float32))
    rate = np.asarray(plan.fractional_delay_table)
    np.testing.assert_allclose(-rate * dm, centre_lags(dm), rtol=1e-5, atol=1e-3)
    lags = centre_lags(dm)
    s, nsamps = 3000, 4096
    t = np.arange(nsamps)
    # A band-limited (sinc) impulse at the exact fractional lag.
    x = np.sinc(t[None, :] - (s - lags[:, None])).astype(np.float32)
    engine = DDMTFFT(F_MIN, F_MAX, NCHANS, TSAMP, np.array([dm], dtype=np.float32))
    y = np.asarray(engine.execute(x)).ravel()
    offset = engine.plan.time_offset
    assert engine.plan.f_ref == F_MIN
    assert y.argmax() == s - offset
    assert y[s - offset] == pytest.approx(NCHANS, rel=2e-2)


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
        assert r.f_ref == F_MIN, engine
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


def ref_delays(dm: float, f_ref: float) -> np.ndarray:
    """Channel-centre delays (samples) behind ``f_ref`` (negative above it)."""
    centres = F_MIN + DF * (np.arange(NCHANS) + 0.5)
    return DM_CONSTANT * dm * (centres**-2 - f_ref**-2) / TSAMP


F_REFS = [F_MIN, F_MAX]


@pytest.mark.parametrize("cls", [DDMT, SDMT, DDMTFFT])
@pytest.mark.parametrize("f_ref", F_REFS)
def test_direct_engines_take_band_edge_references(cls: type, f_ref: float) -> None:
    dm, s, nsamps = 37.3, 3000, 6144
    delays = ref_delays(dm, f_ref)
    t = np.arange(nsamps)
    if cls is DDMTFFT:
        x = np.sinc(t[None, :] - (s + delays[:, None])).astype(np.float32)
    else:
        x = np.zeros((NCHANS, nsamps), dtype=np.float32)
        x[np.arange(NCHANS), s + np.rint(delays).astype(int)] = 1
    engine = cls(F_MIN, F_MAX, NCHANS, TSAMP, np.array([dm], np.float32), f_ref=f_ref)
    assert engine.plan.f_ref == pytest.approx(f_ref)
    offset = engine.plan.time_offset
    if f_ref == F_MAX:
        assert offset == 0
    y = np.asarray(engine.execute(x)).ravel()
    assert y.argmax() == s - offset
    assert y[s - offset] == pytest.approx(NCHANS, rel=2e-2)


def test_dedisperse_top_reference_starts_at_sample_zero() -> None:
    nchans, nsamps, dm, toa = 256, 8192, 60.0, 5000.0
    dt = float(dispersion_delay(dm, F_MIN, F_MAX)) / TSAMP
    wf, _ = generate_pure_frb(nchans, nsamps, F_MIN, F_MAX, dt=dt, pulse_toa=toa)
    truth = toa * TSAMP - float(dispersion_delay(dm, F_MIN, np.inf))
    for engine in ("ddmt", "sdmt", "ddmt_fft"):
        r = dedisperse(wf, F_MIN, F_MAX, TSAMP, [dm], engine=engine, f_ref=F_MAX)
        assert r.start == 0
        assert r.f_ref == F_MAX
        y = r.data[0]
        top = np.flatnonzero(y >= y.max() - 1e-3 * abs(y.max()))
        t_inf = np.interp(top.mean(), np.arange(y.size), r.arrival_times()[0])
        assert abs(t_inf - truth) <= 1.5 * TSAMP, engine
    with pytest.raises(ValueError, match="refers times to f_min"):
        dedisperse(wf, F_MIN, F_MAX, TSAMP, [dm], engine="fdmt", f_ref=F_MAX)


def test_bad_reference_is_rejected() -> None:
    for bad in (0.0, -5.0, 600.0, F_MAX - DF / 2, np.inf, np.nan):
        with pytest.raises(ValueError, match="f_ref"):
            DDMTPlan(
                F_MIN, F_MAX, NCHANS, TSAMP, np.array([1.0], np.float32), f_ref=bad
            )
