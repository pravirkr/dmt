import numpy as np
import pytest
from dmtlib import FDMTCPU, FDMTFFTCPU, compute_fdmt_fft, libdmt


class TestFDMTFFT:
    def test_fdmt_fft_initialise_shape(self) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 128
        nsamples = 512
        tsamp = 0.001
        dt_max = 64

        thefdmt = FDMTFFTCPU(f_min, f_max, nchans, nsamples, tsamp, dt_max)
        waterfall = np.ones((nchans, nsamples), dtype=np.float32)
        dmt_output = thefdmt.execute(waterfall)

        assert dmt_output.shape == (thefdmt.dt_grid_final.size, nsamples)
        assert thefdmt.nbeams == 1
        assert thefdmt.plan.nsamps == nsamples
        assert thefdmt.plan.mode == "valid"

    def test_fdmt_fft_numerical_equivalence_roll(self) -> None:
        f_min = 1200.0
        f_max = 1600.0
        nchans = 64
        nsamples = 256
        tsamp = 0.0005
        dt_max = 48
        dt_min = 0

        rng = np.random.default_rng(42)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        fdmt_fft = FDMTFFTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, dt_min, mode="roll"
        )
        fdmt_roll = FDMTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, dt_min, mode="roll"
        )

        dmt_fft = fdmt_fft.execute(waterfall)
        dmt_roll = fdmt_roll.execute(waterfall)

        assert dmt_fft.shape == dmt_roll.shape
        np.testing.assert_allclose(dmt_fft, dmt_roll, rtol=1e-3, atol=1e-3)

    def test_fdmt_fft_full_mode(self) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 32
        nsamples = 256
        tsamp = 0.001
        dt_max = 32
        rng = np.random.default_rng(7)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        fdmt_fft = FDMTFFTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, mode="full"
        )
        fdmt = FDMTCPU(f_min, f_max, nchans, nsamples, tsamp, dt_max, mode="full")
        a = fdmt_fft.execute(waterfall)
        b = fdmt.execute(waterfall)
        np.testing.assert_allclose(a[:, :nsamples], b[:, :nsamples], rtol=1e-4, atol=1e-4)

    def test_fdmt_fft_valid_streaming(self) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 16
        nsamples = 128
        tsamp = 0.001
        dt_max = 24
        rng = np.random.default_rng(3)

        fdmt_fft = FDMTFFTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, mode="valid"
        )
        fdmt = FDMTCPU(f_min, f_max, nchans, nsamples, tsamp, dt_max, mode="valid")
        fdmt_fft.reset_history()
        fdmt.reset_history()
        for i in range(3):
            waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)
            np.testing.assert_allclose(
                fdmt_fft.execute(waterfall),
                fdmt.execute(waterfall),
                rtol=3e-3,
                atol=3e-3,
            )

    def test_fdmt_fft_odd_channels(self) -> None:
        f_min = 800.0
        f_max = 1200.0
        nchans = 37
        nsamples = 256
        tsamp = 0.001
        dt_max = 32

        rng = np.random.default_rng(123)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        fdmt_fft = FDMTFFTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, mode="roll"
        )
        fdmt_roll = FDMTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, mode="roll"
        )

        np.testing.assert_allclose(
            fdmt_fft.execute(waterfall),
            fdmt_roll.execute(waterfall),
            rtol=1e-3,
            atol=1e-3,
        )

    def test_fdmt_fft_multibeam(self) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 32
        nsamples = 128
        tsamp = 0.001
        dt_max = 20
        nbeams = 2

        rng = np.random.default_rng(999)
        waterfall_beam0 = rng.standard_normal((nchans, nsamples), dtype=np.float32)
        waterfall_beam1 = rng.standard_normal((nchans, nsamples), dtype=np.float32)
        waterfall_multi = np.stack([waterfall_beam0, waterfall_beam1], axis=0)

        fdmt_multi = FDMTFFTCPU(
            f_min,
            f_max,
            nchans,
            nsamples,
            tsamp,
            dt_max,
            nbeams=nbeams,
            mode="roll",
        )
        fdmt_single = FDMTFFTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, mode="roll"
        )

        dmt_multi = fdmt_multi.execute(waterfall_multi)
        dmt_single0 = fdmt_single.execute(waterfall_beam0)
        dmt_single1 = fdmt_single.execute(waterfall_beam1)

        assert dmt_multi.shape == (nbeams, fdmt_multi.dt_grid_final.size, nsamples)
        np.testing.assert_allclose(dmt_multi[0], dmt_single0, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(dmt_multi[1], dmt_single1, rtol=1e-5, atol=1e-5)

    def test_fdmt_fft_convenience_function(self) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 32
        nsamples = 128
        tsamp = 0.001
        dt_max = 20

        rng = np.random.default_rng(777)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        dmt_conv, plan = compute_fdmt_fft(
            waterfall, f_min, f_max, nchans, nsamples, tsamp, dt_max, mode="valid"
        )
        dmt_conv = dmt_conv[: plan.dmt_ndms * plan.dmt_nsamps].reshape(
            plan.dmt_ndms, plan.dmt_nsamps
        )
        fdmt = FDMTFFTCPU(f_min, f_max, nchans, nsamples, tsamp, dt_max, mode="valid")
        dmt_direct = fdmt.execute(waterfall)

        assert dmt_conv.shape == (plan.dmt_ndms, plan.dmt_nsamps)
        np.testing.assert_allclose(dmt_conv, dmt_direct, rtol=1e-5, atol=1e-5)

    def test_fdmt_fft_frb_track_detection(self) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 64
        nsamples = 512
        tsamp = 0.001
        dt_max = 100

        fdmt = FDMTFFTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, mode="roll"
        )
        waterfall = np.zeros((nchans, nsamples), dtype=np.float32)

        target_dm_idx = len(fdmt.dt_grid_final) // 2
        target_toffset = 200
        libdmt.add_frb_track(
            waterfall,
            fdmt.plan,
            dm_idx=target_dm_idx,
            amplitude=10.0,
            toffset=target_toffset,
            width=1,
        )

        dmt = fdmt.execute(waterfall)
        peak_dm, peak_t = np.unravel_index(np.argmax(dmt), dmt.shape)

        assert peak_dm == target_dm_idx
        assert peak_t == target_toffset

    def test_fdmt_fft_custom_grids(self) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 64
        nsamples = 256
        tsamp = 0.001

        dt_arr = np.array([0, 5, 10, 20, 30, 45], dtype=np.int32)
        fdmt_dt = FDMTFFTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_arr=dt_arr, mode="roll"
        )
        waterfall = np.ones((nchans, nsamples), dtype=np.float32)
        dmt = fdmt_dt.execute(waterfall)
        assert dmt.shape[0] == len(dt_arr)
        assert dmt.shape[1] == nsamples

    def test_fdmt_fft_stepper(self) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 32
        nsamples = 128
        tsamp = 0.001
        dt_max = 16
        rng = np.random.default_rng(1)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        fdmt = FDMTFFTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, mode="roll"
        )
        oneshot = fdmt.execute(waterfall)
        fdmt.reset(waterfall)
        fdmt.advance_until_remaining(1)
        assert fdmt.num_subbands == 2
        fdmt.advance_until_remaining(0)
        stepped = fdmt.finalize()
        np.testing.assert_allclose(stepped, oneshot, rtol=1e-5, atol=1e-5)

    def test_fdmt_fft_valid_small_blocks(self) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 16
        nsamples = 8
        tsamp = 0.001
        dt_max = 24
        rng = np.random.default_rng(4)
        fdmt_fft = FDMTFFTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, mode="valid"
        )
        fdmt = FDMTCPU(f_min, f_max, nchans, nsamples, tsamp, dt_max, mode="valid")
        fdmt_fft.reset_history()
        fdmt.reset_history()
        for _ in range(4):
            waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)
            np.testing.assert_allclose(
                fdmt_fft.execute(waterfall),
                fdmt.execute(waterfall),
                rtol=5e-3,
                atol=5e-3,
            )

    @pytest.mark.cuda
    def test_fdmt_fft_gpu_optional(self) -> None:
        from dmtlib import FDMTFFTCUDA

        f_min = 1000.0
        f_max = 1500.0
        nchans = 16
        nsamples = 64
        tsamp = 0.001
        dt_max = 8
        rng = np.random.default_rng(2)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)
        cpu = FDMTFFTCPU(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, mode="roll"
        )
        gpu = FDMTFFTCUDA(
            f_min, f_max, nchans, nsamples, tsamp, dt_max, mode="roll"
        )
        np.testing.assert_allclose(
            gpu.execute(waterfall), cpu.execute(waterfall), rtol=2e-3, atol=2e-3
        )
