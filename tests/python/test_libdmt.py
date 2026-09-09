import numpy as np
import pytest
from dmtlib import libdmt


class TestFDMT:
    def test_initialise_ones(self) -> None:
        nchans = 500
        nsamples = 1024
        dt_max = 512
        thefdmt = libdmt.FDMTCPU(1000, 1500, nchans, nsamples, 0.001, dt_max)
        waterfall = np.ones((nchans, nsamples), dtype=np.float32)
        dmt_output = thefdmt.execute(waterfall)
        np.testing.assert_equal(
            dmt_output.shape,
            (thefdmt.dt_grid_final.size, thefdmt.plan.dmt_nsamps),
        )

    def test_stepper_bit_exact(self) -> None:
        nchans = 128
        nsamples = 512
        dt_max = 64
        thefdmt = libdmt.FDMTCPU(1000.0, 1500.0, nchans, nsamples, 0.001, dt_max)

        rng = np.random.default_rng(42)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        # Synchronous execution
        sync_output = thefdmt.execute(waterfall)

        # Stepper execution
        thefdmt.reset(waterfall)
        thefdmt.advance_until_remaining(0)
        assert thefdmt.is_finished
        step_output = thefdmt.finalize()

        np.testing.assert_allclose(step_output, sync_output, rtol=1e-6, atol=1e-6)

    def test_subband_views_one_level_remaining(self) -> None:
        nchans = 256
        nsamples = 1024
        dt_max = 128
        thefdmt = libdmt.FDMTCPU(1000.0, 1500.0, nchans, nsamples, 0.001, dt_max)

        rng = np.random.default_rng(123)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        thefdmt.reset(waterfall)
        assert thefdmt.current_level == 0
        assert thefdmt.remaining_levels == thefdmt.total_levels - 1

        # Advance until 1 level before root -> 2 children subbands
        thefdmt.advance_until_remaining(1)
        assert thefdmt.remaining_levels == 1
        assert thefdmt.num_subbands == 2
        assert not thefdmt.is_finished

        sub0 = thefdmt.view_subband(0)
        sub1 = thefdmt.view_subband(1)

        # Verify subband metadata
        assert sub0.subband_idx == 0
        assert sub1.subband_idx == 1
        assert np.isclose(sub0.f_start, 1000.0)
        assert np.isclose(sub0.f_end, sub1.f_start)
        assert np.isclose(sub1.f_end, 1500.0)

        assert sub0.ndt > 0
        assert sub1.ndt > 0
        assert sub0.nsamps == sub1.nsamps

        # Check 2D data views
        assert sub0.data.shape == (sub0.ndt, sub0.nsamps)
        assert sub1.data.shape == (sub1.ndt, sub1.nsamps)

        # Check view_subband_data matches view_subband(s).data
        sub0_data = thefdmt.view_subband_data(0)
        np.testing.assert_array_equal(sub0_data, sub0.data)

        # Resume and finalize
        res = thefdmt.finalize()
        sync = thefdmt.execute(waterfall)
        np.testing.assert_allclose(res, sync, rtol=1e-6, atol=1e-6)

    def test_subband_views_two_levels_remaining(self) -> None:
        nchans = 256
        nsamples = 1024
        dt_max = 128
        thefdmt = libdmt.FDMTCPU(1000.0, 1500.0, nchans, nsamples, 0.001, dt_max)

        rng = np.random.default_rng(456)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        thefdmt.reset(waterfall)
        thefdmt.advance_until_remaining(2)

        assert thefdmt.remaining_levels == 2
        assert thefdmt.num_subbands == 4

        # Verify contiguous frequency coverage across the 4 subbands
        subbands = [thefdmt.view_subband(s) for s in range(4)]
        assert np.isclose(subbands[0].f_start, 1000.0)
        for s in range(3):
            assert np.isclose(subbands[s].f_end, subbands[s + 1].f_start)
        assert np.isclose(subbands[3].f_end, 1500.0)

        # Check shapes
        for s in range(4):
            assert subbands[s].data.shape == (subbands[s].ndt, subbands[s].nsamps)

        # Finalize and verify exact match
        res = thefdmt.finalize()
        sync = thefdmt.execute(waterfall)
        np.testing.assert_allclose(res, sync, rtol=1e-6, atol=1e-6)

    def test_step_by_step_advance(self) -> None:
        nchans = 64
        nsamples = 256
        dt_max = 32
        thefdmt = libdmt.FDMTCPU(1000.0, 1500.0, nchans, nsamples, 0.001, dt_max)

        rng = np.random.default_rng(789)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        thefdmt.reset(waterfall)
        levels_stepped = 0
        while not thefdmt.is_finished:
            thefdmt.advance(1)
            levels_stepped += 1
            assert thefdmt.current_level == levels_stepped

        assert thefdmt.remaining_levels == 0
        assert thefdmt.is_finished

        res = thefdmt.finalize()
        sync = thefdmt.execute(waterfall)
        np.testing.assert_allclose(res, sync, rtol=1e-6, atol=1e-6)

    def test_custom_dmt_buffer(self) -> None:
        nchans = 64
        nsamples = 256
        dt_max = 32
        thefdmt = libdmt.FDMTCPU(1000.0, 1500.0, nchans, nsamples, 0.001, dt_max)

        rng = np.random.default_rng(999)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        # Allocate custom buffer matching buffer_size
        custom_buf = np.zeros(thefdmt.plan.buffer_size, dtype=np.float32)
        thefdmt.reset(waterfall, custom_buf)
        thefdmt.advance_until_remaining(1)
        res = thefdmt.finalize()

        sync = thefdmt.execute(waterfall)
        np.testing.assert_allclose(res, sync, rtol=1e-6, atol=1e-6)
        # Verify that the final result is in custom_buf directly
        np.testing.assert_array_equal(res.ravel(), custom_buf[: res.size])

    def test_stepper_error_handling(self) -> None:
        nchans = 64
        nsamples = 256
        dt_max = 32
        thefdmt = libdmt.FDMTCPU(1000.0, 1500.0, nchans, nsamples, 0.001, dt_max)

        rng = np.random.default_rng(101)
        waterfall = rng.standard_normal((nchans, nsamples), dtype=np.float32)

        # Calling advance or view before reset should raise
        with pytest.raises(RuntimeError):
            thefdmt.advance()
        with pytest.raises(RuntimeError):
            thefdmt.view_subband(0)

        # Buffer too small
        undersized_buf = np.zeros(10, dtype=np.float32)
        with pytest.raises(ValueError):
            thefdmt.reset(waterfall, undersized_buf)

        # Invalid waterfall dimension
        with pytest.raises(RuntimeError):
            thefdmt.reset(waterfall.ravel())

        # Subband index out of range
        thefdmt.reset(waterfall)
        with pytest.raises(IndexError):
            thefdmt.view_subband(9999)

        # Advancing beyond remaining levels clamps cleanly at root
        thefdmt.advance(100)
        assert thefdmt.is_finished
        assert thefdmt.remaining_levels == 0
        res = thefdmt.finalize()
        sync = thefdmt.execute(waterfall)
        np.testing.assert_allclose(res, sync, rtol=1e-6, atol=1e-6)
