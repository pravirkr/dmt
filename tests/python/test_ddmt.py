import numpy as np
import pytest
from dmtlib import DDMTCPU, DDMTPlan, LevinConfig


class TestDDMT:
    def test_execute_ones(self) -> None:
        nchans = 8
        nsamps = 64
        ddmt = DDMTCPU(1000.0, 1500.0, nchans, 0.001, 10.0, 5.0, 0.0)
        waterfall = np.ones((nchans, nsamps), dtype=np.float32)
        dmt = ddmt.execute(waterfall)
        assert dmt.ndim == 2
        assert dmt.shape[0] >= 1
        assert dmt.shape[1] <= nsamps
        np.testing.assert_allclose(dmt[0], nchans, rtol=0, atol=1e-5)

    def test_custom_dm_arr(self) -> None:
        nchans = 8
        nsamps = 32
        dm_arr = np.array([0.0, 2.5], dtype=np.float32)
        ddmt = DDMTCPU(1000.0, 1500.0, nchans, 0.001, dm_arr)
        waterfall = np.ones((nchans, nsamps), dtype=np.float32)
        dmt = ddmt.execute(waterfall)
        assert dmt.shape[0] == 2

    def test_levin_grid_generation_and_plan(self) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 16
        tsamp = 0.001
        dm_start = 0.0
        dm_end = 50.0
        pw = 0.001
        tol = 1.25

        grid = DDMTPlan.generate_levin_dm_grid(
            dm_start, dm_end, tsamp, pw, f_min, f_max, nchans, tol
        )
        assert len(grid) > 1
        assert grid[0] == 0.0
        assert grid[-1] >= dm_end
        assert np.all(np.diff(grid) > 0)

        levin = LevinConfig(dm_start, dm_end, pw, tol)
        plan = DDMTPlan(f_min, f_max, nchans, tsamp, levin)
        np.testing.assert_allclose(plan.dm_arr, grid, rtol=1e-5)

        ddmt = DDMTCPU(f_min, f_max, nchans, tsamp, levin)
        np.testing.assert_allclose(ddmt.plan.dm_arr, grid, rtol=1e-5)

    def test_kill_mask(self) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 8
        nsamps = 64
        tsamp = 0.001

        mask = np.ones(nchans, dtype=np.uint8)
        mask[0] = 0
        mask[2] = 0

        ddmt = DDMTCPU(
            f_min, f_max, nchans, tsamp, 10.0, 5.0, 0.0, kill_mask=mask
        )
        waterfall = np.ones((nchans, nsamps), dtype=np.float32)
        dmt = ddmt.execute(waterfall)
        # DM=0 sums unmasked channels only (8 - 2 = 6)
        np.testing.assert_allclose(dmt[0], nchans - 2, rtol=0, atol=1e-5)

    @pytest.mark.parametrize("nbits", [1, 2, 4, 8, 16])
    def test_packed_and_time_major_execution(self, nbits: int) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 16
        nsamps = 64
        tsamp = 0.001
        dm_arr = np.array([0.0, 5.0, 15.0], dtype=np.float32)

        plan = DDMTPlan(f_min, f_max, nchans, tsamp, dm_arr, nbits=nbits)
        ddmt = DDMTCPU(plan)

        mask = (1 << nbits) - 1
        raw = (np.arange(nchans * nsamps, dtype=np.uint64) * 37 + 11) & mask
        raw = raw.reshape((nchans, nsamps))

        # Channel-major packing
        chan_row_bytes = (nsamps * nbits + 7) // 8
        chan_major = np.zeros((nchans, chan_row_bytes), dtype=np.uint8)
        if nbits == 16:
            chan_major.view(np.uint16)[:] = raw
        elif nbits == 8:
            chan_major[:] = raw.astype(np.uint8)
        else:
            for c in range(nchans):
                for s in range(nsamps):
                    val = int(raw[c, s])
                    bit_off = s * nbits
                    byte_idx = bit_off // 8
                    bit_sub = bit_off % 8
                    chan_major[c, byte_idx] |= (val << bit_sub)

        # Time-major packing
        time_samp_bytes = (nchans * nbits + 7) // 8
        time_major = np.zeros((nsamps, time_samp_bytes), dtype=np.uint8)
        if nbits == 16:
            time_major.view(np.uint16)[:] = raw.T
        elif nbits == 8:
            time_major[:] = raw.T.astype(np.uint8)
        else:
            for s in range(nsamps):
                for c in range(nchans):
                    val = int(raw[c, s])
                    bit_off = c * nbits
                    byte_idx = bit_off // 8
                    bit_sub = bit_off % 8
                    time_major[s, byte_idx] |= (val << bit_sub)

        out_chan = ddmt.execute(chan_major, nsamps)
        out_time = ddmt.execute_time_major(time_major, nsamps)

        np.testing.assert_array_equal(out_time, out_chan)

    def test_streaming_history(self) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 8
        tsamp = 0.001
        dms = np.array([0.0, 4.0], dtype=np.float32)

        ddmt = DDMTCPU(f_min, f_max, nchans, tsamp, dms)

        total_nsamps = 256
        rng = np.random.default_rng(42)
        full_waterfall = rng.standard_normal((nchans, total_nsamps)).astype(np.float32)

        # 1. Monolithic one-shot execute
        ddmt_mono = DDMTCPU(f_min, f_max, nchans, tsamp, dms)
        dmt_expected = ddmt_mono.execute(full_waterfall)

        # 2. Streamed execution across two chunks
        chunk1_samps = 100
        chunk1 = full_waterfall[:, :chunk1_samps]
        chunk2 = full_waterfall[:, chunk1_samps:]

        ddmt.reset_history()
        out1 = ddmt.execute(chunk1)
        out2 = ddmt.execute(chunk2)

        dmt_streamed = np.concatenate([out1, out2], axis=1)
        np.testing.assert_allclose(dmt_streamed, dmt_expected, rtol=1e-5, atol=1e-5)

        # 3. Save / load history
        hist = ddmt.save_history()
        assert len(hist) == ddmt.history_state_size()

        ddmt.reset_history()
        ddmt.load_history(hist)
        chunk3 = rng.standard_normal((nchans, 32)).astype(np.float32)
        out3 = ddmt.execute(chunk3)
        assert out3.shape[1] == 32

    def test_multi_beam_float(self) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 8
        nsamps = 64
        nbeams = 3
        tsamp = 0.001
        dms = np.array([0.0, 5.0, 10.0], dtype=np.float32)

        ddmt_multi = DDMTCPU(f_min, f_max, nchans, tsamp, dms, nbeams=nbeams)
        assert ddmt_multi.nbeams == nbeams

        rng = np.random.default_rng(123)
        waterfall = rng.standard_normal((nbeams, nchans, nsamps)).astype(np.float32)

        out_multi = ddmt_multi.execute(waterfall)
        assert out_multi.ndim == 3
        assert out_multi.shape[0] == nbeams
        assert out_multi.shape[1] == len(dms)

        # Compare each beam against single-beam DDMTCPU
        for b in range(nbeams):
            ddmt_single = DDMTCPU(f_min, f_max, nchans, tsamp, dms)
            out_single = ddmt_single.execute(waterfall[b])
            np.testing.assert_allclose(out_multi[b], out_single, rtol=1e-5, atol=1e-5)

    def test_multi_beam_packed_and_time_major(self) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 8
        nsamps = 48
        nbeams = 2
        tsamp = 0.001
        dms = np.array([0.0, 4.0], dtype=np.float32)

        plan = DDMTPlan(f_min, f_max, nchans, tsamp, dms, nbits=8)
        ddmt_multi = DDMTCPU(plan, nbeams=nbeams)

        raw = ((np.arange(nbeams * nchans * nsamps, dtype=np.uint64) * 17 + 3) % 256).astype(np.uint8)
        chan_major = raw.reshape((nbeams, nchans, nsamps))
        time_major = np.ascontiguousarray(
            chan_major.transpose((0, 2, 1)) # (nbeams, nsamps, nchans)
        )

        out_chan = ddmt_multi.execute(chan_major, nsamps)
        out_time = ddmt_multi.execute_time_major(time_major, nsamps)
        np.testing.assert_array_equal(out_time, out_chan)
        assert out_chan.shape[0] == nbeams

    def test_multi_beam_streaming(self) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 8
        nsamps = 128
        nbeams = 2
        tsamp = 0.001
        dms = np.array([0.0, 3.0, 6.0], dtype=np.float32)

        rng = np.random.default_rng(999)
        waterfall = rng.standard_normal((nbeams, nchans, nsamps)).astype(np.float32)

        mono = DDMTCPU(f_min, f_max, nchans, tsamp, dms, nbeams=nbeams)
        out_expected = mono.execute(waterfall)

        streamed = DDMTCPU(f_min, f_max, nchans, tsamp, dms, nbeams=nbeams)
        split = 50
        out1 = streamed.execute(waterfall[:, :, :split])
        out2 = streamed.execute(waterfall[:, :, split:])

        out_cat = np.concatenate([out1, out2], axis=2)
        np.testing.assert_allclose(out_cat, out_expected, rtol=1e-5, atol=1e-5)

        # Multi-beam save / load history
        hist = streamed.save_history()
        assert len(hist) == streamed.history_state_size()
        streamed.reset_history()
        streamed.load_history(hist)
        out3 = streamed.execute(waterfall[:, :, :30])
        assert out3.shape[2] == 30

    @pytest.mark.parametrize("nbits", [1, 2, 4, 8, 16])
    def test_packed_streaming(self, nbits: int) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 8
        nsamps = 120
        split = 67  # unaligned sub-byte boundary
        tsamp = 0.001
        dm_arr = np.array([0.0, 5.0, 10.0], dtype=np.float32)

        plan = DDMTPlan(f_min, f_max, nchans, tsamp, dm_arr, nbits=nbits)
        mono = DDMTCPU(plan)

        mask = (1 << nbits) - 1
        raw = (np.arange(nchans * nsamps, dtype=np.uint64) * 31 + 7) & mask
        raw = raw.reshape((nchans, nsamps))

        # Helper to pack 2D array (nchans, count)
        def pack_data(arr: np.ndarray) -> np.ndarray:
            c_cnt, s_cnt = arr.shape
            r_bytes = (s_cnt * nbits + 7) // 8 if nbits < 8 else s_cnt * (nbits // 8)
            buf = np.zeros((c_cnt, r_bytes), dtype=np.uint8)
            if nbits == 16:
                buf.view(np.uint16)[:] = arr.astype(np.uint16)
            elif nbits == 8:
                buf[:] = arr.astype(np.uint8)
            else:
                for c in range(c_cnt):
                    for s in range(s_cnt):
                        val = int(arr[c, s])
                        bit_off = s * nbits
                        byte_idx = bit_off // 8
                        bit_sub = bit_off % 8
                        buf[c, byte_idx] |= (val & mask) << bit_sub
            return buf

        mono_packed = pack_data(raw)
        out_mono = mono.execute(mono_packed, nsamps)

        streamed = DDMTCPU(plan)
        c1_packed = pack_data(raw[:, :split])
        c2_packed = pack_data(raw[:, split:])

        out1 = streamed.execute(c1_packed, split)
        out2 = streamed.execute(c2_packed, nsamps - split)

        out_streamed = np.concatenate([out1, out2], axis=1)
        np.testing.assert_array_equal(out_streamed, out_mono)

    def test_packed_history_save_load(self) -> None:
        f_min = 1000.0
        f_max = 1500.0
        nchans = 4
        nbits = 2
        tsamp = 0.001
        dms = np.array([0.0, 5.0], dtype=np.float32)

        plan = DDMTPlan(f_min, f_max, nchans, tsamp, dms, nbits=nbits)
        ddmt1 = DDMTCPU(plan)
        state_sz = ddmt1.history_state_size()

        # Before warm-up, save_history raises runtime_error
        with pytest.raises(RuntimeError):
            ddmt1.save_history()

        chunk1_samps = 64
        mask = (1 << nbits) - 1
        raw1 = ((np.arange(nchans * chunk1_samps, dtype=np.uint64) * 11 + 3) & mask).reshape((nchans, chunk1_samps))
        r_bytes1 = (chunk1_samps * nbits + 7) // 8
        c1 = np.zeros((nchans, r_bytes1), dtype=np.uint8)
        for c in range(nchans):
            for s in range(chunk1_samps):
                val = int(raw1[c, s])
                bit_off = s * nbits
                c1[c, bit_off // 8] |= (val & mask) << (bit_off % 8)

        ddmt1.execute(c1, chunk1_samps)

        # Save history
        hist = ddmt1.save_history()
        assert hist.dtype == np.uint8
        assert hist.size == state_sz

        # Load into fresh ddmt2
        ddmt2 = DDMTCPU(plan)
        assert ddmt2.get_output_nsamps(32) < 32  # cold
        ddmt2.load_history(hist)
        assert ddmt2.get_output_nsamps(32) == 32  # warm

        # Feed chunk2 to both
        chunk2_samps = 48
        raw2 = ((np.arange(nchans * chunk2_samps, dtype=np.uint64) * 19 + 5) & mask).reshape((nchans, chunk2_samps))
        r_bytes2 = (chunk2_samps * nbits + 7) // 8
        c2 = np.zeros((nchans, r_bytes2), dtype=np.uint8)
        for c in range(nchans):
            for s in range(chunk2_samps):
                val = int(raw2[c, s])
                bit_off = s * nbits
                c2[c, bit_off // 8] |= (val & mask) << (bit_off % 8)

        out1 = ddmt1.execute(c2, chunk2_samps)
        out2 = ddmt2.execute(c2, chunk2_samps)
        np.testing.assert_array_equal(out1, out2)

        # reset_history clears state
        ddmt2.reset_history()
        assert ddmt2.get_output_nsamps(32) < 32


