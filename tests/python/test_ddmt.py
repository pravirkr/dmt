import numpy as np
from dmtlib import libdmt


class TestDDMT:
    def test_execute_ones(self) -> None:
        nchans = 8
        nsamps = 64
        ddmt = libdmt.DDMTCPU(1000.0, 1500.0, nchans, 0.001, 10.0, 5.0, 0.0)
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
        ddmt = libdmt.DDMTCPU(1000.0, 1500.0, nchans, 0.001, dm_arr)
        waterfall = np.ones((nchans, nsamps), dtype=np.float32)
        dmt = ddmt.execute(waterfall)
        assert dmt.shape[0] == 2
