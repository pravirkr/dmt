import numpy as np
from dmt import libdmt


class TestFDMT:
    def test_initialise_ones(self) -> None:
        nchans = 500
        nsamples = 1024
        dt_max = 512
        thefdmt = libdmt.FDMT(1000, 1500, nchans, nsamples, 0.001, dt_max)
        waterfall = np.ones((nchans, nsamples), dtype=np.float32)
        thefdmt_init = thefdmt.initialise(waterfall)
        np.testing.assert_equal(
            thefdmt_init.shape,
            (nchans, thefdmt.dt_grid_init.size, nsamples),
        )
        """
        np.testing.assert_equal(
            thefdmt_init,
            np.ones((nchans, thefdmt.dt_grid_init.size, nsamples), dtype=np.float32),
        )
        """
