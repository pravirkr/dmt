import numpy as np
import pytest
from dmtlib import FDMTCPU, FDMTFFTCPU, CohFDMTCPU


@pytest.mark.cuda
def test_fdmt_gpu_matches_cpu() -> None:
    from dmtlib import FDMTGPU

    rng = np.random.default_rng(0)
    waterfall = rng.standard_normal((16, 64), dtype=np.float32)
    cpu = FDMTCPU(1000.0, 1500.0, 16, 64, 0.001, 8, mode="roll")
    gpu = FDMTGPU(1000.0, 1500.0, 16, 64, 0.001, 8, mode="roll")
    np.testing.assert_allclose(
        gpu.execute(waterfall), cpu.execute(waterfall), rtol=2e-3, atol=2e-3
    )


@pytest.mark.cuda
def test_fdmt_fft_gpu_matches_cpu() -> None:
    from dmtlib import FDMTFFTGPU

    rng = np.random.default_rng(1)
    waterfall = rng.standard_normal((16, 64), dtype=np.float32)
    cpu = FDMTFFTCPU(1000.0, 1500.0, 16, 64, 0.001, 8, mode="roll")
    gpu = FDMTFFTGPU(1000.0, 1500.0, 16, 64, 0.001, 8, mode="roll")
    np.testing.assert_allclose(
        gpu.execute(waterfall), cpu.execute(waterfall), rtol=2e-3, atol=2e-3
    )


@pytest.mark.cuda
def test_coh_fdmt_gpu_matches_cpu() -> None:
    from dmtlib import CohFDMTGPU

    f_center = 1250.0
    bw_sub = 25.0
    nsub = 4
    tbin = 1.0e-6
    nbin = 1 << 10
    nfft = 2
    t_p = tbin * 4
    dm_max = 5.0
    noverlap = 32
    cpu = CohFDMTCPU(
        f_center, bw_sub, nsub, tbin, nbin, nfft, t_p, dm_max, 0.0, noverlap
    )
    gpu = CohFDMTGPU(
        f_center, bw_sub, nsub, tbin, nbin, nfft, t_p, dm_max, 0.0, noverlap
    )
    in_size = 2 * 2 * cpu.plan.nsamp * nsub
    rng = np.random.default_rng(2)
    data = rng.integers(0, 256, size=in_size, dtype=np.uint8)
    np.testing.assert_allclose(
        gpu.execute(data), cpu.execute(data), rtol=2e-2, atol=2e-2
    )
