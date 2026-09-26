import numpy as np
import pytest

from dmtlib import FDMTCPU, FDMTFFTCPU, CohFDMTCPU


@pytest.mark.cuda
def test_fdmt_gpu_matches_cpu() -> None:
    from dmtlib import FDMTCUDA

    rng = np.random.default_rng(0)
    waterfall = rng.standard_normal((16, 64), dtype=np.float32)
    cpu = FDMTCPU(1000.0, 1500.0, 16, 64, 0.001, 8, mode="roll")
    gpu = FDMTCUDA(1000.0, 1500.0, 16, 64, 0.001, 8, mode="roll")
    np.testing.assert_allclose(
        gpu.execute(waterfall), cpu.execute(waterfall), rtol=2e-3, atol=2e-3
    )


@pytest.mark.cuda
def test_fdmt_fft_gpu_matches_cpu() -> None:
    from dmtlib import FDMTFFTCUDA

    rng = np.random.default_rng(1)
    waterfall = rng.standard_normal((16, 64), dtype=np.float32)
    cpu = FDMTFFTCPU(1000.0, 1500.0, 16, 64, 0.001, 8, mode="roll")
    gpu = FDMTFFTCUDA(1000.0, 1500.0, 16, 64, 0.001, 8, mode="roll")
    np.testing.assert_allclose(
        gpu.execute(waterfall), cpu.execute(waterfall), rtol=2e-3, atol=2e-3
    )


@pytest.mark.cuda
def test_coh_fdmt_gpu_matches_cpu() -> None:
    from dmtlib import CohFDMTCUDA

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
    gpu = CohFDMTCUDA(
        f_center, bw_sub, nsub, tbin, nbin, nfft, t_p, dm_max, 0.0, noverlap
    )
    in_size = 2 * 2 * cpu.plan.nsamp * nsub
    rng = np.random.default_rng(2)
    data = rng.integers(0, 256, size=in_size, dtype=np.uint8)
    np.testing.assert_allclose(
        gpu.execute(data), cpu.execute(data), rtol=2e-2, atol=2e-2
    )


@pytest.mark.cuda
@pytest.mark.parametrize("nbits", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("int_tree", [False, True])
def test_fdmt_gpu_packed_matches_cpu(nbits: int, int_tree: bool) -> None:
    # Integer-valued input: every partial sum is exact, so GPU and CPU agree
    # bit-for-bit (fast-math reassociation cannot change exact sums).
    from dmtlib import FDMTCUDA

    nchans, nsamps = 64, 203
    rng = np.random.default_rng(nbits)
    values = rng.integers(0, 2**nbits, size=(nchans, nsamps), dtype=np.uint32)
    if nbits >= 8:
        packed = values.astype(np.uint8 if nbits == 8 else "<u2").view(np.uint8)
        packed = packed.reshape(nchans, -1)
    else:
        per_byte = 8 // nbits
        row_bytes = (nsamps * nbits + 7) // 8
        padded = np.zeros((nchans, row_bytes * per_byte), dtype=np.uint32)
        padded[:, :nsamps] = values
        shifts = (np.arange(per_byte, dtype=np.uint32) * nbits)[None, None, :]
        packed = (
            (padded.reshape(nchans, row_bytes, per_byte) << shifts)
            .sum(axis=2)
            .astype(np.uint8)
        )
    cpu = FDMTCPU(1000.0, 1500.0, nchans, nsamps, 0.001, 40)
    gpu = FDMTCUDA(1000.0, 1500.0, nchans, nsamps, 0.001, 40, int_tree=int_tree)
    ref = cpu.execute(values.astype(np.float32))
    np.testing.assert_array_equal(gpu.execute(packed, nbits), ref)


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["full", "roll", "valid"])
def test_fdmt_gpu_fused_matches_unfused(mode: str) -> None:
    # Level fusion repeats the unfused kernels' float additions exactly.
    from dmtlib import FDMTCUDA

    nchans, nsamps = 256, 700
    rng = np.random.default_rng(4)
    data = rng.standard_normal((nchans, nsamps), dtype=np.float32)
    ref = FDMTCUDA(1000.0, 1500.0, nchans, nsamps, 0.001, 64, mode=mode, fuse_levels=0)
    assert ref.fuse_levels == 0
    expected = ref.execute(data)
    for fuse in (1, 3, None):
        gpu = FDMTCUDA(
            1000.0, 1500.0, nchans, nsamps, 0.001, 64, mode=mode, fuse_levels=fuse
        )
        assert 0 <= gpu.fuse_levels <= gpu.plan.niters
        assert gpu.memory_usage.total > 0
        np.testing.assert_array_equal(gpu.execute(data), expected)


@pytest.mark.cuda
@pytest.mark.parametrize("packed", [False, True])
def test_fdmt_gpu_stepper_matches_cpu(packed: bool) -> None:
    from dmtlib import FDMTCUDA

    nchans, nsamps = 32, 128
    rng = np.random.default_rng(5)
    values = rng.integers(0, 4, size=(nchans, nsamps), dtype=np.uint8)
    # 2-bit LSB-first packing (4 samples per byte), matching the engines.
    packed_wf = np.zeros((nchans, nsamps // 4), dtype=np.uint8)
    for k in range(4):
        packed_wf |= values[:, k::4] << (2 * k)
    waterfall = values.astype(np.float32)

    cpu = FDMTCPU(1000.0, 1500.0, nchans, nsamps, 0.001, 32, int_tree=False)
    gpu = FDMTCUDA(1000.0, 1500.0, nchans, nsamps, 0.001, 32, int_tree=False)
    if packed:
        cpu.reset(packed_wf, 2)
        gpu.reset(packed_wf, 2)
    else:
        cpu.reset(waterfall)
        gpu.reset(waterfall)
    while not cpu.is_finished:
        assert gpu.current_level == cpu.current_level
        assert gpu.num_subbands == cpu.num_subbands
        for s in range(cpu.num_subbands):
            np.testing.assert_allclose(
                gpu.view_subband_data(s), cpu.view_subband_data(s),
                rtol=1e-5, atol=1e-4,
            )
        cpu.advance()
        gpu.advance()
    np.testing.assert_allclose(gpu.finalize(), cpu.finalize(), rtol=1e-5, atol=1e-4)

    # Same streaming hard fail as the CPU.
    gpu.reset(waterfall)
    gpu.advance_until_remaining(2)
    with pytest.raises(RuntimeError):
        gpu.reset(waterfall)
    gpu.reset_history()
    np.testing.assert_allclose(
        gpu.get_effective_sigma_grid(), cpu.get_effective_sigma_grid()
    )


@pytest.mark.cuda
def test_fdmt_gpu_execute_out_buffer() -> None:
    from dmtlib import FDMTCUDA

    rng = np.random.default_rng(6)
    waterfall = rng.standard_normal((2, 16, 64), dtype=np.float32)
    gpu = FDMTCUDA(1000.0, 1500.0, 16, 64, 0.001, 8, nbeams=2)
    cpu = FDMTCPU(1000.0, 1500.0, 16, 64, 0.001, 8, nbeams=2)
    out = np.empty(2 * gpu.plan.buffer_size, dtype=np.float32)
    got = gpu.execute(waterfall, out=out)
    assert np.shares_memory(got, out)
    np.testing.assert_allclose(got, cpu.execute(waterfall), rtol=2e-3, atol=2e-3)
