import dmtlib
import numpy as np
import pytest

from dmtlib import DDMT, DDMTFFT, FDMT, FDMTFFT, SDMT, CohFDMT

# Every test runs on the GPU backend of this build (the `gpu_backend`
# parameter of conftest.py: "cuda" or "hip") and compares with the CPU.


def on(backend: str, cls: type, *args: object, **kwargs: object) -> object:
    return cls(*args, backend=backend, **kwargs)


def test_fdmt_gpu_matches_cpu(gpu_backend: str) -> None:
    rng = np.random.default_rng(0)
    waterfall = rng.standard_normal((16, 64), dtype=np.float32)
    cpu = FDMT(1000.0, 1500.0, 16, 64, 0.001, 8, mode="roll")
    gpu = on(gpu_backend, FDMT, 1000.0, 1500.0, 16, 64, 0.001, 8, mode="roll")
    np.testing.assert_allclose(
        gpu.execute(waterfall), cpu.execute(waterfall), rtol=2e-3, atol=2e-3
    )


def test_fdmt_fft_gpu_matches_cpu(gpu_backend: str) -> None:
    rng = np.random.default_rng(1)
    waterfall = rng.standard_normal((16, 64), dtype=np.float32)
    cpu = FDMTFFT(1000.0, 1500.0, 16, 64, 0.001, 8, mode="roll")
    gpu = on(gpu_backend, FDMTFFT, 1000.0, 1500.0, 16, 64, 0.001, 8, mode="roll")
    np.testing.assert_allclose(
        gpu.execute(waterfall), cpu.execute(waterfall), rtol=2e-3, atol=2e-3
    )


def test_coh_fdmt_gpu_matches_cpu(gpu_backend: str) -> None:
    cfg = dmtlib.CohFDMTConfig(400.0, 1.0, 16, 4.0e-6, 10.0, 11.0)
    cpu = CohFDMT(cfg, 4)
    gpu = on(gpu_backend, CohFDMT, cfg)
    rng = np.random.default_rng(2)
    data = rng.normal(0.0, 8.0, size=cpu.input_size()).astype(np.int8)
    want = cpu.execute(data)
    got = gpu.execute(data)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-4 * np.abs(want).max())


@pytest.mark.parametrize("nbits", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("int_tree", [False, True])
def test_fdmt_gpu_packed_matches_cpu(
    nbits: int, int_tree: bool, gpu_backend: str
) -> None:
    # Integer-valued input: every partial sum is exact, so GPU and CPU agree
    # bit-for-bit (fast-math reassociation cannot change exact sums).
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
    cpu = FDMT(1000.0, 1500.0, nchans, nsamps, 0.001, 40)
    gpu = on(
        gpu_backend, FDMT, 1000.0, 1500.0, nchans, nsamps, 0.001, 40, int_tree=int_tree
    )
    ref = cpu.execute(values.astype(np.float32))
    np.testing.assert_array_equal(gpu.execute(packed, nbits), ref)


@pytest.mark.parametrize("mode", ["full", "roll", "valid"])
def test_fdmt_gpu_fused_matches_unfused(mode: str, gpu_backend: str) -> None:
    # Level fusion repeats the unfused kernels' float additions exactly.
    nchans, nsamps = 256, 700
    rng = np.random.default_rng(4)
    data = rng.standard_normal((nchans, nsamps), dtype=np.float32)
    ref = on(
        gpu_backend,
        FDMT,
        1000.0,
        1500.0,
        nchans,
        nsamps,
        0.001,
        64,
        mode=mode,
        fuse_levels=0,
    )
    assert ref.fuse_levels == 0
    expected = ref.execute(data)
    for fuse in (1, 3, None):
        gpu = on(
            gpu_backend,
            FDMT,
            1000.0,
            1500.0,
            nchans,
            nsamps,
            0.001,
            64,
            mode=mode,
            fuse_levels=fuse,
        )
        assert 0 <= gpu.fuse_levels <= gpu.plan.niters
        assert gpu.memory_usage.total > 0
        np.testing.assert_array_equal(gpu.execute(data), expected)


@pytest.mark.parametrize("packed", [False, True])
def test_fdmt_gpu_stepper_matches_cpu(packed: bool, gpu_backend: str) -> None:
    nchans, nsamps = 32, 128
    rng = np.random.default_rng(5)
    values = rng.integers(0, 4, size=(nchans, nsamps), dtype=np.uint8)
    # 2-bit LSB-first packing (4 samples per byte), matching the engines.
    packed_wf = np.zeros((nchans, nsamps // 4), dtype=np.uint8)
    for k in range(4):
        packed_wf |= values[:, k::4] << (2 * k)
    waterfall = values.astype(np.float32)

    cpu = FDMT(1000.0, 1500.0, nchans, nsamps, 0.001, 32, int_tree=False)
    gpu = on(
        gpu_backend, FDMT, 1000.0, 1500.0, nchans, nsamps, 0.001, 32, int_tree=False
    )
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
                gpu.view_subband_data(s),
                cpu.view_subband_data(s),
                rtol=1e-5,
                atol=1e-4,
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


def test_fdmt_gpu_execute_out_buffer(gpu_backend: str) -> None:
    rng = np.random.default_rng(6)
    waterfall = rng.standard_normal((2, 16, 64), dtype=np.float32)
    gpu = on(gpu_backend, FDMT, 1000.0, 1500.0, 16, 64, 0.001, 8, nbeams=2)
    cpu = FDMT(1000.0, 1500.0, 16, 64, 0.001, 8, nbeams=2)
    out = np.empty(2 * gpu.plan.buffer_size, dtype=np.float32)
    got = gpu.execute(waterfall, out=out)
    assert np.shares_memory(got, out)
    np.testing.assert_allclose(got, cpu.execute(waterfall), rtol=2e-3, atol=2e-3)


def test_ddmt_gpu_matches_cpu(gpu_backend: str) -> None:
    nchans, nsamps = 64, 1024
    rng = np.random.default_rng(7)
    waterfall = rng.standard_normal((nchans, nsamps), dtype=np.float32)
    cpu = DDMT(1000.0, 1500.0, nchans, 0.001, 50.0, 1.0)
    gpu = on(gpu_backend, DDMT, 1000.0, 1500.0, nchans, 0.001, 50.0, 1.0)
    np.testing.assert_allclose(
        gpu.execute(waterfall), cpu.execute(waterfall), rtol=1e-5, atol=1e-4
    )
    # Second block exercises the carried history on both backends.
    np.testing.assert_allclose(
        gpu.execute(waterfall[:, ::-1].copy()),
        cpu.execute(waterfall[:, ::-1].copy()),
        rtol=1e-5,
        atol=1e-4,
    )


def test_ddmt_gpu_packed_matches_cpu_exactly(gpu_backend: str) -> None:
    nchans, nsamps = 64, 1024
    rng = np.random.default_rng(8)
    packed = rng.integers(0, 256, size=(nchans, nsamps), dtype=np.uint8)
    cpu = DDMT(1000.0, 1500.0, nchans, 0.001, 50.0, 1.0, nbits=8)
    gpu = on(gpu_backend, DDMT, 1000.0, 1500.0, nchans, 0.001, 50.0, 1.0, nbits=8)
    # Integer accumulation: bit-identical.
    np.testing.assert_array_equal(
        gpu.execute(packed, nsamps), cpu.execute(packed, nsamps)
    )


# SDMT: a dense grid over many channels, so the GPU runs its shared-sum
# kernel (the DDMT kernel otherwise; the results are the same either way).
SDMT_ARGS = (1000.0, 1500.0, 256, 0.001, (0.4 * np.arange(400)).astype(np.float32))


def test_sdmt_gpu_matches_cpu(gpu_backend: str) -> None:
    nsamps = 1500
    rng = np.random.default_rng(9)
    waterfall = rng.standard_normal((256, nsamps), dtype=np.float32)
    cpu = DDMT(*SDMT_ARGS)
    gpu = on(gpu_backend, SDMT, *SDMT_ARGS)
    assert gpu.backend == gpu_backend
    for block in (waterfall, waterfall[:, ::-1].copy()):
        np.testing.assert_allclose(
            gpu.execute(block), cpu.execute(block), rtol=1e-5, atol=1e-3
        )


@pytest.mark.parametrize("nbits", [2, 8, 16])
def test_sdmt_gpu_packed_matches_ddmt_exactly(nbits: int, gpu_backend: str) -> None:
    nsamps = 1600
    rng = np.random.default_rng(10 + nbits)
    row_bytes = (nsamps * nbits + 7) // 8
    packed = rng.integers(0, 256, size=(256, row_bytes), dtype=np.uint8)
    cpu = DDMT(*SDMT_ARGS, nbits=nbits)
    gpu = on(gpu_backend, SDMT, *SDMT_ARGS, nbits=nbits)
    # Integer sums: bit-identical to brute force, also when streaming.
    for part in (packed[:, : row_bytes // 2], packed[:, row_bytes // 2 :]):
        n = part.shape[1] * 8 // nbits
        part = np.ascontiguousarray(part)
        np.testing.assert_array_equal(gpu.execute(part, n), cpu.execute(part, n))


@pytest.mark.parametrize("mode", ["valid", "full", "roll"])
@pytest.mark.parametrize("frac", [True, False])
def test_fdmt_fft_gpu_modes_match_cpu(mode: str, frac: bool, gpu_backend: str) -> None:
    rng = np.random.default_rng(11)
    args = (1100.0, 1500.0, 64, 1024, 0.001, 48)
    cpu = FDMTFFT(*args, mode=mode, fractional_delays=frac)
    gpu = on(gpu_backend, FDMTFFT, *args, mode=mode, fractional_delays=frac)
    for _ in range(2):
        waterfall = rng.standard_normal((64, 1024), dtype=np.float32)
        a = cpu.execute(waterfall)
        b = gpu.execute(waterfall)
        np.testing.assert_allclose(b, a, rtol=0, atol=2e-5 * np.abs(a).max())


def test_fdmt_fft_gpu_packed_kill_mask_history(gpu_backend: str) -> None:
    rng = np.random.default_rng(12)
    args = (1100.0, 1500.0, 64, 1024, 0.001, 48)
    mask = np.ones(64, dtype=np.uint8)
    mask[[1, 30]] = 0
    cpu = FDMTFFT(*args, kill_mask=mask)
    gpu = on(gpu_backend, FDMTFFT, *args, kill_mask=mask)
    packed = rng.integers(0, 256, size=(64, 1024), dtype=np.uint8)
    for _ in range(2):
        a = cpu.execute(packed, 8)
        b = gpu.execute(packed, 8)
        np.testing.assert_allclose(b, a, rtol=0, atol=2e-5 * np.abs(a).max())
    np.testing.assert_array_equal(gpu.save_history(), cpu.save_history())


@pytest.mark.parametrize("method", ["brute", "nufft"])
def test_ddmt_fft_gpu_matches_cpu(method: str, gpu_backend: str) -> None:
    rng = np.random.default_rng(13)
    args = (1100.0, 1500.0, 64, 0.001, 30.0, 0.25)
    cpu = DDMTFFT(*args, nbeams=2, method=method)
    gpu = on(gpu_backend, DDMTFFT, *args, nbeams=2, method=method)
    assert gpu.method_used == cpu.method_used
    for _ in range(3):
        waterfall = rng.standard_normal((2, 64, 2000), dtype=np.float32)
        a = cpu.execute(waterfall)
        b = gpu.execute(waterfall)
        np.testing.assert_allclose(b, a, rtol=0, atol=2e-5 * np.abs(a).max())
