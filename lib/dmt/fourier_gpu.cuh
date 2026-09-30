#pragma once

// Helpers shared by the Fourier-domain GPU engines (FDMTFFT, DDMTFFT):
// grow-only device buffers, exact fixed-point phases and a few kernels.
//
// Phases: a delay of `shift` samples in an N-point transform turns bin k by
// shift * k / N turns. phase_fixed() stores (-shift / N mod 1) as a 0.64
// fixed-point number s; phasor(s, k) then multiplies by k modulo 2^64
// (exact reduction, no FP64) and evaluates exp(2 pi i t) from the top 32
// bits, t in [-1/2, 1/2): W^(shift k) = exp(-2 pi i shift k / N) to ~2^-32
// turns.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>

#include "dmt/gpu_compat.cuh"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/types.hpp"
#include "dmt/gpu_utils.cuh"

namespace dmt::fourier_gpu {

inline constexpr int kBlock         = 256;
inline constexpr unsigned kMaxGridY = 65535;

/// Grow-only raw device buffer: no default-stream work (unlike
/// thrust::device_vector), contents undefined after a growth.
template <typename T> class DevBuf {
public:
    DevBuf() = default;
    ~DevBuf() { release(); }
    DevBuf(const DevBuf&)            = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    DevBuf(DevBuf&& o) noexcept : m_ptr(o.m_ptr), m_cap(o.m_cap) {
        o.m_ptr = nullptr;
        o.m_cap = 0;
    }
    DevBuf& operator=(DevBuf&& o) noexcept {
        if (this != &o) {
            release();
            m_ptr   = o.m_ptr;
            m_cap   = o.m_cap;
            o.m_ptr = nullptr;
            o.m_cap = 0;
        }
        return *this;
    }

    void reserve(SizeType n) {
        if (n <= m_cap) {
            return;
        }
        release();
        gpu_utils::check_gpu_call(
            cudaMalloc(reinterpret_cast<void**>(&m_ptr),
                       std::max<SizeType>(n, 1) * sizeof(T)),
            "device allocation failed");
        m_cap = n;
    }
    /// Synchronous upload (set-up only).
    void upload(std::span<const T> h) {
        reserve(h.size());
        if (!h.empty()) {
            gpu_utils::check_gpu_call(cudaMemcpy(m_ptr, h.data(),
                                                 h.size_bytes(),
                                                 cudaMemcpyHostToDevice),
                                      "device upload failed");
        }
    }
    [[nodiscard]] T* data() const noexcept { return m_ptr; }
    [[nodiscard]] SizeType capacity() const noexcept { return m_cap; }
    void release() noexcept {
        if (m_ptr != nullptr) {
            cudaFree(m_ptr);
            m_ptr = nullptr;
            m_cap = 0;
        }
    }

private:
    T* m_ptr{nullptr};
    SizeType m_cap{0};
};

/// -shift / n turns per bin modulo 1, in 0.64 fixed point (see file
/// comment). Exact for the integer part of shift.
inline unsigned long long phase_fixed(double shift, SizeType n) {
    const double fl    = std::floor(shift);
    const auto nn      = static_cast<long long>(n);
    const long long ni = ((static_cast<long long>(fl) % nn) + nn) % nn;
    const auto q       = static_cast<unsigned long long>(
        (static_cast<unsigned __int128>(ni) << 64U) / n);
    const double f = shift - fl; // [0, 1)
    const auto qf  = static_cast<unsigned long long>(
        std::ldexp(f / static_cast<double>(n), 64));
    return 0ULL - (q + qf);
}

inline unsigned blocks_for(int64_t n, int block = kBlock) {
    return static_cast<unsigned>(std::max<int64_t>(1, (n + block - 1) / block));
}

/// Grid y extent for n rows: kernels loop rows with stride gridDim.y.
inline unsigned grid_y(int64_t rows) {
    return static_cast<unsigned>(
        std::clamp<int64_t>(rows, 1, static_cast<int64_t>(kMaxGridY)));
}

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return {fmaf(a.x, b.x, -a.y * b.y), fmaf(a.x, b.y, a.y * b.x)};
}

/// t + h * w
__device__ __forceinline__ float2 cfma(float2 t, float2 h, float2 w) {
    return {fmaf(h.x, w.x, fmaf(-h.y, w.y, t.x)),
            fmaf(h.x, w.y, fmaf(h.y, w.x, t.y))};
}

/// Turn in [-1/2, 1/2) of frac(s * k / 2^64).
__device__ __forceinline__ float fixed_turn(unsigned long long s,
                                            unsigned long long k) {
    const unsigned long long p = s * k;
    return static_cast<float>(static_cast<int>(p >> 32U)) *
           2.3283064365386963e-10F;
}

/// exp(2 pi i frac(s * k / 2^64)), accurate (sincospi).
__device__ __forceinline__ float2 phasor(unsigned long long s,
                                         unsigned long long k) {
    float sn = 0.0F;
    float cs = 0.0F;
    sincospif(2.0F * fixed_turn(s, k), &sn, &cs);
    return {cs, sn};
}

/// As phasor(), through the special-function unit (~4e-7 absolute on the
/// reduced angle).
__device__ __forceinline__ float2 phasor_fast(unsigned long long s,
                                              unsigned long long k) {
    float sn = 0.0F;
    float cs = 0.0F;
    __sincosf(6.28318530717958647692F * fixed_turn(s, k), &sn, &cs);
    return {cs, sn};
}

/// A real signal's spectrum has real DC and (even N) Nyquist bins. Phase
/// ramps of fractional shifts leave an imaginary part there, which FFTW's
/// C2R ignores but cuFFT's does not: drop it so every backend inverts the
/// same Hermitian spectrum.
template <typename T>
__global__ void kernel_hermitian_edges(T* __restrict__ spec,
                                       int64_t nrows,
                                       int64_t row_dist,
                                       int64_t n_bins,
                                       int even) {
    const auto row =
        (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (row >= nrows) {
        return;
    }
    auto* r = reinterpret_cast<float2*>(spec + (row * row_dist));
    r[0].y  = 0.0F;
    if (even != 0) {
        r[n_bins - 1].y = 0.0F;
    }
}

template <typename T>
void hermitian_edges(T* spec,
                     SizeType nrows,
                     SizeType row_dist,
                     SizeType n_bins,
                     SizeType n_fft,
                     cudaStream_t s) {
    if (nrows == 0) {
        return;
    }
    kernel_hermitian_edges<T>
        <<<blocks_for(static_cast<int64_t>(nrows)), kBlock, 0, s>>>(
            spec, static_cast<int64_t>(nrows), static_cast<int64_t>(row_dist),
            static_cast<int64_t>(n_bins), (n_fft % 2 == 0) ? 1 : 0);
}

/// Time-major packed (nbeams, nsamps, nchans) -> channel-major float
/// (nbeams, nchans, nsamps), through a shared-memory tile (coalesced both
/// ways). Launch: block (32, 8), grid (ceil(nsamps / 32), ceil(nchans /
/// 32), nbeams).
template <int NB>
__global__ void kernel_unpack_time_major(const uint8_t* __restrict__ packed,
                                         float* __restrict__ out,
                                         int nchans,
                                         int64_t nsamps,
                                         int64_t samp_bytes) {
    __shared__ float tile[32][33];
    const auto beam = static_cast<int64_t>(blockIdx.z);
    const auto t0   = static_cast<int64_t>(blockIdx.x) * 32;
    const int c0    = static_cast<int>(blockIdx.y) * 32;
    for (auto r = static_cast<int>(threadIdx.y); r < 32;
         r += static_cast<int>(blockDim.y)) {
        const auto t = t0 + r;
        const int c  = c0 + static_cast<int>(threadIdx.x);
        if (t < nsamps && c < nchans) {
            const auto* row = packed + (((beam * nsamps) + t) * samp_bytes);
            tile[r][threadIdx.x] =
                static_cast<float>(bit_pack_utils::read_packed_sample<NB>(
                    row, static_cast<SizeType>(c)));
        }
    }
    __syncthreads();
    for (auto r = static_cast<int>(threadIdx.y); r < 32;
         r += static_cast<int>(blockDim.y)) {
        const int c  = c0 + r;
        const auto t = t0 + static_cast<int64_t>(threadIdx.x);
        if (t < nsamps && c < nchans) {
            out[(((beam * nchans) + c) * nsamps) + t] = tile[threadIdx.x][r];
        }
    }
}

template <int NB>
void launch_unpack_time_major(const uint8_t* packed,
                              float* out,
                              SizeType nbeams,
                              SizeType nchans,
                              SizeType nsamps,
                              cudaStream_t s) {
    const auto samp_bytes = static_cast<int64_t>(
        bit_pack_utils::packed_row_bytes(nchans, static_cast<SizeType>(NB)));
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned>((nsamps + 31) / 32),
                    static_cast<unsigned>((nchans + 31) / 32),
                    static_cast<unsigned>(nbeams));
    gpu_utils::check_kernel_launch_params(grid, block);
    kernel_unpack_time_major<NB>
        <<<grid, block, 0, s>>>(packed, out, static_cast<int>(nchans),
                                static_cast<int64_t>(nsamps), samp_bytes);
}

/// Time-major packed input at `nbits` -> channel-major float rows.
inline void unpack_time_major(const uint8_t* packed,
                              float* out,
                              SizeType nbits,
                              SizeType nbeams,
                              SizeType nchans,
                              SizeType nsamps,
                              cudaStream_t s) {
    switch (nbits) {
    case 1:
        launch_unpack_time_major<1>(packed, out, nbeams, nchans, nsamps, s);
        break;
    case 2:
        launch_unpack_time_major<2>(packed, out, nbeams, nchans, nsamps, s);
        break;
    case 4:
        launch_unpack_time_major<4>(packed, out, nbeams, nchans, nsamps, s);
        break;
    case 8:
        launch_unpack_time_major<8>(packed, out, nbeams, nchans, nsamps, s);
        break;
    default:
        launch_unpack_time_major<16>(packed, out, nbeams, nchans, nsamps, s);
        break;
    }
}

} // namespace dmt::fourier_gpu
