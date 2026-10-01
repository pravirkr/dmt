#pragma once

// Fused coherent stage of the CohFDMT GPU engine: for one coarse trial, each
// thread block takes one channel and a run of its FFT blocks and, per FFT
// block, gathers both polarisations' bins times the trial's chirp, inverse
// transforms them in shared memory, detects, normalises, trims the overlap
// and writes the delay-aligned waterfall row: one read of the spectrum and
// one write of the waterfall per trial (the cuFFT path takes three passes
// through a work buffer). Same maths and layout as the CPU engine's
// coherent_trial() (lib/cpu/cfdmt_cpu.cpp).
//
// The inverse FFT is an in-place radix-4 (+ one radix-2 stage for odd
// log2 N) decimation-in-frequency transform: natural-order input,
// bit-reversed output, twiddles exp(+2 pi i k / N) from a double-accurate
// table (the unnormalised backward transform of cuFFT/FFTW). Shared memory
// only, no warp-size assumptions (HIP-portable).

#include <cstdint>

#include "dmt/gpu_compat.cuh"

#include "dmt/fourier_gpu.cuh"

namespace dmt::cfdmt_gpu {

/// Threads per block of the fused kernel.
inline constexpr int kFusedThreads = 256;
/// FFT lengths the fused kernel is instantiated for.
inline constexpr int kFusedMinLog2 = 4;  // 16
inline constexpr int kFusedMaxLog2 = 12; // 4096
/// Points per polarisation held in shared memory per round (short
/// transforms process several FFT blocks at once).
inline constexpr int kFusedRoundPoints = 1024;

/// Padded shared-memory slot of point p: one spare slot every 32 spreads
/// the bit-reversed read-out over the banks.
__host__ __device__ constexpr uint32_t fused_pad(uint32_t p) {
    return p + (p >> 5U);
}

/// FFT blocks a thread block holds at once for length 2^log2n.
__host__ __device__ constexpr int fused_round_blocks(int log2n) {
    return (1 << log2n) >= kFusedRoundPoints ? 1 : kFusedRoundPoints >> log2n;
}

/// Dynamic shared memory (bytes) of the fused kernel for length 2^log2n.
__host__ __device__ constexpr uint64_t fused_smem_bytes(int log2n) {
    const uint64_t n = uint64_t{1} << log2n;
    return uint64_t{2} * static_cast<uint64_t>(fused_round_blocks(log2n)) *
           fused_pad(static_cast<uint32_t>(n)) * sizeof(float2);
}

struct FusedArgs {
    const float2* spec;     // (sub, ifft, pol, nbin) forward spectrum
    const float2* chirp;    // (nchans, mbin): taper / nbin x exp(i phase)
    const float2* twiddle;  // (mbin / 2): exp(+2 pi i k / mbin)
    const int64_t* shifts;  // (nchans) delay of this trial (tsamp)
    const float* mean;      // (nchans)
    const float* inv_sigma; // (nchans)
    float* waterfall;       // (nchans, nf)
    uint64_t nchans;
    uint64_t n_p;
    uint64_t nfft;
    uint64_t nbin;
    uint64_t novc;           // overlap per side (channel samples)
    int64_t lc;              // kept channel samples per FFT block
    int64_t a;               // FDMT window start (aligned samples)
    int64_t nf;              // FDMT window length
    int64_t msamp;           // channel samples of the block
    uint64_t blocks_per_cta; // FFT blocks per thread block (multiple of the
                             // round size)
    uint64_t ngroups;        // thread blocks per channel
    bool norm;
};

namespace detail {

__device__ __forceinline__ int64_t imin(int64_t a, int64_t b) {
    return a < b ? a : b;
}
__device__ __forceinline__ float2 cadd(float2 a, float2 b) {
    return {a.x + b.x, a.y + b.y};
}
__device__ __forceinline__ float2 csub(float2 a, float2 b) {
    return {a.x - b.x, a.y - b.y};
}

// In-place radix-2 DIF butterfly: (a, b) -> (a + b, (a - b) w).
__device__ __forceinline__ void bfly2(float2& a, float2& b, float2 w) {
    const float2 s = cadd(a, b);
    b              = fourier_gpu::cmul(csub(a, b), w);
    a              = s;
}

} // namespace detail

/**
 * Coherent stage of one coarse trial, fused. Grid: nchans * ngroups thread
 * blocks of kFusedThreads; thread block (c, g) handles FFT blocks
 * [g * blocks_per_cta, (g + 1) * blocks_per_cta) of channel c's window and
 * (g == 0) zero-fills the window samples outside the block.
 */
template <int Log2N>
__global__ void __launch_bounds__(kFusedThreads)
    fused_coherent_kernel(const FusedArgs p) {
    constexpr uint32_t kN     = 1U << Log2N;
    constexpr uint32_t kB     = fused_round_blocks(Log2N);
    constexpr uint32_t kS     = fused_pad(kN); // slots per FFT block
    constexpr uint32_t kQuart = kN / 4;
    extern __shared__ float2 smem[];
    float2* s0 = smem;             // pol 0: kB blocks of kS slots
    float2* s1 = smem + (kB * kS); // pol 1

    const uint64_t c   = blockIdx.x / p.ngroups;
    const uint64_t g   = blockIdx.x % p.ngroups;
    const int64_t s    = p.shifts[c];
    const int64_t lo   = p.a - s; // channel sample at aligned window start
    const int64_t m_lo = lo > 0 ? lo : 0;
    const int64_t m_hi = (lo + p.nf) < p.msamp ? (lo + p.nf) : p.msamp;
    float* row         = p.waterfall + (c * static_cast<uint64_t>(p.nf));

    if (g == 0) {
        // Window samples outside the block are zero.
        const int64_t head = m_lo < m_hi ? m_lo - lo : p.nf;
        const int64_t tail = m_lo < m_hi ? m_hi - lo : p.nf;
        for (int64_t u = threadIdx.x; u < head; u += kFusedThreads) {
            row[u] = 0.0F;
        }
        for (int64_t u = tail + threadIdx.x; u < p.nf; u += kFusedThreads) {
            row[u] = 0.0F;
        }
    }
    if (m_lo >= m_hi) {
        return;
    }
    const int64_t j_first = m_lo / p.lc;
    const int64_t nblk    = ((m_hi - 1) / p.lc) - j_first + 1;
    const auto q_begin    = static_cast<int64_t>(g * p.blocks_per_cta);
    const int64_t q_end =
        detail::imin(nblk, q_begin + static_cast<int64_t>(p.blocks_per_cta));
    if (q_begin >= q_end) {
        return;
    }

    const uint64_t isub   = c / p.n_p;
    const uint64_t ichan  = c % p.n_p;
    const uint64_t off    = ((ichan * kN) + (p.nbin / 2)) % p.nbin;
    const float2* chirp   = p.chirp + (c * kN);
    const float mean      = p.norm ? p.mean[c] : 0.0F;
    const float inv_sigma = p.norm ? p.inv_sigma[c] : 1.0F;
    const auto lc         = static_cast<uint32_t>(p.lc);

    for (int64_t q0 = q_begin; q0 < q_end; q0 += kB) {
        const auto nb = static_cast<uint32_t>(
            detail::imin(kB, q_end - q0)); // blocks this round

        // Gather x chirp (bins of channel ichan; fftshift folded in).
        for (uint32_t idx = threadIdx.x; idx < nb * kN; idx += kFusedThreads) {
            const uint32_t qb = idx >> Log2N;
            const uint32_t b  = idx & (kN - 1U);
            const auto j      = static_cast<uint64_t>(j_first + q0 + qb);
            const float2* x   = p.spec + (((isub * p.nfft) + j) * 2 * p.nbin);
            uint64_t bin      = off + b;
            bin               = bin >= p.nbin ? bin - p.nbin : bin;
            const float2 h    = __ldg(chirp + b);
            s0[(qb * kS) + fused_pad(b)] = fourier_gpu::cmul(__ldg(x + bin), h);
            s1[(qb * kS) + fused_pad(b)] =
                fourier_gpu::cmul(__ldg(x + p.nbin + bin), h);
        }
        __syncthreads();

        // Radix-4 stages: radix-2 spans h and h / 2 fused.
        for (uint32_t h = kN / 2; h >= 2; h /= 4) {
            const uint32_t half = h / 2;
            const uint32_t tstr = kN / (2 * h); // W_{2h}^j = tw[j * tstr]
            for (uint32_t w = threadIdx.x; w < nb * kQuart;
                 w += kFusedThreads) {
                const uint32_t qb  = w / kQuart;
                const uint32_t r   = w % kQuart;
                const uint32_t grp = r / half;
                const uint32_t j   = r % half;
                const uint32_t i0  = (grp * 2 * h) + j;
                const uint32_t i1  = i0 + half;
                const uint32_t i2  = i0 + h;
                const uint32_t i3  = i2 + half;
                const float2 w1    = __ldg(p.twiddle + (j * tstr));
                const float2 w2    = __ldg(p.twiddle + (j * tstr) + (kN / 4));
                const float2 w3    = __ldg(p.twiddle + (2 * j * tstr));
                float2* pols[2]    = {s0 + (qb * kS), s1 + (qb * kS)};
#pragma unroll
                for (int pol = 0; pol < 2; ++pol) {
                    float2* x = pols[pol];
                    float2 x0 = x[fused_pad(i0)];
                    float2 x1 = x[fused_pad(i1)];
                    float2 x2 = x[fused_pad(i2)];
                    float2 x3 = x[fused_pad(i3)];
                    detail::bfly2(x0, x2, w1); // span h
                    detail::bfly2(x1, x3, w2);
                    detail::bfly2(x0, x1, w3); // span h / 2
                    detail::bfly2(x2, x3, w3);
                    x[fused_pad(i0)] = x0;
                    x[fused_pad(i1)] = x1;
                    x[fused_pad(i2)] = x2;
                    x[fused_pad(i3)] = x3;
                }
            }
            __syncthreads();
        }
        if constexpr ((Log2N % 2) == 1) {
            // Final radix-2 stage (span 1, unit twiddle).
            for (uint32_t w = threadIdx.x; w < nb * (kN / 2);
                 w += kFusedThreads) {
                const uint32_t qb = w / (kN / 2);
                const uint32_t i0 = 2 * (w % (kN / 2));
                float2* pols[2]   = {s0 + (qb * kS), s1 + (qb * kS)};
#pragma unroll
                for (int pol = 0; pol < 2; ++pol) {
                    float2* x            = pols[pol];
                    float2 a             = x[fused_pad(i0)];
                    float2 b             = x[fused_pad(i0 + 1)];
                    x[fused_pad(i0)]     = detail::cadd(a, b);
                    x[fused_pad(i0 + 1)] = detail::csub(a, b);
                }
            }
            __syncthreads();
        }

        // Detect the kept samples [novc, novc + lc) of each block inside
        // the window; output sample i sits at slot bitrev(i).
        for (uint32_t idx = threadIdx.x; idx < nb * lc; idx += kFusedThreads) {
            const uint32_t qb = idx / lc;
            const uint32_t ii = idx - (qb * lc);
            const int64_t m   = ((j_first + q0 + qb) * p.lc) + ii;
            if (m < m_lo || m >= m_hi) {
                continue;
            }
            const uint32_t i = static_cast<uint32_t>(p.novc) + ii;
            const uint32_t slot =
                fused_pad(__brev(i) >> (32U - static_cast<uint32_t>(Log2N)));
            const float2 y0 = s0[(qb * kS) + slot];
            const float2 y1 = s1[(qb * kS) + slot];
            const float v =
                (y0.x * y0.x) + (y0.y * y0.y) + (y1.x * y1.x) + (y1.y * y1.y);
            row[m - lo] = (v - mean) * inv_sigma;
        }
        __syncthreads();
    }
}

namespace {

/// chirp[c, b] = taper[b] * exp(i 2 pi frac(base[c, b] + k inc[c, b])):
/// the trial's exact fixed-point phases (lib/dmt/cfdmt_common.hpp).
__global__ void chirp_kernel(const unsigned long long* __restrict__ base,
                             const unsigned long long* __restrict__ inc,
                             const float* __restrict__ taper,
                             unsigned long long k,
                             uint64_t nchans,
                             uint64_t mbin,
                             float2* __restrict__ chirp) {
    const uint64_t total = nchans * mbin;
    for (uint64_t idx =
             (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
         idx < total; idx += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        const unsigned long long ph = base[idx] + (k * inc[idx]);
        float sn                    = 0.0F;
        float cs                    = 0.0F;
        sincospif(2.0F * static_cast<float>(static_cast<long long>(ph)) *
                      5.42101086242752217e-20F,
                  &sn, &cs);
        const float w = taper[idx % mbin];
        chirp[idx]    = {w * cs, w * sn};
    }
}

} // namespace

} // namespace dmt::cfdmt_gpu
