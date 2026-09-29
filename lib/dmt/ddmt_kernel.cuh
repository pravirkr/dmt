#pragma once

#include <cstdint>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/types.hpp"

/**
 * @file ddmt_kernel.cuh
 * @brief DDMT (brute-force dedispersion) GPU kernels.
 *
 * Every kernel reads its input through a DDMTSegments view: a stream of
 * samples per (beam, channel) row made of two back-to-back pieces, A (the
 * retained history tail) followed by B (the new block). This lets a
 * streaming call dedisperse history + new data without first copying them
 * into one combined buffer.
 *
 * Samples are summed per output in ascending channel order starting from
 * zero, exactly as the CPU engine does, so float results are bitwise
 * identical across backends and kernels, and integer results are exact.
 */

namespace dmt::algorithms::ddmt_gpu {

/**
 * @brief Two-segment view of a beam-major (nbeams*nchans rows) input.
 *
 * Stream sample g of row r is A[r][a_off + g] for g < a_len and
 * B[r][b_off + g - a_len] otherwise; samples g >= total read as zero.
 * Rows are addressed in bytes (row r at ptr + r * row_bytes) so float and
 * packed inputs share the view. For packed NBITS < 8, a_off/b_off are sample
 * offsets inside the first byte-aligned row.
 */
struct DDMTSegments {
    const uint8_t* a{nullptr};
    SizeType a_row_bytes{0};
    int a_len{0};
    int a_off{0};
    const uint8_t* b{nullptr};
    SizeType b_row_bytes{0};
    int b_off{0};
    int total{0};
};

/// Kind of a DDMT input sample; NBITS == 32 means float32.
template <unsigned NBITS>
using DDMTAcc = std::conditional_t<NBITS == 32, float, int32_t>;

template <unsigned NBITS>
__device__ __forceinline__ DDMTAcc<NBITS>
fetch_sample(const DDMTSegments& seg, SizeType row, int g) {
    if (g >= seg.total) {
        return 0;
    }
    const uint8_t* base = nullptr;
    int idx             = 0;
    if (g < seg.a_len) {
        base = seg.a + (row * seg.a_row_bytes);
        idx  = seg.a_off + g;
    } else {
        base = seg.b + (row * seg.b_row_bytes);
        idx  = seg.b_off + (g - seg.a_len);
    }
    if constexpr (NBITS == 32) {
        return reinterpret_cast<const float*>(base)[idx];
    } else if constexpr (NBITS == 16) {
        // Rows of a device buffer are 2-byte aligned (row_bytes is even).
        return reinterpret_cast<const uint16_t*>(base)[idx];
    } else {
        return static_cast<int32_t>(
            bit_pack_utils::read_packed_sample<NBITS>(base, idx));
    }
}

/**
 * @brief Tiling of one tiled-kernel instantiation.
 *
 * A block computes BT = TX * SPW * V output samples for TDM = TY * DPT
 * consecutive DM trials. Thread (tx, ty) owns the sample words tx + k*TX
 * (k < SPW, V samples each) of the DMs ty*DPT .. ty*DPT + DPT-1.
 */
template <int TX_, int TY_, int SPW_, int DPT_, int CHUNK_, int MINB_ = 1>
struct TileCfg {
    static constexpr int kMinBlocks = MINB_;
    static constexpr int kTX        = TX_;
    static constexpr int kTY        = TY_;
    static constexpr int kSPW       = SPW_;
    static constexpr int kDPT       = DPT_;
    static constexpr int kChunk     = CHUNK_;
    static_assert(DPT_ % 4 == 0, "DPT must be a multiple of 4");
    static_assert(CHUNK_ <= 256, "CHUNK <= 256 keeps SWAR lanes exact");
};

/// Samples per shared-memory word: float reads 2 (float2), 16-bit reads 4
/// (one 64-bit word), bytes read 4 (one 32-bit word).
template <unsigned NBITS> constexpr int kWordSamples = NBITS == 32 ? 2 : 4;

template <unsigned NBITS>
using DDMTWord = std::conditional_t<
    NBITS == 32,
    float2,
    std::conditional_t<NBITS == 16, unsigned long long, uint32_t>>;

/// Tile parameters derived from a config, shared by host and device code.
template <unsigned NBITS, class Cfg> struct TileShape {
    static constexpr int kV   = kWordSamples<NBITS>;
    static constexpr int kBT  = Cfg::kTX * Cfg::kSPW * kV;
    static constexpr int kTDM = Cfg::kTY * Cfg::kDPT;
    static constexpr int kNT  = Cfg::kTX * Cfg::kTY;

    /// Words per shifted copy of a channel window with @p spread.
    DMT_HD static constexpr int words(int spread) {
        return ((kBT + spread + kV - 1) / kV) + 1;
    }
    DMT_HD static constexpr SizeType offs_bytes() {
        return ((static_cast<SizeType>(Cfg::kChunk * kTDM) * 2 + 15) / 16) * 16;
    }
    /// Dynamic shared memory for a window of @p spread samples.
    DMT_HD static constexpr SizeType smem_bytes(int spread) {
        return offs_bytes() + (static_cast<SizeType>(Cfg::kChunk) * kV *
                               words(spread) * sizeof(DDMTWord<NBITS>));
    }
};

/**
 * @brief Pack V consecutive samples (starting at window sample j0) into a
 * word, V = kWordSamples<NBITS>.
 */
template <unsigned NBITS>
__device__ __forceinline__ DDMTWord<NBITS>
load_word(const DDMTSegments& seg, SizeType row, int g0) {
    if constexpr (NBITS == 32) {
        return make_float2(fetch_sample<32>(seg, row, g0),
                           fetch_sample<32>(seg, row, g0 + 1));
    } else if constexpr (NBITS == 16) {
        unsigned long long x = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            x |= static_cast<unsigned long long>(
                     fetch_sample<16>(seg, row, g0 + b))
                 << (16 * b);
        }
        return x;
    } else {
        uint32_t x = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            x |= static_cast<uint32_t>(fetch_sample<NBITS>(seg, row, g0 + b))
                 << (8 * b);
        }
        return x;
    }
}

/// Whether a tiled kernel stages a chunk in one pass (each thread loads two
/// words) or two (copy 0, then the shifted copies from shared memory).
template <unsigned NBITS> constexpr bool kOnePassStage = NBITS == 32;

/// Word of copy r (window shifted by r samples) from copy-0 words x, y.
template <unsigned NBITS>
__device__ __forceinline__ DDMTWord<NBITS>
shift_word(DDMTWord<NBITS> x, DDMTWord<NBITS> y, int r) {
    if constexpr (NBITS == 32) {
        (void)r; // only r == 1
        return make_float2(x.y, y.x);
    } else if constexpr (NBITS == 16) {
        return (x >> (16 * r)) | (y << (64 - (16 * r)));
    } else {
        const auto xy =
            static_cast<uint64_t>(x) | (static_cast<uint64_t>(y) << 32);
        return static_cast<uint32_t>(xy >> (8 * r));
    }
}

/**
 * @brief DM-tiled brute-force dedispersion kernel.
 * @details
 * For each chunk of active channels the block stages, per channel c, the
 * input window [t0 + base(c), t0 + base(c) + BT + spread) into shared
 * memory once and every DM of the tile reads it with its own offset
 * delay(dm, c) - base(c) (host-precomputed, int16, laid out
 * [tile][channel][TDM]). The window is stored V times, shifted by 0..V-1
 * samples, so a read at any offset is one aligned V-sample word.
 *
 * Consecutive DMs of a thread frequently share a delay in higher-frequency
 * channels; the loaded word is then reused instead of reloaded. Offsets are
 * uniform across a warp row (TX == warp width), so that branch is uniform.
 *
 * Byte-family inputs (NBITS <= 8) accumulate with SWAR: even and odd bytes
 * of each word go to two 16-bit lanes of one 32-bit register, flushed to
 * int32 after each chunk of at most 256 channels (255 * 256 < 65536).
 */
template <unsigned NBITS, class Cfg>
__global__
__launch_bounds__(Cfg::kTX* Cfg::kTY, Cfg::kMinBlocks) void ddmt_tiled_kernel(
    DDMTSegments seg,
    DDMTAcc<NBITS>* __restrict__ d_out,
    SizeType out_dm_stride,
    SizeType out_beam_stride,
    const int16_t* __restrict__ offs_tab,
    const int* __restrict__ tile_base,
    const int* __restrict__ active,
    int nact,
    int nchans,
    int ndm,
    int n_out,
    int Q) {
    using Shape        = TileShape<NBITS, Cfg>;
    using Word         = DDMTWord<NBITS>;
    using Acc          = DDMTAcc<NBITS>;
    constexpr int V    = Shape::kV;
    constexpr int TX   = Cfg::kTX;
    constexpr int SPW  = Cfg::kSPW;
    constexpr int DPT  = Cfg::kDPT;
    constexpr int NT   = Shape::kNT;
    constexpr int BT   = Shape::kBT;
    constexpr int TDM  = Shape::kTDM;
    constexpr int CH   = Cfg::kChunk;
    constexpr bool kSW = NBITS <= 8;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto* offs = reinterpret_cast<int16_t*>(smem_raw);
    auto* win  = reinterpret_cast<Word*>(smem_raw + Shape::offs_bytes());

    const int tx  = static_cast<int>(threadIdx.x);
    const int ty  = static_cast<int>(threadIdx.y);
    const int tid = (ty * TX) + tx;
    const int t0  = static_cast<int>(blockIdx.x) * BT;
    const int dm0 = static_cast<int>(blockIdx.y) * TDM;
    const auto row0 =
        static_cast<SizeType>(blockIdx.z) * static_cast<SizeType>(nchans);
    const int* tb = tile_base + (static_cast<SizeType>(blockIdx.y) * nact);
    const int16_t* ot =
        offs_tab + (static_cast<SizeType>(blockIdx.y) * nact * TDM);

    // acc[d][s][v]: sample tx*V + s*TX*V + v of DM ty*DPT + d.
    Acc acc[DPT][SPW][V];
#pragma unroll
    for (int d = 0; d < DPT; ++d) {
#pragma unroll
        for (int s = 0; s < SPW; ++s) {
#pragma unroll
            for (int v = 0; v < V; ++v) {
                acc[d][s][v] = 0;
            }
        }
    }

    for (int cb = 0; cb < nact; cb += CH) {
        const int nc = min(CH, nact - cb);
        __syncthreads();
        for (int i = tid; i < nc * TDM; i += NT) {
            offs[i] = ot[(static_cast<SizeType>(cb) * TDM) + i];
        }
        if constexpr (kOnePassStage<NBITS>) {
            // One pass: each thread loads its word and the next one (the
            // overlap mostly hits L1) and writes all V shifted copies.
            int cc = tid / Q;
            int q  = tid - (cc * Q);
            while (cc < nc) {
                const auto row = row0 + active[cb + cc];
                const int g    = t0 + tb[cb + cc] + (V * q);
                const Word x   = load_word<NBITS>(seg, row, g);
                const Word y   = load_word<NBITS>(seg, row, g + V);
                Word* wr       = win + (cc * V * Q);
                wr[q]          = x;
#pragma unroll
                for (int r = 1; r < V; ++r) {
                    wr[(r * Q) + q] = shift_word<NBITS>(x, y, r);
                }
                q += NT;
                while (q >= Q) {
                    q -= Q;
                    ++cc;
                }
            }
            __syncthreads();
        } else {
            // Copy 0: word q of channel cc = window samples [V*q, V*q + V).
            {
                int cc = tid / Q;
                int q  = tid - (cc * Q);
                while (cc < nc) {
                    const auto row = row0 + active[cb + cc];
                    win[(cc * V * Q) + q] =
                        load_word<NBITS>(seg, row, t0 + tb[cb + cc] + (V * q));
                    q += NT;
                    while (q >= Q) {
                        q -= Q;
                        ++cc;
                    }
                }
            }
            __syncthreads();
            // Copies 1..V-1 from neighbouring copy-0 words.
            {
                int cc = tid / Q;
                int q  = tid - (cc * Q);
                while (cc < nc) {
                    Word* wr     = win + (cc * V * Q);
                    const Word x = wr[q];
                    const Word y = q + 1 < Q ? wr[q + 1] : Word{};
#pragma unroll
                    for (int r = 1; r < V; ++r) {
                        wr[(r * Q) + q] = shift_word<NBITS>(x, y, r);
                    }
                    q += NT;
                    while (q >= Q) {
                        q -= Q;
                        ++cc;
                    }
                }
            }
            __syncthreads();
        }

        // SWAR lanes for byte inputs: lo = bytes 0,2; hi = bytes 1,3.
        uint32_t lo[kSW ? DPT : 1][kSW ? SPW : 1];
        uint32_t hi[kSW ? DPT : 1][kSW ? SPW : 1];
        if constexpr (kSW) {
#pragma unroll
            for (int d = 0; d < DPT; ++d) {
#pragma unroll
                for (int s = 0; s < SPW; ++s) {
                    lo[d][s] = 0;
                    hi[d][s] = 0;
                }
            }
        }
        for (int c2 = 0; c2 < nc; ++c2) {
            int o[DPT];
#pragma unroll
            for (int d = 0; d < DPT; d += 4) {
                const auto v4 = *reinterpret_cast<const short4*>(
                    offs + (c2 * TDM) + (ty * DPT) + d);
                o[d]     = v4.x;
                o[d + 1] = v4.y;
                o[d + 2] = v4.z;
                o[d + 3] = v4.w;
            }
            const Word* wr = win + (c2 * V * Q) + tx;
            Word w[SPW];
#pragma unroll
            for (int d = 0; d < DPT; ++d) {
                if (d == 0 || o[d] != o[d - 1]) {
                    const auto od = static_cast<unsigned>(o[d]);
                    const Word* p = wr + (static_cast<int>(od % V) * Q) +
                                    static_cast<int>(od / V);
#pragma unroll
                    for (int s = 0; s < SPW; ++s) {
                        w[s] = p[s * TX];
                    }
                }
#pragma unroll
                for (int s = 0; s < SPW; ++s) {
                    if constexpr (NBITS == 32) {
                        acc[d][s][0] += w[s].x;
                        acc[d][s][1] += w[s].y;
                    } else if constexpr (NBITS == 16) {
#pragma unroll
                        for (int v = 0; v < 4; ++v) {
                            acc[d][s][v] += static_cast<int32_t>(
                                (w[s] >> (16 * v)) & 0xFFFFU);
                        }
                    } else {
                        lo[d][s] += w[s] & 0x00FF00FFU;
                        hi[d][s] += (w[s] >> 8) & 0x00FF00FFU;
                    }
                }
            }
        }
        if constexpr (kSW) {
#pragma unroll
            for (int d = 0; d < DPT; ++d) {
#pragma unroll
                for (int s = 0; s < SPW; ++s) {
                    acc[d][s][0] += static_cast<int32_t>(lo[d][s] & 0xFFFFU);
                    acc[d][s][1] += static_cast<int32_t>(hi[d][s] & 0xFFFFU);
                    acc[d][s][2] += static_cast<int32_t>(lo[d][s] >> 16);
                    acc[d][s][3] += static_cast<int32_t>(hi[d][s] >> 16);
                }
            }
        }
    }

    auto* out_b = d_out + (static_cast<SizeType>(blockIdx.z) * out_beam_stride);
#pragma unroll
    for (int d = 0; d < DPT; ++d) {
        const int dm = dm0 + (ty * DPT) + d;
        if (dm >= ndm) {
            continue;
        }
        auto* orow = out_b + (static_cast<SizeType>(dm) * out_dm_stride);
#pragma unroll
        for (int s = 0; s < SPW; ++s) {
            const int t = t0 + (V * (tx + (s * TX)));
#pragma unroll
            for (int v = 0; v < V; ++v) {
                if (t + v < n_out) {
                    orow[t + v] = acc[d][s][v];
                }
            }
        }
    }
}

/**
 * @brief Direct (untiled) dedispersion kernel: the fallback for DM grids
 * whose per-tile delay spread does not fit the tiled kernel's shared
 * memory. One thread per (sample, DM), channels ascending.
 */
template <unsigned NBITS>
__global__ void ddmt_direct_kernel(DDMTSegments seg,
                                   DDMTAcc<NBITS>* __restrict__ d_out,
                                   SizeType out_dm_stride,
                                   SizeType out_beam_stride,
                                   const int* __restrict__ delay_table,
                                   const int* __restrict__ active,
                                   int nact,
                                   int nchans,
                                   int ndm,
                                   int n_out) {
    const int t = static_cast<int>((blockIdx.x * blockDim.x) + threadIdx.x);
    if (t >= n_out) {
        return;
    }
    const auto row0 =
        static_cast<SizeType>(blockIdx.z) * static_cast<SizeType>(nchans);
    auto* out_b = d_out + (static_cast<SizeType>(blockIdx.z) * out_beam_stride);
    for (auto idm = static_cast<int>(blockIdx.y); idm < ndm;
         idm += static_cast<int>(gridDim.y)) {
        const int* delays = delay_table + (static_cast<SizeType>(idm) * nchans);
        DDMTAcc<NBITS> sum = 0;
        for (int k = 0; k < nact; ++k) {
            const int c = active[k];
            sum += fetch_sample<NBITS>(seg, row0 + c, t + delays[c]);
        }
        out_b[(static_cast<SizeType>(idm) * out_dm_stride) + t] = sum;
    }
}

/**
 * @brief Copies stream samples [start, start + len) of every row of @p seg
 * into @p dst (row r at dst + r * dst_row_bytes, packed from sample 0).
 * One thread per destination byte (packed NBITS < 8) or sample.
 */
template <unsigned NBITS>
__global__ void ddmt_copy_tail_kernel(DDMTSegments seg,
                                      int start,
                                      int len,
                                      uint8_t* __restrict__ dst,
                                      SizeType dst_row_bytes,
                                      SizeType nrows) {
    constexpr int kPer = NBITS < 8 ? static_cast<int>(8 / NBITS) : 1;
    const int nunits   = (len + kPer - 1) / kPer;
    const auto unit =
        (static_cast<SizeType>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const auto row = static_cast<SizeType>(blockIdx.y) +
                     (static_cast<SizeType>(blockIdx.z) * gridDim.y);
    if (unit >= static_cast<SizeType>(nunits) || row >= nrows) {
        return;
    }
    uint8_t* drow = dst + (row * dst_row_bytes);
    const int u   = static_cast<int>(unit);
    if constexpr (NBITS == 32) {
        reinterpret_cast<float*>(drow)[u] =
            fetch_sample<32>(seg, row, start + u);
    } else if constexpr (NBITS == 16) {
        reinterpret_cast<uint16_t*>(drow)[u] =
            static_cast<uint16_t>(fetch_sample<16>(seg, row, start + u));
    } else if constexpr (NBITS == 8) {
        drow[u] = static_cast<uint8_t>(fetch_sample<8>(seg, row, start + u));
    } else {
        uint32_t byte = 0;
        for (int i = 0; i < kPer; ++i) {
            const int s = (u * kPer) + i;
            if (s < len) {
                byte |= static_cast<uint32_t>(
                            fetch_sample<NBITS>(seg, row, start + s))
                        << (i * NBITS);
            }
        }
        drow[u] = static_cast<uint8_t>(byte);
    }
}

/**
 * @brief Transposes a time-major packed block (nsamps rows of samp_bytes,
 * channels packed LSB-first within a time sample) into channel-major packed
 * rows (row c at dst + (beam*nchans + c) * dst_row_bytes, sample 0 at bit
 * 0). blockIdx.z selects the beam.
 *
 * A block handles a 32-channel x (32 * kPer)-sample tile through shared
 * memory, so both the reads (consecutive channels of one sample) and the
 * writes (consecutive bytes of one channel row) are coalesced.
 */
template <unsigned NBITS>
__global__ void ddmt_transpose_kernel(const uint8_t* __restrict__ src,
                                      SizeType samp_bytes,
                                      SizeType src_beam_stride,
                                      int nsamps,
                                      int nchans,
                                      uint8_t* __restrict__ dst,
                                      SizeType dst_row_bytes) {
    constexpr int kPer = NBITS < 8 ? static_cast<int>(8 / NBITS) : 1;
    constexpr int kTS  = 32 * kPer; // samples per tile
    __shared__ uint16_t tile[32][kTS + 1];
    const int c0         = static_cast<int>(blockIdx.y) * 32;
    const int s0         = static_cast<int>(blockIdx.x) * kTS;
    const int tx         = static_cast<int>(threadIdx.x);
    const int ty         = static_cast<int>(threadIdx.y); // 0..7
    const auto beam      = static_cast<SizeType>(blockIdx.z);
    const uint8_t* src_b = src + (beam * src_beam_stride);
    for (int s = ty; s < kTS; s += 8) {
        const int gs = s0 + s;
        const int gc = c0 + tx;
        uint32_t v   = 0;
        if (gs < nsamps && gc < nchans) {
            v = bit_pack_utils::read_packed_sample<NBITS>(
                src_b + (static_cast<SizeType>(gs) * samp_bytes),
                static_cast<SizeType>(gc));
        }
        tile[tx][s] = static_cast<uint16_t>(v);
    }
    __syncthreads();
    constexpr int kSampBytes = NBITS >= 8 ? static_cast<int>(NBITS / 8) : 1;
    constexpr int kUnits     = 32 * kSampBytes; // dst bytes per tile row
    for (int c = ty; c < 32; c += 8) {
        const int gc = c0 + c;
        if (gc >= nchans) {
            continue;
        }
        uint8_t* drow = dst + ((beam * static_cast<SizeType>(nchans)) +
                               static_cast<SizeType>(gc)) *
                                  dst_row_bytes;
        for (int u = tx; u < kUnits; u += 32) {
            if constexpr (NBITS == 16) {
                const int s = u / 2;
                if (s0 + s < nsamps) {
                    drow[(static_cast<SizeType>(s0) * 2) + u] =
                        static_cast<uint8_t>(tile[c][s] >> (8 * (u & 1)));
                }
            } else if constexpr (NBITS == 8) {
                if (s0 + u < nsamps) {
                    drow[s0 + u] = static_cast<uint8_t>(tile[c][u]);
                }
            } else {
                const int first = u * kPer;
                if (s0 + first < nsamps) {
                    uint32_t byte = 0;
                    for (int i = 0; i < kPer; ++i) {
                        if (s0 + first + i < nsamps) {
                            byte |= static_cast<uint32_t>(tile[c][first + i])
                                    << (i * NBITS);
                        }
                    }
                    drow[(s0 / kPer) + u] = static_cast<uint8_t>(byte);
                }
            }
        }
    }
}

} // namespace dmt::algorithms::ddmt_gpu
