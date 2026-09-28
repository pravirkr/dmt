#pragma once

/**
 * @file ddmt_cpu_engine.hpp
 * @brief Shared parts of the DDMT and SDMT CPU engines: input staging, the
 * DM-tile layout, and the streaming engine (history, entry points) around an
 * algorithm's dedisperse().
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>

#include "ddmt_cpu_kernels.hpp"
#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/engines.hpp"

namespace dmt::algorithms::ddmt_cpu {

// Matches the GPU engine's default; see DDMT::set_gulp_size().
inline constexpr SizeType kDefaultGulpSamples = 65536;
// Largest DM tile (trials sharing one set of staged windows).
inline constexpr int kMaxTileDM = 256;
// Byte-family inputs accumulate in uint16 lanes, flushed to int32 before
// they could overflow: after at most this many channels.
template <unsigned NBITS>
inline constexpr int kU16FlushChans =
    static_cast<int>(65535U / ((1U << NBITS) - 1));

inline bool is_packed_width(SizeType nbits) {
    return nbits == 1 || nbits == 2 || nbits == 4 || nbits == 8 || nbits == 16;
}

template <typename F> void dispatch_nbits(SizeType nbits, F&& f) {
    switch (nbits) {
    case 1:
        f.template operator()<1>();
        break;
    case 2:
        f.template operator()<2>();
        break;
    case 4:
        f.template operator()<4>();
        break;
    case 8:
        f.template operator()<8>();
        break;
    case 16:
        f.template operator()<16>();
        break;
    case 32:
        f.template operator()<32>();
        break;
    default:
        throw std::invalid_argument(
            std::format("DDMT: unsupported nbits={}", nbits));
    }
}

inline SizeType row_bytes_for(SizeType nsamps, SizeType nbits) {
    return nbits == 32 ? nsamps * sizeof(float)
                       : bit_pack_utils::packed_row_bytes(nsamps, nbits);
}

template <unsigned NBITS>
using Acc = std::conditional_t<NBITS == 32, float, int32_t>;
/// Staged window and partial-sum lane type: every kernel adds like types.
/// Byte-family samples widen to uint16 (exact up to kU16FlushChans
/// channels), 16-bit samples to uint32.
template <unsigned NBITS>
using Win =
    std::conditional_t<NBITS == 32,
                       float,
                       std::conditional_t<NBITS == 16, uint32_t, uint16_t>>;

/**
 * @brief Two-segment view of a beam-major input (see the GPU engine's
 * DDMTSegments): stream sample g of row r is A[r][g] for g < a_len and
 * B[r][g - a_len] otherwise.
 */
struct Segments {
    const uint8_t* a{nullptr};
    SizeType a_row_bytes{0};
    SizeType a_len{0};
    const uint8_t* b{nullptr};
    SizeType b_row_bytes{0};
    SizeType total{0};
};

/// Unpacks samples [first, first + count) of a packed row into @p out.
template <unsigned NBITS, typename T>
void unpack_range(const uint8_t* row, SizeType first, SizeType count, T* out) {
    if constexpr (NBITS == 32) {
        std::memcpy(out, reinterpret_cast<const float*>(row) + first,
                    count * sizeof(float));
    } else if constexpr (NBITS >= 8) {
        bit_pack_utils::unpack_row<NBITS>(row + (first * (NBITS / 8)), count,
                                          out);
    } else {
        constexpr SizeType kPer = 8 / NBITS;
        SizeType i              = 0;
        for (; i < count && (first + i) % kPer != 0; ++i) {
            out[i] = static_cast<T>(
                bit_pack_utils::read_packed_sample<NBITS>(row, first + i));
        }
        if (i < count) {
            if constexpr (std::is_same_v<T, uint16_t>) {
                const auto* bytes = row + ((first + i) / kPer);
                const auto nfull  = (count - i) / kPer;
                ddmt_cpu::unpack_bytes_u16<NBITS>(bytes, nfull, out + i);
                for (SizeType s = i + (nfull * kPer); s < count; ++s) {
                    out[s] = static_cast<T>(
                        bit_pack_utils::read_packed_sample<NBITS>(row,
                                                                  first + s));
                }
            } else {
                bit_pack_utils::unpack_row<NBITS>(row + ((first + i) / kPer),
                                                  count - i, out + i);
            }
        }
    }
}

/// Writes stream samples [g0, g0 + len) of row @p r into @p out (zeros
/// past the end of the stream).
template <unsigned NBITS>
void stage_window(const Segments& seg,
                  SizeType r,
                  SizeType g0,
                  SizeType len,
                  Win<NBITS>* out) {
    const auto end = g0 + len;
    auto g         = g0;
    if (g < seg.a_len) {
        const auto n = std::min(end, seg.a_len) - g;
        unpack_range<NBITS>(seg.a + (r * seg.a_row_bytes), g, n, out);
        g += n;
    }
    if (g < end && g < seg.total) {
        const auto n = std::min(end, seg.total) - g;
        unpack_range<NBITS>(seg.b + (r * seg.b_row_bytes), g - seg.a_len, n,
                            out + (g - g0));
        g += n;
    }
    if (g < end) {
        std::fill(out + (g - g0), out + len, Win<NBITS>{0});
    }
}

/// Copies stream samples [start, start + len) of every row into @p dst
/// (rows of @p dst_row_bytes, packed from sample 0).
template <unsigned NBITS>
void copy_tail(const Segments& seg,
               SizeType nrows,
               SizeType start,
               SizeType len,
               uint8_t* dst,
               SizeType dst_row_bytes,
               int nthreads) {
    if (len == 0) {
        return;
    }
#pragma omp parallel for num_threads(nthreads) schedule(static)
    for (SizeType r = 0; r < nrows; ++r) {
        auto* drow = dst + (r * dst_row_bytes);
        auto copy  = [&](const uint8_t* src, SizeType from, SizeType to,
                         SizeType n) {
            if constexpr (NBITS == 32) {
                std::memcpy(reinterpret_cast<float*>(drow) + to,
                            reinterpret_cast<const float*>(src) + from,
                            n * sizeof(float));
            } else {
                bit_pack_utils::copy_packed_samples<NBITS>(src, from, drow, to,
                                                           n);
            }
        };
        SizeType g = start;
        if (g < seg.a_len) {
            const auto n = std::min(start + len, seg.a_len) - g;
            copy(seg.a + (r * seg.a_row_bytes), g, 0, n);
            g += n;
        }
        if (g < start + len) {
            copy(seg.b + (r * seg.b_row_bytes), g - seg.a_len, g - start,
                 start + len - g);
        }
    }
}

/**
 * @brief DM-tile layout shared by the CPU engines.
 * @details
 * Trials are ordered by DM (tile slot i holds trial order[i]), so unsorted
 * grids still tile, and cut into tiles of the widest power of two (up to
 * kMaxTileDM) whose per-channel delay spread is at most one time tile. For
 * each tile and active channel, base is the smallest delay (the start of the
 * channel's staged window) and spreads the largest minus the smallest.
 */
struct Tiling {
    std::vector<int> active;     // unmasked channels, ascending
    std::vector<SizeType> order; // tile slot -> trial
    SizeType ndm{0};
    int tdm{1};               // DM trials per tile
    int spread{0};            // largest delay offset within a DM tile
    std::vector<int> base;    // [tile][active]
    std::vector<int> spreads; // [tile][active]

    Tiling(const plans::DDMTPlan& plan, int time_tile) {
        const auto& pc = plan.get_container();
        ndm            = pc.dm_arr.size();
        for (SizeType c = 0; c < pc.nchans; ++c) {
            if (pc.kill_mask[c] != 0) {
                active.push_back(static_cast<int>(c));
            }
        }
        order.resize(ndm);
        for (SizeType i = 0; i < ndm; ++i) {
            order[i] = i;
        }
        std::ranges::stable_sort(order, [&](SizeType a, SizeType b) {
            return pc.dm_arr[a] < pc.dm_arr[b];
        });
        if (ndm == 0 || active.empty()) {
            return;
        }
        for (int t = kMaxTileDM; t >= 2; t /= 2) {
            if (static_cast<SizeType>(t) >= 2 * ndm) {
                continue;
            }
            const auto s = tile_spread(pc, t);
            if (s <= static_cast<SizeType>(time_tile)) {
                tdm    = t;
                spread = static_cast<int>(s);
                break;
            }
        }
        const auto nact = active.size();
        const auto nt   = ntile();
        const auto td   = static_cast<SizeType>(tdm);
        base.assign(nt * nact, 0);
        spreads.assign(nt * nact, 0);
        for (SizeType t = 0; t < nt; ++t) {
            const auto d_end = std::min(td, ndm - (t * td));
            for (SizeType k = 0; k < nact; ++k) {
                const auto c = static_cast<SizeType>(active[k]);
                auto mn      = pc.delay_table[(order[t * td] * pc.nchans) + c];
                auto mx      = mn;
                for (SizeType d = 1; d < d_end; ++d) {
                    const auto v =
                        pc.delay_table[(order[(t * td) + d] * pc.nchans) + c];
                    mn = std::min(mn, v);
                    mx = std::max(mx, v);
                }
                base[(t * nact) + k]    = static_cast<int>(mn);
                spreads[(t * nact) + k] = static_cast<int>(mx - mn);
            }
        }
    }

    [[nodiscard]] SizeType ntile() const noexcept {
        const auto td = static_cast<SizeType>(tdm);
        return (ndm + td - 1) / td;
    }

    /// Largest delay offset inside DM tiles of @p t trials.
    [[nodiscard]] SizeType tile_spread(const plans::DDMTPlanContainer& pc,
                                       int t) const {
        const auto n = static_cast<int>(ndm);
        SizeType s   = 0;
        for (int t0 = 0; t0 < n; t0 += t) {
            const int t1 = std::min(t0 + t, n);
            for (const int c : active) {
                auto mn = pc.delay_table[(order[t0] * pc.nchans) + c];
                auto mx = mn;
                for (int dm = t0 + 1; dm < t1; ++dm) {
                    const auto v = pc.delay_table[(order[dm] * pc.nchans) + c];
                    mn           = std::min(mn, v);
                    mx           = std::max(mx, v);
                }
                s = std::max(s, mx - mn);
            }
        }
        return s;
    }
};

/// Adds the uint16 partial sums into the int32 accumulators (rows @p stride
/// apart), zeroing the partials.
template <typename P, typename A>
inline void flush(P* part, A* acc, SizeType nd, int cnt, SizeType stride) {
    for (SizeType d = 0; d < nd; ++d) {
        P* __restrict__ p = part + (d * stride);
        A* __restrict__ a = acc + (d * stride);
#pragma omp simd
        for (int s = 0; s < cnt; ++s) {
            a[s] += static_cast<A>(p[s]);
            p[s] = 0;
        }
    }
}

/**
 * @brief Streaming CPU engine around an algorithm.
 * @details
 * Holds the stream history and implements every entry point; each call
 * dedisperses the retained history followed by the new samples through
 * `Algo::dedisperse<NBITS>(Segments, n_out, out)`. `Algo` provides
 * `kName`, `kTimeTile` and a constructor from (plan, tiling, nthreads,
 * nbeams).
 */
template <class Algo>
class CpuEngine final : public algorithms::detail::DDMTEngine {
public:
    CpuEngine(const plans::DDMTPlan& plan,
              const algorithms::detail::DDMTEngineConfig& cfg)
        : m_plan(plan),
          m_nthreads(std::max(1, cfg.exec.nthreads)),
          m_nbeams(cfg.nbeams),
          m_tiling(plan, Algo::kTimeTile),
          m_algo(plan, m_tiling, m_nthreads, m_nbeams) {
        const auto& pc   = m_plan.get_container();
        m_ndm            = pc.dm_arr.size();
        m_max_delay      = pc.delay_table.empty()
                               ? 0
                               : *std::ranges::max_element(pc.delay_table);
        m_hist_row_bytes = row_bytes_for(m_max_delay, pc.nbits);
        m_hist.assign(m_nbeams * pc.nchans * m_hist_row_bytes, 0);
        m_hist_next.assign(m_hist.size(), 0);
    }

    // Number of output samples execute() will produce for a block of
    // `input_nsamps` new samples, given the currently retained history
    // (see reset_history()'s doc comment for the streaming model).
    [[nodiscard]] SizeType
    get_output_nsamps(SizeType input_nsamps) const noexcept override {
        const auto total = m_history_len + input_nsamps;
        return total > m_max_delay ? total - m_max_delay : 0;
    }

    void reset_history() noexcept override { m_history_len = 0; }

    void set_gulp_size(SizeType gulp_size) override {
        m_gulp = gulp_size == 0 ? kDefaultGulpSamples : gulp_size;
    }
    [[nodiscard]] SizeType get_gulp_size() const noexcept override {
        return m_gulp;
    }

    [[nodiscard]] SizeType history_state_size() const noexcept override {
        const auto rows = m_nbeams * m_plan.get_nchans();
        if (m_plan.get_nbits() == 32) {
            return rows * m_max_delay;
        }
        return rows * m_hist_row_bytes;
    }

    void save_history(std::span<float> out) const override {
        check_float("save_history(float)");
        check_warm();
        check_size(out.size(), history_state_size(), "save_history");
        std::memcpy(out.data(), m_hist.data(), out.size() * sizeof(float));
    }

    void save_history(std::span<uint8_t> out) const override {
        check_packed("save_history(uint8_t)");
        check_warm();
        check_size(out.size(), history_state_size(), "save_history");
        std::memcpy(out.data(), m_hist.data(), out.size());
    }

    void load_history(std::span<const float> in) override {
        check_float("load_history(float)");
        check_size(in.size(), history_state_size(), "load_history");
        std::memcpy(m_hist.data(), in.data(), in.size() * sizeof(float));
        m_history_len = m_max_delay;
    }

    void load_history(std::span<const uint8_t> in) override {
        check_packed("load_history(uint8_t)");
        check_size(in.size(), history_state_size(), "load_history");
        std::memcpy(m_hist.data(), in.data(), in.size());
        m_history_len = m_max_delay;
    }

    void execute(std::span<const float> waterfall,
                 std::span<float> dmt) override {
        check_float("execute(float)");
        const auto rows   = m_nbeams * m_plan.get_nchans();
        const auto nsamps = waterfall.size() / rows;
        check_size(dmt.size(), out_size(nsamps), "execute (output)");
        core<32>(reinterpret_cast<const uint8_t*>(waterfall.data()),
                 nsamps * sizeof(float), nsamps, dmt.data());
    }

    void execute(std::span<const uint8_t> waterfall_packed,
                 SizeType nsamps,
                 std::span<int32_t> dmt) override {
        check_packed("execute(packed)");
        const auto nbits     = m_plan.get_nbits();
        const auto rows      = m_nbeams * m_plan.get_nchans();
        const auto row_bytes = bit_pack_utils::packed_row_bytes(nsamps, nbits);
        if (waterfall_packed.size() != rows * row_bytes) {
            throw std::invalid_argument(std::format(
                "{}: Packed input buffer size mismatch: expected {} "
                "bytes ({} beams x {} chans x {} bytes/row), got {}",
                Algo::kName, rows * row_bytes, m_nbeams, m_plan.get_nchans(),
                row_bytes, waterfall_packed.size()));
        }
        check_size(dmt.size(), out_size(nsamps), "execute (output)");
        dispatch_nbits(nbits, [&]<unsigned NBITS>() {
            if constexpr (NBITS != 32) {
                core<NBITS>(waterfall_packed.data(), row_bytes, nsamps,
                            dmt.data());
            }
        });
    }

    // Time-major input is transposed to channel-major packed rows, then
    // streamed exactly like execute(packed): it shares the stream history.
    void execute_time_major(std::span<const uint8_t> filterbank_packed,
                            SizeType nsamps,
                            std::span<int32_t> dmt) override {
        check_packed("execute_time_major");
        const auto nbits      = m_plan.get_nbits();
        const auto nchans     = m_plan.get_nchans();
        const auto samp_bytes = bit_pack_utils::packed_row_bytes(nchans, nbits);
        if (filterbank_packed.size() != m_nbeams * nsamps * samp_bytes) {
            throw std::invalid_argument(std::format(
                "{}: Time-major input buffer size mismatch: expected {}, "
                "got {}",
                Algo::kName, m_nbeams * nsamps * samp_bytes,
                filterbank_packed.size()));
        }
        check_size(dmt.size(), out_size(nsamps), "execute_time_major (output)");
        const auto row_bytes = bit_pack_utils::packed_row_bytes(nsamps, nbits);
        const auto nrows     = m_nbeams * nchans;
        std::vector<uint8_t> cm(nrows * row_bytes, 0);
        dispatch_nbits(nbits, [&]<unsigned NBITS>() {
            if constexpr (NBITS != 32) {
#pragma omp parallel for num_threads(m_nthreads) schedule(static)
                for (SizeType r = 0; r < nrows; ++r) {
                    const auto b = r / nchans;
                    const auto c = r % nchans;
                    const auto* src =
                        filterbank_packed.data() + (b * nsamps * samp_bytes);
                    auto* drow = cm.data() + (r * row_bytes);
                    for (SizeType s = 0; s < nsamps; ++s) {
                        bit_pack_utils::write_packed_sample<NBITS>(
                            drow, s,
                            bit_pack_utils::read_packed_sample<NBITS>(
                                src + (s * samp_bytes), c));
                    }
                }
                core<NBITS>(cm.data(), row_bytes, nsamps, dmt.data());
            }
        });
    }

protected:
    [[nodiscard]] Backend backend() const noexcept override {
        return Backend::kCPU;
    }

private:
    const plans::DDMTPlan& m_plan; // owned by the DDMT facade
    int m_nthreads;
    SizeType m_nbeams;
    SizeType m_max_delay{0};
    SizeType m_ndm{0};
    SizeType m_gulp{kDefaultGulpSamples};
    Tiling m_tiling;
    Algo m_algo;

    // Retained history (m_history_len samples/row, row stride
    // m_hist_row_bytes; floats stored as bytes) and its double buffer.
    std::vector<uint8_t> m_hist;
    std::vector<uint8_t> m_hist_next;
    SizeType m_hist_row_bytes{0};
    SizeType m_history_len{0};

    [[nodiscard]] SizeType out_size(SizeType nsamps) const noexcept {
        return m_nbeams * m_ndm * get_output_nsamps(nsamps);
    }

    void check_float(std::string_view what) const {
        if (m_plan.get_nbits() != 32) {
            throw std::invalid_argument(std::format(
                "{}::{}: plan nbits={} != 32; use the packed-integer "
                "overload instead",
                Algo::kName, what, m_plan.get_nbits()));
        }
    }
    void check_packed(std::string_view what) const {
        if (!is_packed_width(m_plan.get_nbits())) {
            throw std::invalid_argument(std::format(
                "{}::{}: plan nbits={} is not a supported packed width "
                "(1,2,4,8,16); use the float overload for nbits==32",
                Algo::kName, what, m_plan.get_nbits()));
        }
    }
    void check_warm() const {
        if (m_history_len != m_max_delay) {
            throw std::logic_error(
                std::format("{}::save_history: stream is not fully "
                            "warmed up yet ({} of {} history samples/channel)",
                            Algo::kName, m_history_len, m_max_delay));
        }
    }
    static void
    check_size(SizeType got, SizeType expected, std::string_view what) {
        if (got != expected) {
            throw std::invalid_argument(
                std::format("{}: {} buffer size mismatch: expected {}, "
                            "got {}",
                            Algo::kName, what, expected, got));
        }
    }

    /**
     * @brief One streaming step: dedisperses the retained history followed
     * by @p n_new new samples (rows of @p new_row_bytes at @p data) into
     * @p out (nbeams, ndm, n_out), then retains the last max_delay samples.
     */
    template <unsigned NBITS>
    void core(const uint8_t* data,
              SizeType new_row_bytes,
              SizeType n_new,
              Acc<NBITS>* out) {
        Segments seg;
        seg.a           = m_hist.data();
        seg.a_row_bytes = m_hist_row_bytes;
        seg.a_len       = m_history_len;
        seg.b           = data;
        seg.b_row_bytes = new_row_bytes;
        seg.total       = m_history_len + n_new;
        const auto n_out =
            seg.total > m_max_delay ? seg.total - m_max_delay : 0;
        if (n_out > 0 && m_ndm > 0) {
            m_algo.template dedisperse<NBITS>(seg, n_out, out);
        }
        const auto new_len = std::min(seg.total, m_max_delay);
        copy_tail<NBITS>(seg, m_nbeams * m_plan.get_nchans(),
                         seg.total - new_len, new_len, m_hist_next.data(),
                         m_hist_row_bytes, m_nthreads);
        std::swap(m_hist, m_hist_next);
        m_history_len = new_len;
    }
};

} // namespace dmt::algorithms::ddmt_cpu
