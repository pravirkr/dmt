#include "dmt/algorithms/ddmt.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/common/types.hpp"
#include "dmt/engines.hpp"

namespace dmt::algorithms {

namespace {

// Matches the GPU engine's default; see DDMT::set_gulp_size().
constexpr SizeType kDefaultGulpSamples = 65536;

// Output samples per work unit (time tile) and the largest DM tile. A unit's
// accumulators (up to kMaxTileDM x kTimeTile values) stay in L2 and each
// staged channel window in L1.
constexpr int kTimeTile  = 256;
constexpr int kMaxTileDM = 256;
// Channels summed in registers per accumulator load/store.
constexpr int kChanGroup = 8;
// Byte-family inputs accumulate in uint16 lanes, flushed to int32 after at
// most this many channels (255 * 256 < 65536).
constexpr int kU16FlushChans = 256;

bool is_packed_width(SizeType nbits) {
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

SizeType row_bytes_for(SizeType nsamps, SizeType nbits) {
    return nbits == 32 ? nsamps * sizeof(float)
                       : bit_pack_utils::packed_row_bytes(nsamps, nbits);
}

template <unsigned NBITS>
using Acc = std::conditional_t<NBITS == 32, float, int32_t>;
/// Unpacked window sample type.
template <unsigned NBITS>
using Win =
    std::conditional_t<NBITS == 32,
                       float,
                       std::conditional_t<NBITS == 16, uint16_t, uint8_t>>;
/// Per-channel-block partial accumulator type.
template <unsigned NBITS>
using Part = std::conditional_t<(NBITS <= 8), uint16_t, Acc<NBITS>>;

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
            bit_pack_utils::unpack_row<NBITS>(row + ((first + i) / kPer),
                                              count - i, out + i);
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

class DDMTCpuEngine final : public detail::DDMTEngine {
public:
    DDMTCpuEngine(const plans::DDMTPlan& plan,
                  const detail::DDMTEngineConfig& cfg)
        : m_plan(plan),
          m_nthreads(std::max(1, cfg.exec.nthreads)),
          m_nbeams(cfg.nbeams) {
        init();
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
                "DDMT: Packed input buffer size mismatch: expected {} "
                "bytes ({} beams x {} chans x {} bytes/row), got {}",
                rows * row_bytes, m_nbeams, m_plan.get_nchans(), row_bytes,
                waterfall_packed.size()));
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
                "DDMT: Time-major input buffer size mismatch: expected {}, "
                "got {}",
                m_nbeams * nsamps * samp_bytes, filterbank_packed.size()));
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

    std::vector<int> m_active;    // unmasked channels, ascending
    int m_tdm{1};                 // DM trials per tile
    int m_spread{0};              // largest delay offset within a DM tile
    std::vector<int> m_base;      // [tile][active]: smallest delay in tile
    std::vector<uint16_t> m_offs; // [tile][d][active]: delay - base

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
                "DDMT::{}: plan nbits={} != 32; use the packed-integer "
                "overload instead",
                what, m_plan.get_nbits()));
        }
    }
    void check_packed(std::string_view what) const {
        if (!is_packed_width(m_plan.get_nbits())) {
            throw std::invalid_argument(std::format(
                "DDMT::{}: plan nbits={} is not a supported packed width "
                "(1,2,4,8,16); use the float overload for nbits==32",
                what, m_plan.get_nbits()));
        }
    }
    void check_warm() const {
        if (m_history_len != m_max_delay) {
            throw std::logic_error(
                std::format("DDMT::save_history: stream is not fully "
                            "warmed up yet ({} of {} history samples/channel)",
                            m_history_len, m_max_delay));
        }
    }
    static void
    check_size(SizeType got, SizeType expected, std::string_view what) {
        if (got != expected) {
            throw std::invalid_argument(
                std::format("DDMT: {} buffer size mismatch: expected {}, "
                            "got {}",
                            what, expected, got));
        }
    }

    void init() {
        const auto& pc   = m_plan.get_container();
        m_ndm            = pc.dm_arr.size();
        m_max_delay      = pc.delay_table.empty()
                               ? 0
                               : *std::ranges::max_element(pc.delay_table);
        m_hist_row_bytes = row_bytes_for(m_max_delay, pc.nbits);
        m_hist.assign(m_nbeams * pc.nchans * m_hist_row_bytes, 0);
        m_hist_next.assign(m_hist.size(), 0);
        for (SizeType c = 0; c < pc.nchans; ++c) {
            if (pc.kill_mask[c] != 0) {
                m_active.push_back(static_cast<int>(c));
            }
        }
        build_tiles();
    }

    /// Largest delay offset inside DM tiles of @p tdm trials.
    [[nodiscard]] SizeType tile_spread(int tdm) const {
        const auto& delays = m_plan.get_container().delay_table;
        const auto nchans  = m_plan.get_nchans();
        const auto ndm     = static_cast<int>(m_ndm);
        SizeType spread    = 0;
        for (int t0 = 0; t0 < ndm; t0 += tdm) {
            const int t1 = std::min(t0 + tdm, ndm);
            for (const int c : m_active) {
                auto mn = delays[(static_cast<SizeType>(t0) * nchans) + c];
                auto mx = mn;
                for (int dm = t0 + 1; dm < t1; ++dm) {
                    const auto v =
                        delays[(static_cast<SizeType>(dm) * nchans) + c];
                    mn = std::min(mn, v);
                    mx = std::max(mx, v);
                }
                spread = std::max(spread, mx - mn);
            }
        }
        return spread;
    }

    /// Picks the widest DM tile whose delay spread keeps a channel window
    /// within twice the time tile, and builds its base/offset tables.
    void build_tiles() {
        m_tdm    = 1;
        m_spread = 0;
        if (m_ndm == 0 || m_active.empty()) {
            return;
        }
        for (int tdm = kMaxTileDM; tdm >= 2; tdm /= 2) {
            if (static_cast<SizeType>(tdm) >= 2 * m_ndm) {
                continue;
            }
            const auto spread = tile_spread(tdm);
            if (spread <= static_cast<SizeType>(kTimeTile)) {
                m_tdm    = tdm;
                m_spread = static_cast<int>(spread);
                break;
            }
        }
        const auto& delays = m_plan.get_container().delay_table;
        const auto nchans  = m_plan.get_nchans();
        const auto nact    = m_active.size();
        const auto ntile   = (m_ndm + m_tdm - 1) / m_tdm;
        m_base.assign(ntile * nact, 0);
        m_offs.assign(ntile * m_tdm * nact, 0);
        for (SizeType t = 0; t < ntile; ++t) {
            const auto d_end = std::min<SizeType>(m_tdm, m_ndm - (t * m_tdm));
            for (SizeType k = 0; k < nact; ++k) {
                const auto c = static_cast<SizeType>(m_active[k]);
                auto at      = [&](SizeType d) {
                    return delays[(((t * m_tdm) + d) * nchans) + c];
                };
                auto mn = at(0);
                for (SizeType d = 1; d < d_end; ++d) {
                    mn = std::min(mn, at(d));
                }
                m_base[(t * nact) + k] = static_cast<int>(mn);
                for (SizeType d = 0; d < d_end; ++d) {
                    m_offs[(((t * m_tdm) + d) * nact) + k] =
                        static_cast<uint16_t>(at(d) - mn);
                }
            }
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
            dedisperse<NBITS>(seg, n_out, out);
        }
        const auto new_len = std::min(seg.total, m_max_delay);
        copy_tail<NBITS>(seg, m_nbeams * m_plan.get_nchans(),
                         seg.total - new_len, new_len, m_hist_next.data(),
                         m_hist_row_bytes, m_nthreads);
        std::swap(m_hist, m_hist_next);
        m_history_len = new_len;
    }

    /**
     * @brief Brute-force delay-and-sum over output samples [0, n_out).
     * @details
     * Work units are (beam, time tile, DM tile), DM tile fastest so that
     * concurrently running units read overlapping input. Per group of
     * kChanGroup active channels, a unit stages each channel's window
     * [t0 + base, t0 + base + T + spread) once (or points straight into the
     * input) and every DM of the tile adds its delayed slice into its
     * accumulator row, the group's channels summed in registers. Channels
     * are added in ascending order from zero, as on the GPU, so float
     * results match bitwise.
     */
    template <unsigned NBITS>
    void dedisperse(const Segments& seg, SizeType n_out, Acc<NBITS>* out) {
        using A           = Acc<NBITS>;
        using W           = Win<NBITS>;
        using P           = Part<NBITS>;
        constexpr bool kU = !std::is_same_v<P, A>;
        const auto nchans = m_plan.get_nchans();
        const auto nact   = static_cast<int>(m_active.size());
        const auto tdm    = static_cast<SizeType>(m_tdm);
        const auto wlen   = static_cast<SizeType>(kTimeTile + m_spread);
        // Window rows padded to an odd number of cache lines apart.
        constexpr SizeType kLine = 64 / sizeof(W);
        const auto wstride       = (((wlen + kLine - 1) / kLine) | 1) * kLine;
        const auto ntt           = (n_out + kTimeTile - 1) / kTimeTile;
        const auto ntd           = (m_ndm + tdm - 1) / tdm;
        const auto nunits        = m_nbeams * ntt * ntd;
        const auto ndm           = m_ndm;

#pragma omp parallel num_threads(m_nthreads)
        {
            std::vector<W> win(kChanGroup * wstride);
            std::vector<P> part(tdm * kTimeTile);
            std::vector<A> acc(kU ? tdm * kTimeTile : 0);

#pragma omp for schedule(dynamic, 1)
            for (SizeType u = 0; u < nunits; ++u) {
                const auto beam = u / (ntt * ntd);
                const auto rem  = u % (ntt * ntd);
                const auto tt   = rem / ntd;
                const auto td   = rem % ntd;
                const auto t0   = tt * kTimeTile;
                const int cnt =
                    static_cast<int>(std::min<SizeType>(kTimeTile, n_out - t0));
                const auto nd        = std::min(tdm, ndm - (td * tdm));
                const auto row0      = beam * nchans;
                const int* base      = m_base.data() + (td * nact);
                const uint16_t* offs = m_offs.data() + (td * tdm * nact);
                const auto len       = static_cast<SizeType>(cnt + m_spread);

                std::fill_n(part.data(), nd * kTimeTile, P{0});
                if constexpr (kU) {
                    std::fill_n(acc.data(), nd * kTimeTile, A{0});
                }
                int since_flush = 0;
                for (int k = 0; k < nact; k += kChanGroup) {
                    const int ng = std::min(kChanGroup, nact - k);
                    const W* wp[kChanGroup];
                    // Always staged: input rows are often a multiple of
                    // 4 KiB apart, and reading kChanGroup of them in step
                    // thrashes L1 sets (and 4K-aliases the accumulator
                    // stores). The scratch rows are padded apart instead.
                    for (int j = 0; j < ng; ++j) {
                        const auto r  = row0 + m_active[k + j];
                        const auto g0 = t0 + static_cast<SizeType>(base[k + j]);
                        W* dst        = win.data() + (j * wstride);
                        stage_window<NBITS>(seg, r, g0, len, dst);
                        wp[j] = dst;
                    }
                    for (SizeType d = 0; d < nd; ++d) {
                        const uint16_t* od = offs + (d * nact) + k;
                        P* __restrict__ a  = part.data() + (d * kTimeTile);
                        if (ng == kChanGroup) {
                            const W* __restrict__ p0 = wp[0] + od[0];
                            const W* __restrict__ p1 = wp[1] + od[1];
                            const W* __restrict__ p2 = wp[2] + od[2];
                            const W* __restrict__ p3 = wp[3] + od[3];
                            const W* __restrict__ p4 = wp[4] + od[4];
                            const W* __restrict__ p5 = wp[5] + od[5];
                            const W* __restrict__ p6 = wp[6] + od[6];
                            const W* __restrict__ p7 = wp[7] + od[7];
#pragma omp simd
                            for (int s = 0; s < cnt; ++s) {
                                P v = a[s];
                                v += static_cast<P>(p0[s]);
                                v += static_cast<P>(p1[s]);
                                v += static_cast<P>(p2[s]);
                                v += static_cast<P>(p3[s]);
                                v += static_cast<P>(p4[s]);
                                v += static_cast<P>(p5[s]);
                                v += static_cast<P>(p6[s]);
                                v += static_cast<P>(p7[s]);
                                a[s] = v;
                            }
                        } else {
                            for (int j = 0; j < ng; ++j) {
                                const W* __restrict__ p = wp[j] + od[j];
#pragma omp simd
                                for (int s = 0; s < cnt; ++s) {
                                    a[s] = static_cast<P>(a[s] +
                                                          static_cast<P>(p[s]));
                                }
                            }
                        }
                    }
                    if constexpr (kU) {
                        since_flush += ng;
                        if (since_flush + kChanGroup > kU16FlushChans ||
                            k + kChanGroup >= nact) {
                            flush(part.data(), acc.data(), nd, cnt);
                            since_flush = 0;
                        }
                    }
                }
                const A* src = nullptr;
                if constexpr (kU) {
                    src = acc.data();
                } else {
                    src = part.data();
                }
                auto* ob = out + (beam * ndm * n_out);
                for (SizeType d = 0; d < nd; ++d) {
                    const auto dm = (td * tdm) + d;
                    std::copy_n(src + (d * kTimeTile), cnt,
                                ob + (dm * n_out) + t0);
                }
            }
        }
    }

    /// Adds the uint16 partial sums into the int32 accumulators, zeroing
    /// the partials.
    template <typename P, typename A>
    static void flush(P* part, A* acc, SizeType nd, int cnt) {
        for (SizeType d = 0; d < nd; ++d) {
            P* __restrict__ p = part + (d * kTimeTile);
            A* __restrict__ a = acc + (d * kTimeTile);
#pragma omp simd
            for (int s = 0; s < cnt; ++s) {
                a[s] += static_cast<A>(p[s]);
                p[s] = 0;
            }
        }
    }
};

} // namespace

std::unique_ptr<detail::DDMTEngine>
detail::make_ddmt_cpu(const plans::DDMTPlan& plan,
                      const detail::DDMTEngineConfig& cfg) {
    return std::make_unique<DDMTCpuEngine>(plan, cfg);
}

} // namespace dmt::algorithms
