/**
 * @file ddmt_cpu.cpp
 * @brief DDMT CPU engine: direct (brute-force) delay-and-sum. Every trial
 * sums every active channel; nothing is shared between trials.
 */

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "ddmt_cpu_engine.hpp"

namespace dmt::algorithms {

namespace {

using ddmt_cpu::Acc;
using ddmt_cpu::kMaxSources;
using ddmt_cpu::kPad;
using ddmt_cpu::kU16FlushChans;
using ddmt_cpu::Segments;
using ddmt_cpu::stage_window;
using ddmt_cpu::Tiling;
using ddmt_cpu::Win;

// Channels staged together and summed in registers per accumulator pass:
// about 40 bytes of lanes per sample, so the group's windows stay in L1 while
// the accumulator row is read and written once per group (measured on
// AVX-512; 8 channels is ~10% slower, and 16 float windows spill L1).
template <typename W> constexpr int kChanGroup = sizeof(W) == 2 ? 16 : 10;
static_assert(kChanGroup<uint16_t> <= kMaxSources);

class DirectAlgo {
public:
    static constexpr const char* kName = "DDMT";
    // Output samples per work unit (time tile).
    static constexpr int kTimeTile = 512;

    DirectAlgo(const plans::DDMTPlan& plan,
               const Tiling& tiling,
               int nthreads,
               SizeType nbeams)
        : m_plan(plan),
          m_t(tiling),
          m_nthreads(nthreads),
          m_nbeams(nbeams) {
        // offs[tile slot][active] = delay - the channel's window start.
        const auto& pc  = plan.get_container();
        const auto nact = m_t.active.size();
        const auto tdm  = static_cast<SizeType>(m_t.tdm);
        m_offs.assign(m_t.ndm * nact, 0);
        for (SizeType i = 0; i < m_t.ndm; ++i) {
            const auto* row =
                pc.delay_table.data() + (m_t.order[i] * pc.nchans);
            const int* base = m_t.base.data() + ((i / tdm) * nact);
            for (SizeType k = 0; k < nact; ++k) {
                m_offs[(i * nact) + k] = static_cast<uint16_t>(
                    row[m_t.active[k]] - static_cast<SizeType>(base[k]));
            }
        }
    }

    /**
     * @brief Delay-and-sum over output samples [0, n_out).
     * @details
     * Work units are (beam, time tile, DM tile), DM tile fastest so that
     * concurrently running units read overlapping input. Per group of
     * kChanGroup active channels a unit stages each channel's window
     * [t0 + base, t0 + base + T + spread) once (unpacked and widened;
     * two-segment aware), and every trial of the tile adds the group's
     * delayed slices into its accumulator row in one kernel call.
     */
    template <unsigned NBITS>
    void dedisperse(const Segments& seg, SizeType n_out, Acc<NBITS>* out) {
        using A           = Acc<NBITS>;
        using W           = Win<NBITS>;
        constexpr bool kU = NBITS <= 8;
        const auto nchans = m_plan.get_nchans();
        const auto nact   = static_cast<int>(m_t.active.size());
        const auto tdm    = static_cast<SizeType>(m_t.tdm);
        // Window rows padded to an odd number of cache lines apart: input
        // rows are often a multiple of 4 KiB apart, and reading several in
        // step would thrash L1 sets.
        constexpr SizeType kLine = 64 / sizeof(W);
        const auto wlen = static_cast<SizeType>(kTimeTile + m_t.spread + kPad);
        const auto wstride = (((wlen + kLine - 1) / kLine) | 1) * kLine;
        const auto ntt     = (n_out + kTimeTile - 1) / kTimeTile;
        const auto ntd     = (m_t.ndm + tdm - 1) / tdm;
        const auto nunits  = m_nbeams * ntt * ntd;
        const auto ndm     = m_t.ndm;
        const auto nacts   = static_cast<SizeType>(nact);

#pragma omp parallel num_threads(m_nthreads)
        {
            std::vector<W> win(kChanGroup<W> * wstride, W{0});
            std::vector<W> part((tdm * kTimeTile) + kPad, W{0});
            std::vector<A> acc(kU ? tdm * kTimeTile : 0);
            const W* ptr[kChanGroup<W>];

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
                const int* base      = m_t.base.data() + (td * nacts);
                const int* spread    = m_t.spreads.data() + (td * nacts);
                const uint16_t* offs = m_offs.data() + (td * tdm * nacts);

                std::fill_n(part.data(), nd * kTimeTile, W{0});
                if constexpr (kU) {
                    std::fill_n(acc.data(), nd * kTimeTile, A{0});
                }
                int since_flush = 0;
                for (int k0 = 0; k0 < nact; k0 += kChanGroup<W>) {
                    const int ng = std::min(kChanGroup<W>, nact - k0);
                    for (int j = 0; j < ng; ++j) {
                        const auto k   = k0 + j;
                        const auto r   = row0 + m_t.active[k];
                        const auto g0  = t0 + static_cast<SizeType>(base[k]);
                        const auto len = static_cast<SizeType>(cnt + spread[k]);
                        stage_window<NBITS>(seg, r, g0, len,
                                            win.data() + (j * wstride));
                    }
                    for (SizeType d = 0; d < nd; ++d) {
                        const uint16_t* od = offs + (d * nacts) + k0;
                        for (int j = 0; j < ng; ++j) {
                            ptr[j] = win.data() + (j * wstride) + od[j];
                        }
                        ddmt_cpu::sum_into<kChanGroup<W>>(
                            part.data() + (d * kTimeTile), ptr, ng, cnt,
                            /*accumulate=*/true);
                    }
                    if constexpr (kU) {
                        since_flush += ng;
                        if (k0 + kChanGroup<W> >= nact ||
                            since_flush + kChanGroup<W> >
                                kU16FlushChans<NBITS>) {
                            ddmt_cpu::flush(part.data(), acc.data(), nd, cnt,
                                            kTimeTile);
                            since_flush = 0;
                        }
                    }
                }
                auto* ob = out + (beam * ndm * n_out);
                for (SizeType d = 0; d < nd; ++d) {
                    auto* o = ob + (m_t.order[(td * tdm) + d] * n_out) + t0;
                    if constexpr (kU) {
                        std::copy_n(acc.data() + (d * kTimeTile), cnt, o);
                    } else {
                        const W* p = part.data() + (d * kTimeTile);
                        for (int s = 0; s < cnt; ++s) {
                            o[s] = static_cast<A>(p[s]);
                        }
                    }
                }
            }
        }
    }

private:
    const plans::DDMTPlan& m_plan;
    const Tiling& m_t;
    int m_nthreads;
    SizeType m_nbeams;
    std::vector<uint16_t> m_offs; // [tile slot][active]: delay - base
};

} // namespace

std::unique_ptr<detail::DDMTEngine>
detail::make_ddmt_cpu(const plans::DDMTPlan& plan,
                      const detail::DDMTEngineConfig& cfg) {
    return std::make_unique<ddmt_cpu::CpuEngine<DirectAlgo>>(plan, cfg);
}

} // namespace dmt::algorithms
