/**
 * @file sdmt_cpu.cpp
 * @brief SDMT CPU engine: exact delay-and-sum with partial sums shared
 * between DM trials within subbands (see SharedSumAlgo::build_programs()).
 */

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "ddmt_cpu_engine.hpp"

namespace dmt::algorithms {

namespace {

using ddmt_cpu::Acc;
using ddmt_cpu::kPad;
using ddmt_cpu::kU16FlushChans;
using ddmt_cpu::Segments;
using ddmt_cpu::stage_window;
using ddmt_cpu::Tiling;
using ddmt_cpu::Win;

// Active channels per subband: the prefix-tree depth (see build_programs()).
constexpr int kSubband = 16;
// Sources per kernel call in a program (the kernel takes up to
// ddmt_cpu::kMaxSources).
constexpr int kMaxSources = 8;
static_assert(kMaxSources <= ddmt_cpu::kMaxSources);
// Subbands per group and the group's leaf pool (see build_programs()).
constexpr int kGroupSubbands = 8;
constexpr int kPoolSlots     = 128;
// One combine pass per trial and group must fit one kernel call, and a
// group's uint16 partial sums must not overflow.
static_assert(kGroupSubbands <= kMaxSources);
static_assert(kGroupSubbands * kSubband <= kU16FlushChans<8> / 2);

class SharedSumAlgo {
public:
    static constexpr const char* kName = "SDMT";
    // Output samples per work unit (time tile). A unit's accumulators (up
    // to kMaxTileDM x kTimeTile values) stay in L2.
    static constexpr int kTimeTile = 512;

    SharedSumAlgo(const plans::DDMTPlan& plan,
                  const Tiling& tiling,
                  int nthreads,
                  SizeType nbeams)
        : m_plan(plan),
          m_t(tiling),
          m_nthreads(nthreads),
          m_nbeams(nbeams) {
        build_programs();
    }

    /**
     * @brief Delay-and-sum over output samples [0, n_out).
     * @details
     * Work units are (beam, time tile, DM tile), DM tile fastest so that
     * concurrently running units read overlapping input. Per subband a unit
     * stages each channel's window [t0 + base, t0 + base + T + spread) once
     * (unpacked and widened; two-segment aware) and runs the subband's
     * program (build_programs()) over the windows, its slots and the
     * accumulator rows of the tile's trials.
     */
    template <unsigned NBITS>
    void dedisperse(const Segments& seg, SizeType n_out, Acc<NBITS>* out) {
        using A           = Acc<NBITS>;
        using W           = Win<NBITS>;
        constexpr bool kU = NBITS <= 8;
        const auto nchans = m_plan.get_nchans();
        const auto nact   = static_cast<SizeType>(m_t.active.size());
        const auto nsub   = m_sub_start.size() - 1;
        const auto tdm    = static_cast<SizeType>(m_t.tdm);
        // Window and slot rows padded to an odd number of cache lines apart:
        // input rows are often a multiple of 4 KiB apart, and reading several
        // in step would thrash L1 sets.
        constexpr SizeType kLine = 64 / sizeof(W);
        const auto stride        = [](SizeType n) {
            return (((n + kLine - 1) / kLine) | 1) * kLine;
        };
        const auto wstride =
            stride(static_cast<SizeType>(kTimeTile + m_t.spread + kPad));
        const auto ntt    = (n_out + kTimeTile - 1) / kTimeTile;
        const auto ntd    = (m_t.ndm + tdm - 1) / tdm;
        const auto nunits = m_nbeams * ntt * ntd;
        const auto ndm    = m_t.ndm;
        const auto nslots = static_cast<SizeType>(m_nslots);

#pragma omp parallel num_threads(m_nthreads)
        {
            std::vector<W> win(kSubband * wstride, W{0});
            std::vector<W> slots(nslots * wstride, W{0});
            std::vector<W> part((tdm * kTimeTile) + kPad, W{0});
            std::vector<A> acc(kU ? tdm * kTimeTile : 0);
            const W* ptr[kMaxSources];

#pragma omp for schedule(dynamic, 1)
            for (SizeType u = 0; u < nunits; ++u) {
                const auto beam = u / (ntt * ntd);
                const auto rem  = u % (ntt * ntd);
                const auto tt   = rem / ntd;
                const auto td   = rem % ntd;
                const auto t0   = tt * kTimeTile;
                const int cnt =
                    static_cast<int>(std::min<SizeType>(kTimeTile, n_out - t0));
                const auto nd     = std::min(tdm, ndm - (td * tdm));
                const auto row0   = beam * nchans;
                const int* base   = m_t.base.data() + (td * nact);
                const int* spread = m_t.spreads.data() + (td * nact);

                std::fill_n(part.data(), nd * kTimeTile, W{0});
                if constexpr (kU) {
                    std::fill_n(acc.data(), nd * kTimeTile, A{0});
                }
                int since_flush = 0;
                for (SizeType sb = 0; sb < nsub; ++sb) {
                    const int* chans = m_sub_chan.data() + m_sub_start[sb];
                    const int g      = m_sub_start[sb + 1] - m_sub_start[sb];
                    for (int j = 0; j < g; ++j) {
                        const auto k   = static_cast<SizeType>(chans[j]);
                        const auto r   = row0 + m_t.active[k];
                        const auto g0  = t0 + static_cast<SizeType>(base[k]);
                        const auto len = static_cast<SizeType>(cnt + spread[k]);
                        stage_window<NBITS>(seg, r, g0, len,
                                            win.data() + (j * wstride));
                    }
                    const auto p0 = m_prog[(td * nsub) + sb];
                    const auto p1 = m_prog[(td * nsub) + sb + 1];
                    for (auto i = p0; i < p1; ++i) {
                        const Op& op   = m_ops[i];
                        const Src* src = m_srcs.data() + op.src;
                        for (int k = 0; k < op.nsrc; ++k) {
                            const W* b =
                                src[k].slot != 0
                                    ? slots.data() + (src[k].ref * wstride)
                                    : win.data() + (src[k].ref * wstride);
                            ptr[k] = b + src[k].off;
                        }
                        const bool to_row = (op.flags & kToRow) != 0;
                        W* dst = to_row ? part.data() + (op.dst * kTimeTile)
                                        : slots.data() + (op.dst * wstride);
                        ddmt_cpu::sum_into<kMaxSources>(
                            dst, ptr, op.nsrc, to_row ? cnt : cnt + op.span,
                            (op.flags & kAccumulate) != 0);
                    }
                    if constexpr (kU) {
                        // Deferred leaves land at group ends: flush only
                        // there, before a further group could overflow.
                        since_flush += g;
                        if (m_group_end[(td * nsub) + sb] != 0 &&
                            (sb + 1 == nsub ||
                             since_flush + (kGroupSubbands * kSubband) >
                                 kU16FlushChans<NBITS>)) {
                            ddmt_cpu::flush(part.data(), acc.data(), nd, cnt,
                                            kTimeTile);
                            since_flush = 0;
                        }
                    }
                }
                auto* ob = out + (beam * ndm * n_out);
                for (SizeType d = 0; d < nd; ++d) {
                    const auto dm = m_t.order[(td * tdm) + d];
                    auto* o       = ob + (dm * n_out) + t0;
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

    /// One kernel call of a subband program (see build_programs()).
    struct Op {
        uint32_t src;  // first entry in m_srcs
        uint16_t dst;  // slot, or DM trial within the tile (kToRow)
        uint16_t span; // slot length is span + time-tile length
        uint8_t nsrc;  // 1..kMaxSources
        uint8_t flags; // kAccumulate | kToRow
    };
    /// A kernel source: staged channel window or slot, read from off.
    struct Src {
        int32_t off;
        uint16_t ref; // window (position in the subband) or slot
        uint8_t slot; // 1: ref is a slot
    };
    static constexpr uint8_t kAccumulate = 1;
    static constexpr uint8_t kToRow      = 2;
    // Slots [0, kPoolBase) hold prefix-tree nodes, [kPoolBase, ...) the
    // shared subband sums (leaves) of the current group.
    static constexpr int kPoolBase = kSubband + 1;

    std::vector<int> m_sub_start;     // [subband + 1] into m_sub_chan
    std::vector<int> m_sub_chan;      // active-channel indices, tree order
    std::vector<uint32_t> m_prog;     // [tile * nsub + subband + 1] into m_ops
    std::vector<uint8_t> m_group_end; // [tile * nsub + subband]
    std::vector<Op> m_ops;
    std::vector<Src> m_srcs;
    int m_nslots{1}; // slots any program uses

    /**
     * @brief Builds the per-(DM tile, subband) kernel programs.
     * @details
     * Active channels are split into subbands of kSubband channels, ordered
     * by increasing delay. Within a subband write each trial's delays as
     * base(d) + r(d, j). Trials whose r agree on the first j channels share
     * the partial sum over those channels exactly, so the program walks the
     * prefix tree of the r rows: a branching node's partial sum is written
     * once into a slot, which its children then extend. A trial with a
     * prefix of its own adds the rest of its channels straight into its
     * accumulator row. A trial reads its subband sum at offset base(d).
     * Nothing is approximated: every trial still sums
     * x[c][t + delay(d, c)] over the active channels; only the order of
     * the float additions differs from a channel-by-channel sum.
     *
     * A subband sum shared by several trials (a leaf) goes to the group's
     * leaf pool. Once a group of up to kGroupSubbands subbands is done, one
     * pass per trial adds all its pooled leaves into its row, which saves
     * most of the accumulator traffic.
     *
     * When sharing does not pay (sparse or coarse DM grids), a subband falls
     * back to the plain program: every trial sums its channels directly. The
     * choice uses an estimate of loads and stores per sample.
     */
    void build_programs() {
        const auto& pc    = m_plan.get_container();
        const auto nchans = pc.nchans;
        const auto nact   = static_cast<int>(m_t.active.size());
        m_sub_start.clear();
        m_sub_chan.clear();
        m_prog.assign(1, 0);
        m_group_end.clear();
        m_ops.clear();
        m_srcs.clear();
        m_nslots = 1;
        if (m_t.ndm == 0 || nact == 0) {
            m_sub_start.push_back(0);
            return;
        }
        // Subbands of consecutive active channels, each ordered by delay.
        const auto& frac = pc.fractional_delay_table;
        for (int k0 = 0; k0 < nact; k0 += kSubband) {
            m_sub_start.push_back(static_cast<int>(m_sub_chan.size()));
            const int k1 = std::min(k0 + kSubband, nact);
            std::vector<int> ks(static_cast<SizeType>(k1 - k0));
            for (int k = k0; k < k1; ++k) {
                ks[static_cast<SizeType>(k - k0)] = k;
            }
            if (frac.size() == nchans) {
                std::ranges::stable_sort(ks, [&](int x, int y) {
                    return frac[static_cast<SizeType>(m_t.active[x])] <
                           frac[static_cast<SizeType>(m_t.active[y])];
                });
            }
            m_sub_chan.insert(m_sub_chan.end(), ks.begin(), ks.end());
        }
        m_sub_start.push_back(static_cast<int>(m_sub_chan.size()));
        const auto nsub  = m_sub_start.size() - 1;
        const auto tdm   = static_cast<SizeType>(m_t.tdm);
        const auto ntile = (m_t.ndm + tdm - 1) / tdm;

        // Tiles are independent: build them in parallel, then concatenate.
        std::vector<TileProgram> tiles(ntile);
#pragma omp parallel for num_threads(m_nthreads) schedule(dynamic, 1)
        for (SizeType t = 0; t < ntile; ++t) {
            tiles[t] = build_tile(t, nsub);
        }
        for (const auto& tp : tiles) {
            const auto op0 = static_cast<uint32_t>(m_ops.size());
            const auto s0  = static_cast<uint32_t>(m_srcs.size());
            for (auto op : tp.ops) {
                op.src += s0;
                m_ops.push_back(op);
            }
            m_srcs.insert(m_srcs.end(), tp.srcs.begin(), tp.srcs.end());
            for (const auto end : tp.prog_end) {
                m_prog.push_back(op0 + end);
            }
            m_group_end.insert(m_group_end.end(), tp.group_end.begin(),
                               tp.group_end.end());
            m_nslots = std::max(m_nslots, tp.nslots);
        }
    }

    /// Programs of one DM tile (op and source indices local to the tile).
    struct TileProgram {
        std::vector<Op> ops;
        std::vector<Src> srcs;
        std::vector<uint32_t> prog_end; // [subband]: end of its ops
        std::vector<uint8_t> group_end; // [subband]
        int nslots{1};

        void append(const std::vector<Op>& o, const std::vector<Src>& s) {
            const auto s0 = static_cast<uint32_t>(srcs.size());
            for (auto op : o) {
                op.src += s0;
                ops.push_back(op);
            }
            srcs.insert(srcs.end(), s.begin(), s.end());
        }
    };

    [[nodiscard]] TileProgram build_tile(SizeType t, SizeType nsub) const {
        const auto& delays = m_plan.get_container().delay_table;
        const auto nchans  = m_plan.get_nchans();
        const auto nact    = m_t.active.size();
        const auto tdm     = static_cast<SizeType>(m_t.tdm);
        const auto nd = static_cast<int>(std::min(tdm, m_t.ndm - (t * tdm)));
        TileProgram tp;
        std::vector<int64_t> dly;  // [d][j]
        std::vector<int64_t> base; // [d]
        std::vector<int> order;
        std::vector<Op> plain_ops;
        std::vector<Src> plain_srcs;
        std::vector<Op> tree_ops;
        std::vector<Src> tree_srcs;
        std::vector<std::pair<int, Src>> tree_defer;
        std::vector<std::vector<Src>> group(static_cast<SizeType>(nd));
        int pool_used = 0;
        int in_group  = 0;
        for (SizeType sb = 0; sb < nsub; ++sb) {
            const int* chans = m_sub_chan.data() + m_sub_start[sb];
            const int g      = m_sub_start[sb + 1] - m_sub_start[sb];
            const auto gs    = static_cast<SizeType>(g);
            dly.assign(static_cast<SizeType>(nd) * gs, 0);
            base.assign(static_cast<SizeType>(nd), 0);
            for (int d = 0; d < nd; ++d) {
                const auto row = m_t.order[(t * tdm) + d] * nchans;
                int64_t mn     = INT64_MAX;
                for (int j = 0; j < g; ++j) {
                    const auto v = static_cast<int64_t>(
                        delays[row + m_t.active[chans[j]]]);
                    dly[(d * gs) + j] = v;
                    mn                = std::min(mn, v);
                }
                base[d] = mn;
            }
            ProgramBuilder pb{.g     = g,
                              .chans = chans,
                              .bch   = m_t.base.data() + (t * nact),
                              .dly   = dly.data(),
                              .gs    = gs,
                              .base  = base.data()};
            // Plain program: every trial sums its own channels.
            plain_ops.clear();
            plain_srcs.clear();
            pb.ops  = &plain_ops;
            pb.srcs = &plain_srcs;
            for (int d = 0; d < nd; ++d) {
                pb.chain.clear();
                for (int j = 0; j < g; ++j) {
                    pb.chain.push_back(j);
                }
                pb.to_row(d, -1, 0);
            }
            const auto plain_cost = pb.cost;
            // Prefix-tree program.
            order.resize(static_cast<SizeType>(nd));
            for (int d = 0; d < nd; ++d) {
                order[d] = d;
            }
            std::ranges::sort(order, [&](int x, int y) {
                const auto* a = dly.data() + (x * gs);
                const auto* b = dly.data() + (y * gs);
                for (int j = 0; j < g; ++j) {
                    const auto ra = a[j] - base[x];
                    const auto rb = b[j] - base[y];
                    if (ra != rb) {
                        return ra < rb;
                    }
                }
                return x < y;
            });
            tree_ops.clear();
            tree_srcs.clear();
            tree_defer.clear();
            pb.ops       = &tree_ops;
            pb.srcs      = &tree_srcs;
            pb.defer     = &tree_defer;
            pb.pool_used = pool_used;
            pb.cost      = 0;
            pb.order     = order.data();
            pb.chain.clear();
            pb.walk(0, nd, 0, -1, 0, 0);
            const bool use_tree = pb.cost < plain_cost;
            if (use_tree) {
                tp.nslots = std::max(tp.nslots, pb.max_slot + 1);
                pool_used = pb.pool_used;
                for (const auto& [d, src] : tree_defer) {
                    group[static_cast<SizeType>(d)].push_back(src);
                }
                tp.append(tree_ops, tree_srcs);
            } else {
                tp.append(plain_ops, plain_srcs);
            }
            // Close the group: one pass per trial adds its deferred leaves
            // (at most one per subband) into its row.
            ++in_group;
            const bool end = sb + 1 == nsub || in_group == kGroupSubbands ||
                             pool_used > (kPoolSlots * 3) / 4;
            if (end) {
                std::vector<Op> cops;
                std::vector<Src> csrcs;
                pb.ops  = &cops;
                pb.srcs = &csrcs;
                for (int d = 0; d < nd; ++d) {
                    auto& list = group[static_cast<SizeType>(d)];
                    if (!list.empty()) {
                        pb.emit(static_cast<uint16_t>(d), kAccumulate | kToRow,
                                0, list);
                        list.clear();
                    }
                }
                tp.append(cops, csrcs);
                pool_used = 0;
                in_group  = 0;
            }
            tp.prog_end.push_back(static_cast<uint32_t>(tp.ops.size()));
            tp.group_end.push_back(end ? 1 : 0);
        }
        return tp;
    }

    /// Emits one subband program; see build_programs().
    struct ProgramBuilder {
        int g{0};
        const int* chans{nullptr};   // active-channel index per position
        const int* bch{nullptr};     // window start (tile's smallest delay)
        const int64_t* dly{nullptr}; // [d][j]
        SizeType gs{0};
        const int64_t* base{nullptr}; // [d]
        const int* order{nullptr};
        std::vector<Op>* ops{nullptr};
        std::vector<Src>* srcs{nullptr};
        // Shared leaves read at group end: (trial, pool slot + offset).
        std::vector<std::pair<int, Src>>* defer{nullptr};
        int pool_used{0};
        std::vector<int> chain{}; // pending channel positions of the node
        // Estimated loads + stores (x4) per time-tile sample.
        int64_t cost{0};
        int max_slot{0};

        [[nodiscard]] int64_t r(int d, int j) const {
            return dly[(d * gs) + j] - base[d];
        }
        [[nodiscard]] int64_t win_off(int64_t lo, int d, int j) const {
            return lo + r(d, j) - bch[chans[j]];
        }

        /// Emits dst (+)= sum of sources in ops of at most kMaxSources.
        void emit(uint16_t dst,
                  uint8_t flags,
                  uint16_t span,
                  const std::vector<Src>& in) {
            const auto len = static_cast<int64_t>(kTimeTile) + span;
            for (SizeType i = 0; i < in.size(); i += kMaxSources) {
                const auto n = std::min<SizeType>(kMaxSources, in.size() - i);
                ops->push_back(Op{static_cast<uint32_t>(srcs->size()), dst,
                                  span, static_cast<uint8_t>(n), flags});
                srcs->insert(srcs->end(), in.begin() + static_cast<long>(i),
                             in.begin() + static_cast<long>(i + n));
                const auto rmw = (flags & kAccumulate) != 0 ? 2 : 1;
                cost += 4 * (static_cast<int64_t>(n) + rmw) * len;
                flags |= kAccumulate;
            }
        }

        /// Sources of a node starting at lo: slot (or none) plus the chain;
        /// @p d is any trial under the node (they share r on the chain).
        std::vector<Src> sources(int slot, int64_t lo_slot, int64_t lo, int d) {
            std::vector<Src> in;
            if (slot >= 0) {
                in.push_back(Src{static_cast<int32_t>(lo - lo_slot),
                                 static_cast<uint16_t>(slot), 1});
            }
            for (const int j : chain) {
                in.push_back(Src{static_cast<int32_t>(win_off(lo, d, j)),
                                 static_cast<uint16_t>(j), 0});
            }
            return in;
        }

        /// Trial d adds its node's sum (slot + chain) into its row.
        void to_row(int d, int slot, int64_t lo_slot) {
            const auto in = sources(slot, lo_slot, base[d], d);
            if (!in.empty()) {
                emit(static_cast<uint16_t>(d), kAccumulate | kToRow, 0, in);
            }
        }

        /// Writes the node of trials order[a, b) into slot @p dst; returns
        /// the slot's first sample (the smallest base).
        int64_t to_slot(int a, int b, int dst, int slot, int64_t lo_slot) {
            int64_t lo = INT64_MAX;
            int64_t hi = INT64_MIN;
            for (int i = a; i < b; ++i) {
                lo = std::min(lo, base[order[i]]);
                hi = std::max(hi, base[order[i]]);
            }
            emit(static_cast<uint16_t>(dst), 0, static_cast<uint16_t>(hi - lo),
                 sources(slot, lo_slot, lo, order[a]));
            max_slot = std::max(max_slot, dst);
            return lo;
        }

        /// Trials order[a, b) share channels [0, j): their sum is slot
        /// (starting at lo_slot, or zero if slot < 0) plus `chain`.
        void walk(int a, int b, int j, int slot, int64_t lo_slot, int next) {
            if (j == g) {
                if (b - a == 1) {
                    to_row(order[a], slot, lo_slot);
                    return;
                }
                if (!chain.empty() && defer != nullptr &&
                    pool_used < kPoolSlots) {
                    // A shared subband sum: kept in the pool and added to
                    // each trial's row together with the group's others.
                    const int dst = kPoolBase + pool_used++;
                    const auto lo = to_slot(a, b, dst, slot, lo_slot);
                    for (int i = a; i < b; ++i) {
                        const int d = order[i];
                        defer->emplace_back(
                            d, Src{static_cast<int32_t>(base[d] - lo),
                                   static_cast<uint16_t>(dst), 1});
                        // About a quarter of a shared combine pass.
                        cost += 5 * static_cast<int64_t>(kTimeTile);
                    }
                    return;
                }
                if (!chain.empty()) {
                    lo_slot = to_slot(a, b, next, slot, lo_slot);
                    slot    = next;
                }
                const auto saved = chain;
                chain.clear();
                for (int i = a; i < b; ++i) {
                    to_row(order[i], slot, lo_slot);
                }
                chain = saved;
                return;
            }
            // Children: runs of equal r(., j) (order is lexicographic).
            if (r(order[a], j) == r(order[b - 1], j)) {
                chain.push_back(j);
                walk(a, b, j + 1, slot, lo_slot, next);
                chain.pop_back();
                return;
            }
            const auto saved = chain;
            if (!chain.empty()) {
                lo_slot = to_slot(a, b, next, slot, lo_slot);
                slot    = next;
                ++next;
            }
            for (int i = a; i < b;) {
                int e = i + 1;
                while (e < b && r(order[e], j) == r(order[i], j)) {
                    ++e;
                }
                chain.assign(1, j);
                walk(i, e, j + 1, slot, lo_slot, next);
                i = e;
            }
            chain = saved;
        }
    };
};

} // namespace

std::unique_ptr<detail::DDMTEngine>
detail::make_sdmt_cpu(const plans::DDMTPlan& plan,
                      const detail::DDMTEngineConfig& cfg) {
    return std::make_unique<ddmt_cpu::CpuEngine<SharedSumAlgo>>(plan, cfg);
}

} // namespace dmt::algorithms
