#include "dmt/fdmt_fft_tree.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <format>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/dm_utils.hpp"
#include "dmt/fdmt_fft_common.hpp"
#include "dmt/fft.hpp"
#include "dmt/modes.hpp"

namespace dmt::algorithms::fdmt_fft {

namespace {

std::uint32_t to_u32(SizeType v) {
    if (v >= std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("FDMTFFT: tree index exceeds 32 bits");
    }
    return static_cast<std::uint32_t>(v);
}

SizeType index_of(SizeType buf_offset, SizeType nsamps) {
    if (nsamps == 0) {
        throw std::logic_error("FDMTFFT: coordinate with zero samples");
    }
    return buf_offset / nsamps;
}

// Merge work per node relative to one byte of device-memory traffic, per
// bin: a node is ~24 bytes of on-chip traffic plus a sincos, against ~1 TB/s
// of device memory.
constexpr double kOpCost = 2.0;

// Greedy cone chunking of one stage.
class StageBuilder {
public:
    StageBuilder(const TreeDag& dag, SizeType lo, SizeType hi)
        : m_dag(dag),
          m_lo(lo),
          m_hi(hi),
          m_stamp(hi - lo),
          m_list(hi - lo),
          m_local(hi - lo) {
        for (SizeType l = lo; l < hi; ++l) {
            m_stamp[l - lo].assign(dag.ncoords[l], 0);
            m_local[l - lo].assign(dag.ncoords[l], 0);
        }
    }

    std::optional<TreeStage> build(SizeType budget, SizeType max_out) {
        TreeStage st;
        st.level_in  = m_lo;
        st.level_out = m_hi;
        st.lvl_off.push_back(0);
        const auto n_out = m_dag.ncoords[m_hi];
        SizeType i       = 0;
        while (i < n_out) {
            begin();
            SizeType j = i;
            while (j < n_out && (j - i) < max_out) {
                add(j);
                if (inner_max() > budget) {
                    break;
                }
                ++j;
            }
            if (j == i) {
                return std::nullopt; // one node's cone exceeds the budget
            }
            if (j < n_out && (j - i) < max_out) {
                // Node j overflowed: rebuild [i, j) without it.
                begin();
                for (SizeType n = i; n < j; ++n) {
                    add(n);
                }
            }
            emit(st, i, j);
            i = j;
        }
        return st;
    }

private:
    const TreeDag& m_dag;
    SizeType m_lo;
    SizeType m_hi;
    std::uint32_t m_cur{0};
    std::vector<std::vector<std::uint32_t>> m_stamp; // levels lo..hi-1
    std::vector<std::vector<std::uint32_t>> m_list;  // nodes of the chunk
    std::vector<std::vector<std::uint32_t>> m_local; // node -> local index
    std::vector<std::uint32_t> m_new;
    std::vector<std::uint32_t> m_next;

    void begin() {
        ++m_cur;
        for (auto& l : m_list) {
            l.clear();
        }
    }

    void mark(SizeType l, std::uint32_t node) {
        auto& s = m_stamp[l - m_lo][node];
        if (s != m_cur) {
            s = m_cur;
            m_list[l - m_lo].push_back(node);
            m_next.push_back(node);
        }
    }

    void add(SizeType out) {
        m_new.assign(1, static_cast<std::uint32_t>(out));
        for (SizeType l = m_hi; l > m_lo; --l) {
            m_next.clear();
            for (const auto x : m_new) {
                const auto& op = m_dag.prod[l][x];
                mark(l - 1, op.tail);
                if (op.head != kTreeCopy) {
                    mark(l - 1, op.head);
                }
            }
            std::swap(m_new, m_next);
        }
    }

    [[nodiscard]] SizeType inner_max() const {
        SizeType m = 0;
        for (SizeType l = m_lo + 1; l < m_hi; ++l) {
            m = std::max<SizeType>(m, m_list[l - m_lo].size());
        }
        return m;
    }

    void emit(TreeStage& st, SizeType i, SizeType j) {
        for (SizeType l = m_lo + 1; l < m_hi; ++l) {
            auto& list = m_list[l - m_lo];
            std::ranges::sort(list);
            for (SizeType r = 0; r < list.size(); ++r) {
                m_local[l - m_lo][list[r]] = static_cast<std::uint32_t>(r);
            }
        }
        const auto map = [&](SizeType l, std::uint32_t node) {
            return (l == m_lo) ? node : m_local[l - m_lo][node];
        };
        for (SizeType l = m_lo + 1; l <= m_hi; ++l) {
            const auto emit_node = [&](std::uint32_t x) {
                const auto& op = m_dag.prod[l][x];
                st.ops.push_back({
                    .tail = map(l - 1, op.tail),
                    .head = (op.head == kTreeCopy) ? kTreeCopy
                                                   : map(l - 1, op.head),
                    .sid  = op.sid,
                });
            };
            if (l == m_hi) {
                for (SizeType x = i; x < j; ++x) {
                    emit_node(static_cast<std::uint32_t>(x));
                }
            } else {
                for (const auto x : m_list[l - m_lo]) {
                    emit_node(x);
                }
            }
            st.lvl_off.push_back(to_u32(st.ops.size()));
        }
        st.out_begin.push_back(to_u32(i));
        st.max_nodes   = std::max(st.max_nodes, inner_max());
        const auto& in = m_list[0];
        if (m_lo == 0) {
            // Level-0 nodes are read as channel spectra.
            std::vector<std::uint32_t> chans;
            chans.reserve(in.size());
            for (const auto x : in) {
                chans.push_back(m_dag.l0_chan[x]);
            }
            std::ranges::sort(chans);
            st.in_rows += static_cast<SizeType>(
                std::ranges::unique(chans).begin() - chans.begin());
        } else {
            st.in_rows += in.size();
        }
    }
};

double stage_cost(const TreeDag& dag, const TreeStage& st) {
    return (8.0 * static_cast<double>(st.in_rows + dag.ncoords[st.level_out])) +
           (kOpCost * static_cast<double>(st.ops.size()));
}

// Rough relative FFT cost per n * log2(n) from the length's factors, used
// only to shortlist candidate lengths before asking FFTW.
double radix_penalty(SizeType n) {
    double p = 1.0;
    for (const SizeType f : {3U, 5U, 7U}) {
        while (n % f == 0) {
            n /= f;
            p *= (f == 3) ? 1.03 : 1.1;
        }
    }
    return p;
}

// Shortest overlap-save segment considered (unless capped).
constexpr SizeType kMinSegment = 2048;

} // namespace

SizeType TreeDag::nops() const noexcept {
    SizeType n = 0;
    for (const auto& l : prod) {
        n += l.size();
    }
    return n;
}

TreeDag make_tree_dag(const plans::FDMTPlan& plan, bool fractional, bool box) {
    const auto& pc    = plan.get_container();
    const auto niters = plan.get_niters();
    TreeDag dag;
    dag.nchans = plan.get_nchans();
    dag.ncoords.resize(niters + 1);
    for (SizeType l = 0; l <= niters; ++l) {
        dag.ncoords[l] = pc.state_shape[l].ncoords;
    }
    dag.l0_chan.assign(dag.ncoords[0], 0);
    dag.l0_shift.assign(dag.ncoords[0], 0);
    for (SizeType c = 0; c < pc.grids[0].size(); ++c) {
        const auto& g = pc.grids[0][c];
        for (SizeType i = 0; i < g.ndt; ++i) {
            const auto s = static_cast<SizeType>(std::abs(g.dt_grid[i]));
            dag.l0_chan[g.coord_offset + i]  = to_u32(c);
            dag.l0_shift[g.coord_offset + i] = to_u32(s);
            dag.dt0_max                      = std::max(dag.dt0_max, s);
        }
    }
    const auto frac =
        fractional ? fractional_delays(plan, box) : FractionalDelays{};
    dag.prod.resize(niters + 1);
    SizeType max_delay = 0;
    for (SizeType l = 1; l <= niters; ++l) {
        auto& prod = dag.prod[l];
        prod.assign(dag.ncoords[l], TreeOp{kTreeCopy, kTreeCopy, 0});
        std::vector<bool> seen(dag.ncoords[l], false);
        const auto& sums = pc.coordinates_sum[l];
        for (SizeType i = 0; i < sums.size(); ++i) {
            const auto& c   = sums[i];
            const auto cur  = index_of(c.buf_offset, c.nsamps);
            std::uint32_t s = 0;
            if (fractional) {
                s = to_u32(dag.shift.size());
                dag.shift.push_back(frac.shift[l][i]);
            } else {
                s         = to_u32(c.delay);
                max_delay = std::max(max_delay, c.delay);
            }
            prod[cur] = {
                .tail = to_u32(index_of(c.tail_buf_offset, c.tail_nsamps)),
                .head = to_u32(index_of(c.head_buf_offset, c.head_nsamps)),
                .sid  = s,
            };
            seen[cur] = true;
        }
        for (const auto& c : pc.coordinates_copy[l]) {
            const auto cur = index_of(c.buf_offset, c.nsamps);
            prod[cur]      = {
                .tail = to_u32(index_of(c.tail_buf_offset, c.tail_nsamps)),
                .head = kTreeCopy,
                .sid  = 0,
            };
            seen[cur] = true;
        }
        if (!std::ranges::all_of(seen, [](bool b) { return b; })) {
            throw std::logic_error(std::format(
                "FDMTFFT: level {} has a node without producer", l));
        }
    }
    if (!fractional) {
        dag.shift.resize(max_delay + 1);
        for (SizeType s = 0; s <= max_delay; ++s) {
            dag.shift[s] = static_cast<double>(s);
        }
    }
    if (dag.shift.empty()) {
        dag.shift.push_back(0.0);
    }
    return dag;
}

std::optional<TreeStage> make_tree_stage(const TreeDag& dag,
                                         SizeType level_in,
                                         SizeType level_out,
                                         SizeType budget,
                                         SizeType max_out) {
    if (level_in >= level_out || level_out > dag.levels()) {
        throw std::invalid_argument("FDMTFFT: bad tree stage levels");
    }
    StageBuilder b(dag, level_in, level_out);
    return b.build(budget, std::max<SizeType>(max_out, 1));
}

std::vector<TreeStage>
plan_tree_stages(const TreeDag& dag, SizeType budget, SizeType max_out) {
    const auto nl = dag.levels();
    if (nl == 0) {
        return {};
    }
    // best[h]: cheapest stages 0 -> h; from[h]: the last stage's input level.
    // A wider stage has a larger cone, so once (l, h) does not fit, no l' < l
    // fits either.
    std::vector<double> best(nl + 1, std::numeric_limits<double>::max());
    std::vector<SizeType> from(nl + 1, 0);
    best[0] = 0.0;
    for (SizeType h = 1; h <= nl; ++h) {
        for (SizeType l = h; l-- > 0;) {
            const auto st = make_tree_stage(dag, l, h, budget, max_out);
            if (!st) {
                break;
            }
            const double c = best[l] + stage_cost(dag, *st);
            if (c < best[h]) {
                best[h] = c;
                from[h] = l;
            }
        }
    }
    std::vector<TreeStage> stages;
    for (SizeType h = nl; h > 0; h = from[h]) {
        stages.push_back(*make_tree_stage(dag, from[h], h, budget, max_out));
    }
    std::ranges::reverse(stages);
    return stages;
}

std::vector<TreeStage> level_tree_stages(const TreeDag& dag, SizeType max_out) {
    std::vector<TreeStage> stages;
    for (SizeType l = 1; l <= dag.levels(); ++l) {
        stages.push_back(*make_tree_stage(dag, l - 1, l, 0, max_out));
    }
    return stages;
}

Segments choose_segments(const Geometry& geom,
                         FDMTMode mode,
                         SizeType nsamps_out,
                         SizeType tree_nodes,
                         SizeType nchans,
                         SizeType ndms,
                         SizeType max_len) {
    const auto n_single = geom.n_fft;
    Segments best_seg{
        .n_fft = n_single,
        .nseg  = 1,
        .hop   = nsamps_out,
        .skip  = geom.skip,
        .zeros = 0,
    };
    if (mode == FDMTMode::kRoll) {
        return best_seg; // cyclic: the block is the period
    }
    const bool capped = max_len > 0 && max_len < n_single;
    // A segment keeps transform samples [skip, skip + hop): at least the
    // tree support behind the first (the single transform's skip in valid
    // mode, which starts with the history) and, with fractional delays,
    // `guard` samples of look-ahead past the last.
    const auto discard = std::max(geom.skip, geom.support);
    const auto ahead   = geom.guard;
    // FFT work of (nchans + ndms) rows vs tree work of `nodes` nodes on n/2
    // bins, a node costing about 1.5 FFT butterflies' worth.
    const double tree = 1.5 * 0.33 * static_cast<double>(tree_nodes) /
                        static_cast<double>(nchans + ndms);
    const auto model  = [&](SizeType n, SizeType nseg, double fft) {
        const double nd = static_cast<double>(n);
        return static_cast<double>(nseg) * nd * ((std::log2(nd) * fft) + tree);
    };
    struct Cand {
        SizeType n;
        SizeType nseg;
        double rough;
    };
    std::vector<Cand> cands;
    const auto n_min = (2 * (discard + ahead)) + 2;
    for (SizeType n = utils::next_fft_size(
             capped ? n_min : std::max(kMinSegment, n_min));
         n < n_single && (!capped || n <= max_len);
         n = utils::next_fft_size(n + 1)) {
        const auto hop  = n - discard - ahead;
        const auto nseg = (nsamps_out + hop - 1) / hop;
        if (nseg < 2 && !capped) {
            break;
        }
        cands.push_back({n, nseg, model(n, nseg, 0.33 * radix_penalty(n))});
    }
    if (capped && cands.empty()) {
        throw std::invalid_argument(std::format(
            "FDMTFFT: segment cap {} is shorter than the minimum overlap-save "
            "segment {}",
            max_len, utils::next_fft_size(n_min)));
    }
    std::ranges::sort(cands, {}, &Cand::rough);
    if (cands.size() > 8) {
        cands.resize(8);
    }
    double best = capped
                      ? std::numeric_limits<double>::max()
                      : model(n_single, 1, utils::r2c_cost_per_nlogn(n_single));
    for (const auto& c : cands) {
        const double cost = model(c.n, c.nseg, utils::r2c_cost_per_nlogn(c.n));
        // Segment only for a clear predicted gain: the estimate is rough and
        // every segment adds fixed per-pass costs.
        if (cost < 0.9 * best || (capped && cost < best)) {
            best     = cost;
            best_seg = {
                .n_fft = c.n,
                .nseg  = c.nseg,
                .hop   = c.n - discard - ahead,
                .skip  = discard,
                .zeros = discard - geom.skip,
            };
        }
    }
    return best_seg;
}

} // namespace dmt::algorithms::fdmt_fft
