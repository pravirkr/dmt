#include "dmt/algorithms/fdmt.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

#include "dmt/bit_pack_utils.hpp"
#include "dmt/engines.hpp"
#include "dmt/fdmt_box_weights.hpp"
#include "dmt/fdmt_chunk.hpp"
#include "dmt/fdmt_history.hpp"
#include "dmt/fdmt_int_tree.hpp"
#include "dmt/fdmt_skew.hpp"
#include "dmt/modes.hpp"

#include "fdmt_cpu_engine.hpp"

// CPU FDMT engine: construction, path and memory planning.

namespace dmt::algorithms::detail::fdmt_cpu {

FDMTCpuEngine::FDMTCpuEngine(const plans::FDMTPlan& plan,
                             const detail::FDMTEngineConfig& cfg,
                             bool skew,
                             SizeType chunk,
                             SizeType skew_leaf)
    : m_use_box_smearing(cfg.use_box_smearing),
      m_int_tree(cfg.int_tree),
      m_mode(cfg.mode),
      m_nbeams(cfg.nbeams),
      m_nthreads(std::max(1, cfg.exec.nthreads)),
      m_plan(plan),
      m_chunk(chunk),
      m_search_nbits(cfg.search_nbits),
      m_search_upper(cfg.search_upper_levels),
      m_search_u32(detail::int_search_fits_u32(m_plan,
                                               m_use_box_smearing,
                                               m_search_nbits)),
      m_skew_leaf(skew_leaf) {
    if (m_nbeams == 0) {
        throw std::invalid_argument("FDMT: nbeams must be at least 1");
    }
    plan_working_memory(cfg.fuse_levels, cfg.max_fuse_levels, skew);
    if (m_mode == FDMTMode::kValid && !m_skew) {
        m_hist_hash = detail::fdmt_tree_history_hash(m_plan, m_nbeams,
                                                     m_use_box_smearing);
    }
    if (!cfg.defer_allocation) {
        allocate();
    }
}

void FDMTCpuEngine::allocate() {
    if (m_allocated) {
        return;
    }
    if (m_skew) {
        m_skew->allocate();
        m_allocated = true;
        return;
    }
    m_state_internal.assign(m_planned.state, m_nthreads);
    if (m_mode == FDMTMode::kValid) {
        m_history.assign(m_planned.history, 0.0F);
        if (m_search_nbits == 32) {
            m_tree_history.assign(m_planned.tree_history, m_nthreads);
        } else if (m_search_u32) {
            m_integer32_history.assign(m_planned.tree_history, m_nthreads);
        } else {
            m_integer_history.assign(m_planned.tree_history, m_nthreads);
        }
    }
    m_workspace.assign(m_planned.workspace, std::byte{0});
    m_ws.base = m_workspace.data();
    // What was allocated must be what get_memory_usage() reported.
    const SizeType history = (history_state_size() * sizeof(float)) +
                             (m_integer_history.size() * sizeof(uint64_t)) +
                             (m_integer32_history.size() * sizeof(uint32_t));
    const auto planned     = get_memory_usage();
    if (m_state_internal.size() != planned.state ||
        history != planned.history || m_workspace.size() != planned.workspace) {
        throw std::logic_error(
            "FDMT (cpu): allocation differs from the planned sizes");
    }
    m_allocated = true;
}

std::string FDMTCpuEngine::summary() const {
    const auto mem = get_memory_usage();
    const auto mib = [](SizeType b) {
        return static_cast<double>(b) / 1048576.0;
    };
    const auto summary = fmt::format(
        "*** FDMT (cpu) ***\n"
        "Mode: {}, {}, nbeams {}, chunk {} of {} samples, fuse_levels "
        "{}{}, int_tree {}\n"
        "Host memory: plan {:.1f} MiB, state {:.1f} MiB, history {:.1f} "
        "MiB, workspace {:.1f} MiB (total {:.1f} MiB); output buffer "
        "{:.1f} MiB per execute()\n",
        fdmt_mode_to_string(m_mode), fmt::format("{} threads", m_nthreads),
        m_nbeams, m_chunk, m_plan.get_nsamps(), m_fuse_levels,
        (m_fuse_tile > 0) ? fmt::format(" (time tiles of {})", m_fuse_tile)
                          : std::string{},
        m_int_tree ? "on" : "off", mib(mem.plan), mib(mem.state),
        mib(mem.history), mib(mem.workspace), mib(mem.total()),
        mib(mem.output));
    return m_plan.summary() + summary;
}

FDMTMemoryUsage FDMTCpuEngine::get_memory_usage() const noexcept {
    if (m_skew) {
        const auto sm = m_skew->memory_usage();
        return {
            .plan      = m_plan.get_container().get_memory_usage() +
                         m_skew->plan_bytes(),
            .state     = sm.state,
            .history   = sm.ring + sm.rings,
            .workspace = sm.stripes,
            .output    = m_nbeams * m_plan.get_dmt_size() * sizeof(float),
        };
    }
    return {
        .plan  = m_plan.get_container().get_memory_usage() +
                 (m_ops_elems * sizeof(FDMTMergeOp)),
        .state = m_planned.state,
        // history_state_size() (with its header) plus the integer rings.
        .history = planned_history_words() * sizeof(float) +
                   (m_search_nbits != 32 ? m_planned.tree_history *
                                               (m_search_u32 ? sizeof(uint32_t)
                                                             : sizeof(uint64_t))
                                         : 0),
        .workspace = m_planned.workspace,
        .output    = m_nbeams * m_plan.get_dmt_size() * sizeof(float),
    };
}

SizeType FDMTCpuEngine::skew_fuse_request(SizeType fuse_levels,
                                          SizeType max_fuse_levels) const {
    return (fuse_levels == kFDMTAutoFuse)
               ? SizeType{0}
               : std::min({fuse_levels, max_fuse_levels,
                           m_plan.get_max_fuse_levels()});
}

std::vector<std::pair<SizeType, SizeType>>
FDMTCpuEngine::skew_candidates(SizeType ring_elem) {
    if (ring_elem == sizeof(float)) {
        return {{4096, 1}, {2048, 1}, {1024, 1}, {1024, 0},
                {512, 1},  {512, 0},  {256, 1},  {256, 0}};
    }
    return {{4096, 2}, {4096, 4}, {2048, 4}, {1024, 0}, {512, 0}, {256, 0}};
}

void FDMTCpuEngine::plan_working_memory(SizeType fuse_levels,
                                        SizeType max_fuse_levels,
                                        bool skew) {
    const SizeType niters = m_plan.get_niters();
    if (skew) {
        const SizeType req_fuse =
            skew_fuse_request(fuse_levels, max_fuse_levels);
        m_skew = std::make_unique<detail::FDMTSkewCpu>(
            m_plan, m_use_box_smearing, m_nthreads, m_chunk, SizeType{4},
            m_search_nbits, req_fuse, m_search_nbits != 32, m_search_u32,
            m_search_upper ? m_plan.get_subband_depth() : SizeType{0},
            m_skew_leaf, /*defer_allocation=*/true); // see allocate()
        m_fuse_levels = m_skew->get_fuse_level();
        return;
    }
    max_fuse_levels = (m_search_nbits == 32) ? max_fuse_levels : 0;
    build_merge_ops();
    const SizeType state_elem = (m_search_nbits == 32 || m_search_u32)
                                    ? sizeof(uint32_t)
                                    : sizeof(uint64_t);
    m_planned.state           = 2 * m_nbeams * m_half_elems * state_elem;
    if (m_mode == FDMTMode::kValid) {
        m_planned.history      = m_nbeams * m_plan.get_history_size();
        m_planned.tree_history = m_nbeams * m_plan.get_tree_history_size();
    }
    if (m_use_box_smearing) {
        const auto weights = m_plan.get_box_weights();
        const auto box_max = weights.size(); // L_max
        auto tables        = detail::make_fdmt_box_tables(
            weights, m_plan.get_box_fraction_bits(), m_plan.get_box_mean_bits(),
            m_search_nbits != 32);
        m_int_unit  = tables.int_unit;
        m_box_w_f32 = std::move(tables.w_f32);
        m_box_round = std::move(tables.round);
        for (const SizeType nbits : {1, 2, 4, 8, 16}) {
            const auto largest = detail::box_round_max(
                bit_pack_utils::max_sample_value(nbits), box_max);
            m_box_round_bytes[nbits] = (largest <= 0xFF)         ? 1
                                       : (largest <= 0xFFFF)     ? 2
                                       : (largest <= 0xFFFFFFFF) ? 4
                                                                 : 8;
        }
        if (m_search_nbits != 32) {
            m_box_round_bytes[m_search_nbits] = 8;
        }
        m_box_weights = {.w_f32 = m_box_w_f32.data(),
                         .round = m_box_round.data(),
                         .unit  = m_int_unit};
    }
    m_float_levels.assign(niters + 1, Elem::kF32);
    for (const SizeType nbits : {1, 2, 4, 8, 16}) {
        m_int_levels[nbits] =
            detail::int_tree_level_types(m_plan, m_use_box_smearing, nbits);
    }
    m_levels.resize(niters + 1);

    if (fuse_levels == kFDMTAutoFuse) {
        m_fuse_levels = auto_fuse_levels();
    } else {
        m_fuse_levels = std::min(fuse_levels, niters);
    }

    m_fuse_levels = std::min(
        {m_fuse_levels, max_fuse_levels, m_plan.get_max_fuse_levels()});

    m_ws.row_bytes =
        round_up_64(m_row_len[0] *
                    (m_search_nbits == 32 ? sizeof(float) : sizeof(uint64_t)));
    m_fuse_tile = (m_mode == FDMTMode::kValid && m_fuse_levels > 0)
                      ? fuse_tile_for(m_fuse_levels)
                      : 0;
    if (m_fuse_tile > 0) {
        m_tile_ring_pos.assign(m_fuse_levels + 1, {});
        for (SizeType l = 1; l <= m_fuse_levels; ++l) {
            m_tile_ring_pos[l].assign(m_ops[l].size(), 0);
        }
    }
    m_ws.fused_bytes =
        (m_fuse_levels > 0)
            ? round_up_64(fused_scratch_bytes(m_fuse_levels, m_fuse_tile))
            : 0;
    m_ws.stride         = (3 * m_ws.row_bytes) + (2 * m_ws.fused_bytes);
    m_ws.nthreads       = m_nthreads;
    m_planned.workspace = m_nthreads * m_ws.stride;
}

void FDMTCpuEngine::build_merge_ops() {
    const auto& pc        = m_plan.get_container();
    const SizeType niters = m_plan.get_niters();
    const bool valid      = m_mode == FDMTMode::kValid;
    m_row_len.resize(niters + 1);
    for (SizeType l = 0; l <= niters; ++l) {
        m_row_len[l] =
            (valid && l < niters) ? m_chunk : pc.state_shape[l].nsamps;
    }
    m_half_elems = 0;
    for (SizeType l = 0; l < niters; ++l) {
        const SizeType n = m_search_upper ? detail::fdmt_regular_coords(pc, l)
                                          : pc.state_shape[l].ncoords;
        m_half_elems     = std::max(m_half_elems, n * m_row_len[l]);
    }
    const auto relayout = [&](SizeType offset, SizeType l) {
        const auto plan_len = pc.state_shape[l].nsamps;
        return (l == niters) ? offset : (offset / plan_len) * m_row_len[l];
    };
    m_ops.assign(niters + 1, {});
    m_ops_elems = 0;
    for (SizeType l = 1; l <= niters; ++l) {
        const auto& coords     = pc.coordinates[l];
        const auto& coords_sum = pc.coordinates_sum[l];
        auto& ops              = m_ops[l];
        ops.reserve(coords.size());
        SizeType j = 0;
        for (const auto& c : coords) {
            const bool is_sum = c.i_coord_head != SIZE_MAX;
            // Only coordinates_sum carries the tree-history slot.
            const auto& src = is_sum ? coords_sum[j++] : c;
            const auto row  = [&](SizeType offset, SizeType lv) {
                return offset / pc.state_shape[lv].nsamps;
            };
            ops.push_back({
                .out  = relayout(src.buf_offset, l),
                .tail = relayout(src.tail_buf_offset, l - 1),
                .head =
                    is_sum ? relayout(src.head_buf_offset, l - 1) : SIZE_MAX,
                .nsamps      = src.nsamps,
                .tail_nsamps = src.tail_nsamps,
                .delay       = is_sum ? src.delay : 0,
                .hist        = is_sum ? src.hist_offset : 0,
                .out_row     = row(src.buf_offset, l),
                .tail_row    = row(src.tail_buf_offset, l - 1),
                .head_row    = is_sum ? row(src.head_buf_offset, l - 1) : 0,
            });
        }
        m_ops_elems += ops.size();
    }
}

SizeType FDMTCpuEngine::fused_scratch_bytes(SizeType fuse,
                                            SizeType tile) const noexcept {
    const auto& pc         = m_plan.get_container();
    const SizeType ngroups = pc.state_shape[fuse].nchans;
    SizeType max_bytes     = 0;
    for (SizeType l = 0; l < fuse; ++l) {
        const SizeType len = (tile > 0) ? tile : m_row_len[l];
        for (SizeType g = 0; g < ngroups; ++g) {
            const auto [cb, ce] = fused_coord_range(l, g, fuse);
            max_bytes = std::max(max_bytes, (ce - cb) * len * sizeof(float));
        }
    }
    return max_bytes;
}

double FDMTCpuEngine::fused_tile_working_set(SizeType fuse,
                                             SizeType tile) const noexcept {
    const auto& pc         = m_plan.get_container();
    const SizeType ngroups = pc.state_shape[fuse].nchans;
    double total           = 0.0;
    for (SizeType g = 0; g < ngroups; ++g) {
        SizeType rows = 0;
        SizeType ring = 0;
        for (SizeType l = 0; l <= fuse; ++l) {
            const auto [cb, ce] = fused_coord_range(l, g, fuse);
            if (l < fuse) {
                rows = std::max(rows, ce - cb);
            }
            if (l > 0) {
                for (SizeType k = cb; k < ce; ++k) {
                    ring += m_ops[l][k].delay;
                }
            }
        }
        total += static_cast<double>((2 * rows * tile) + ring) *
                 static_cast<double>(sizeof(float));
    }
    return total / static_cast<double>(ngroups);
}

SizeType FDMTCpuEngine::fuse_tile_for(SizeType fuse) const noexcept {
    SizeType tile = kFuseTiles.back();
    for (const SizeType t : kFuseTiles) {
        if (fused_tile_working_set(fuse, t) <=
            static_cast<double>(kFuseTileBudget)) {
            tile = t;
            break;
        }
    }
    return std::min(tile, m_chunk);
}

SizeType FDMTCpuEngine::auto_fuse_levels() const noexcept {
    if (m_mode == FDMTMode::kValid) {
        const auto& pc   = m_plan.get_container();
        const auto min_g = kFuseGroupsPerThread *
                           static_cast<SizeType>(std::max(m_nthreads, 1));
        SizeType best    = 0;
        for (SizeType f = 1; f <= m_plan.get_niters(); ++f) {
            if (fused_tile_working_set(f, kFuseTiles.back()) >
                static_cast<double>(kFuseTileBudget)) {
                break;
            }
            if (f > 1 && pc.state_shape[f].nchans < min_g) {
                break;
            }
            best = f;
        }
        return best;
    }
    const SizeType budget =
        std::max(kFuseBudgetTotal / std::max<SizeType>(m_nthreads, 1),
                 kFuseBudgetPerThread);
    SizeType best = 0;
    for (SizeType f = 1; f <= m_plan.get_niters(); ++f) {
        if (2 * fused_scratch_bytes(f) > budget) {
            break;
        }
        best = f;
    }
    return best;
}

} // namespace dmt::algorithms::detail::fdmt_cpu

namespace dmt::algorithms {

detail::FDMTEngineAttempt
detail::try_make_fdmt_cpu(const plans::FDMTPlan& plan,
                          const detail::FDMTEngineConfig& cfg) {
    if (cfg.nbeams == 0) {
        throw std::invalid_argument("FDMT: nbeams must be at least 1");
    }
    const bool search_u32 = detail::int_search_fits_u32(
        plan, cfg.use_box_smearing, cfg.search_nbits);
    const SizeType state_elem = (cfg.search_nbits == 32 || search_u32)
                                    ? sizeof(uint32_t)
                                    : sizeof(uint64_t);
    const SizeType auto_chunk = detail::fdmt_chunk_nsamps(
        plan, cfg.mode, cfg.nbeams, cfg.chunk_nsamps, cfg.search_upper_levels,
        state_elem);

    const SizeType niters = plan.get_niters();
    const SizeType nsamps = plan.get_nsamps();
    const SizeType depth =
        cfg.search_upper_levels ? plan.get_subband_depth() : SizeType{0};
    const bool skew_eligible =
        cfg.mode == FDMTMode::kValid && cfg.nbeams == 1 && niters >= 5 &&
        cfg.fuse_levels != 0 && cfg.max_fuse_levels >= 4 &&
        // Searched subband levels sit above the leaf and fused levels.
        (!cfg.search_upper_levels || niters >= depth + 5);
    const SizeType ring_elem = cfg.search_nbits == 32   ? sizeof(float)
                               : cfg.search_nbits == 16 ? 2
                                                        : 1;
    const auto skew_list =
        detail::fdmt_cpu::FDMTCpuEngine::skew_candidates(ring_elem);

    struct Candidate {
        bool skew{false};
        SizeType chunk{0};
        SizeType leaf{0};
    };
    std::vector<Candidate> candidates;
    const auto add_skew_explicit = [&](SizeType c) {
        if (!skew_eligible || c >= nsamps) {
            return;
        }
        if (cfg.skew_leaf != 0) {
            candidates.push_back({true, c, cfg.skew_leaf});
            return;
        }
        for (const auto& [sc, leaf] : skew_list) {
            if (sc == c && leaf != 0) {
                candidates.push_back({true, c, leaf});
            }
        }
        candidates.push_back({true, c, 0});
    };

    if (cfg.path == FDMTPath::kTree || cfg.path == FDMTPath::kDirect) {
        candidates.push_back({false, auto_chunk, 0});
    } else if (detail::is_skewed_path(cfg.path)) {
        if (cfg.chunk_nsamps != 0) {
            add_skew_explicit(auto_chunk);
        } else if (skew_eligible) {
            for (const auto& [c, leaf] : skew_list) {
                if (c < nsamps) {
                    candidates.push_back({true, c, leaf});
                }
            }
        }
    } else if (cfg.chunk_nsamps != 0) {
        // Explicit chunk under kAuto: tree first, then skewed at that chunk.
        candidates.push_back({false, auto_chunk, 0});
        add_skew_explicit(auto_chunk);
    } else {
        // Automatic chunk under kAuto: packed input takes the long skewed
        // chunks before the tree; float takes the tree first.
        const bool skew_first = cfg.search_nbits != 32;
        constexpr SizeType kMinFirst =
            detail::fdmt_cpu::FDMTCpuEngine::kSkewFirstMinChunk;
        if (skew_first && skew_eligible) {
            for (const auto& [c, leaf] : skew_list) {
                if (c >= kMinFirst && c < nsamps) {
                    candidates.push_back({true, c, leaf});
                }
            }
        }
        candidates.push_back({false, auto_chunk, 0});
        if (skew_eligible) {
            for (const auto& [c, leaf] : skew_list) {
                if ((!skew_first || c < kMinFirst) && c < nsamps) {
                    candidates.push_back({true, c, leaf});
                }
            }
        } else if (cfg.mode == FDMTMode::kValid) {
            // Tree-only plans (too few levels, multi-beam, or unfused):
            // halve the automatic chunk down to 256, matching the GPU path.
            for (SizeType c = auto_chunk / 2; c >= 256; c /= 2) {
                candidates.push_back({false, c, 0});
            }
        }
    }

    auto deferred_cfg             = cfg;
    deferred_cfg.defer_allocation = true;
    SizeType smallest             = std::numeric_limits<SizeType>::max();
    SizeType tree_bytes           = 0;
    bool tried_skew               = false;
    for (const auto& cand : candidates) {
        std::unique_ptr<detail::fdmt_cpu::FDMTCpuEngine> eng;
        try {
            eng = std::make_unique<detail::fdmt_cpu::FDMTCpuEngine>(
                plan, deferred_cfg, cand.skew, cand.chunk, cand.leaf);
        } catch (const std::length_error&) {
            continue;
        }
        const SizeType used = eng->get_memory_usage().total();
        if (!cand.skew) {
            tree_bytes = used;
        } else {
            tried_skew = true;
        }
        if (used <= cfg.max_memory_bytes) {
            if (!cfg.defer_allocation) {
                eng->allocate();
            }
            return {.engine = std::move(eng), .needed = used, .error = {}};
        }
        smallest = std::min(smallest, used);
    }

    std::string msg;
    if (smallest == std::numeric_limits<SizeType>::max()) {
        msg = "FDMT (cpu): no streaming path is available for this "
              "configuration";
    } else if (tree_bytes > 0 && tried_skew && cfg.chunk_nsamps == 0) {
        msg = fmt::format(
            "FDMT (cpu): no streaming path fits max_memory_bytes ({:.2f} GiB): "
            "the tree path needs {:.2f} GiB and the skewed-window path more "
            "than the budget at every chunk length",
            static_cast<double>(cfg.max_memory_bytes) / (1 << 30),
            static_cast<double>(tree_bytes) / (1 << 30));
    } else {
        msg = fmt::format("FDMT (cpu): this configuration needs {:.2f} GiB, "
                          "over max_memory_bytes ({:.2f} GiB){}",
                          static_cast<double>(smallest) / (1 << 30),
                          static_cast<double>(cfg.max_memory_bytes) / (1 << 30),
                          tried_skew ? "; use a shorter chunk_nsamps" : "");
    }
    return {.engine = nullptr, .needed = smallest, .error = std::move(msg)};
}

std::unique_ptr<detail::FDMTEngine>
detail::make_fdmt_cpu(const plans::FDMTPlan& plan,
                      const detail::FDMTEngineConfig& cfg) {
    auto res = try_make_fdmt_cpu(plan, cfg);
    if (!res.engine) {
        throw std::invalid_argument(res.error);
    }
    return std::move(res.engine);
}

} // namespace dmt::algorithms
