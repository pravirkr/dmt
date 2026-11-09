#include "dmt/algorithms/fdmt.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
                             const detail::FDMTEngineConfig& cfg)
    : m_use_box_smearing(cfg.use_box_smearing),
      m_int_tree(cfg.int_tree),
      m_mode(cfg.mode),
      m_nbeams(cfg.nbeams),
      m_nthreads(std::max(1, cfg.exec.nthreads)),
      m_plan(plan) {
    if (m_nbeams == 0) {
        throw std::invalid_argument("FDMT: nbeams must be at least 1");
    }
    m_search_nbits = cfg.search_nbits;
    m_search_upper = cfg.search_upper_levels;
    // Packed search state in uint32_t when the plan bounds every tree
    // value below 2^32 (as the GPU engine): half the bytes per merge.
    m_search_u32 =
        detail::int_search_fits_u32(m_plan, m_use_box_smearing, m_search_nbits);
    m_chunk = detail::fdmt_chunk_nsamps(m_plan, m_mode, m_nbeams,
                                        cfg.chunk_nsamps, m_search_upper);
    if (m_mode == FDMTMode::kValid && cfg.chunk_nsamps == 0 &&
        (m_plan.get_history_size() + m_plan.get_tree_history_size()) *
                sizeof(float) >
            detail::kFDMTChunkStateBudget) {
        m_chunk = std::min<SizeType>(1024, m_plan.get_nsamps());
    } else if (m_search_nbits != 32 && !m_search_u32 && cfg.chunk_nsamps == 0) {
        m_chunk =
            std::min(m_plan.get_nsamps(), std::max<SizeType>(64, m_chunk / 2));
    }
    const bool skew = choose_skew(cfg);
    allocate_working_memory(cfg.fuse_levels, cfg.max_fuse_levels, skew);
    if (m_mode == FDMTMode::kValid && !m_skew) {
        m_hist_hash = detail::fdmt_tree_history_hash(m_plan, m_nbeams,
                                                     m_use_box_smearing);
    }
    const SizeType used = get_memory_usage().total();
    if (used > cfg.max_memory_bytes) {
        throw std::invalid_argument(
            fmt::format("FDMT (cpu): this configuration needs {:.2f} GiB, over "
                        "max_memory_bytes ({:.2f} GiB){}",
                        static_cast<double>(used) / (1 << 30),
                        static_cast<double>(cfg.max_memory_bytes) / (1 << 30),
                        skew ? "; use a shorter chunk_nsamps" : ""));
    }
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
        .plan      = m_plan.get_container().get_memory_usage() +
                     (m_ops_elems * sizeof(FDMTMergeOp)),
        .state     = m_state_internal.size(),
        .history   = history_state_size() * sizeof(float) +
                     m_integer_history.size() * sizeof(uint64_t) +
                     m_integer32_history.size() * sizeof(uint32_t),
        .workspace = m_workspace.size(),
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
        return {{4096, 1}, {4096, 2}, {2048, 1}, {4096, 3}, {2048, 2},
                {2048, 3}, {4096, 4}, {1024, 1}, {2048, 4}, {1024, 0},
                {512, 1},  {512, 0},  {256, 1},  {256, 0}};
    }
    return {{4096, 2}, {4096, 3}, {4096, 4}, {4096, 1}, {2048, 4}, {2048, 2},
            {2048, 3}, {2048, 1}, {1024, 0}, {512, 0},  {256, 0}};
}

bool FDMTCpuEngine::choose_skew(const detail::FDMTEngineConfig& cfg) {
    const SizeType niters = m_plan.get_niters();
    const SizeType nsamps = m_plan.get_nsamps();
    const SizeType depth =
        m_search_upper ? m_plan.get_subband_depth() : SizeType{0};
    const bool eligible =
        m_mode == FDMTMode::kValid && m_nbeams == 1 && niters >= 5 &&
        cfg.fuse_levels != 0 && cfg.max_fuse_levels >= 4 &&
        // Searched subband levels sit above the leaf and fused levels.
        (!m_search_upper || niters >= depth + 5);
    if (!eligible) {
        return false;
    }
    const SizeType hist_floats =
        m_plan.get_history_size() + m_plan.get_tree_history_size();
    if (cfg.skew_history_bytes != detail::FDMTEngineConfig::kSkewHistoryAuto) {
        m_skew_leaf = cfg.skew_leaf;
        return m_chunk < nsamps &&
               hist_floats * sizeof(float) > cfg.skew_history_bytes;
    }
    const auto& pc            = m_plan.get_container();
    const bool wide           = m_search_nbits != 32 && !m_search_u32;
    const SizeType state_elem = wide ? sizeof(uint64_t) : sizeof(float);
    const SizeType plan_bytes = pc.get_memory_usage();
    SizeType max_rows         = 0;
    for (SizeType l = 0; l < niters; ++l) {
        max_rows = std::max(max_rows, m_search_upper
                                          ? detail::fdmt_regular_coords(pc, l)
                                          : pc.state_shape[l].ncoords);
    }
    const SizeType tree_bytes = (hist_floats * state_elem) +
                                (2 * max_rows * m_chunk * state_elem) +
                                plan_bytes;
    if (tree_bytes <= cfg.max_memory_bytes) {
        return false;
    }
    const SizeType ring_elem = m_search_nbits == 32   ? sizeof(float)
                               : m_search_nbits == 16 ? 2
                                                      : 1;
    const SizeType req_fuse =
        skew_fuse_request(cfg.fuse_levels, cfg.max_fuse_levels);
    const auto fits = [&](SizeType c, SizeType leaf) {
        const auto e = detail::estimate_skew_memory(
            m_plan, 4, c, ring_elem, m_nthreads, m_use_box_smearing, req_fuse,
            depth, leaf);
        const SizeType bytes =
            e.ring + ((e.state + e.rings) * state_elem / sizeof(float)) +
            e.stripes + plan_bytes;
        // A tenth of headroom for the engine's merge tables.
        return bytes + (bytes / 10) <= cfg.max_memory_bytes;
    };
    if (cfg.chunk_nsamps != 0) {
        // Explicit chunk: the fastest leaf level that fits it (else the
        // automatic one); checked against the budget once allocated.
        if (m_chunk >= nsamps) {
            return false;
        }
        m_skew_leaf = 0;
        for (const auto& [c, leaf] : skew_candidates(ring_elem)) {
            if (c == m_chunk && leaf != 0 && fits(c, leaf)) {
                m_skew_leaf = leaf;
                break;
            }
        }
        return true;
    }
    for (const auto& [c, leaf] : skew_candidates(ring_elem)) {
        if (c < nsamps && fits(c, leaf)) {
            m_chunk     = c;
            m_skew_leaf = leaf;
            return true;
        }
    }
    throw std::invalid_argument(fmt::format(
        "FDMT (cpu): no streaming path fits max_memory_bytes ({:.2f} GiB): "
        "the tree path needs {:.2f} GiB and the skewed-window path more "
        "than the budget at every chunk length",
        static_cast<double>(cfg.max_memory_bytes) / (1 << 30),
        static_cast<double>(tree_bytes) / (1 << 30)));
}

void FDMTCpuEngine::allocate_working_memory(SizeType fuse_levels,
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
            m_skew_leaf);
        m_fuse_levels = m_skew->get_fuse_level();
        return;
    }
    max_fuse_levels = (m_search_nbits == 32) ? max_fuse_levels : 0;
    build_merge_ops();
    const SizeType state_elem = (m_search_nbits == 32 || m_search_u32)
                                    ? sizeof(uint32_t)
                                    : sizeof(uint64_t);
    m_state_internal.assign(2 * m_nbeams * m_half_elems * state_elem,
                            m_nthreads);
    if (m_mode == FDMTMode::kValid) {
        m_history.assign(m_nbeams * m_plan.get_history_size(), 0.0F);
        if (m_search_nbits == 32) {
            m_tree_history.assign(m_nbeams * m_plan.get_tree_history_size(),
                                  m_nthreads);
        } else if (m_search_u32) {
            m_integer32_history.assign(
                m_nbeams * m_plan.get_tree_history_size(), m_nthreads);
        } else {
            m_integer_history.assign(m_nbeams * m_plan.get_tree_history_size(),
                                     m_nthreads);
        }
    }
    if (m_use_box_smearing) {
        const auto weights = m_plan.get_box_weights();
        const auto box_max = weights.size(); // L_max
        const auto scale_f = m_plan.get_box_mean_bits();
        const float w_unit =
            std::ldexp(1.0F, -static_cast<int>(m_plan.get_box_fraction_bits()));
        m_int_unit = std::ldexp(1.0F, -static_cast<int>(scale_f));
        m_box_w_f32.resize(box_max);
        m_box_round.resize(box_max);
        for (SizeType s = 0; s < box_max; ++s) {
            m_box_w_f32[s] = static_cast<float>(weights[s]) * w_unit;
            m_box_round[s] = m_search_nbits == 32
                                 ? detail::box_round(s + 1, scale_f)
                                 : detail::FDMTBoxRound{.mul    = weights[s],
                                                        .add    = 0,
                                                        .rshift = 0,
                                                        .lshift = 0};
        }
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
    m_ws.stride   = (3 * m_ws.row_bytes) + (2 * m_ws.fused_bytes);
    m_ws.nthreads = m_nthreads;
    m_workspace.assign(m_nthreads * m_ws.stride, std::byte{0});
    m_ws.base = m_workspace.data();
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
std::unique_ptr<detail::FDMTEngine>
detail::make_fdmt_cpu(const plans::FDMTPlan& plan,
                      const detail::FDMTEngineConfig& cfg) {
    return std::make_unique<detail::fdmt_cpu::FDMTCpuEngine>(plan, cfg);
}

} // namespace dmt::algorithms
