#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/common/backend.hpp"
#include "dmt/common/types.hpp"

#include "dmt/fdmt_history.hpp"

#include "fdmt_cpu_engine.hpp"

// CPU FDMT engine: the stepper and the stream history.

namespace dmt::algorithms::detail::fdmt_cpu {

void FDMTCpuEngine::reset(std::span<const float> waterfall,
                          std::span<float> dmt) {
    start_stepper(float_input(waterfall), m_float_levels, dmt);
}

void FDMTCpuEngine::reset(std::span<const uint8_t> waterfall_packed,
                          SizeType nbits,
                          std::span<float> dmt) {
    const auto input = packed_input(waterfall_packed, nbits);
    start_stepper(input, packed_levels(nbits), dmt);
}

void FDMTCpuEngine::advance(SizeType levels, Stream /*stream*/) {
    if (!m_is_initialized) {
        throw std::logic_error(
            "FDMT: Stepper is not initialized. Call reset() first.");
    }
    const auto total_lvl = total_levels();
    while (levels > 0 && m_current_level < total_lvl - 1) {
        const SizeType next_level = m_current_level + 1;
        execute_iter(next_level);
        m_current_level = next_level;
        --levels;
    }
}

void FDMTCpuEngine::advance_until_remaining(SizeType remaining_levels,
                                            Stream stream) {
    if (!m_is_initialized) {
        throw std::logic_error(
            "FDMT: Stepper is not initialized. Call reset() first.");
    }
    const auto total_lvl = total_levels();
    if (remaining_levels >= total_lvl) {
        return;
    }
    const SizeType target_level = total_lvl - 1 - remaining_levels;
    if (target_level > m_current_level) {
        advance(target_level - m_current_level, stream);
    }
}

FDMTSubbandView FDMTCpuEngine::view_subband(SizeType subband_idx) const {
    if (!m_is_initialized) {
        throw std::logic_error(
            "FDMT: Stepper is not initialized. Call reset() first.");
    }
    const auto& plan_c = m_plan.get_container();
    if (subband_idx >= plan_c.grids[m_current_level].size()) {
        throw std::out_of_range(fmt::format(
            "FDMT: Subband index {} out of range (current level has {} "
            "subbands)",
            subband_idx, plan_c.grids[m_current_level].size()));
    }
    const auto& grid  = plan_c.grids[m_current_level][subband_idx];
    const auto len    = m_row_len[m_current_level];
    const auto offset = grid.coord_offset * len;
    const auto count  = grid.ndt * len;
    return FDMTSubbandView{
        .data = std::span<const float>(current_level_f32() + offset, count),
        .subband_idx = subband_idx,
        .ndt         = grid.ndt,
        .nsamps      = len,
        .f_start     = grid.f_start,
        .f_end       = grid.f_end,
        .dt_grid     = std::span<const IndexType>(grid.dt_grid.data(),
                                                  grid.dt_grid.size()),
        .chan_begin  = grid.chan_begin,
        .chan_end    = grid.chan_end,
        .staggered   = grid.staggered,
    };
}

void FDMTCpuEngine::finalize(Stream stream) {
    if (!m_is_initialized) {
        throw std::logic_error(
            "FDMT: Stepper is not initialized. Call reset() first.");
    }
    if (!is_finished()) {
        advance_until_remaining(0, stream);
    }
    m_is_initialized = false;
}

void FDMTCpuEngine::reset_history() noexcept {
    if (m_skew) {
        m_skew->reset_history();
        return;
    }
    std::ranges::fill(m_history, 0.0F);
    std::ranges::fill(m_tree_history, 0.0F);
    std::ranges::fill(m_integer_history, uint64_t{0});
    std::ranges::fill(m_integer32_history, uint32_t{0});
    m_stream_pos    = 0;
    m_hist_pos_base = 0;
    // Abandons any unfinished stepper block along with its stream.
    m_is_initialized = false;
}

void FDMTCpuEngine::save_history(std::span<float> out) const {
    if (out.size() != history_state_size()) {
        throw std::invalid_argument(
            fmt::format("FDMT::save_history: Invalid output size. "
                        "Expected {}, got {}",
                        history_state_size(), out.size()));
    }
    if (m_skew) { // the skewed-window engine's own layout
        m_skew->save_history(out);
        return;
    }
    if (out.empty()) {
        return;
    }
    detail::write_fdmt_tree_history_header(out, m_hist_hash,
                                           m_hist_pos_base + m_stream_pos);
    const auto payload = out.subspan(detail::kFDMTHistHeader);
    std::ranges::copy(m_history, payload.begin());
    const auto tree = payload.subspan(m_history.size());
    for_each_ring(
        [&](SizeType beam_off, SizeType off, SizeType d, SizeType pos) {
            const float* ring = m_tree_history.data() + beam_off + off;
            float* fifo       = tree.data() + beam_off + off;
            std::copy(ring + pos, ring + d, fifo);
            std::copy(ring, ring + pos, fifo + (d - pos));
        });
}

void FDMTCpuEngine::load_history(std::span<const float> in) {
    if (m_skew) { // either ring layout; the engine checks the size
        m_skew->load_history(in);
        m_is_initialized = false;
        return;
    }
    if (in.size() != history_state_size()) {
        detail::throw_fdmt_history_size(history_state_size(), in.size());
    }
    uint64_t pos_base = 0;
    if (!in.empty()) {
        pos_base = detail::check_fdmt_tree_history_header(in, m_hist_hash);
        const auto payload = in.subspan(detail::kFDMTHistHeader);
        std::copy_n(payload.begin(), m_history.size(), m_history.begin());
        // FIFO order is the ring at stream position 0.
        std::copy_n(payload.begin() +
                        static_cast<std::ptrdiff_t>(m_history.size()),
                    m_tree_history.size(), m_tree_history.begin());
    }
    m_stream_pos     = 0;
    m_hist_pos_base  = pos_base;
    m_is_initialized = false;
}

void FDMTCpuEngine::start_stepper(const FDMTInput& input,
                                  const std::vector<Elem>& types,
                                  std::span<float> dmt) {
    if (m_chunk < m_plan.get_nsamps()) {
        throw std::logic_error(fmt::format(
            "FDMT: the stepper runs whole blocks, but this engine "
            "executes blocks of {} samples in chunks of {} "
            "(chunk_nsamps); construct with chunk_nsamps >= nsamps to "
            "step through levels.",
            m_plan.get_nsamps(), m_chunk));
    }
    check_dmt_size(dmt);
    check_previous_block_finished();
    start_chunk(input, types, dmt, 0, m_plan.get_nsamps());
    initialise();
    m_current_level  = 0;
    m_is_initialized = true;
}

template <typename F> void FDMTCpuEngine::for_each_ring(F&& f) const {
    if (m_tree_history.empty()) {
        return;
    }
    const auto stride = m_plan.get_tree_history_size();
    for (SizeType b = 0; b < m_nbeams; ++b) {
        for (SizeType l = 1; l < m_ops.size(); ++l) {
            for (const auto& op : m_ops[l]) {
                if (op.head != SIZE_MAX && op.delay > 0) {
                    f(b * stride, op.hist, op.delay, m_stream_pos % op.delay);
                }
            }
        }
    }
}

} // namespace dmt::algorithms::detail::fdmt_cpu
