#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

#include "dmt/dm_utils.hpp"

#include "fdmt_plan_impl.hpp"

// FDMTPlan: shapes, queries and the public forwarders.

namespace dmt::plans {

constexpr std::string_view FDMTShape::header_fmt() {
    return "{ncoords} ({nchans}x[{ndt_min}..{ndt_max}]) x "
           "{nsamps}, {nelements}";
}

std::string FDMTShape::to_string() const {
    return fmt::format("{:5d} ({:5d}x[{:5d}..{:5d}]) x {:6d}, {:10d}", ncoords,
                       nchans, ndt_min, ndt_max, nsamps, nelements);
}

std::string FDMTComplexity::to_string() const {
    std::string result = fmt::format(
        "FDMT Complexity:\n"
        "  Output DM trials (N_dt)      : {}\n"
        "  Input channels   (N_chans)   : {}\n"
        "  Brute-force ops / time samp  : {} (N_dt x N_chans)\n"
        "  FDMT tree nodes  (Levels 1..M): {}\n"
        "  FDMT additions   (Levels 1..M): {} (offset_add operations)\n"
        "  FDMT copy nodes  (Levels 1..M): {}\n"
        "  Theoretical Speedup Factor   : {:.2f}x (brute_force / "
        "fdmt_additions)",
        n_dt, n_chans, brute_force_ops, total_tree_nodes, sum_additions,
        copy_nodes, ops_ratio);
    if (!ops_by_iter.empty()) {
        result += "\n  Operations by iteration (0..M): [";
        for (size_t i = 0; i < ops_by_iter.size(); ++i) {
            result += std::to_string(ops_by_iter[i]);
            if (i + 1 < ops_by_iter.size()) {
                result += ", ";
            }
        }
        result += ']';
    }
    if (ops_ratio > 0.0F && ops_ratio < 2.0F) {
        result += "\n  Note: Speedup factor is < 2.0x (coarse dt spacing "
                  "approaches brute-force complexity)";
    }
    return result;
}

FDMTPlanContainer::FDMTPlanContainer(SizeType niters) {
    state_shape.resize(niters + 1);
    grids.resize(niters + 1);
    coordinates.resize(niters + 1);
    coordinates_sum.resize(niters + 1);
    coordinates_copy.resize(niters + 1);
    dt_grid_sub_top.resize(niters + 1);
}

SizeType FDMTPlanContainer::get_memory_usage() const noexcept {
    SizeType mem_use = 0;
    mem_use += state_shape.size() * sizeof(FDMTShape);
    for (const auto& grid_iter : grids) {
        mem_use += grid_iter.size() * sizeof(FDMTCoordGrid);
        for (const auto& grid : grid_iter) {
            mem_use += grid.dt_grid.size() * sizeof(IndexType);
        }
    }
    // coordinates, plus coordinates_sum and coordinates_copy (a partition of
    // the same operations).
    for (const auto& coord : coordinates) {
        mem_use += 2 * coord.size() * sizeof(FDMTCoord);
    }
    for (const auto& dt_grid : dt_grid_sub_top) {
        mem_use += dt_grid.size() * sizeof(IndexType);
    }
    return mem_use;
}

SizeType FDMTPlanContainer::get_buffer_size() const noexcept {
    auto max_elements = std::ranges::max_element(
        state_shape, [](const FDMTShape& a, const FDMTShape& b) {
            return a.nelements < b.nelements;
        });
    return max_elements->nelements;
}

FDMTComplexity
FDMTPlan::Impl::get_complexity(bool use_box_smearing) const noexcept {
    const auto n_dt            = get_dt_grid_final().size();
    const auto n_chans         = m_nchans;
    const auto brute_force_ops = n_dt * n_chans;
    SizeType total_tree_nodes  = 0;
    SizeType sum_additions     = 0;
    SizeType copy_nodes        = 0;

    std::vector<SizeType> ops_by_iter(m_niters + 1, 0);
    std::vector<SizeType> nodes_by_iter(m_niters + 1, 0);

    // Level 0: Active coordinates and boxcar additions.
    nodes_by_iter[0] = m_container.coordinates[0].size();
    if (use_box_smearing) {
        SizeType lvl0_ops = 0;
        for (SizeType c = 0; c < m_nchans; ++c) {
            const auto& dt_grid = m_container.grids[0][c].dt_grid;
            if (!dt_grid.empty()) {
                lvl0_ops +=
                    (dt_grid.size() - 1) +
                    ((dt_grid.front() != 0) ? SizeType{2} : SizeType{0});
            }
        }
        ops_by_iter[0] = lvl0_ops;
    }

    for (SizeType l = 1; l <= m_niters; ++l) {
        const auto n_coords = m_container.coordinates[l].size();
        const auto n_sum    = m_container.coordinates_sum[l].size();
        const auto n_copy   = m_container.coordinates_copy[l].size();
        total_tree_nodes += n_coords;
        sum_additions += n_sum;
        copy_nodes += n_copy;
        nodes_by_iter[l] = n_coords;
        ops_by_iter[l]   = n_sum;
    }

    const float ops_ratio = sum_additions > 0
                                ? static_cast<float>(brute_force_ops) /
                                      static_cast<float>(sum_additions)
                                : 0.0F;

    return FDMTComplexity{
        .n_dt             = n_dt,
        .n_chans          = n_chans,
        .brute_force_ops  = brute_force_ops,
        .total_tree_nodes = total_tree_nodes,
        .sum_additions    = sum_additions,
        .copy_nodes       = copy_nodes,
        .ops_ratio        = ops_ratio,
        .ops_by_iter      = std::move(ops_by_iter),
        .nodes_by_iter    = std::move(nodes_by_iter),
    };
}

std::vector<float> FDMTPlan::Impl::get_dm_grid_final() const noexcept {
    const float dm_conv       = utils::get_dmconv(m_f_min, m_f_max, m_tsamp);
    const auto& dt_grid_final = get_dt_grid_final();
    std::vector<float> dm_grid_final(dt_grid_final.size());
    std::ranges::transform(
        dt_grid_final, dm_grid_final.begin(),
        [dm_conv](auto& dt) { return static_cast<float>(dt) * dm_conv; });
    return dm_grid_final;
}

std::vector<float> FDMTPlan::Impl::get_smearing_grid_final() const noexcept {
    const auto ndms = get_dmt_ndms();
    std::vector<float> smearing_grid(ndms * m_nchans, 0.0F);
    if (m_niters == 0 || m_nchans == 0) {
        return smearing_grid;
    }

    // Helper lambda to recursively trace coordinates down to level 0
    auto trace_coord = [&](auto& self, SizeType level, SizeType coord_idx,
                           SizeType dm_idx) -> void {
        if (level == 0) {
            if (coord_idx < m_container.coordinates[0].size()) {
                const auto& coord0 = m_container.coordinates[0][coord_idx];
                const auto chan    = coord0.i_sub;
                if (chan < m_nchans &&
                    coord0.i_dt < m_container.grids[0][chan].dt_grid.size()) {
                    const auto dt =
                        m_container.grids[0][chan].dt_grid[coord0.i_dt];
                    smearing_grid[(dm_idx * m_nchans) + chan] =
                        std::abs(static_cast<float>(dt));
                }
            }
            return;
        }

        if (level < m_container.coordinates.size() &&
            coord_idx < m_container.coordinates[level].size()) {
            const auto& coord = m_container.coordinates[level][coord_idx];
            if (coord.i_coord_tail != SIZE_MAX) {
                self(self, level - 1, coord.i_coord_tail, dm_idx);
            }
            if (coord.i_coord_head != SIZE_MAX) {
                self(self, level - 1, coord.i_coord_head, dm_idx);
            }
        }
    };

    for (SizeType i_dm = 0; i_dm < ndms; ++i_dm) {
        trace_coord(trace_coord, m_niters, i_dm, i_dm);
    }
    return smearing_grid;
}

SizeType FDMTPlan::Impl::get_startup_samples() const {
    IndexType lag = 0;
    for (SizeType i_dm = 0; i_dm < get_dmt_ndms(); ++i_dm) {
        for (const auto shift : trace_dm(i_dm)) {
            lag = std::max(lag, -shift);
        }
    }
    return static_cast<SizeType>(lag);
}

std::vector<IndexType> FDMTPlan::Impl::trace_dm(SizeType dm_idx) const {
    std::vector<IndexType> shifts(m_nchans, 0);
    if (m_niters == 0 || m_nchans == 0) {
        return shifts;
    }
    const auto ndms = get_dmt_ndms();
    if (dm_idx >= ndms) {
        throw std::out_of_range("dm_idx " + std::to_string(dm_idx) +
                                " out of range [0, " + std::to_string(ndms) +
                                ")");
    }

    auto trace = [&](auto& self, SizeType level, SizeType coord_idx,
                     IndexType target) -> void {
        if (level == 0) {
            if (coord_idx < m_container.coordinates[0].size()) {
                const auto& coord0 = m_container.coordinates[0][coord_idx];
                const auto chan    = coord0.i_sub;
                if (chan < m_nchans &&
                    coord0.i_dt < m_container.grids[0][chan].dt_grid.size()) {
                    const auto dt0 =
                        m_container.grids[0][chan].dt_grid[coord0.i_dt];
                    shifts[chan] =
                        target - std::abs(static_cast<IndexType>(dt0));
                }
            }
            return;
        }
        if (level >= m_container.coordinates.size() ||
            coord_idx >= m_container.coordinates[level].size()) {
            return;
        }
        const auto& coord = m_container.coordinates[level][coord_idx];
        if (coord.i_coord_head == SIZE_MAX) {
            // Pure copy/passthrough of an unpaired odd sub-band: target
            // is forwarded unchanged.
            if (coord.i_coord_tail != SIZE_MAX) {
                self(self, level - 1, coord.i_coord_tail, target);
            }
            return;
        }
        const auto dt_val =
            m_container.grids[level][coord.i_sub].dt_grid[coord.i_dt];
        const auto delay = static_cast<IndexType>(coord.delay);
        if (coord.i_coord_tail != SIZE_MAX) {
            self(self, level - 1, coord.i_coord_tail,
                 target - ((dt_val >= 0) ? IndexType{0} : delay));
        }
        self(self, level - 1, coord.i_coord_head,
             target - ((dt_val >= 0) ? delay : IndexType{0}));
    };
    trace(trace, m_niters, dm_idx, IndexType{0});
    return shifts;
}

SizeType FDMTPlan::Impl::compute_fft_support() const {
    const auto& grid0 = m_container.grids[0];
    std::vector<SizeType> span(m_container.state_shape[0].ncoords, 0);
    for (const auto& g : grid0) {
        for (SizeType i = 0; i < g.ndt; ++i) {
            span[g.coord_offset + i] =
                static_cast<SizeType>(std::abs(g.dt_grid[i]));
        }
    }
    for (SizeType i_iter = 1; i_iter <= m_niters; ++i_iter) {
        std::vector<SizeType> next(m_container.state_shape[i_iter].ncoords, 0);
        for (const auto& c : m_container.coordinates_sum[i_iter]) {
            const auto cur  = c.buf_offset / c.nsamps;
            const auto tail = c.tail_buf_offset / c.tail_nsamps;
            const auto head = c.head_buf_offset / c.head_nsamps;
            next[cur]       = std::max(span[tail], c.delay + span[head]);
        }
        for (const auto& c : m_container.coordinates_copy[i_iter]) {
            next[c.buf_offset / c.nsamps] =
                span[c.tail_buf_offset / c.tail_nsamps];
        }
        span = std::move(next);
    }
    return span.empty() ? 0 : std::ranges::max(span);
}

SizeType FDMTPlan::Impl::compute_fft_size() const {
    if (m_mode == FDMTMode::kRoll) {
        return m_nsamps;
    }
    // Circular convolution equals the linear one on the kept samples as
    // long as nothing the tree spreads past the transform length wraps
    // back onto them: valid keeps [L, L + nsamps) of an (L + nsamps)
    // window, full keeps [0, dmt_nsamps) of an nsamps window. Round up
    // to a length with only small prime factors.
    const auto support = m_fft_support;
    const auto n_min   = (m_mode == FDMTMode::kValid)
                             ? m_nsamps + std::max(get_fft_overlap(), support)
                             : std::max(m_nsamps + support, get_dmt_nsamps());
    return utils::next_fft_size(n_min);
}

std::vector<ComplexType> FDMTPlan::Impl::get_fft_phasor_table() const {
    const auto n_bins = get_fft_n_bins();
    const auto n_fft  = get_fft_size();
    const auto max_s  = get_max_shift();
    std::vector<ComplexType> phasors((max_s + 1) * n_bins);
    const float two_pi_over_n =
        static_cast<float>(2.0 * std::numbers::pi) / static_cast<float>(n_fft);
    for (SizeType s = 0; s <= max_s; ++s) {
        const auto s_float = static_cast<float>(s);
        for (SizeType k = 0; k < n_bins; ++k) {
            const float angle =
                -two_pi_over_n * s_float * static_cast<float>(k);
            phasors[(s * n_bins) + k] =
                ComplexType(std::cos(angle), std::sin(angle));
        }
    }
    return phasors;
}

std::string FDMTPlan::Impl::summary(std::string_view prefix) const {
    auto size_in_mb = [](SizeType count, SizeType size) {
        return static_cast<float>(count * size) / 1024.0F / 1024.0F;
    };
    const auto& state_shape   = m_container.state_shape;
    const auto waterfall_size = state_shape[0].nchans * state_shape[0].nsamps;
    const auto dmt_size       = get_dmt_size();
    const auto history_size   = get_history_size();
    const auto tree_size      = get_tree_history_size();
    const auto buffer_size    = m_buffer_size;
    const auto niters         = state_shape.size() - 1;

    const auto waterfall_size_mb = size_in_mb(waterfall_size, sizeof(float));
    const auto dmt_size_mb       = size_in_mb(dmt_size, sizeof(float));
    const auto history_size_mb   = size_in_mb(history_size, sizeof(float));
    const auto tree_size_mb      = size_in_mb(tree_size, sizeof(float));
    const auto buffer_size_mb    = size_in_mb(buffer_size, sizeof(float));
    const auto plan_use_mb = size_in_mb(m_container.get_memory_usage(), 1);

    std::array<std::string, 5> lines = {
        fmt::format("*** FDMT Plan Summary ***"),
        fmt::format("Input: Waterfall Size: ({} x {}; {:.1f} MB ), DMT "
                    "Size: ({} x {}; {:.1f} MB )",
                    state_shape[0].nchans, state_shape[0].nsamps,
                    waterfall_size_mb, state_shape[niters].ncoords,
                    state_shape[niters].nsamps, dmt_size_mb),
        fmt::format("Plan Memory Usage: {:.1f} MB", plan_use_mb),
        fmt::format("Largest Level: ({}; {:.1f} MB ), History: level-0 "
                    "({}; {:.1f} MB ), tree ({}; {:.1f} MB )",
                    buffer_size, buffer_size_mb, history_size, history_size_mb,
                    tree_size, tree_size_mb),
        fmt::format("Plan Details: {}", FDMTShape::header_fmt()),
    };
    std::string out;
    for (const auto& line : lines) {
        out += fmt::format("{}{}\n", prefix, line);
    }
    for (SizeType i_iter = 0; i_iter < niters + 1; ++i_iter) {
        out += fmt::format("{}Iteration {:2d}: {}\n", prefix, i_iter,
                           state_shape[i_iter].to_string());
    }
    out += fmt::format("{}{:*>80}\n", prefix, "");
    return out;
}

FDMTPlan::FDMTPlan(float f_min,
                   float f_max,
                   SizeType nchans,
                   SizeType nsamps,
                   float tsamp,
                   IndexType dt_max,
                   IndexType dt_min,
                   SizeType dt_step,
                   std::string_view mode,
                   SizeType subband_depth)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    nsamps,
                                    tsamp,
                                    dt_max,
                                    dt_min,
                                    dt_step,
                                    mode,
                                    subband_depth)) {}
FDMTPlan::FDMTPlan(float f_min,
                   float f_max,
                   SizeType nchans,
                   SizeType nsamps,
                   float tsamp,
                   const std::vector<IndexType>& dt_grid,
                   std::string_view mode,
                   SizeType subband_depth)
    : m_impl(std::make_unique<Impl>(
          f_min, f_max, nchans, nsamps, tsamp, dt_grid, mode, subband_depth)) {}

FDMTPlan::FDMTPlan(float f_min,
                   float f_max,
                   SizeType nchans,
                   SizeType nsamps,
                   float tsamp,
                   const std::vector<float>& dm_grid,
                   std::string_view mode,
                   SizeType subband_depth)
    : m_impl(std::make_unique<Impl>(
          f_min, f_max, nchans, nsamps, tsamp, dm_grid, mode, subband_depth)) {}

FDMTPlan::~FDMTPlan()                              = default;
FDMTPlan::FDMTPlan(FDMTPlan&&) noexcept            = default;
FDMTPlan& FDMTPlan::operator=(FDMTPlan&&) noexcept = default;
FDMTPlan::FDMTPlan(const FDMTPlan& other)
    : m_impl(std::make_unique<Impl>(*other.m_impl)) {}

FDMTPlan& FDMTPlan::operator=(const FDMTPlan& other) {
    if (this != &other) {
        m_impl = std::make_unique<Impl>(*other.m_impl);
    }
    return *this;
}
float FDMTPlan::get_f_min() const noexcept { return m_impl->get_f_min(); }
float FDMTPlan::get_f_max() const noexcept { return m_impl->get_f_max(); }
float FDMTPlan::get_f_ref() const noexcept { return m_impl->get_f_ref(); }
SizeType FDMTPlan::get_nchans() const noexcept { return m_impl->get_nchans(); }
SizeType FDMTPlan::get_nsamps() const noexcept { return m_impl->get_nsamps(); }
float FDMTPlan::get_tsamp() const noexcept { return m_impl->get_tsamp(); }
IndexType FDMTPlan::get_dt_max() const noexcept { return m_impl->get_dt_max(); }
IndexType FDMTPlan::get_dt_min() const noexcept { return m_impl->get_dt_min(); }
SizeType FDMTPlan::get_dt_step() const noexcept {
    return m_impl->get_dt_step();
}
std::string_view FDMTPlan::get_mode() const noexcept {
    return m_impl->get_mode();
}
bool FDMTPlan::is_custom_grid() const noexcept {
    return m_impl->is_custom_grid();
}
float FDMTPlan::get_df() const noexcept { return m_impl->get_df(); }
SizeType FDMTPlan::get_niters() const noexcept { return m_impl->get_niters(); }
SizeType FDMTPlan::get_subband_depth() const noexcept {
    return m_impl->get_subband_depth();
}
SizeType FDMTPlan::get_max_fuse_levels() const noexcept {
    return m_impl->get_max_fuse_levels();
}
const FDMTPlanContainer& FDMTPlan::get_container() const noexcept {
    return m_impl->get_container();
}
FDMTComplexity FDMTPlan::get_complexity(bool use_box_smearing) const noexcept {
    return m_impl->get_complexity(use_box_smearing);
}
std::vector<SizeType>
FDMTPlan::get_operations_by_iteration(bool use_box_smearing) const {
    return m_impl->get_operations_by_iteration(use_box_smearing);
}
std::vector<SizeType> FDMTPlan::get_nodes_by_iteration() const {
    return m_impl->get_nodes_by_iteration();
}
const std::vector<FDMTShape>& FDMTPlan::get_state_shapes() const noexcept {
    return m_impl->get_state_shapes();
}
SizeType FDMTPlan::get_total_operations(bool use_box_smearing) const noexcept {
    return m_impl->get_total_operations(use_box_smearing);
}
SizeType FDMTPlan::get_total_flops(bool use_box_smearing) const noexcept {
    return m_impl->get_total_flops(use_box_smearing);
}
float FDMTPlan::get_theoretical_gflops(bool use_box_smearing) const noexcept {
    return m_impl->get_theoretical_gflops(use_box_smearing);
}
std::vector<IndexType> FDMTPlan::get_dt_grid_final() const noexcept {
    return m_impl->get_dt_grid_final();
}
std::vector<float> FDMTPlan::get_dm_grid_final() const noexcept {
    return m_impl->get_dm_grid_final();
}
std::vector<float> FDMTPlan::get_smearing_grid_final() const noexcept {
    return m_impl->get_smearing_grid_final();
}
SizeType FDMTPlan::get_box_fraction_bits() const noexcept {
    return m_impl->get_box_fraction_bits();
}
std::vector<uint32_t> FDMTPlan::get_box_weights() const {
    return m_impl->get_box_weights();
}
SizeType FDMTPlan::get_box_mean_bits() const noexcept {
    return m_impl->get_box_mean_bits();
}
std::vector<IndexType> FDMTPlan::trace_dm(SizeType dm_idx) const {
    return m_impl->trace_dm(dm_idx);
}
SizeType FDMTPlan::get_startup_samples() const {
    return m_impl->get_startup_samples();
}
std::vector<float>
FDMTPlan::get_subband_variance_grid(SizeType level,
                                    SizeType subband,
                                    SizeType boxcar_width,
                                    bool use_box_smearing,
                                    std::span<const uint8_t> kill_mask) const {
    return m_impl->get_subband_variance_grid(level, subband, boxcar_width,
                                             use_box_smearing, kill_mask);
}
std::vector<float>
FDMTPlan::get_subband_sigma_grid(SizeType level,
                                 SizeType subband,
                                 SizeType boxcar_width,
                                 bool use_box_smearing,
                                 std::span<const uint8_t> kill_mask) const {
    auto grid = get_subband_variance_grid(level, subband, boxcar_width,
                                          use_box_smearing, kill_mask);
    for (auto& v : grid) {
        v = std::sqrt(v);
    }
    return grid;
}
float FDMTPlan::get_effective_variance(
    SizeType dm_idx,
    SizeType boxcar_width,
    bool use_box_smearing,
    std::span<const uint8_t> kill_mask) const {
    return m_impl->get_effective_variance(dm_idx, boxcar_width,
                                          use_box_smearing, kill_mask);
}
float FDMTPlan::get_effective_sigma(SizeType dm_idx,
                                    SizeType boxcar_width,
                                    bool use_box_smearing,
                                    std::span<const uint8_t> kill_mask) const {
    return m_impl->get_effective_sigma(dm_idx, boxcar_width, use_box_smearing,
                                       kill_mask);
}
std::vector<float> FDMTPlan::get_effective_variance_grid(
    SizeType boxcar_width,
    bool use_box_smearing,
    std::span<const uint8_t> kill_mask) const {
    return m_impl->get_effective_variance_grid(boxcar_width, use_box_smearing,
                                               kill_mask);
}
std::vector<float>
FDMTPlan::get_effective_sigma_grid(SizeType boxcar_width,
                                   bool use_box_smearing,
                                   std::span<const uint8_t> kill_mask) const {
    return m_impl->get_effective_sigma_grid(boxcar_width, use_box_smearing,
                                            kill_mask);
}
SizeType FDMTPlan::get_dmt_ndms() const noexcept {
    return m_impl->get_dmt_ndms();
}
SizeType FDMTPlan::get_dmt_nsamps() const noexcept {
    return m_impl->get_dmt_nsamps();
}
SizeType FDMTPlan::get_dmt_size() const noexcept {
    return m_impl->get_dmt_size();
}
SizeType FDMTPlan::get_buffer_size() const noexcept {
    return m_impl->get_buffer_size();
}
SizeType FDMTPlan::get_history_size() const noexcept {
    return m_impl->get_history_size();
}

SizeType FDMTPlan::get_tree_history_size() const noexcept {
    return m_impl->get_tree_history_size();
}
SizeType FDMTPlan::get_fft_overlap() const noexcept {
    return m_impl->get_fft_overlap();
}
SizeType FDMTPlan::get_fft_size() const noexcept {
    return m_impl->get_fft_size();
}
SizeType FDMTPlan::get_fft_support() const noexcept {
    return m_impl->get_fft_support();
}
SizeType FDMTPlan::get_fft_n_bins() const noexcept {
    return m_impl->get_fft_n_bins();
}
SizeType FDMTPlan::get_fft_buffer_size() const noexcept {
    return m_impl->get_fft_buffer_size();
}
SizeType FDMTPlan::get_max_shift() const noexcept {
    return m_impl->get_max_shift();
}
std::vector<ComplexType> FDMTPlan::get_fft_phasor_table() const {
    return m_impl->get_fft_phasor_table();
}
std::string FDMTPlan::summary(std::string_view prefix) const {
    return m_impl->summary(prefix);
}

// --- Definitions for DDMTPlan ---

} // namespace dmt::plans
