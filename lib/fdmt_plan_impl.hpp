#pragma once

// Private: FDMTPlan's implementation. Declared here and defined in
// fdmt_plan.cpp (queries, shapes and the public forwarders),
// fdmt_plan_build.cpp (the tree and its merge plan) and
// fdmt_plan_noise.cpp (the noise model of every row).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

#include "dmt/fdmt_box_weights.hpp"
#include "dmt/modes.hpp"

namespace dmt::plans {

class FDMTPlan::Impl {
public:
    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         IndexType dt_max,
         IndexType dt_min,
         SizeType dt_step,
         std::string_view mode,
         SizeType subband_depth);

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         const std::vector<IndexType>& dt_grid,
         std::string_view mode,
         SizeType subband_depth);

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         SizeType nsamps,
         float tsamp,
         const std::vector<float>& dm_grid,
         std::string_view mode,
         SizeType subband_depth);

    ~Impl()                                = default;
    Impl(Impl&& other) noexcept            = default;
    Impl& operator=(Impl&& other) noexcept = default;
    Impl(const Impl& other)                = default;
    Impl& operator=(const Impl& other)     = default;

    float get_f_min() const noexcept { return m_f_min; }
    float get_f_max() const noexcept { return m_f_max; }
    float get_f_ref() const noexcept { return m_f_min; }
    SizeType get_nchans() const noexcept { return m_nchans; }
    SizeType get_nsamps() const noexcept { return m_nsamps; }
    float get_tsamp() const noexcept { return m_tsamp; }
    IndexType get_dt_max() const noexcept { return m_dt_max; }
    IndexType get_dt_min() const noexcept { return m_dt_min; }
    SizeType get_dt_step() const noexcept { return m_dt_step; }
    // m_mode is a validated enum: fdmt_mode_to_string() cannot throw.
    // NOLINTNEXTLINE(bugprone-exception-escape)
    std::string_view get_mode() const noexcept {
        return fdmt_mode_to_string(m_mode);
    }
    bool is_custom_grid() const noexcept { return m_is_custom_grid; }
    float get_df() const noexcept { return m_df; }
    SizeType get_niters() const noexcept { return m_niters; }
    SizeType get_subband_depth() const noexcept { return m_subband_depth; }
    SizeType get_max_fuse_levels() const noexcept {
        return m_subband_depth == 0 ? m_niters : m_niters - m_subband_depth - 1;
    }
    const FDMTPlanContainer& get_container() const noexcept {
        return m_container;
    }
    FDMTComplexity get_complexity(bool use_box_smearing = true) const noexcept;
    std::vector<SizeType>
    get_operations_by_iteration(bool use_box_smearing = true) const {
        return get_complexity(use_box_smearing).ops_by_iter;
    }
    std::vector<SizeType> get_nodes_by_iteration() const {
        return get_complexity().nodes_by_iter;
    }
    const std::vector<FDMTShape>& get_state_shapes() const noexcept {
        return m_container.state_shape;
    }
    SizeType get_total_operations(bool use_box_smearing = true) const noexcept {
        const auto ops = get_operations_by_iteration(use_box_smearing);
        SizeType total = 0;
        for (auto v : ops) {
            total += v;
        }
        return total;
    }
    SizeType get_total_flops(bool use_box_smearing = true) const noexcept {
        return get_total_operations(use_box_smearing) * m_nsamps;
    }
    float get_theoretical_gflops(bool use_box_smearing = true) const noexcept {
        if (m_tsamp <= 0.0F) {
            return 0.0F;
        }
        return static_cast<float>(get_total_operations(use_box_smearing)) /
               (m_tsamp * 1e9F);
    }
    std::vector<IndexType> get_dt_grid_final() const noexcept {
        return m_container.grids[m_niters][0].dt_grid;
    }
    std::vector<float> get_dm_grid_final() const noexcept;
    std::vector<float> get_smearing_grid_final() const noexcept;
    /**
     * @brief Per-channel input-sample shift needed to *place* a synthetic
     * pulse so it reconstructs at a given DM/dt trial, i.e. the inverse of
     * "where does this channel's data end up" — see `algorithms::
     * add_frb_track`, which injects a unit impulse into channel c at
     * `toffset + trace_dm(dm_idx)[c]` for every channel to make the FDMT
     * transform peak at exactly `(dm_idx, toffset)`.
     *
     * Derivation: `offset_add`'s hot loop computes
     * `out[k] = tail[k] + head[k - delay_shift]`, i.e. an impulse sitting at
     * position p in the HEAD child's own buffer surfaces in the parent's
     * output at position `p + delay_shift`. Turned around: for the parent's
     * output to peak at a chosen target position T, the HEAD child's own
     * data must be placed at `T - delay_shift`, while the TAIL child's data
     * stays at `T` unchanged. Walking this rule recursively from the root
     * (target = toffset) down to level 0 gives every leaf channel's
     * required placement: descending into a tail subtree leaves the running
     * target unchanged, descending into a head subtree subtracts that
     * merge's `coord.delay`. Which child is "tail" vs "head" in this sense
     * (not which is numerically lower/higher frequency) flips with the sign
     * of that merge's own dt, exactly mirroring `make_plan`'s tail/head
     * reference swap for negative dt. A copy/passthrough coordinate
     * (unpaired odd sub-band, `i_coord_head == SIZE_MAX`) forwards its
     * target unchanged. Level 0 applies one more magnitude-only shift of
     * `|dt|` in the same "look backward" direction as a head shift (see
     * `fdmt_init_subband`'s doc comment in fdmt.cpp), so the running target
     * is reduced by `|dt|` once more at the leaf itself.
     */
    SizeType get_startup_samples() const;

    std::vector<IndexType> trace_dm(SizeType dm_idx) const;
    /**
     * @brief variance of a boxcar matched filter of
     * width @p w convolved with one channel's own intra-channel smearing
     * kernel of width L_c = `smearing_row[c] + 1` samples, weighted by the
     * channel's box-mean weight round(2^b / L_c) * 2^-b and summed over
     * channels: sum_c weight_c^2 * V(L_c, w), with V the trapezoid
     * sum of squares of the box (x) boxcar kernel.
     *
     * When there is no smearing at all (every `smearing_row[c] == 0`), each
     * weight is exactly 1 and each term collapses to `w`, so this is a strict
     * superset of the `use_box_smearing = false` fast path (`nchans * w`)
     * rather than a separate special case of it.
     */
    float variance_from_smearing_row(
        const float* smearing_row,
        float w,
        std::span<const uint8_t> kill_mask) const noexcept;
    SizeType get_box_fraction_bits() const noexcept {
        return algorithms::detail::box_fraction_bits(
            m_container.state_shape[0].dt_max);
    }
    SizeType get_box_mean_bits() const noexcept {
        return algorithms::detail::box_mean_bits(
            m_container.state_shape[0].dt_max + 1);
    }
    std::vector<uint32_t> get_box_weights() const;
    static bool is_active(std::span<const uint8_t> kill_mask,
                          SizeType c) noexcept {
        return kill_mask.empty() || kill_mask[c] != 0;
    }
    void check_kill_mask(std::span<const uint8_t> kill_mask) const;
    // Active channels in [begin, end).
    static SizeType count_active(std::span<const uint8_t> kill_mask,
                                 SizeType begin,
                                 SizeType end) noexcept;
    float get_effective_variance(SizeType dm_idx,
                                 SizeType boxcar_width,
                                 bool use_box_smearing,
                                 std::span<const uint8_t> kill_mask) const;
    float get_effective_sigma(SizeType dm_idx,
                              SizeType boxcar_width,
                              bool use_box_smearing,
                              std::span<const uint8_t> kill_mask) const;
    std::vector<float>
    get_effective_variance_grid(SizeType boxcar_width,
                                bool use_box_smearing,
                                std::span<const uint8_t> kill_mask) const;
    std::vector<float>
    get_effective_sigma_grid(SizeType boxcar_width,
                             bool use_box_smearing,
                             std::span<const uint8_t> kill_mask) const;
    std::vector<float>
    get_subband_variance_grid(SizeType level,
                              SizeType subband,
                              SizeType boxcar_width,
                              bool use_box_smearing,
                              std::span<const uint8_t> kill_mask) const;
    SizeType get_dmt_ndms() const noexcept {
        return m_container.state_shape[m_niters].ncoords;
    }
    SizeType get_dmt_nsamps() const noexcept {
        return m_container.state_shape[m_niters].nsamps;
    }
    SizeType get_dmt_size() const noexcept {
        return m_container.state_shape[m_niters].nelements;
    }
    SizeType get_buffer_size() const noexcept {
        return m_container.get_buffer_size();
    }
    SizeType get_history_size() const noexcept {
        // Level 0 reads at most dt0_max samples before a block: the raw
        // input history of every channel.
        return m_nchans * m_container.state_shape[0].dt_max;
    }
    SizeType get_tree_history_size() const noexcept {
        return m_container.tree_history_size;
    }
    SizeType get_fft_overlap() const noexcept {
        return static_cast<SizeType>(
            std::max(std::abs(m_dt_min), std::abs(m_dt_max)));
    }
    SizeType get_fft_support() const noexcept { return m_fft_support; }
    SizeType get_fft_size() const noexcept { return m_fft_size; }
    // Longest delay any root coordinate applies to any input sample, in the
    // Fourier engines' convention: a level-0 node spans shifts [0, |dt|] and
    // a merge adds `delay` to its head, so span = max(tail, delay + head).
    SizeType compute_fft_support() const;
    SizeType compute_fft_size() const;
    SizeType get_fft_n_bins() const noexcept {
        return (get_fft_size() / 2) + 1;
    }
    SizeType get_fft_buffer_size() const noexcept {
        SizeType max_coords = 0;
        for (const auto& shape : m_container.state_shape) {
            max_coords = std::max(max_coords, shape.ncoords);
        }
        return max_coords * get_fft_n_bins();
    }
    SizeType get_max_shift() const noexcept {
        SizeType max_s = m_container.state_shape[0].dt_max;
        for (SizeType i_iter = 1; i_iter <= m_niters; ++i_iter) {
            for (const auto& coord : m_container.coordinates_sum[i_iter]) {
                max_s = std::max(max_s, coord.delay);
            }
        }
        return max_s;
    }
    std::vector<ComplexType> get_fft_phasor_table() const;

    std::string summary(std::string_view prefix) const;

private:
    float m_f_min;
    float m_f_max;
    SizeType m_nchans;
    SizeType m_nsamps;
    float m_tsamp;
    IndexType m_dt_max;
    IndexType m_dt_min;
    SizeType m_dt_step;
    FDMTMode m_mode;
    SizeType m_subband_depth{0};
    bool m_is_custom_grid{false};
    std::vector<IndexType> m_dt_grid_target;

    float m_df{};
    SizeType m_niters{};
    FDMTPlanContainer m_container;
    SizeType m_buffer_size{};
    SizeType m_fft_support{};
    SizeType m_fft_size{};

    void validate_band() const;

    void validate_inputs() const;

    /// One subband of the merge tree at one level (see FDMTCoordGrid).
    struct TreeNode {
        SizeType chan_begin;
        SizeType chan_end;
        SizeType tail;
        SizeType head;
        bool staggered;
    };

    /// The merge tree, level 0 (channels) to m_niters (the band). The band
    /// is halved top-down, the lower child taking the extra channel of an
    /// odd count; a single channel above level 0 is copied. Depths
    /// 1..m_subband_depth gain the staggered merges of adjacent subbands of
    /// different parents, after the tree subbands of their level.
    std::vector<std::vector<TreeNode>> build_tree() const;

    float chan_freq(SizeType chan) const noexcept {
        return (m_df * static_cast<float>(chan)) + m_f_min;
    }

    /// Lower edge, split and upper edge (MHz) of a merge or copy node.
    std::array<float, 3> node_freqs(const TreeNode& node,
                                    const std::vector<TreeNode>& prev) const;

    std::vector<std::vector<std::vector<IndexType>>> determine_active_grids(
        const std::vector<std::vector<TreeNode>>& tree) const;

    void configure_plan();

    void
    make_plan_iter0(const std::vector<std::vector<IndexType>>& active_level0);

    void make_plan(SizeType i_iter,
                   const std::vector<std::vector<TreeNode>>& tree,
                   const std::vector<std::vector<IndexType>>& active_level);
};

} // namespace dmt::plans
