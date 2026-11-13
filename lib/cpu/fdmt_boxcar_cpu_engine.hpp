#pragma once

// Private: the CPU FDMTBoxcar engine (detail::make_fdmt_boxcar_cpu). The class
// is declared here and defined across:
// - fdmt_boxcar_cpu.cpp           (lifecycle, memory, block noise, dispatch)
// - fdmt_boxcar_cpu_prefix.cpp    (kPrefix & kNone prefix-sum screen + scorer)
// - fdmt_boxcar_cpu_kadane.cpp    (kKadane SIMD bound gate + lazy tile scorer)
// - fdmt_boxcar_cpu_surrogate.cpp (kKadaneSurrogate multi-strip scorer)

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "dmt/boxcar.hpp"
#include "dmt/kadane_monoid.hpp"
#include "dmt/kadane_plan.hpp"
#include "dmt/kadane_simd.hpp"
#include "dmt/kadane_surrogate.hpp"

namespace dmt::algorithms::detail::fdmt_boxcar_cpu {

using kadane::LagWindow;

inline constexpr SizeType kGateTile             = 256;
inline constexpr SizeType kKadaneExactPrefixCap = 4096;

[[nodiscard]] inline bool is_finite_f32(float x) noexcept {
    const volatile uint32_t bits = std::bit_cast<uint32_t>(x);
    return (bits & 0x7F800000U) != 0x7F800000U;
}

/// Scoring tile: each tile rebuilds its running sums and screen minima over
/// the max_width - 1 lookback samples before it, so wide boxcars take longer
/// tiles (about 4x the lookback, 1024 to 4096 samples) to amortize that.
[[nodiscard]] inline SizeType scoring_tile(SizeType max_width) noexcept {
    SizeType tile = 1024;
    while (tile < 4096 && tile < 4 * max_width) {
        tile *= 2;
    }
    return tile;
}

class FDMTBoxcarCpuEngine final : public FDMTBoxcarEngine {
public:
    FDMTBoxcarCpuEngine(const plans::FDMTPlan& plan,
                        const BoxcarConfig& config,
                        const std::vector<plans::BoxcarPlanePlan>& planes,
                        const FDMTEngineConfig& cfg,
                        std::unique_ptr<BoxcarDriver> driver,
                        std::vector<std::pair<SizeType, SizeType>> kill_runs);

    CandidateResultView execute(std::span<const float> waterfall) override;
    CandidateDeviceResultView execute(DeviceSpan<const float> waterfall,
                                      Stream stream) override;
    CandidateResultView execute(std::span<const uint8_t> waterfall) override;
    CandidateDeviceResultView execute(DeviceSpan<const uint8_t> waterfall,
                                      Stream stream) override;
    CandidateDeviceResultView flush_device(Stream stream) override;
    CandidateResultView copy_result(Stream stream) override;
    CandidateResultView flush() override;
    void set_search_stop(uint64_t stop) noexcept override;
    [[nodiscard]] uint64_t stream_position() const noexcept override;
    [[nodiscard]] uint64_t search_latency() const noexcept override;
    [[nodiscard]] uint64_t open_begin_min() const noexcept override;
    [[nodiscard]] SizeType block_noise_bytes() const noexcept override;
    void enable_block_noise(
        std::shared_ptr<const std::vector<uint64_t>> support) override;
    void push_block_noise(BlockNoise entry) override;
    void drop_block_noise() noexcept override;
    void reset_history(Stream stream) override;
    void consume(SizeType level, const FDMTSearchInput& input) override;
    [[nodiscard]] FDMTBoxcarMemoryUsage memory_usage() const noexcept override;
    [[nodiscard]] SizeType chunk_nsamps() const noexcept override;
    [[nodiscard]] std::string search_path() const override;
    [[nodiscard]] SizeType fuse_levels() const noexcept override;

private:
    /// Thresholds and inverse sigmas of a float row's tile: one table
    /// (`thr`, `inv`: the row's entries), or, when `mixed`, a table per
    /// sample and width chosen from the block ring (support: the row's
    /// support with every channel active).
    struct TileTables {
        const float* thr = nullptr;
        const float* inv = nullptr;
        bool mixed       = false;
        uint64_t support = 0;
    };

    /// Under mixed tables: the table of endpoint `sample` for width
    /// `width` and the first sample after it that may use another.
    struct SampleTable {
        const NoiseTables* table;
        uint64_t until;
    };

    [[nodiscard]] static const float* table_thr(const NoiseTables* t,
                                                const float* plane_row,
                                                SizeType row,
                                                SizeType filters) noexcept {
        return t == nullptr ? plane_row : t->thresh.data() + (row * filters);
    }
    [[nodiscard]] static const float* table_inv(const NoiseTables* t,
                                                const float* plane_row,
                                                SizeType row,
                                                SizeType filters) noexcept {
        return t == nullptr ? plane_row : t->inv_sigma.data() + (row * filters);
    }

    // Core result, noise-ring, and staging helpers (fdmt_boxcar_cpu.cpp)
    void begin_result(SizeType samples);
    [[nodiscard]] CandidateResultView view() const;
    void emit(SizeType row, SizeType beam, const BoxcarPeak& peak);
    [[nodiscard]] RawCandidate
    candidate_of(SizeType row, SizeType beam, const BoxcarPeak& peak) const;
    void keep(SizeType index, uint64_t& count, const RawCandidate& candidate);
    void finish_result();
    const float* tile_tables(const plans::BoxcarPlanePlan& plane,
                             SizeType dm,
                             SizeType row_idx,
                             uint64_t start,
                             SizeType n,
                             SizeType tid,
                             TileTables& tables);
    [[nodiscard]] SampleTable sample_table(uint64_t sample,
                                           SizeType width,
                                           uint64_t support) const noexcept;
    template <typename T>
    std::span<const T> masked(std::span<const T> block, std::vector<T>& buffer);
    void update_open_begin();
    void reset_block_noise();
    void commit_block_noise();
    [[nodiscard]] SizeType ring_index(uint64_t sample) const noexcept;

    // Prefix-sum screen & exact tile scorer (fdmt_boxcar_cpu_prefix.cpp)
    void make_screen_groups(const std::vector<uint32_t>& widths);
    [[nodiscard]] SizeType screen_slots() const noexcept {
        return m_screen_levels + (m_config.input_nbits != 32 ? 1 : 0);
    }
    template <typename T, typename Prefix>
    static void prefix_tile(const FDMTSearchInput& input,
                            const BoxcarRow& row,
                            SizeType beam,
                            bool root,
                            SizeType ring0,
                            SizeType t0,
                            SizeType n,
                            Prefix* out) noexcept;
    template <typename T, typename C>
    void score(const plans::BoxcarPlanePlan& plane,
               SizeType row_idx,
               SizeType dm,
               SizeType beam,
               const FDMTSearchInput& input);
    [[nodiscard]] bool screen_tile(const plans::BoxcarPlanePlan& plane,
                                   const float* thr_row,
                                   const BoxcarRow& row,
                                   const double* prefix,
                                   SizeType lookback,
                                   uint64_t start,
                                   SizeType n,
                                   bool cold,
                                   const double* group_thresh,
                                   double* range_min) const;
    [[nodiscard]] bool screen_tile_int(const plans::BoxcarPlanePlan& plane,
                                       SizeType dm,
                                       const BoxcarRow& row,
                                       const uint64_t* prefix,
                                       SizeType lookback,
                                       uint64_t start,
                                       SizeType n,
                                       bool cold,
                                       const double* group_thresh,
                                       double* range_min) const;
    template <typename T, typename Prefix>
    void score_tile(const plans::BoxcarPlanePlan& plane,
                    SizeType dm,
                    const BoxcarRow& row,
                    SizeType beam,
                    const Prefix* prefix,
                    SizeType lookback,
                    uint64_t start,
                    SizeType n,
                    BoxcarPeak* peaks,
                    SizeType row_idx,
                    const TileTables& tables);

    // Kadane gate & shared row staging (fdmt_boxcar_cpu_kadane.cpp)
    void init_kadane(bool use_box_smearing);
    void populate_row_int_m1();
    static const float* stage_row_f32(const FDMTSearchInput& in,
                                      const BoxcarRow& row,
                                      SizeType beam,
                                      SizeType t0,
                                      SizeType n,
                                      float* dst) noexcept;
    static const uint32_t* stage_row_u32(const FDMTSearchInput& in,
                                         const BoxcarRow& row,
                                         SizeType beam,
                                         SizeType t0,
                                         SizeType n,
                                         uint32_t* dst) noexcept;
    const float* stage_row_int_centered_f32(const FDMTSearchInput& in,
                                            const BoxcarRow& row,
                                            SizeType row_idx,
                                            SizeType beam,
                                            SizeType t0,
                                            SizeType n,
                                            float* dst) const noexcept;
    template <typename T, typename C>
    void save_row_carry(SizeType row_idx,
                        SizeType beam,
                        const FDMTSearchInput& in) noexcept;
    template <typename T, typename C>
    void lazy_score_tile(SizeType row_idx,
                         SizeType beam,
                         const FDMTSearchInput& in,
                         SizeType t0,
                         SizeType n,
                         const TileTables* precomputed_tables);
    void consume_kadane(SizeType row_lo,
                        SizeType row_hi,
                        SizeType first_beam,
                        SizeType beams,
                        const FDMTSearchInput& input);

    // Standalone multi-strip Kadane surrogate (fdmt_boxcar_cpu_surrogate.cpp)
    void init_surrogate(bool use_box_smearing);
    void emit_surrogate_raw(SizeType row,
                            SizeType beam,
                            uint64_t begin_sample,
                            uint64_t end_sample,
                            uint64_t peak_sample,
                            float snr,
                            uint32_t filter_idx,
                            uint32_t width);
    void
    emit_surrogate_peak(SizeType row, SizeType beam, const BoxcarPeak& peak);
    [[nodiscard]] const NoiseTables* active_surrogate_table(
        SizeType row_idx, uint64_t sample, uint32_t filter_idx) const noexcept;
    [[nodiscard]] float
    eval_surrogate_excursion_snr(SizeType row_idx,
                                 const kadane::SurrogateExcursion& ex,
                                 const TileTables* tables,
                                 uint64_t tile_start,
                                 uint64_t tile_end,
                                 uint32_t& out_filter_idx) const noexcept;
    void cache_surrogate_pass_snr(
        SizeType row_idx,
        kadane::KadaneSurrogatePassState& st,
        uint32_t pass_idx,
        const TileTables* tables,
        uint64_t tile_start,
        uint64_t tile_end,
        const kadane::KadaneSurrogatePassState* prev_st) const noexcept;
    void handle_surrogate_excursion(SizeType row_idx,
                                    SizeType beam,
                                    const kadane::SurrogateExcursion& ex,
                                    const kadane::KadaneSurrogateRowState& st,
                                    const TileTables* tables,
                                    uint64_t tile_start,
                                    uint64_t tile_end);
    void
    maybe_finalize_surrogate_pending(SizeType row_idx,
                                     SizeType beam,
                                     uint64_t tile_end,
                                     const kadane::KadaneSurrogateRowState& st);
    const float* stage_surrogate_row(const FDMTSearchInput& in,
                                     SizeType row_idx,
                                     SizeType beam,
                                     SizeType t0,
                                     SizeType n,
                                     float* dst_f32,
                                     uint32_t* tmp_u32) const noexcept;
    std::span<const float> build_surrogate_lookback(SizeType tid,
                                                    SizeType row_idx,
                                                    SizeType beam,
                                                    const FDMTSearchInput& in,
                                                    SizeType t0) noexcept;
    void save_surrogate_carry(SizeType tid,
                              SizeType row_idx,
                              SizeType beam,
                              const FDMTSearchInput& in) noexcept;
    void compute_surrogate_row_triggers(
        SizeType row_idx,
        const float* screen_thr,
        std::array<float, kadane::kMaxSurrogatePasses>& out_triggers)
        const noexcept;
    void consume_surrogate(SizeType row_lo,
                           SizeType row_hi,
                           SizeType first_beam,
                           SizeType beams,
                           const FDMTSearchInput& input);
    void sync_surrogate_row_at_pos(SizeType tid,
                                   SizeType row_idx,
                                   SizeType beam,
                                   uint64_t pos);
    void update_open_begin_surrogate();
    void flush_surrogate();

    const plans::FDMTPlan& m_plan;
    BoxcarConfig m_config;
    const std::vector<plans::BoxcarPlanePlan>& m_planes;
    std::vector<BoxcarRow> m_rows;
    SizeType m_beams;
    int m_threads;
    SizeType m_width;
    SizeType m_tile;
    SizeType m_filters;
    SizeType m_peak_count;
    std::unique_ptr<BoxcarDriver> m_driver;
    /// Killed channel runs and the staging block they are zeroed in.
    std::vector<std::pair<SizeType, SizeType>> m_kill_runs;
    std::vector<uint8_t> m_killed; ///< per channel, 1 = killed
    std::vector<float> m_masked;
    std::vector<uint8_t> m_masked_packed;
    std::vector<SizeType> m_offsets;
    /// Row range [first, second) of each level's planes, and each row's plane.
    std::vector<std::pair<SizeType, SizeType>> m_level_rows;
    std::vector<uint32_t> m_row_plane;
    std::vector<float> m_carry;
    std::vector<BoxcarPeak> m_peaks;
    std::vector<uint64_t> m_per_dm;
    /// kKeepBrightest: per row, a heap of max_cands_per_dm candidates
    /// (m_per_dm entries used); m_keep_q 0 under kDropNew.
    std::vector<RawCandidate> m_keep;
    SizeType m_keep_q = 0;
    bool m_keep_open  = false;
    std::vector<RawCandidate> m_candidates;
    std::vector<double> m_prefix;
    std::vector<uint64_t> m_integer_prefix, m_integer_carry;
    std::vector<uint32_t> m_integer_carry32;
    bool m_carry32{false};
    std::vector<double> m_integer_means, m_integer_scales;
    double m_integer_unit = 1;
    std::vector<float> m_best;
    std::vector<uint32_t> m_best_filter;
    std::vector<BoxcarScreenGroup> m_groups;
    SizeType m_screen_levels = 0;
    std::vector<double> m_range_min;
    std::vector<double> m_group_thresh;
    CandidateCounts m_counts{};
    std::atomic<uint64_t> m_next_slot{0};
    uint64_t m_position       = 0;
    uint64_t m_search_stop    = UINT64_MAX; // drain(): end of stream
    uint64_t m_result_begin   = 0;
    SizeType m_result_samples = 0;
    // Per-call kill masks: off until the first one (then the per-block
    // tables below replace the planes' tables).
    uint64_t m_open_begin = UINT64_MAX; ///< see open_begin_min()
    bool m_noise_on       = false;
    std::shared_ptr<const std::vector<uint64_t>> m_support;
    std::vector<BlockNoise> m_ring; ///< oldest first, at most m_ring_size
    std::optional<BlockNoise> m_pending;
    /// The entry the last commit evicted, kept until its call completes.
    std::optional<BlockNoise> m_evicted;
    bool m_commit_open   = false;
    SizeType m_ring_size = 1;
    std::vector<float> m_screen_thr; ///< per thread: min thresholds of a tile
    SizeType m_kadane_exact_prefix_cap = kKadaneExactPrefixCap;
    SizeType m_prefix_stride           = 0;
    SizeType m_surrogate_carry_ring    = 0;
    kadane::KadaneGatePlan m_kadane_plan;
    bool m_kadane_int32 = false;
    bool m_any_edge_lag = false;
    std::vector<float> m_kadane_fc;
    std::vector<int32_t> m_kadane_ic;
    std::vector<LagWindow<float>> m_lag_f32;
    std::vector<LagWindow<int32_t>> m_lag_i32;
    std::vector<uint8_t> m_open;
    kadane::KadaneSurrogatePlan m_surrogate_plan;
    bool m_surrogate_int_u32 = false;
    std::vector<float> m_surrogate_c;
    std::vector<float> m_surrogate_dec;
    std::vector<kadane::KadaneSurrogateRowState> m_surrogate_rows;
    std::vector<uint8_t> m_surrogate_open;
    std::vector<uint64_t> m_surrogate_pos;
    std::vector<float> m_surrogate_chk_c;
    std::vector<float> m_surrogate_chk_dec;
    std::vector<float> m_surrogate_lb;
    std::vector<double> m_row_int_m1;
    std::vector<float> m_tile_f32;
    std::vector<uint32_t> m_tile_u32;
    std::vector<float> m_thread_cuts;
};

} // namespace dmt::algorithms::detail::fdmt_boxcar_cpu
