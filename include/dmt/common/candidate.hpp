#pragma once

#include <cstdint>
#include <span>

#include "dmt/common/backend.hpp"
#include "dmt/common/types.hpp"

namespace dmt {

/// Reduction of contiguous threshold crossings; no gap bridging or clustering.
enum class PeakFindingMode : uint8_t {
    kContiguousPeakPerWidth,
    kContiguousPeakAcrossWidths,
    kAllAboveThreshold,
};

/// What a call keeps when more candidates cross the threshold than fit
/// (BoxcarConfig::overflow_policy, max_cands_per_dm and max_candidates).
enum class OverflowPolicy : uint8_t {
    /// First come (default): once a row has max_cands_per_dm candidates in
    /// the call, or the call max_candidates, later ones are dropped. Which
    /// candidates survive an overflow is unspecified.
    kDropNew,
    /// The brightest, exactly: each row (plane, beam, DM) keeps its
    /// max_cands_per_dm highest-S/N candidates of the call, and of those
    /// the call keeps the max_candidates highest. Ties go to the earlier
    /// peak sample, then beam, subband, DM row and width. A store of
    /// max_cands_per_dm candidates per row counts in the memory budget, so
    /// max_cands_per_dm must be positive (and is meant to be small, e.g.
    /// 16). The surviving set is deterministic.
    kKeepBrightest,
};

/// Analytical search of independent input noise of known common mean/sigma.
/// The subband search depth is the plan's (FDMTPlan subband_depth).
struct BoxcarConfig {
    BoxcarConfig() = default;
    /// Settings only; FDMTBoxcar validates them together with its plan.
    explicit constexpr BoxcarConfig(
        SizeType max_boxcar,
        float wtsp                = 2.0F,
        float detect_thresh       = 6.0F,
        float noise_sigma         = 1.0F,
        PeakFindingMode peak_mode = PeakFindingMode::kContiguousPeakPerWidth,
        SizeType max_candidates   = 131072,
        SizeType max_cands_per_dm = 4096,
        bool skip_cold_prefix     = true,
        SizeType input_nbits      = 32,
        float input_mean          = 0.0F,
        bool use_trapezoid        = false,
        OverflowPolicy overflow_policy = OverflowPolicy::kDropNew) noexcept
        : max_boxcar(max_boxcar),
          wtsp(wtsp),
          detect_thresh(detect_thresh),
          noise_sigma(noise_sigma),
          peak_mode(peak_mode),
          max_candidates(max_candidates),
          max_cands_per_dm(max_cands_per_dm),
          skip_cold_prefix(skip_cold_prefix),
          input_nbits(input_nbits),
          input_mean(input_mean),
          use_trapezoid(use_trapezoid),
          overflow_policy(overflow_policy) {}

    /// Widest box searched, 1..4096; always the last width of the bank.
    SizeType max_boxcar = 4096;
    /// Width spacing >= 1: from 1, next = max(w + 1, ceil(w * wtsp)) while
    /// it is <= max_boxcar, then max_boxcar when the steps stop below it
    /// (plans::generate_box_width_trials).
    float wtsp                = 2.0F;
    float detect_thresh       = 6.0F;
    float noise_sigma         = 1.0F;
    PeakFindingMode peak_mode = PeakFindingMode::kContiguousPeakPerWidth;
    SizeType max_candidates   = 131072;
    /// Per plane/beam/DM/call; 0 disables (kDropNew). Under kKeepBrightest
    /// the size of each row's store of brightest candidates (must be > 0).
    SizeType max_cands_per_dm = 4096;
    /// Default true: a window reaching before the row's cold support
    /// (warmup_samples) sums an incomplete sweep, whose S/N the noise model
    /// does not describe, so it is not searched.
    bool skip_cold_prefix = true;
    SizeType input_nbits  = 32;   ///< 32: float32; otherwise packed 1/2/4/8/16.
    float input_mean      = 0.0F; ///< Common mean in input code units.
    /// Half-weight edge taper for widths >= 3, replacing the boxcar there.
    /// Default false (boxcar-only); searching trapezoids alongside the
    /// boxcars would come as a separate option.
    bool use_trapezoid = false;
    /// Default kDropNew; see OverflowPolicy.
    OverflowPolicy overflow_policy = OverflowPolicy::kDropNew;
};

/**
 * @brief A raw matched-filter detection, without astrophysical selection.
 *
 * Sample indices are absolute stream indices of trailing-window endpoints;
 * begin/end delimit an inclusive contiguous excursion. Channel extents are
 * half-open in FDMT input order (low frequency first); subband_level is the
 * search depth (0: full band) and subband_idx the subband's index at that
 * level, staggered subbands following the 2^depth tree subbands. A carried
 * peak can
 * refer to a previous execute() block. Parallel output order is unspecified.
 */
struct RawCandidate {
    uint64_t peak_sample   = 0;
    uint64_t begin_sample  = 0;
    uint64_t end_sample    = 0;
    float snr              = 0.0F;
    float dm               = 0.0F;
    uint32_t dm_idx        = 0;
    uint32_t filter_idx    = 0;
    uint32_t width         = 1;
    uint32_t beam_idx      = 0;
    uint32_t subband_level = 0;
    uint32_t subband_idx   = 0;
    uint32_t chan_begin    = 0;
    uint32_t chan_end      = 0;
    float f_low_mhz        = 0.0F;
    float f_high_mhz       = 0.0F;
    uint32_t staggered     = 0; ///< 1: overlap subband straddling a boundary
};

/// Counters describe closed excursions (or individual points in all mode).
struct CandidateCounts {
    uint64_t emitted_count  = 0;
    uint64_t total_detected = 0;
    uint64_t dropped_global = 0;
    uint64_t dropped_per_dm = 0;

    [[nodiscard]] bool overflowed() const noexcept {
        return total_detected != emitted_count;
    }
};

/// Engine-owned storage; invalidated by the next result-changing operation.
struct CandidateResultView : CandidateCounts {
    std::span<const RawCandidate> candidates;
    uint64_t input_sample_begin = 0;
    SizeType input_nsamps       = 0; ///< 0 for flush().
};

/// Asynchronous result: synchronize the supplied stream before reading.
struct CandidateDeviceResultView {
    DeviceSpan<const RawCandidate> candidates; ///< Capacity, not count.
    DeviceSpan<const CandidateCounts> counts;  ///< One device object.
    uint64_t input_sample_begin = 0;
    SizeType input_nsamps       = 0;
};

} // namespace dmt
