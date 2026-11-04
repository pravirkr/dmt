#pragma once

#include <memory>
#include <span>
#include <string>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/common/boxcar_plans.hpp"
#include "dmt/common/candidate.hpp"

namespace dmt::algorithms {

/// Allocated memory in bytes. The full DM-time output is never allocated.
struct FDMTBoxcarMemoryUsage {
    SizeType plan       = 0;
    SizeType state      = 0;
    SizeType history    = 0; ///< FDMT and boxcar sample history.
    SizeType peaks      = 0;
    SizeType workspace  = 0;
    SizeType candidates = 0; ///< Includes host result storage on GPUs.
    SizeType staging    = 0;
    [[nodiscard]] SizeType total() const noexcept {
        return plan + state + history + peaks + workspace + candidates +
               staging;
    }
};

/**
 * @brief Streaming valid-mode FDMT followed by analytical boxcar detection.
 *
 * Input is finite float32, (nbeams, nchans, nsamps), low frequency first,
 * normalized to zero mean and known common noise sigma. Analytical S/N
 * assumes independent channel/time noise before the modeled FDMT filtering;
 * masking or additional correlations require upstream calibration.
 * For packed unsigned input set config.input_nbits (1/2/4/8/16), input_mean
 * and noise_sigma in code units, and execute byte spans. Packed rows follow
 * FDMT byte order with independent row padding; 32 denotes float32 input.
 *
 * Subband search: a plan built with subband_depth d is searched at depths
 * 0..d. Depth k covers 2^k equal tree subbands and the 2^k - 1 staggered
 * subbands of the same width centred on their boundaries, so a narrowband
 * burst anywhere in the band falls mostly inside one searched subband
 * (d = 1: full band, two halves and the central half). Every subband row
 * has its own analytical noise model; candidates carry the subband's depth,
 * index, channel extent and staggered flag.
 *
 * Consecutive non-overlapping blocks form one stream. Root samples are
 * merged directly into the search without a DM-time output plane. Pending
 * contiguous excursions cross blocks; flush() emits them at end of stream.
 * Host calls block on every backend. Device calls are asynchronous; order
 * operations on one stream, or synchronize explicitly when changing streams.
 * GPU reset_history() is also ordered asynchronously on its supplied stream.
 * An instance is not safe for concurrent calls. CPU device calls throw.
 */
class FDMTBoxcar {
public:
    explicit FDMTBoxcar(plans::FDMTPlan plan,
                        BoxcarConfig config   = {},
                        bool use_box_smearing = true,
                        Exec exec             = {},
                        SizeType nbeams       = 1,
                        SizeType fuse_levels  = kFDMTAutoFuse,
                        SizeType chunk_nsamps = 0);
    ~FDMTBoxcar();
    FDMTBoxcar(FDMTBoxcar&&) noexcept;
    FDMTBoxcar& operator=(FDMTBoxcar&&) noexcept;
    FDMTBoxcar(const FDMTBoxcar&)            = delete;
    FDMTBoxcar& operator=(const FDMTBoxcar&) = delete;

    [[nodiscard]] CandidateResultView execute(std::span<const float> waterfall);
    [[nodiscard]] CandidateDeviceResultView
    execute(DeviceSpan<const float> waterfall, Stream stream = {});
    /// Packed rows follow FDMT byte order and have independent byte padding.
    [[nodiscard]] CandidateResultView
    execute(std::span<const uint8_t> waterfall);
    [[nodiscard]] CandidateDeviceResultView
    execute(DeviceSpan<const uint8_t> waterfall, Stream stream = {});
    [[nodiscard]] CandidateResultView flush();
    [[nodiscard]] CandidateDeviceResultView flush_device(Stream stream = {});
    [[nodiscard]] CandidateResultView copy_result(Stream stream = {});
    /// Discard all histories and pending peaks; restart stream sample 0.
    void reset_history(Stream stream = {});

    [[nodiscard]] const plans::FDMTPlan& get_plan() const noexcept;
    [[nodiscard]] const BoxcarConfig& get_config() const noexcept;
    [[nodiscard]] std::span<const plans::BoxcarPlanePlan>
    get_search_plans() const noexcept;
    [[nodiscard]] SizeType get_nbeams() const noexcept;
    [[nodiscard]] SizeType get_chunk_nsamps() const noexcept;
    [[nodiscard]] SizeType get_fuse_levels() const noexcept;
    [[nodiscard]] Backend backend() const noexcept;
    [[nodiscard]] int device() const noexcept;
    [[nodiscard]] int nthreads() const noexcept;
    [[nodiscard]] FDMTBoxcarMemoryUsage get_memory_usage() const noexcept;
    [[nodiscard]] std::string summary() const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace dmt::algorithms
