#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "dmt/algorithms/fdmt_boxcar.hpp"
#include "dmt/common/boxcar_plans.hpp"
#include "dmt/common/candidate.hpp"

namespace dmt::algorithms {

/**
 * @brief Streaming valid-mode FDMT-FFT followed by analytical boxcar
 * detection.
 *
 * The Fourier-domain counterpart of FDMTBoxcar, with the same configuration,
 * candidates and result views. Each overlap-save segment's DM rows are
 * searched straight from the inverse transform, so no DM-time plane is
 * allocated. Input is float32 (nbeams, nchans, nsamps), low frequency first,
 * normalized to zero mean and known common noise sigma; packed input and
 * subband search (plan subband_depth > 0) are not supported.
 *
 * The analytical S/N is FDMTBoxcar's: a merge's shift, integer or
 * fractional, is an all-pass operation that leaves each channel's noise
 * spectrum unchanged, so a row's boxcar variance depends only on its
 * channels' level-0 boxes. Channels with kill_mask[c] == 0 read as zero and
 * are excluded from it.
 *
 * Candidate sample indices are input-stream indices, as in FDMTBoxcar. With
 * fractional delays (the default) the search lags the input by
 * get_output_latency() samples: a block searches stream samples
 * [block_start - latency, block_end - latency). Pending excursions cross
 * blocks; flush() emits them at end of stream. Host calls block; device
 * calls are asynchronous on the given stream. Not safe for concurrent calls.
 */
class FDMTFFTBoxcar {
public:
    explicit FDMTFFTBoxcar(plans::FDMTPlan plan,
                           BoxcarConfig config                = {},
                           bool use_box_smearing              = true,
                           Exec exec                          = {},
                           SizeType nbeams                    = 1,
                           bool fractional_delays             = true,
                           std::span<const uint8_t> kill_mask = {});
    ~FDMTFFTBoxcar();
    FDMTFFTBoxcar(FDMTFFTBoxcar&&) noexcept;
    FDMTFFTBoxcar& operator=(FDMTFFTBoxcar&&) noexcept;
    FDMTFFTBoxcar(const FDMTFFTBoxcar&)            = delete;
    FDMTFFTBoxcar& operator=(const FDMTFFTBoxcar&) = delete;

    [[nodiscard]] CandidateResultView execute(std::span<const float> waterfall);
    [[nodiscard]] CandidateDeviceResultView
    execute(DeviceSpan<const float> waterfall, Stream stream = {});
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
    /// Output samples per overlap-save segment, the unit of one search pass.
    [[nodiscard]] SizeType get_chunk_nsamps() const noexcept;
    /// Samples by which the search lags the input (0 with integer delays).
    [[nodiscard]] SizeType get_output_latency() const noexcept;
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
