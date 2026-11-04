#pragma once

#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "dmt/algorithms/fdmt_boxcar.hpp"

#include "dmt/engines.hpp"

namespace dmt::algorithms::detail {

/// Search planes of `plan`. Channels with kill_mask[c] == 0 are excluded
/// from the noise model (empty: keep all); `extra_warmup` is added to every
/// row's cold support (FDMT-FFT's interpolation guard).
[[nodiscard]] std::vector<plans::BoxcarPlanePlan>
make_boxcar_plans(const plans::FDMTPlan& plan,
                  const BoxcarConfig& config,
                  bool smear,
                  std::span<const uint8_t> kill_mask = {},
                  uint64_t extra_warmup              = 0);

/// Per-plane row description, shared by CPU and device root sample readers.
struct BoxcarRow {
    SizeType row    = 0;
    SizeType tail   = 0;
    SizeType head   = SIZE_MAX;
    SizeType delay  = 0;
    SizeType hist   = 0;
    uint64_t warmup = 0;
    RawCandidate candidate{};
};

[[nodiscard]] std::vector<BoxcarRow>
make_boxcar_rows(const plans::FDMTPlan& plan,
                 std::span<const plans::BoxcarPlanePlan> planes);

/// The same root addition as FDMT; history is not mutated by the reader.
DMT_HD inline float boxcar_sample(const FDMTSearchInput& input,
                                  const BoxcarRow& row,
                                  SizeType beam,
                                  SizeType t,
                                  bool root) noexcept {
    const float* state = input.state + beam * input.beam_stride;
    if (!root || input.materialized_root) {
        return state[row.row * input.row_stride + t];
    }
    const float tail = state[row.tail * input.row_stride + t];
    if (row.head == SIZE_MAX) {
        return tail;
    }
    float head;
    if (t >= row.delay) {
        head = state[row.head * input.row_stride + t - row.delay];
    } else {
        const SizeType slot =
            input.history_ring ? (input.sample_begin + t) % row.delay : t;
        head = input.history[beam * input.history_stride + row.hist -
                             input.history_base + slot];
    }
    return tail + head;
}

struct BoxcarPeak {
    uint64_t begin   = 0;
    uint64_t end     = 0;
    uint64_t peak    = 0;
    float snr        = 0.0F;
    uint32_t filter  = 0;
    bool open        = false;
    uint64_t raw_sum = 0;
};

DMT_HD inline bool boxcar_better(float snr,
                                 uint64_t sample,
                                 uint32_t filter,
                                 const BoxcarPeak& peak) noexcept {
    return !peak.open || snr > peak.snr ||
           (snr == peak.snr && (sample < peak.peak ||
                                (sample == peak.peak && filter < peak.filter)));
}

/// Feed one point; emission is supplied by the CPU/GPU buffer owner.
template <typename Emit>
DMT_HD void boxcar_peak_step(BoxcarPeak& peak,
                             bool above,
                             float snr,
                             uint64_t sample,
                             uint32_t filter,
                             Emit emit) {
    if (!above) {
        if (peak.open) {
            emit(peak);
            peak.open = false;
        }
        return;
    }
    const bool better = boxcar_better(snr, sample, filter, peak);
    if (!peak.open) {
        peak.begin = sample;
    }
    peak.end = sample;
    if (better) {
        peak.snr    = snr;
        peak.peak   = sample;
        peak.filter = filter;
    }
    peak.open = true;
}

DMT_HD inline uint64_t boxcar_integer_sample(const FDMTSearchInput& input,
                                             const BoxcarRow& row,
                                             SizeType beam,
                                             SizeType t,
                                             bool root) noexcept {
    const auto* state = input.integer_state + beam * input.beam_stride;
    if (!root || input.materialized_root) {
        return state[row.row * input.row_stride + t];
    }
    const uint64_t tail = state[row.tail * input.row_stride + t];
    if (row.head == SIZE_MAX) {
        return tail;
    }
    const SizeType slot = input.history_ring && t < row.delay
                              ? (input.sample_begin + t) % row.delay
                              : t;
    const uint64_t head =
        t >= row.delay
            ? state[row.head * input.row_stride + t - row.delay]
            : input.integer_history[beam * input.history_stride + row.hist -
                                    input.history_base + slot];
    return tail + head;
}

DMT_HD inline bool boxcar_integer_better(uint64_t sum,
                                         uint64_t sample,
                                         uint32_t filter,
                                         const BoxcarPeak& peak) noexcept {
    return !peak.open || sum > peak.raw_sum ||
           (sum == peak.raw_sum &&
            (sample < peak.peak ||
             (sample == peak.peak && filter < peak.filter)));
}

template <typename Emit>
DMT_HD void boxcar_integer_peak_step(BoxcarPeak& peak,
                                     bool above,
                                     uint64_t sum,
                                     uint64_t sample,
                                     uint32_t filter,
                                     Emit emit) {
    if (!above) {
        if (peak.open) {
            emit(peak);
            peak.open = false;
        }
        return;
    }
    const bool better = boxcar_integer_better(sum, sample, filter, peak);
    if (!peak.open) {
        peak.begin = sample;
    }
    peak.end = sample;
    if (better) {
        peak.raw_sum = sum;
        peak.peak    = sample;
        peak.filter  = filter;
    }
    peak.open = true;
}

/// The transform feeding a boxcar engine: FDMT (merged root rows) or
/// FDMT-FFT (materialized root rows per overlap-save segment).
class BoxcarDriver {
public:
    BoxcarDriver()                               = default;
    virtual ~BoxcarDriver()                      = default;
    BoxcarDriver(const BoxcarDriver&)            = delete;
    BoxcarDriver& operator=(const BoxcarDriver&) = delete;
    BoxcarDriver(BoxcarDriver&&)                 = delete;
    BoxcarDriver& operator=(BoxcarDriver&&)      = delete;

    virtual void search(std::span<const float>, FDMTSearchSink&)          = 0;
    virtual void search(DeviceSpan<const float>, FDMTSearchSink&, Stream) = 0;
    virtual void search(std::span<const uint8_t>, SizeType, FDMTSearchSink&) {
        throw std::invalid_argument("boxcar search: packed input unsupported");
    }
    virtual void
    search(DeviceSpan<const uint8_t>, SizeType, FDMTSearchSink&, Stream) {
        throw std::invalid_argument("boxcar search: packed input unsupported");
    }
    virtual void reset(Stream)                                    = 0;
    [[nodiscard]] virtual FDMTMemoryUsage memory() const noexcept = 0;
    [[nodiscard]] virtual SizeType chunk_nsamps() const noexcept  = 0;
    [[nodiscard]] virtual SizeType fuse_levels() const noexcept   = 0;
};

[[nodiscard]] std::unique_ptr<BoxcarDriver>
make_boxcar_driver(std::unique_ptr<FDMTEngine> fdmt);
[[nodiscard]] std::unique_ptr<BoxcarDriver>
make_boxcar_driver(std::unique_ptr<FDMTFFTEngine> fft);

class FDMTBoxcarEngine : public FDMTSearchSink {
public:
    virtual CandidateResultView execute(std::span<const float>)   = 0;
    virtual CandidateDeviceResultView execute(DeviceSpan<const float>,
                                              Stream)             = 0;
    virtual CandidateResultView execute(std::span<const uint8_t>) = 0;
    virtual CandidateDeviceResultView execute(DeviceSpan<const uint8_t>,
                                              Stream)             = 0;
    virtual CandidateResultView flush()                           = 0;
    virtual CandidateDeviceResultView flush_device(Stream)        = 0;
    virtual CandidateResultView copy_result(Stream)               = 0;
    virtual void reset_history(Stream)                            = 0;
    virtual FDMTBoxcarMemoryUsage memory_usage() const noexcept   = 0;
    virtual SizeType chunk_nsamps() const noexcept                = 0;
    virtual SizeType fuse_levels() const noexcept                 = 0;
};

/// The boxcar engines own `driver`, which calls them back per chunk.
std::unique_ptr<FDMTBoxcarEngine>
make_fdmt_boxcar_cpu(const plans::FDMTPlan&,
                     const BoxcarConfig&,
                     const std::vector<plans::BoxcarPlanePlan>&,
                     const FDMTEngineConfig&,
                     std::unique_ptr<BoxcarDriver> driver);
std::unique_ptr<FDMTBoxcarEngine>
make_fdmt_boxcar_gpu(const plans::FDMTPlan&,
                     const BoxcarConfig&,
                     const std::vector<plans::BoxcarPlanePlan>&,
                     const FDMTEngineConfig&,
                     std::unique_ptr<BoxcarDriver> driver);

} // namespace dmt::algorithms::detail
