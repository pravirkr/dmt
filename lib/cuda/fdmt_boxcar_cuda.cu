#include "dmt/boxcar.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <thrust/device_vector.h>
#include "dmt/gpu_compat.cuh"

#include "dmt/boxcar_runs.hpp"
#include "dmt/gpu_utils.cuh"

#include "dmt/bit_pack_utils.hpp"

namespace dmt::algorithms::detail {
namespace {

constexpr int kThreads = 128;
constexpr int kItems   = 4;
constexpr int kTile    = kThreads * kItems;
constexpr int kRing    = 8192; // max width + one tile + boundary, rounded up.

template <typename T> struct BoxcarShared {
    // 4609 uint64 boundaries leave space for run summaries within 48 KiB.
    // Float keeps the original power-of-two ring and scan arithmetic.
    static constexpr int ring =
        std::is_integral_v<T> ? 4096 + kTile + 1 : kRing;
    using Scan = std::conditional_t<std::is_integral_v<T>, uint64_t, double>;
    T prefix[ring];
    Scan scan[kThreads];
    BoxcarRunSummary runs[kThreads];
};
static_assert(sizeof(BoxcarShared<float>) <= 48 * 1024);
static_assert(sizeof(BoxcarShared<uint64_t>) <= 48 * 1024);

DMT_D uint64_t increment(uint64_t* count) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(count), 1ULL);
}

struct CandidateWriter {
    RawCandidate* candidates;
    CandidateCounts* counts;
    uint64_t* slots;
    uint64_t* per_dm;
    const uint32_t* widths;
    RawCandidate prototype;
    SizeType capacity;
    SizeType dm_cap;
    const double* means = nullptr;
    const float* scales = nullptr;
    double unit         = 1;
    bool integer        = false;

    DMT_D void operator()(const BoxcarPeak& peak) const {
        increment(&counts->total_detected);
        if (dm_cap != 0 && increment(per_dm) >= dm_cap) {
            increment(&counts->dropped_per_dm);
            return;
        }
        const uint64_t slot = increment(slots);
        if (slot >= capacity) {
            increment(&counts->dropped_global);
            return;
        }
        RawCandidate candidate = prototype;
        candidate.peak_sample  = peak.peak;
        candidate.begin_sample = peak.begin;
        candidate.end_sample   = peak.end;
        candidate.snr =
            integer ? static_cast<float>((static_cast<double>(peak.raw_sum) -
                                          means[peak.filter]) *
                                         (scales[peak.filter] * unit))
                    : peak.snr;
        candidate.filter_idx = peak.filter;
        candidate.width      = widths[peak.filter];
        candidates[slot]     = candidate;
        increment(&counts->emitted_count);
    }
};

// Ordered block prefix of each thread's four consecutive input samples.
// Double scan intermediates bound rounding in the prefix construction; the
// shared ring remains float32 and is rebased after every scoring tile.
template <typename T>
DMT_D void prefix_tile(BoxcarShared<T>& shared,
                       const T (&x)[kItems],
                       SizeType position,
                       T base) {
    const int tid = static_cast<int>(threadIdx.x);
    using Scan    = typename BoxcarShared<T>::Scan;
    Scan total    = 0;
    for (int j = 0; j < kItems; ++j) {
        total += x[j];
    }
    shared.scan[tid] = total;
    __syncthreads();
    for (int offset = 1; offset < kThreads; offset *= 2) {
        const Scan left = tid >= offset ? shared.scan[tid - offset] : 0;
        __syncthreads();
        total += left;
        shared.scan[tid] = total;
        __syncthreads();
    }
    Scan running = base + (tid > 0 ? shared.scan[tid - 1] : Scan{0});
    for (int j = 0; j < kItems; ++j) {
        running += x[j];
        shared
            .prefix[(position + tid * kItems + j + 1) % BoxcarShared<T>::ring] =
            static_cast<T>(running);
    }
    __syncthreads();
}

// Scan trailing-run summaries to give each worker the state of every
// preceding sample. Each worker then emits only excursions ending within
// its own four samples. The last worker saves the open trailing excursion.
template <typename T>
DMT_D void reduce_runs(BoxcarShared<T>& shared,
                       const bool (&above)[kItems],
                       const float (&snr)[kItems],
                       const uint32_t (&filters)[kItems],
                       uint64_t start,
                       int n,
                       BoxcarPeak* persistent,
                       const CandidateWriter& writer,
                       const uint64_t (&raw_sum)[kItems],
                       bool integer) {
    const int tid   = static_cast<int>(threadIdx.x);
    const int first = tid * kItems;
    BoxcarRunSummary summary;
    for (int j = 0; j < kItems && first + j < n; ++j) {
        if (!above[j]) {
            summary.all           = false;
            summary.trailing.open = false;
        } else {
            const auto ignore = [] DMT_D(const BoxcarPeak&) {};
            if (integer) {
                boxcar_integer_peak_step(summary.trailing, true, raw_sum[j],
                                         start + first + j, filters[j], ignore);
            } else {
                boxcar_peak_step(summary.trailing, true, snr[j],
                                 start + first + j, filters[j], ignore);
            }
        }
    }
    const BoxcarRunSummary carried{*persistent, true};
    shared.runs[tid] = summary;
    __syncthreads();
    for (int offset = 1; offset < kThreads; offset *= 2) {
        const auto left =
            tid >= offset ? shared.runs[tid - offset] : BoxcarRunSummary{};
        __syncthreads();
        if (tid >= offset) {
            summary = boxcar_join(left, summary, integer);
        }
        shared.runs[tid] = summary;
        __syncthreads();
    }
    BoxcarPeak incoming =
        tid == 0 ? carried.trailing
                 : boxcar_join(carried, shared.runs[tid - 1], integer).trailing;
    for (int j = 0; j < kItems && first + j < n; ++j) {
        if (integer) {
            boxcar_integer_peak_step(incoming, above[j], raw_sum[j],
                                     start + first + j, filters[j], writer);
        } else {
            boxcar_peak_step(incoming, above[j], snr[j], start + first + j,
                             filters[j], writer);
        }
    }
    // All threads have read persistent and shared summaries before writing
    // the state for the next width/tile.
    __syncthreads();
    if (tid == kThreads - 1) {
        *persistent = boxcar_join(carried, summary, integer).trailing;
    }
    __syncthreads();
}

template <typename T>
__global__ void kernel_fdmt_boxcar(FDMTSearchInput input,
                                   const BoxcarRow* rows,
                                   SizeType row_base,
                                   SizeType ndm,
                                   SizeType total_rows,
                                   const uint32_t* widths,
                                   SizeType nfilters,
                                   SizeType max_width,
                                   const float* thresholds,
                                   const float* inv_sigma,
                                   T* carry_all,
                                   BoxcarPeak* peaks_all,
                                   SizeType peak_count,
                                   RawCandidate* candidates,
                                   CandidateCounts* counts,
                                   uint64_t* slots,
                                   uint64_t* per_dm_all,
                                   SizeType capacity,
                                   SizeType dm_cap,
                                   PeakFindingMode mode,
                                   bool skip_cold,
                                   bool root,
                                   const uint64_t* integer_thresholds,
                                   const double* means,
                                   double unit) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto& shared             = *reinterpret_cast<BoxcarShared<T>*>(storage);
    const SizeType task      = blockIdx.x;
    const SizeType beam      = task / ndm;
    const SizeType d         = task % ndm;
    const SizeType row_index = row_base + d;
    const SizeType index     = beam * total_rows + row_index;
    const BoxcarRow row      = rows[row_index];
    T* carry                 = carry_all + index * max_width;
    BoxcarPeak* peaks        = peaks_all + index * peak_count;
    RawCandidate prototype   = row.candidate;
    prototype.beam_idx       = static_cast<uint32_t>(beam);
    CandidateWriter writer{candidates, counts,    slots,    per_dm_all + index,
                           widths,     prototype, capacity, dm_cap};
    writer.means   = means ? means + row_index * nfilters : nullptr;
    writer.scales  = inv_sigma + row_index * nfilters;
    writer.unit    = unit;
    writer.integer = std::is_integral_v<T> &&
                     mode != PeakFindingMode::kContiguousPeakAcrossWidths;
    const int tid  = static_cast<int>(threadIdx.x);
    const SizeType lookback = max_width - 1;
    if (tid == 0) {
        shared.prefix[0] = 0;
    }
    __syncthreads();
    // Bootstrap only the short raw carry, never an entire DM-time row.
    for (SizeType p = 0; p < lookback; p += kTile) {
        const T base = shared.prefix[p % BoxcarShared<T>::ring];
        T x[kItems];
        for (int j = 0; j < kItems; ++j) {
            const SizeType t    = p + tid * kItems + j;
            const SizeType back = t < lookback ? lookback - t : 0;
            x[j] = t < lookback && input.sample_begin >= back
                       ? carry[(input.sample_begin - back) % max_width]
                       : T{0};
        }
        prefix_tile(shared, x, p, base);
    }
    for (SizeType t0 = 0; t0 < input.nsamps; t0 += kTile) {
        const int n = static_cast<int>(
            input.nsamps - t0 < kTile ? input.nsamps - t0 : kTile);
        const SizeType position = lookback + t0;
        const T base = shared.prefix[position % BoxcarShared<T>::ring];
        T x[kItems];
        float best[kItems];
        uint32_t best_filter[kItems];
        for (int j = 0; j < kItems; ++j) {
            const SizeType local = tid * kItems + j;
            x[j]                 = 0;
            if (local < static_cast<SizeType>(n)) {
                if constexpr (std::is_integral_v<T>) {
                    x[j] = boxcar_integer_sample(input, row, beam, t0 + local,
                                                 root);
                } else {
                    x[j] = boxcar_sample(input, row, beam, t0 + local, root);
                }
            }
            best[j]        = -1.0F;
            best_filter[j] = 0;
        }
        prefix_tile(shared, x, position, base);
        T current[kItems];
        uint64_t raw_sums[kItems]{};
        for (int j = 0; j < kItems; ++j) {
            current[j] = shared.prefix[(position + tid * kItems + j + 1) %
                                       BoxcarShared<T>::ring];
        }
        // Width tables live in global read-only storage and are traversed
        // without a fixed 32-filter array or a filter-dependent launch size.
        for (SizeType f = 0; f < nfilters; ++f) {
            bool above[kItems];
            float snr[kItems];
            uint32_t filters[kItems];
            const SizeType width  = widths[f];
            const float threshold = thresholds[(row_base + d) * nfilters + f];
            const float scale     = inv_sigma[(row_base + d) * nfilters + f];
            for (int j = 0; j < kItems; ++j) {
                const SizeType local  = tid * kItems + j;
                const uint64_t sample = input.sample_begin + t0 + local;
                const SizeType p      = position + local + 1;
                const T sum =
                    current[j] -
                    shared.prefix[(p - width) % BoxcarShared<T>::ring];
                above[j] =
                    local < static_cast<SizeType>(n) &&
                    (!skip_cold || sample >= row.warmup + width - 1) && [&] {
                        if constexpr (std::is_integral_v<T>) {
                            return sum >=
                                   integer_thresholds[row_index * nfilters + f];
                        } else {
                            return sum >= threshold;
                        }
                    }();
                snr[j] = 0;
                if constexpr (std::is_integral_v<T>) {
                    raw_sums[j] = sum;
                    if (above[j] &&
                        mode == PeakFindingMode::kContiguousPeakAcrossWidths) {
                        snr[j] = static_cast<float>(
                            (static_cast<double>(sum) -
                             means[row_index * nfilters + f]) *
                            (scale * unit));
                    }
                } else {
                    snr[j] = above[j] ? sum * scale : 0.0F;
                }
                filters[j] = static_cast<uint32_t>(f);
                if (mode == PeakFindingMode::kContiguousPeakAcrossWidths) {
                    if (above[j] && snr[j] > best[j]) {
                        best[j]        = snr[j];
                        best_filter[j] = filters[j];
                    }
                } else if (mode == PeakFindingMode::kAllAboveThreshold &&
                           above[j]) {
                    BoxcarPeak peak{sample, sample,     sample,
                                    snr[j], filters[j], true};
                    if constexpr (std::is_integral_v<T>) {
                        peak.raw_sum = sum;
                    }
                    writer(peak);
                }
            }
            if (mode == PeakFindingMode::kContiguousPeakPerWidth) {
                reduce_runs(shared, above, snr, filters,
                            input.sample_begin + t0, n, peaks + f, writer,
                            raw_sums, std::is_integral_v<T>);
            }
        }
        if (mode == PeakFindingMode::kContiguousPeakAcrossWidths) {
            bool above[kItems];
            for (int j = 0; j < kItems; ++j) {
                above[j] = best[j] >= 0;
            }
            reduce_runs(shared, above, best, best_filter,
                        input.sample_begin + t0, n, peaks, writer, raw_sums,
                        false);
        }
        __syncthreads();
        // Recenter the retained prefix boundaries after all scorers finish.
        const SizeType end = position + n;
        const T origin =
            shared.prefix[(end - lookback) % BoxcarShared<T>::ring];
        __syncthreads();
        for (SizeType j = tid; j <= lookback; j += kThreads) {
            shared.prefix[(end - lookback + j) % BoxcarShared<T>::ring] -=
                origin;
        }
        __syncthreads();
    }
    // The root's old FDMT history remains available until this kernel ends.
    const SizeType keep = (input.nsamps < max_width ? input.nsamps : max_width);
    for (SizeType t = tid; t < keep; t += kThreads) {
        const SizeType sample = input.nsamps - keep + t;
        if constexpr (std::is_integral_v<T>) {
            carry[(input.sample_begin + sample) % max_width] =
                boxcar_integer_sample(input, row, beam, sample, root);
        } else {
            carry[(input.sample_begin + sample) % max_width] =
                boxcar_sample(input, row, beam, sample, root);
        }
    }
}

__global__ void kernel_flush_boxcar(const BoxcarRow* rows,
                                    SizeType nrows,
                                    SizeType peak_count,
                                    BoxcarPeak* peaks,
                                    const uint32_t* widths,
                                    RawCandidate* candidates,
                                    CandidateCounts* counts,
                                    uint64_t* slots,
                                    uint64_t* per_dm,
                                    SizeType capacity,
                                    SizeType dm_cap,
                                    SizeType tasks,
                                    const double* means,
                                    const float* scales,
                                    SizeType nfilters,
                                    double unit,
                                    bool integer) {
    const SizeType index =
        static_cast<SizeType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tasks) {
        return;
    }
    const SizeType row_index = index % nrows;
    auto prototype           = rows[row_index].candidate;
    prototype.beam_idx       = static_cast<uint32_t>(index / nrows);
    CandidateWriter writer{candidates, counts,    slots,    per_dm + index,
                           widths,     prototype, capacity, dm_cap};
    writer.means   = means ? means + row_index * nfilters : nullptr;
    writer.scales  = scales + row_index * nfilters;
    writer.unit    = unit;
    writer.integer = integer;
    for (SizeType f = 0; f < peak_count; ++f) {
        auto& peak = peaks[index * peak_count + f];
        if (peak.open) {
            writer(peak);
            peak.open = false;
        }
    }
}

template <typename T> T* raw(thrust::device_vector<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}

class FDMTBoxcarCudaEngine final : public FDMTBoxcarEngine {
public:
    FDMTBoxcarCudaEngine(const plans::FDMTPlan& plan,
                         const BoxcarConfig& config,
                         const std::vector<plans::BoxcarPlanePlan>& planes,
                         const FDMTEngineConfig& cfg)
        : m_plan(plan),
          m_config(config),
          m_planes(planes),
          m_beams(cfg.nbeams),
          m_device(cfg.exec.device),
          m_rows(make_boxcar_rows(plan, planes)),
          m_width(planes.front().widths.back()),
          m_filters(planes.front().widths.size()),
          m_peak_count(config.peak_mode ==
                               PeakFindingMode::kContiguousPeakPerWidth
                           ? m_filters
                           : 1) {
        gpu_utils::set_device(m_device);
        if (m_rows.size() > SIZE_MAX / m_beams / m_width ||
            m_rows.size() > SIZE_MAX / m_beams / m_peak_count ||
            m_rows.size() * m_beams > static_cast<SizeType>(INT32_MAX)) {
            throw std::invalid_argument(
                "FDMTBoxcar: GPU search extent exceeds limits");
        }
        m_fdmt = make_fdmt_gpu(plan, cfg);
        std::vector<float> thresholds, scales;
        m_offsets.push_back(0);
        for (const auto& p : planes) {
            thresholds.insert(thresholds.end(), p.raw_thresh_lut.begin(),
                              p.raw_thresh_lut.end());
            scales.insert(scales.end(), p.inv_sigma_lut.begin(),
                          p.inv_sigma_lut.end());
            m_offsets.push_back(m_offsets.back() + p.dm_grid.size());
        }
        m_rows_d             = m_rows;
        m_widths_d           = planes.front().widths;
        m_thresholds_d       = thresholds;
        m_scales_d           = scales;
        const SizeType tasks = m_rows.size() * m_beams;
        m_integer_unit =
            cfg.use_box_smearing
                ? std::ldexp(1.0,
                             -static_cast<int>(plan.get_box_fraction_bits()))
                : 1.0;
        if (config.input_nbits == 32) {
            m_carry_d.resize(tasks * m_width, 0);
        } else {
            m_integer_carry.resize(tasks * m_width, 0);
        }
        std::vector<double> means;
        std::vector<uint64_t> cutoffs;
        for (const auto& p : planes) {
            means.insert(means.end(), p.integer_mean_lut.begin(),
                         p.integer_mean_lut.end());
            cutoffs.insert(cutoffs.end(), p.integer_thresh_lut.begin(),
                           p.integer_thresh_lut.end());
        }
        m_integer_means      = means;
        m_integer_thresholds = cutoffs;
        m_peaks_d.resize(tasks * m_peak_count);
        m_per_dm_d.resize(tasks, 0);
        m_candidates_d.resize(config.max_candidates);
        m_counts_d.resize(1);
        m_slots_d.resize(1, 0);
        m_candidates_h.resize(config.max_candidates);
    }
    ~FDMTBoxcarCudaEngine() override {
        // Destruction must not release search buffers while queued work
        // still references them, including on a nondefault stream.
        cudaSetDevice(m_device);
        m_work.wait();
    }
    CandidateResultView execute(std::span<const float> waterfall) override {
        if (m_config.input_nbits != 32) {
            throw std::invalid_argument(
                "FDMTBoxcar: expected packed uint8 input");
        }
        validate_input(waterfall.size());
        gpu_utils::set_device(m_device);
        m_work.wait();
        if (m_input_d.size() != waterfall.size()) {
            m_input_d.resize(waterfall.size());
        }
        gpu_utils::check_gpu_call(cudaMemcpyAsync(
            raw(m_input_d), waterfall.data(), waterfall.size_bytes(),
            cudaMemcpyHostToDevice, nullptr));
        (void)execute(DeviceSpan<const float>(raw(m_input_d), waterfall.size(),
                                              {kGPUBackend, m_device}),
                      {});
        return copy_result({});
    }
    CandidateDeviceResultView execute(DeviceSpan<const float> waterfall,
                                      Stream queue) override {
        if (m_config.input_nbits != 32) {
            throw std::invalid_argument(
                "FDMTBoxcar: expected packed uint8 input");
        }
        validate_input(waterfall.size());
        begin_result(m_plan.get_nsamps(), queue);
        m_fdmt->execute_search(waterfall, *this, queue);
        m_position += m_plan.get_nsamps();
        m_work.mark(static_cast<cudaStream_t>(queue.native));
        return device_view();
    }
    CandidateResultView execute(std::span<const uint8_t> waterfall) override {
        validate_packed(waterfall.size());
        gpu_utils::set_device(m_device);
        m_work.wait();
        if (m_packed_input.size() != waterfall.size()) {
            m_packed_input.resize(waterfall.size());
        }
        gpu_utils::check_gpu_call(cudaMemcpyAsync(
            raw(m_packed_input), waterfall.data(), waterfall.size_bytes(),
            cudaMemcpyHostToDevice, nullptr));
        (void)execute(DeviceSpan<const uint8_t>(raw(m_packed_input),
                                                waterfall.size(),
                                                {kGPUBackend, m_device}),
                      {});
        return copy_result({});
    }
    CandidateDeviceResultView execute(DeviceSpan<const uint8_t> waterfall,
                                      Stream queue) override {
        validate_packed(waterfall.size());
        begin_result(m_plan.get_nsamps(), queue);
        m_fdmt->execute_search(waterfall, m_config.input_nbits, *this, queue);
        m_position += m_plan.get_nsamps();
        m_work.mark(static_cast<cudaStream_t>(queue.native));
        return device_view();
    }
    CandidateResultView flush() override {
        gpu_utils::set_device(m_device);
        m_work.wait();
        (void)flush_device({});
        return copy_result({});
    }
    CandidateDeviceResultView flush_device(Stream queue) override {
        begin_result(0, queue);
        const auto tasks  = m_rows.size() * m_beams;
        const auto stream = static_cast<cudaStream_t>(queue.native);
        kernel_flush_boxcar<<<(tasks + 127) / 128, 128, 0, stream>>>(
            raw(m_rows_d), m_rows.size(), m_peak_count, raw(m_peaks_d),
            raw(m_widths_d), raw(m_candidates_d), raw(m_counts_d),
            raw(m_slots_d), raw(m_per_dm_d), m_config.max_candidates,
            m_config.max_cands_per_dm, tasks, raw(m_integer_means),
            raw(m_scales_d), m_filters, m_integer_unit,
            m_config.input_nbits != 32 &&
                m_config.peak_mode !=
                    PeakFindingMode::kContiguousPeakAcrossWidths);
        gpu_utils::check_last_gpu_error("FDMTBoxcar flush failed");
        m_work.mark(stream);
        return device_view();
    }
    CandidateResultView copy_result(Stream queue) override {
        gpu_utils::set_device(m_device);
        const auto stream = static_cast<cudaStream_t>(queue.native);
        m_work.order(stream);
        gpu_utils::check_gpu_call(cudaMemcpyAsync(
            &m_counts_h, raw(m_counts_d), sizeof(CandidateCounts),
            cudaMemcpyDeviceToHost, stream));
        gpu_utils::check_gpu_call(cudaStreamSynchronize(stream));
        if (m_counts_h.emitted_count > 0) {
            gpu_utils::check_gpu_call(
                cudaMemcpyAsync(m_candidates_h.data(), raw(m_candidates_d),
                                m_counts_h.emitted_count * sizeof(RawCandidate),
                                cudaMemcpyDeviceToHost, stream));
            gpu_utils::check_gpu_call(cudaStreamSynchronize(stream));
        }
        return {m_counts_h,
                {m_candidates_h.data(),
                 static_cast<SizeType>(m_counts_h.emitted_count)},
                m_result_begin,
                m_result_samples};
    }
    void reset_history(Stream queue) override {
        gpu_utils::set_device(m_device);
        const auto stream = static_cast<cudaStream_t>(queue.native);
        m_fdmt->reset_search_history(queue);
        gpu_utils::check_gpu_call(cudaMemsetAsync(
            raw(m_carry_d), 0, m_carry_d.size() * sizeof(float), stream));
        gpu_utils::check_gpu_call(cudaMemsetAsync(
            raw(m_peaks_d), 0, m_peaks_d.size() * sizeof(BoxcarPeak), stream));
        if (!m_integer_carry.empty()) {
            gpu_utils::check_gpu_call(cudaMemsetAsync(
                raw(m_integer_carry), 0,
                m_integer_carry.size() * sizeof(uint64_t), stream));
        }
        m_position = 0;
        begin_result(0, queue);
        m_work.mark(stream);
    }
    void consume(SizeType level, const FDMTSearchInput& input) override {
        for (SizeType p = 0; p < m_planes.size(); ++p) {
            const auto& plane = m_planes[p];
            if (plane.level != level) {
                continue;
            }
            const dim3 grid(plane.dm_grid.size() * m_beams);
            const dim3 block(kThreads);
            gpu_utils::check_kernel_launch_params(grid, block);
            if (m_config.input_nbits == 32) {
                launch_score<float>(input, p, grid, block, raw(m_carry_d));
            } else {
                launch_score<uint64_t>(input, p, grid, block,
                                       raw(m_integer_carry));
            }
            gpu_utils::check_last_gpu_error("FDMTBoxcar scoring failed");
        }
    }
    FDMTBoxcarMemoryUsage memory_usage() const noexcept override {
        const auto fdmt    = m_fdmt->get_memory_usage();
        SizeType host_plan = m_rows.size() * sizeof(BoxcarRow) +
                             m_offsets.size() * sizeof(SizeType);
        for (const auto& p : m_planes) {
            host_plan +=
                sizeof(p) + p.widths.size() * sizeof(uint32_t) +
                p.dm_grid.size() * sizeof(float) +
                (p.warmup_samples.size() + p.integer_thresh_lut.size()) *
                    sizeof(uint64_t) +
                p.integer_mean_lut.size() * sizeof(double) +
                (p.inv_sigma_lut.size() + p.raw_thresh_lut.size()) *
                    sizeof(float);
        }
        return {
            .plan = fdmt.plan + host_plan +
                    m_rows_d.size() * sizeof(BoxcarRow) +
                    m_widths_d.size() * sizeof(uint32_t) +
                    m_integer_means.size() * sizeof(double) +
                    m_integer_thresholds.size() * sizeof(uint64_t) +
                    (m_thresholds_d.size() + m_scales_d.size()) * sizeof(float),
            .state     = fdmt.state,
            .history   = fdmt.history + m_carry_d.size() * sizeof(float) +
                         m_integer_carry.size() * sizeof(uint64_t),
            .peaks     = m_peaks_d.size() * sizeof(BoxcarPeak),
            .workspace = fdmt.workspace + m_per_dm_d.size() * sizeof(uint64_t) +
                         sizeof(CandidateCounts) + sizeof(uint64_t),
            .candidates = (m_candidates_d.size() + m_candidates_h.size()) *
                          sizeof(RawCandidate),
            .staging =
                m_input_d.size() * sizeof(float) + m_packed_input.size()};
    }
    SizeType chunk_nsamps() const noexcept override {
        return m_fdmt->get_chunk_nsamps();
    }
    SizeType fuse_levels() const noexcept override {
        return m_fdmt->get_fuse_levels();
    }

private:
    template <typename T>
    void launch_score(const FDMTSearchInput& input,
                      SizeType p,
                      dim3 grid,
                      dim3 block,
                      T* carry) {
        kernel_fdmt_boxcar<T>
            <<<grid, block, sizeof(BoxcarShared<T>),
               static_cast<cudaStream_t>(input.stream.native)>>>(
                input, raw(m_rows_d), m_offsets[p], m_planes[p].dm_grid.size(),
                m_rows.size(), raw(m_widths_d), m_filters, m_width,
                raw(m_thresholds_d), raw(m_scales_d), carry, raw(m_peaks_d),
                m_peak_count, raw(m_candidates_d), raw(m_counts_d),
                raw(m_slots_d), raw(m_per_dm_d), m_config.max_candidates,
                m_config.max_cands_per_dm, m_config.peak_mode,
                m_config.skip_cold_prefix,
                m_planes[p].level == m_plan.get_niters(),
                raw(m_integer_thresholds), raw(m_integer_means),
                m_integer_unit);
    }
    void validate_packed(SizeType size) const {
        if (m_config.input_nbits == 32 ||
            size != m_beams * m_plan.get_nchans() *
                        bit_pack_utils::packed_row_bytes(
                            m_plan.get_nsamps(), m_config.input_nbits)) {
            throw std::invalid_argument(
                "FDMTBoxcar: packed input size or bit depth mismatch");
        }
        if (m_position > UINT64_MAX - m_plan.get_nsamps()) {
            throw std::overflow_error(
                "FDMTBoxcar: stream sample index overflow");
        }
    }
    void validate_input(SizeType size) const {
        if (size != m_beams * m_plan.get_nchans() * m_plan.get_nsamps()) {
            throw std::invalid_argument("FDMTBoxcar: wrong waterfall size");
        }
        if (m_position > UINT64_MAX - m_plan.get_nsamps()) {
            throw std::overflow_error(
                "FDMTBoxcar: stream sample index overflow");
        }
    }
    void begin_result(SizeType samples, Stream queue) {
        gpu_utils::set_device(m_device);
        const auto stream = static_cast<cudaStream_t>(queue.native);
        gpu_utils::check_gpu_call(cudaMemsetAsync(
            raw(m_counts_d), 0, sizeof(CandidateCounts), stream));
        gpu_utils::check_gpu_call(
            cudaMemsetAsync(raw(m_slots_d), 0, sizeof(uint64_t), stream));
        gpu_utils::check_gpu_call(cudaMemsetAsync(
            raw(m_per_dm_d), 0, m_per_dm_d.size() * sizeof(uint64_t), stream));
        m_result_begin   = m_position;
        m_result_samples = samples;
    }
    CandidateDeviceResultView device_view() {
        const Device device{kGPUBackend, m_device};
        return {DeviceSpan<const RawCandidate>(raw(m_candidates_d),
                                               m_candidates_d.size(), device),
                DeviceSpan<const CandidateCounts>(raw(m_counts_d), 1, device),
                m_result_begin, m_result_samples};
    }

    const plans::FDMTPlan& m_plan;
    BoxcarConfig m_config;
    const std::vector<plans::BoxcarPlanePlan>& m_planes;
    SizeType m_beams;
    int m_device;
    std::vector<BoxcarRow> m_rows;
    SizeType m_width;
    SizeType m_filters;
    SizeType m_peak_count;
    std::unique_ptr<FDMTEngine> m_fdmt;
    std::vector<SizeType> m_offsets;
    thrust::device_vector<BoxcarRow> m_rows_d;
    thrust::device_vector<uint32_t> m_widths_d;
    thrust::device_vector<float> m_thresholds_d, m_scales_d, m_carry_d,
        m_input_d;
    thrust::device_vector<uint64_t> m_integer_carry, m_integer_thresholds;
    thrust::device_vector<double> m_integer_means;
    thrust::device_vector<uint8_t> m_packed_input;
    double m_integer_unit = 1;
    thrust::device_vector<BoxcarPeak> m_peaks_d;
    thrust::device_vector<uint64_t> m_per_dm_d, m_slots_d;
    thrust::device_vector<CandidateCounts> m_counts_d;
    thrust::device_vector<RawCandidate> m_candidates_d;
    std::vector<RawCandidate> m_candidates_h;
    CandidateCounts m_counts_h{};
    gpu_utils::DeviceWorkFence m_work;
    uint64_t m_position       = 0;
    uint64_t m_result_begin   = 0;
    SizeType m_result_samples = 0;
};

} // namespace
std::unique_ptr<FDMTBoxcarEngine>
make_fdmt_boxcar_gpu(const plans::FDMTPlan& plan,
                     const BoxcarConfig& config,
                     const std::vector<plans::BoxcarPlanePlan>& planes,
                     const FDMTEngineConfig& cfg) {
    return std::make_unique<FDMTBoxcarCudaEngine>(plan, config, planes, cfg);
}
} // namespace dmt::algorithms::detail
