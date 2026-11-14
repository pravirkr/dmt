#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <thrust/device_vector.h>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/boxcar.hpp"
#include "dmt/boxcar_runs.hpp"
#include "dmt/fdmt_int_tree.hpp"
#include "dmt/gpu_compat.cuh"
#include "dmt/gpu_utils.cuh"
#include "dmt/host_staging.cuh"

#include "fdmt_boxcar_cuda_engine.cuh"

// GPU FDMTBoxcar engine: lifecycle, memory accounting, per-block noise ring,
// kKeepBrightest radix selection, streaming entrypoints, flush, candidate
// frontier, and consume() dispatch across all 4 BoxcarScreenModes.

namespace dmt::algorithms::detail::fdmt_boxcar_cuda {
namespace {

__global__ void kernel_flush_boxcar(const BoxcarRow* rows,
                                    SizeType nrows,
                                    SizeType peak_count,
                                    BoxcarPeak* peaks,
                                    uint8_t* open,
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
                                    bool integer,
                                    BoxcarKeepEntry* keep_all,
                                    SizeType keep_q) {
    const SizeType index =
        static_cast<SizeType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= tasks) {
        return;
    }
    if (keep_q != 0) {
        // kKeepBrightest: this thread owns the row's store.
        const SizeType row_index = index % nrows;
        for (SizeType f = 0; f < peak_count; ++f) {
            auto& peak = peaks[index * peak_count + f];
            if (peak.open) {
                increment(&counts->total_detected);
                const float snr =
                    integer ? static_cast<float>(
                                  (static_cast<double>(peak.raw_sum) -
                                   means[row_index * nfilters + peak.filter]) *
                                  (scales[row_index * nfilters + peak.filter] *
                                   unit))
                            : peak.snr;
                keep_insert(keep_all + index * keep_q, per_dm + index, keep_q,
                            {peak.peak, peak.begin, peak.end, snr,
                             pack_keep_filter(peak.filter)},
                            counts);
                peak.open = false;
            }
        }
        open[index] = 0;
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
    open[index] = 0;
}

// kKeepBrightest, end of a call: the stores' entries become the call's
// candidates, or, when there are more than `capacity`, exactly the
// `capacity` brightest. Keys order the entries as brighter() does:
//   0..3  : S/N (descending)
//   4..11 : peak_sample (ascending)
//   12..15: row rank (beam and row in brighter() order)
//   16..17: filter_idx (ascending)
// On kPrefix/kKadane/kNone, 18 bytes uniquely identify every candidate of a
// call. On kKadaneSurrogate, multi-strip excursions in the same row may tie
// on (snr, peak, filter_idx) and continue with:
//   18..25: begin_sample (ascending)
//   26..33: end_sample (ascending)
//   34..35: surrogate width (ascending)
//   36..37: slot index within the row's store (unique tie-breaker).
DMT_D inline uint8_t
keep_key_byte(const BoxcarKeepEntry& e, uint32_t rank, uint16_t slot, int d) {
    if (d < 4) {
        uint32_t u = __float_as_uint(e.snr);
        u          = (u & 0x80000000U) != 0 ? ~u : (u | 0x80000000U);
        u          = ~u; // larger S/N first
        return static_cast<uint8_t>(u >> (8 * (3 - d)));
    }
    if (d < 12) {
        return static_cast<uint8_t>(e.peak >> (8 * (11 - d)));
    }
    if (d < 16) {
        return static_cast<uint8_t>(rank >> (8 * (15 - d)));
    }
    if (d < 18) {
        const uint32_t f = keep_filter_idx(e);
        return static_cast<uint8_t>(f >> (8 * (17 - d)));
    }
    if (d < 26) {
        return static_cast<uint8_t>(e.begin >> (8 * (25 - d)));
    }
    if (d < 34) {
        return static_cast<uint8_t>(e.end >> (8 * (33 - d)));
    }
    if (d < 36) {
        const uint32_t sw = keep_surrogate_width(e);
        return static_cast<uint8_t>(sw >> (8 * (35 - d)));
    }
    return static_cast<uint8_t>(static_cast<uint32_t>(slot) >> (8 * (37 - d)));
}

// Whether an entry is selected: its key's first `depth` bytes are at most
// the prefix.
DMT_D inline bool keep_selected(const BoxcarKeepEntry& e,
                                uint32_t rank,
                                uint16_t slot,
                                const KeepSelect& sel) {
    for (uint32_t d = 0; d < sel.depth; ++d) {
        const uint8_t b = keep_key_byte(e, rank, slot, static_cast<int>(d));
        if (b != sel.prefix[d]) {
            return b < sel.prefix[d];
        }
    }
    return true;
}

// After kernel_keep_init: the entries in the stores.
__global__ void
kernel_keep_count(const uint64_t* per_dm, SizeType tasks, KeepSelect* sel) {
    uint64_t local = 0;
    for (SizeType i =
             static_cast<SizeType>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < tasks; i += static_cast<SizeType>(gridDim.x) * blockDim.x) {
        local += per_dm[i];
    }
    if (local != 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&sel->n),
                  static_cast<unsigned long long>(local));
    }
}

__global__ void kernel_keep_init(KeepSelect* sel, SizeType capacity) {
    *sel   = KeepSelect{};
    sel->k = capacity;
}

__global__ void kernel_keep_resolve_all(KeepSelect* sel) {
    if (sel->n <= sel->k) {
        sel->resolved = 1; // depth 0: every entry
    }
}

// Histogram of byte `d` of the keys that match the prefix so far.
__global__ void kernel_keep_hist(const BoxcarKeepEntry* keep_all,
                                 const uint64_t* per_dm,
                                 const uint32_t* rank,
                                 SizeType nrows,
                                 SizeType tasks,
                                 SizeType q,
                                 const KeepSelect* sel,
                                 uint64_t* hist,
                                 int d) {
    __shared__ uint32_t local[256];
    if (sel->resolved != 0) {
        return;
    }
    for (int b = static_cast<int>(threadIdx.x); b < 256;
         b += static_cast<int>(blockDim.x)) {
        local[b] = 0;
    }
    __syncthreads();
    const SizeType total = tasks * q;
    for (SizeType s =
             static_cast<SizeType>(blockIdx.x) * blockDim.x + threadIdx.x;
         s < total; s += static_cast<SizeType>(gridDim.x) * blockDim.x) {
        const SizeType index = s / q;
        const SizeType rem   = s % q;
        if (rem >= per_dm[index]) {
            continue;
        }
        const BoxcarKeepEntry& e = keep_all[s];
        const auto r           = static_cast<uint32_t>((index / nrows) * nrows +
                                                       rank[index % nrows]);
        const auto slot_in_row = static_cast<uint16_t>(rem);
        bool match             = true;
        for (int j = 0; j < d && match; ++j) {
            match = keep_key_byte(e, r, slot_in_row, j) == sel->prefix[j];
        }
        if (match) {
            atomicAdd(&local[keep_key_byte(e, r, slot_in_row, d)], 1U);
        }
    }
    __syncthreads();
    for (int b = static_cast<int>(threadIdx.x); b < 256;
         b += static_cast<int>(blockDim.x)) {
        if (local[b] != 0) {
            atomicAdd(reinterpret_cast<unsigned long long*>(&hist[b]),
                      static_cast<unsigned long long>(local[b]));
        }
    }
}

// Byte `d` of the k-th key; clears the histogram for the next byte.
__global__ void
kernel_keep_pick(KeepSelect* sel, uint64_t* hist, int d, int key_bytes) {
    if (sel->resolved == 0) {
        uint64_t below = 0;
        for (int b = 0; b < 256; ++b) {
            if (below + hist[b] >= sel->k) {
                sel->prefix[d] = static_cast<uint8_t>(b);
                sel->k -= below;
                sel->depth = static_cast<uint32_t>(d) + 1;
                // The whole bin is selected: no later byte matters.
                if (hist[b] == sel->k || d + 1 == key_bytes) {
                    sel->resolved = 1;
                }
                break;
            }
            below += hist[b];
        }
    }
    for (int b = 0; b < 256; ++b) {
        hist[b] = 0;
    }
}

// The selected entries as candidates; counts follow.
__global__ void kernel_keep_emit(const BoxcarKeepEntry* keep_all,
                                 const uint64_t* per_dm,
                                 const uint32_t* rank,
                                 const BoxcarRow* rows,
                                 const uint32_t* widths,
                                 SizeType nrows,
                                 SizeType tasks,
                                 SizeType q,
                                 const KeepSelect* sel,
                                 RawCandidate* candidates,
                                 CandidateCounts* counts,
                                 uint64_t* slots) {
    const SizeType total = tasks * q;
    for (SizeType s =
             static_cast<SizeType>(blockIdx.x) * blockDim.x + threadIdx.x;
         s < total; s += static_cast<SizeType>(gridDim.x) * blockDim.x) {
        const SizeType index = s / q;
        const SizeType rem   = s % q;
        if (rem >= per_dm[index]) {
            continue;
        }
        const BoxcarKeepEntry& e = keep_all[s];
        const SizeType row_index = index % nrows;
        const auto r =
            static_cast<uint32_t>((index / nrows) * nrows + rank[row_index]);
        const auto slot_in_row = static_cast<uint16_t>(rem);
        if (!keep_selected(e, r, slot_in_row, *sel)) {
            continue;
        }
        const uint32_t f             = keep_filter_idx(e);
        const uint32_t sw            = keep_surrogate_width(e);
        RawCandidate c               = rows[row_index].candidate;
        c.beam_idx                   = static_cast<uint32_t>(index / nrows);
        c.peak_sample                = e.peak;
        c.begin_sample               = e.begin;
        c.end_sample                 = e.end;
        c.snr                        = e.snr;
        c.filter_idx                 = f;
        c.width                      = sw != 0U ? sw : widths[f];
        candidates[increment(slots)] = c;
        increment(&counts->emitted_count);
    }
}

__global__ void kernel_keep_finish(const KeepSelect* sel,
                                   CandidateCounts* counts) {
    counts->dropped_global = sel->n - counts->emitted_count;
}

// Candidate frontier: the smallest begin sample of the open excursions,
// min-reduced into *oldest (preset to UINT64_MAX). Only rows flagged open
// are read.
__global__ void kernel_open_begin(const BoxcarPeak* peaks,
                                  const uint8_t* open,
                                  SizeType tasks,
                                  SizeType peak_count,
                                  uint64_t* oldest) {
    uint64_t local = UINT64_MAX;
    for (SizeType i =
             static_cast<SizeType>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < tasks; i += static_cast<SizeType>(gridDim.x) * blockDim.x) {
        if (open[i] == 0) {
            continue;
        }
        for (SizeType f = 0; f < peak_count; ++f) {
            const BoxcarPeak& p = peaks[i * peak_count + f];
            if (p.open && p.begin < local) {
                local = p.begin;
            }
        }
    }
    if (local != UINT64_MAX) {
        atomicMin(reinterpret_cast<unsigned long long*>(oldest),
                  static_cast<unsigned long long>(local));
    }
}

} // namespace

FDMTBoxcarCudaEngine::FDMTBoxcarCudaEngine(
    const plans::FDMTPlan& plan,
    const BoxcarConfig& config,
    const std::vector<plans::BoxcarPlanePlan>& planes,
    const FDMTEngineConfig& cfg,
    std::unique_ptr<BoxcarDriver> driver,
    std::vector<std::pair<SizeType, SizeType>> kill_runs)
    : m_plan(plan),
      m_config(config),
      m_planes(planes),
      m_beams(cfg.nbeams),
      m_device(cfg.exec.device),
      m_rows(make_boxcar_rows(plan, planes)),
      m_width(planes.front().widths.back()),
      m_filters(planes.front().widths.size()),
      m_peak_count(
          config.screen_mode == BoxcarScreenMode::kKadaneSurrogate
              ? SizeType{1}
              : (config.peak_mode == PeakFindingMode::kContiguousPeakPerWidth
                     ? m_filters
                     : SizeType{1})),
      m_carry32(
          config.input_nbits != 32 &&
          int_search_fits_u32(plan, cfg.use_box_smearing, config.input_nbits)) {
    m_shared_limit = cfg.boxcar_shared_limit;
    gpu_utils::set_device(m_device);
    if (m_rows.size() > SIZE_MAX / m_beams / m_width ||
        m_rows.size() > SIZE_MAX / m_beams / m_peak_count ||
        m_rows.size() * m_beams > static_cast<SizeType>(INT32_MAX)) {
        throw std::invalid_argument(
            "FDMTBoxcar: GPU search extent exceeds limits");
    }
    m_driver    = std::move(driver);
    m_kill_runs = std::move(kill_runs);
    // FDMT reads killed rows as zero itself (one byte per row); other
    // drivers read a staging copy with the rows zeroed.
    m_keep_in_transform = m_driver->set_search_keep(nullptr);
    if (!m_kill_runs.empty() && !m_keep_in_transform) {
        // The staging block the killed rows are zeroed in.
        if (config.input_nbits == 32) {
            m_input_d.resize(m_beams * plan.get_nchans() * plan.get_nsamps());
        } else {
            m_packed_input.resize(m_beams * plan.get_nchans() *
                                  bit_pack_utils::packed_row_bytes(
                                      plan.get_nsamps(), config.input_nbits));
        }
    }
    std::vector<float> thresholds;
    std::vector<float> scales;
    SizeType row_offset = 0;
    for (const auto& p : planes) {
        thresholds.insert(thresholds.end(), p.raw_thresh_lut.begin(),
                          p.raw_thresh_lut.end());
        scales.insert(scales.end(), p.inv_sigma_lut.begin(),
                      p.inv_sigma_lut.end());
        const SizeType count = p.dm_grid.size();
        if (count > 0) {
            if (m_level_groups.empty() ||
                m_level_groups.back().level != p.level) {
                m_level_groups.push_back({.level     = p.level,
                                          .row_begin = row_offset,
                                          .row_count = count});
            } else {
                m_level_groups.back().row_count += count;
            }
        }
        row_offset += count;
    }
    m_rows_d             = m_rows;
    m_widths_d           = planes.front().widths;
    m_thresholds_d       = thresholds;
    m_scales_d           = scales;
    const SizeType tasks = m_rows.size() * m_beams;
    m_integer_unit =
        cfg.use_box_smearing
            ? std::ldexp(1.0, -static_cast<int>(plan.get_box_fraction_bits()))
            : 1.0;

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
    m_open_d.resize(tasks, 0);
    m_per_dm_d.resize(tasks, 0);

    // Initialize kKeepBrightest state BEFORE kernel shared-memory configuration
    // so configure_variants / configure_kadane_variants opt in the kKeep=true
    // kernel instances as well.
    if (config.overflow_policy == OverflowPolicy::kKeepBrightest) {
        m_bright_q = config.max_cands_per_dm;
        m_bright_d.resize(tasks * m_bright_q);
        // Each row's place in brighter() order among the rows (equal
        // S/N, sample and beam): the key's row rank.
        std::vector<uint32_t> order(m_rows.size());
        std::iota(order.begin(), order.end(), 0U);
        std::ranges::sort(order, [&](uint32_t a, uint32_t b) {
            return brighter(m_rows[a].candidate, m_rows[b].candidate);
        });
        std::vector<uint32_t> rank(m_rows.size());
        for (SizeType i = 0; i < order.size(); ++i) {
            rank[order[i]] = static_cast<uint32_t>(i);
        }
        m_bright_rank_d = rank;
        m_bright_sel_d.resize(1);
        m_bright_hist_d.resize(256, 0);
        // A selection is needed only when the stores can exceed it.
        m_bright_select = tasks * m_bright_q > config.max_candidates;
    }

    if (config.screen_mode != BoxcarScreenMode::kKadaneSurrogate) {
        if (config.input_nbits == 32) {
            m_carry_d.resize(tasks * m_width, 0);
        } else if (m_carry32) {
            // uint32_t whenever every tree value fits (int_search_fits_u32),
            // as the FDMT engines' own state: half the carry's bytes.
            m_integer_carry32.resize(tasks * m_width, 0);
        } else {
            m_integer_carry.resize(tasks * m_width, 0);
        }
    }

    if (config.screen_mode == BoxcarScreenMode::kKadaneSurrogate) {
        init_surrogate(cfg.use_box_smearing, means);
    } else if (config.screen_mode == BoxcarScreenMode::kKadane) {
        init_kadane(cfg.use_box_smearing, means, cutoffs);
    } else {
        if (config.screen_mode == BoxcarScreenMode::kPrefix) {
            init_prefix_screen();
        }
        if (config.input_nbits == 32) {
            configure_shared<float>();
        } else {
            configure_shared<uint64_t>();
        }
    }

    m_candidates_d.resize(config.max_candidates);
    m_counts_d.resize(1);
    m_slots_d.resize(1, 0);
    m_candidates_h.resize(config.max_candidates);
    m_oldest_d.resize(1, UINT64_MAX);
    m_oldest_h.reserve(1);
    *m_oldest_h.data() = UINT64_MAX;
}

FDMTBoxcarCudaEngine::~FDMTBoxcarCudaEngine() {
    // Destruction must not release search buffers while queued work
    // still references them, including on a nondefault stream.
    cudaSetDevice(m_device);
    m_work.wait();
    for (auto& slot : m_slots) {
        if (slot->copied != nullptr) {
            cudaEventDestroy(slot->copied);
        }
        if (slot->released != nullptr) {
            cudaEventDestroy(slot->released);
        }
    }
}

CandidateResultView
FDMTBoxcarCudaEngine::execute(std::span<const float> waterfall) {
    if (m_config.input_nbits != 32) {
        throw std::invalid_argument("FDMTBoxcar: expected packed uint8 input");
    }
    validate_input(waterfall.size());
    gpu_utils::set_device(m_device);
    m_work.wait();
    if (m_input_d.size() != waterfall.size()) {
        m_input_d.resize(waterfall.size());
    }
    gpu_utils::check_gpu_call(cudaMemcpyAsync(raw(m_input_d), waterfall.data(),
                                              waterfall.size_bytes(),
                                              cudaMemcpyHostToDevice, nullptr));
    (void)execute(DeviceSpan<const float>(raw(m_input_d), waterfall.size(),
                                          {kGPUBackend, m_device}),
                  {});
    return copy_result({});
}

CandidateDeviceResultView
FDMTBoxcarCudaEngine::execute(DeviceSpan<const float> waterfall, Stream queue) {
    if (m_config.input_nbits != 32) {
        throw std::invalid_argument("FDMTBoxcar: expected packed uint8 input");
    }
    validate_input(waterfall.size());
    const auto stream = static_cast<cudaStream_t>(queue.native);
    // Before the table uploads, which go to the current device.
    gpu_utils::set_device(m_device);
    commit_block_noise(stream);
    begin_result(m_plan.get_nsamps(), queue);
    m_driver->search(masked(waterfall, m_input_d, queue), *this, queue);
    m_commit_open = false;
    m_position += m_plan.get_nsamps();
    finish_keep(stream);
    release_slots(stream);
    update_open_begin(stream);
    m_work.mark(stream);
    return device_view();
}

CandidateResultView
FDMTBoxcarCudaEngine::execute(std::span<const uint8_t> waterfall) {
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

CandidateDeviceResultView
FDMTBoxcarCudaEngine::execute(DeviceSpan<const uint8_t> waterfall,
                              Stream queue) {
    validate_packed(waterfall.size());
    begin_result(m_plan.get_nsamps(), queue);
    m_driver->search(masked(waterfall, m_packed_input, queue),
                     m_config.input_nbits, *this, queue);
    m_position += m_plan.get_nsamps();
    finish_keep(static_cast<cudaStream_t>(queue.native));
    update_open_begin(static_cast<cudaStream_t>(queue.native));
    m_work.mark(static_cast<cudaStream_t>(queue.native));
    return device_view();
}

CandidateResultView FDMTBoxcarCudaEngine::flush() {
    gpu_utils::set_device(m_device);
    m_work.wait();
    (void)flush_device({});
    return copy_result({});
}

CandidateDeviceResultView FDMTBoxcarCudaEngine::flush_device(Stream queue) {
    begin_result(0, queue);
    const auto tasks  = m_rows.size() * m_beams;
    const auto stream = static_cast<cudaStream_t>(queue.native);
    if (m_config.screen_mode == BoxcarScreenMode::kKadaneSurrogate) {
        launch_flush_surrogate(stream);
    } else {
        kernel_flush_boxcar<<<(tasks + 127) / 128, 128, 0, stream>>>(
            raw(m_rows_d), m_rows.size(), m_peak_count, raw(m_peaks_d),
            raw(m_open_d), raw(m_widths_d), raw(m_candidates_d),
            raw(m_counts_d), raw(m_slots_d), raw(m_per_dm_d),
            m_config.max_candidates, m_config.max_cands_per_dm, tasks,
            raw(m_integer_means), raw(m_scales_d), m_filters, m_integer_unit,
            m_config.input_nbits != 32 &&
                m_config.peak_mode !=
                    PeakFindingMode::kContiguousPeakAcrossWidths,
            raw(m_bright_d), m_bright_q);
        gpu_utils::check_last_gpu_error("FDMTBoxcar flush failed");
    }
    finish_keep(stream);
    update_open_begin(stream);
    m_work.mark(stream);
    return device_view();
}

CandidateResultView FDMTBoxcarCudaEngine::copy_result(Stream queue) {
    gpu_utils::set_device(m_device);
    const auto stream = static_cast<cudaStream_t>(queue.native);
    m_work.order(stream);
    gpu_utils::check_gpu_call(cudaMemcpyAsync(&m_counts_h, raw(m_counts_d),
                                              sizeof(CandidateCounts),
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

void FDMTBoxcarCudaEngine::set_search_stop(uint64_t stop) noexcept {
    m_search_stop = stop;
}

uint64_t FDMTBoxcarCudaEngine::stream_position() const noexcept {
    return m_position;
}

uint64_t FDMTBoxcarCudaEngine::search_latency() const noexcept {
    return m_driver->search_latency();
}

uint64_t FDMTBoxcarCudaEngine::open_begin_min() const noexcept {
    return *m_oldest_h.data();
}

SizeType FDMTBoxcarCudaEngine::block_noise_bytes() const noexcept {
    // Staging (unless allocated), the row supports, and the table slots
    // on the device and in pinned host memory.
    const SizeType full = m_beams * m_plan.get_nchans() * m_plan.get_nsamps();
    const SizeType staging =
        m_keep_in_transform
            ? (m_keep_d.empty() ? m_beams * m_plan.get_nchans() : 0)
        : m_input_d.size() < full ? (full - m_input_d.size()) * sizeof(float)
                                  : 0;
    // Device bytes only (the host halves of the slots are not budgeted).
    return staging + (m_rows.size() * sizeof(uint64_t)) +
           (slot_count() * 2 * table_floats() * sizeof(float));
}

void FDMTBoxcarCudaEngine::enable_block_noise(
    std::shared_ptr<const std::vector<uint64_t>> support) {
    if (m_config.input_nbits != 32) {
        throw std::logic_error(
            "FDMTBoxcar (gpu): per-call kill masks need float input");
    }
    if (ring_size() > 2) {
        // A call's endpoints reach at most two blocks only when the
        // search latency is at most one block.
        throw std::invalid_argument(fmt::format(
            "FDMTBoxcar (gpu): per-call kill masks need a search latency "
            "of at most one block ({} > {} samples)",
            search_latency(), m_plan.get_nsamps()));
    }
    gpu_utils::set_device(m_device);
    const SizeType full = m_beams * m_plan.get_nchans() * m_plan.get_nsamps();
    if (!m_keep_in_transform && m_input_d.size() < full) {
        m_work.wait(); // the staging buffer may still be read
        m_input_d.resize(full);
    }
    m_support_d = *support;
    m_slots.clear();
    for (SizeType i = 0; i < slot_count(); ++i) {
        auto slot = std::make_unique<TableSlot>();
        slot->device.resize(2 * table_floats());
        slot->host.reserve(2 * table_floats());
        gpu_utils::check_gpu_call(
            cudaEventCreateWithFlags(&slot->copied, cudaEventDisableTiming),
            "FDMTBoxcar: event creation failed");
        gpu_utils::check_gpu_call(
            cudaEventCreateWithFlags(&slot->released, cudaEventDisableTiming),
            "FDMTBoxcar: event creation failed");
        m_slots.push_back(std::move(slot));
    }
    m_ring.reserve(3);
    m_noise_on = true;
    reset_block_noise();
}

void FDMTBoxcarCudaEngine::push_block_noise(BlockNoise entry) {
    m_pending = std::move(entry);
}

void FDMTBoxcarCudaEngine::drop_block_noise() noexcept {
    m_pending.reset();
    undo_commit();
}

void FDMTBoxcarCudaEngine::reset_history(Stream queue) {
    gpu_utils::set_device(m_device);
    const auto stream = static_cast<cudaStream_t>(queue.native);
    if (m_noise_on) {
        reset_block_noise();
    }
    m_search_stop = UINT64_MAX;
    m_driver->reset(queue);
    if (!m_carry_d.empty()) {
        gpu_utils::check_gpu_call(cudaMemsetAsync(
            raw(m_carry_d), 0, m_carry_d.size() * sizeof(float), stream));
    }
    gpu_utils::check_gpu_call(cudaMemsetAsync(
        raw(m_peaks_d), 0, m_peaks_d.size() * sizeof(BoxcarPeak), stream));
    gpu_utils::check_gpu_call(
        cudaMemsetAsync(raw(m_open_d), 0, m_open_d.size(), stream));
    if (!m_integer_carry.empty()) {
        gpu_utils::check_gpu_call(
            cudaMemsetAsync(raw(m_integer_carry), 0,
                            m_integer_carry.size() * sizeof(uint64_t), stream));
    }
    if (!m_integer_carry32.empty()) {
        gpu_utils::check_gpu_call(cudaMemsetAsync(
            raw(m_integer_carry32), 0,
            m_integer_carry32.size() * sizeof(uint32_t), stream));
    }
    reset_kadane(stream);
    reset_surrogate(stream);
    m_position = 0;
    begin_result(0, queue);
    update_open_begin(stream);
    m_work.mark(stream);
}

void FDMTBoxcarCudaEngine::consume(SizeType level,
                                   const FDMTSearchInput& input) {
    for (const auto& g : m_level_groups) {
        if (g.level != level) {
            continue;
        }
        const SizeType beams =
            std::min(input.beam_count, m_beams - input.beam_begin);
        if (m_config.screen_mode == BoxcarScreenMode::kKadaneSurrogate) {
            launch_score_surrogate(input, g, beams);
            gpu_utils::check_last_gpu_error("FDMTBoxcar scoring failed");
            continue;
        }
        const dim3 grid(g.row_count * beams);
        const dim3 block(m_block_threads);
        gpu_utils::check_kernel_launch_params(grid, block);
        if (m_config.screen_mode == BoxcarScreenMode::kKadane &&
            !m_kadane_fallback) {
            if (m_config.input_nbits == 32) {
                launch_score_kadane<float>(input, g, grid, block,
                                           raw(m_carry_d));
            } else if (!m_integer_carry32.empty()) {
                launch_score_kadane<uint64_t, uint32_t>(input, g, grid, block,
                                                        raw(m_integer_carry32));
            } else {
                launch_score_kadane<uint64_t>(input, g, grid, block,
                                              raw(m_integer_carry));
            }
        } else {
            if (m_config.input_nbits == 32) {
                launch_score<float>(input, g, grid, block, raw(m_carry_d));
            } else if (!m_integer_carry32.empty()) {
                launch_score<uint64_t, uint32_t>(input, g, grid, block,
                                                 raw(m_integer_carry32));
            } else {
                launch_score<uint64_t>(input, g, grid, block,
                                       raw(m_integer_carry));
            }
        }
        gpu_utils::check_last_gpu_error("FDMTBoxcar scoring failed");
    }
}

FDMTBoxcarMemoryUsage FDMTBoxcarCudaEngine::memory_usage() const noexcept {
    const auto fdmt    = m_driver->memory();
    SizeType host_plan = (m_rows.size() * sizeof(BoxcarRow)) +
                         (m_level_groups.size() * sizeof(LevelGroup)) +
                         m_kadane_plan.memory_bytes() +
                         m_surrogate_plan.memory_bytes();
    for (const auto& p : m_planes) {
        host_plan += sizeof(p) + (p.widths.size() * sizeof(uint32_t)) +
                     (p.dm_grid.size() * sizeof(float)) +
                     ((p.warmup_samples.size() + p.integer_thresh_lut.size()) *
                      sizeof(uint64_t)) +
                     (p.integer_mean_lut.size() * sizeof(double)) +
                     ((p.inv_sigma_lut.size() + p.raw_thresh_lut.size()) *
                      sizeof(float));
    }
    const SizeType kadane_plan_d =
        ((m_kadane_bias_d.size() + m_kadane_cut_d.size() +
          m_surr_bias_d.size() + m_surr_trig_d.size() +
          m_surr_box_inv_d.size() + m_surr_lerp_d.size()) *
         sizeof(float)) +
        ((m_kadane_ibias_d.size() + m_kadane_icut_d.size()) * sizeof(int32_t)) +
        ((m_kadane_disabled_d.size() + m_surr_disabled_d.size()) *
         sizeof(uint8_t)) +
        ((m_row_int_m1_d.size() + m_surr_mean1_d.size()) * sizeof(double)) +
        ((m_surr_nearest_d.size() + m_surr_blo_d.size() + m_surr_bhi_d.size()) *
         sizeof(uint32_t));
    const SizeType kadane_hist_d =
        (m_kadane_fc_d.size() * sizeof(float)) +
        (m_kadane_ic_d.size() * sizeof(int32_t)) +
        (m_lag_f32_d.size() * sizeof(kadane::LagWindow<float>)) +
        (m_lag_i32_d.size() * sizeof(kadane::LagWindow<int32_t>)) +
        (m_surrogate_rows_d.size() * sizeof(kadane::KadaneSurrogateRowState)) +
        (m_surrogate_pos_d.size() * sizeof(uint64_t));
    return {.plan =
                fdmt.plan + host_plan + (m_rows_d.size() * sizeof(BoxcarRow)) +
                (m_widths_d.size() * sizeof(uint32_t)) +
                (m_screen_groups_d.size() * sizeof(BoxcarScreenGroup)) +
                ((m_screen_thresholds_d.size() + m_screen_m1_d.size()) *
                 sizeof(float)) +
                (m_integer_means.size() * sizeof(double)) +
                (m_integer_thresholds.size() * sizeof(uint64_t)) +
                ((m_thresholds_d.size() + m_scales_d.size()) * sizeof(float)) +
                (m_support_d.size() * sizeof(uint64_t)) +
                (m_slots.size() * 4 * table_floats() * sizeof(float)) +
                kadane_plan_d,
            .state   = fdmt.state,
            .history = fdmt.history + (m_carry_d.size() * sizeof(float)) +
                       (m_integer_carry.size() * sizeof(uint64_t)) +
                       (m_integer_carry32.size() * sizeof(uint32_t)) +
                       kadane_hist_d,
            .peaks = (m_peaks_d.size() * sizeof(BoxcarPeak)) + m_open_d.size(),
            .workspace =
                fdmt.workspace + (m_per_dm_d.size() * sizeof(uint64_t)) +
                (m_ring_d.size() * sizeof(uint64_t)) + sizeof(CandidateCounts) +
                sizeof(uint64_t) + (2 * sizeof(uint64_t)), // candidate frontier
            .candidates = ((m_candidates_d.size() + m_candidates_h.size()) *
                           sizeof(RawCandidate)) +
                          (m_bright_d.size() * sizeof(BoxcarKeepEntry)) +
                          (m_bright_rank_d.size() * sizeof(uint32_t)) +
                          (sizeof(KeepSelect) * m_bright_sel_d.size()) +
                          (m_bright_hist_d.size() * sizeof(uint64_t)),
            .staging    = (m_input_d.size() * sizeof(float)) +
                          m_packed_input.size() + m_keep_d.size(),
            .host = host_plan +
                    (m_slots.size() * 2 * table_floats() * sizeof(float)) +
                    (m_candidates_h.size() * sizeof(RawCandidate))};
}

SizeType FDMTBoxcarCudaEngine::chunk_nsamps() const noexcept {
    return m_driver->chunk_nsamps();
}

std::string FDMTBoxcarCudaEngine::search_path() const {
    std::string path = m_driver->search_path();
    if (m_global_ring) {
        path += " (boxcar: device-memory ring)";
    } else if (m_screen_dropped || m_block_shrunk) {
        path += fmt::format(" (boxcar: {}-thread blocks{})", m_block_threads,
                            m_screen_dropped ? ", no width screen" : "");
    }
    return path;
}

SizeType FDMTBoxcarCudaEngine::fuse_levels() const noexcept {
    return m_driver->fuse_levels();
}

template <typename T>
DeviceSpan<const T> FDMTBoxcarCudaEngine::masked(
    DeviceSpan<const T> block, thrust::device_vector<T>& buffer, Stream queue) {
    const auto& runs = m_noise_on ? m_ring.back().kill_runs : m_kill_runs;
    if (m_keep_in_transform) {
        set_keep(runs, queue);
        return block;
    }
    if (runs.empty()) {
        return block;
    }
    const auto stream = static_cast<cudaStream_t>(queue.native);
    T* dst            = raw(buffer);
    if (block.data() != dst) {
        // The previous block's search may still read the buffer.
        m_work.order(stream);
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(dst, block.data(), block.size_bytes(),
                            cudaMemcpyDeviceToDevice, stream));
    }
    const SizeType nchans = m_plan.get_nchans();
    const SizeType row    = block.size_bytes() / (m_beams * nchans);
    for (const auto& [c0, c1] : runs) {
        // Channels [c0, c1) of every beam: rows nchans rows apart.
        gpu_utils::check_gpu_call(cudaMemset2DAsync(
            reinterpret_cast<std::byte*>(dst) + (c0 * row), nchans * row, 0,
            (c1 - c0) * row, m_beams, stream));
    }
    return {dst, block.size(), block.device};
}

void FDMTBoxcarCudaEngine::set_keep(
    const std::vector<std::pair<SizeType, SizeType>>& runs, Stream queue) {
    if (runs.empty()) {
        (void)m_driver->set_search_keep(nullptr);
        return;
    }
    if (m_keep_d.empty() || runs != m_keep_runs) {
        const SizeType nchans = m_plan.get_nchans();
        m_keep_h.assign(m_beams * nchans, 1);
        for (SizeType b = 0; b < m_beams; ++b) {
            for (const auto& [c0, c1] : runs) {
                std::fill_n(m_keep_h.begin() +
                                static_cast<std::ptrdiff_t>(b * nchans + c0),
                            c1 - c0, uint8_t{0});
            }
        }
        if (m_keep_d.empty()) {
            m_keep_d.resize(m_keep_h.size());
        }
        const auto stream = static_cast<cudaStream_t>(queue.native);
        m_work.order(stream);
        // Pageable source: staged before the call returns.
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(raw(m_keep_d), m_keep_h.data(), m_keep_h.size(),
                            cudaMemcpyHostToDevice, stream));
        m_keep_runs = runs;
    }
    (void)m_driver->set_search_keep(raw(m_keep_d));
}

void FDMTBoxcarCudaEngine::finish_keep(cudaStream_t stream) {
    if (m_bright_q == 0) {
        return;
    }
    const SizeType tasks   = m_rows.size() * m_beams;
    constexpr int kThreads = 256;
    const auto grid        = static_cast<unsigned>(std::min<SizeType>(
        (tasks * m_bright_q + kThreads - 1) / kThreads, 4096));
    KeepSelect* sel        = raw(m_bright_sel_d);
    kernel_keep_init<<<1, 1, 0, stream>>>(sel, m_config.max_candidates);
    kernel_keep_count<<<std::min<SizeType>((tasks + 255) / 256, 1024), 256, 0,
                        stream>>>(raw(m_per_dm_d), tasks, sel);
    kernel_keep_resolve_all<<<1, 1, 0, stream>>>(sel);
    if (m_bright_select) {
        const int key_bytes =
            m_config.screen_mode == BoxcarScreenMode::kKadaneSurrogate
                ? kKeepSurrogateKeyBytes
                : kKeepBaseKeyBytes;
        for (int d = 0; d < key_bytes; ++d) {
            kernel_keep_hist<<<grid, kThreads, 0, stream>>>(
                raw(m_bright_d), raw(m_per_dm_d), raw(m_bright_rank_d),
                m_rows.size(), tasks, m_bright_q, sel, raw(m_bright_hist_d), d);
            kernel_keep_pick<<<1, 1, 0, stream>>>(sel, raw(m_bright_hist_d), d,
                                                  key_bytes);
        }
    }
    kernel_keep_emit<<<grid, kThreads, 0, stream>>>(
        raw(m_bright_d), raw(m_per_dm_d), raw(m_bright_rank_d), raw(m_rows_d),
        raw(m_widths_d), m_rows.size(), tasks, m_bright_q, sel,
        raw(m_candidates_d), raw(m_counts_d), raw(m_slots_d));
    kernel_keep_finish<<<1, 1, 0, stream>>>(sel, raw(m_counts_d));
    gpu_utils::check_last_gpu_error("FDMTBoxcar keep-brightest failed");
}

void FDMTBoxcarCudaEngine::update_open_begin(cudaStream_t stream) {
    gpu_utils::check_gpu_call(
        cudaMemsetAsync(raw(m_oldest_d), 0xFF, sizeof(uint64_t), stream));
    if (m_config.screen_mode == BoxcarScreenMode::kKadaneSurrogate) {
        launch_open_begin_surrogate(stream);
    } else if (m_config.peak_mode != PeakFindingMode::kAllAboveThreshold) {
        const SizeType tasks = m_rows.size() * m_beams;
        const auto blocks    = static_cast<unsigned>(
            std::min<SizeType>((tasks + 255) / 256, 1024));
        kernel_open_begin<<<blocks, 256, 0, stream>>>(
            raw(m_peaks_d), raw(m_open_d), tasks, m_peak_count,
            raw(m_oldest_d));
        gpu_utils::check_last_gpu_error("FDMTBoxcar frontier failed");
    }
    gpu_utils::check_gpu_call(cudaMemcpyAsync(m_oldest_h.data(),
                                              raw(m_oldest_d), sizeof(uint64_t),
                                              cudaMemcpyDeviceToHost, stream));
}

SizeType FDMTBoxcarCudaEngine::ring_size() const noexcept {
    const uint64_t ns = m_plan.get_nsamps();
    return 1 + static_cast<SizeType>((search_latency() + ns - 1) / ns);
}

SizeType FDMTBoxcarCudaEngine::slot_count() const noexcept {
    return (2 * ring_size()) + 2;
}

SizeType FDMTBoxcarCudaEngine::table_floats() const noexcept {
    return m_rows.size() * m_filters;
}

void FDMTBoxcarCudaEngine::reset_block_noise() {
    m_pending.reset();
    m_evicted.reset();
    m_commit_open = false;
    m_ring.clear();
    BlockNoise first; // the constructor tables from stream sample 0
    first.kill_runs = m_kill_runs;
    m_ring.push_back(std::move(first));
    m_block_tables = {};
}

const float* FDMTBoxcarCudaEngine::slot_of(const NoiseTables* tables,
                                           cudaStream_t stream) {
    if (tables == nullptr) {
        return nullptr;
    }
    for (auto& slot : m_slots) {
        if (slot->id == tables->id) {
            return raw(slot->device);
        }
    }
    const auto in_ring = [&](uint64_t id) {
        for (const auto& e : m_ring) {
            if ((e.steady && e.steady->id == id) ||
                (e.transition_tables && e.transition_tables->id == id)) {
                return true;
            }
        }
        return false;
    };
    for (auto& slot : m_slots) {
        if (slot->id != 0 && in_ring(slot->id)) {
            continue;
        }
        // The previous copy out of the pinned buffer has finished
        // before it is overwritten; the device buffer is rewritten only
        // after the kernels that read it (stream order, no host wait).
        if (slot->copy_pending) {
            gpu_utils::check_gpu_call(cudaEventSynchronize(slot->copied),
                                      "FDMTBoxcar: table copy failed");
        }
        const SizeType n = table_floats();
        std::copy_n(tables->thresh.data(), n, slot->host.data());
        std::copy_n(tables->inv_sigma.data(), n, slot->host.data() + n);
        if (slot->release_pending) {
            gpu_utils::check_gpu_call(
                cudaStreamWaitEvent(stream, slot->released, 0),
                "FDMTBoxcar: stream wait failed");
        }
        gpu_utils::check_gpu_call(
            cudaMemcpyAsync(raw(slot->device), slot->host.data(),
                            2 * n * sizeof(float), cudaMemcpyHostToDevice,
                            stream),
            "FDMTBoxcar: table upload failed");
        gpu_utils::check_gpu_call(cudaEventRecord(slot->copied, stream),
                                  "FDMTBoxcar: event record failed");
        slot->copy_pending = true;
        slot->id           = tables->id;
        return raw(slot->device);
    }
    throw std::logic_error("FDMTBoxcar (gpu): no free table slot");
}

void FDMTBoxcarCudaEngine::commit_block_noise(cudaStream_t stream) {
    if (!m_noise_on) {
        return;
    }
    BlockNoise entry;
    if (m_pending) {
        entry = std::move(*m_pending);
        m_pending.reset();
    } else {
        entry.steady    = m_ring.back().steady;
        entry.kill_runs = m_kill_runs;
    }
    entry.first_sample = m_position;
    m_evicted.reset();
    if (m_ring.size() == ring_size()) {
        m_evicted = std::move(m_ring.front());
        m_ring.erase(m_ring.begin());
    }
    m_ring.push_back(std::move(entry));
    m_commit_open = true;
    BlockTables bt;
    const SizeType n   = table_floats();
    const SizeType off = m_ring.size() > 2 ? m_ring.size() - 2 : 0;
    bt.count           = static_cast<int>(m_ring.size() - off);
    bt.support         = raw(m_support_d);
    for (int e = 0; e < bt.count; ++e) {
        const auto& r    = m_ring[off + e];
        bt.first[e]      = r.first_sample;
        bt.transition[e] = r.transition;
        const float* st  = slot_of(r.steady.get(), stream);
        bt.thr[e][0]     = st;
        bt.inv[e][0]     = st != nullptr ? st + n : nullptr;
        if (r.transition) {
            const float* tr = slot_of(r.transition_tables.get(), stream);
            bt.thr[e][1]    = tr;
            bt.inv[e][1]    = tr != nullptr ? tr + n : nullptr;
        }
    }
    m_block_tables = bt;
}

void FDMTBoxcarCudaEngine::undo_commit() noexcept {
    if (!m_commit_open) {
        return;
    }
    m_ring.pop_back();
    if (m_evicted) {
        m_ring.insert(m_ring.begin(), std::move(*m_evicted));
        m_evicted.reset();
    }
    m_commit_open = false;
}

void FDMTBoxcarCudaEngine::release_slots(cudaStream_t stream) {
    if (!m_noise_on) {
        return;
    }
    for (auto& slot : m_slots) {
        if (slot->id != 0) {
            gpu_utils::check_gpu_call(cudaEventRecord(slot->released, stream),
                                      "FDMTBoxcar: event record failed");
            slot->release_pending = true;
        }
    }
}

void FDMTBoxcarCudaEngine::validate_packed(SizeType size) const {
    if (m_config.input_nbits == 32 ||
        size != m_beams * m_plan.get_nchans() *
                    bit_pack_utils::packed_row_bytes(m_plan.get_nsamps(),
                                                     m_config.input_nbits)) {
        throw std::invalid_argument(
            "FDMTBoxcar: packed input size or bit depth mismatch");
    }
    if (m_position > UINT64_MAX - m_plan.get_nsamps()) {
        throw std::overflow_error("FDMTBoxcar: stream sample index overflow");
    }
}

void FDMTBoxcarCudaEngine::validate_input(SizeType size) const {
    if (size != m_beams * m_plan.get_nchans() * m_plan.get_nsamps()) {
        throw std::invalid_argument("FDMTBoxcar: wrong waterfall size");
    }
    if (m_position > UINT64_MAX - m_plan.get_nsamps()) {
        throw std::overflow_error("FDMTBoxcar: stream sample index overflow");
    }
}

void FDMTBoxcarCudaEngine::begin_result(SizeType samples, Stream queue) {
    gpu_utils::set_device(m_device);
    const auto stream = static_cast<cudaStream_t>(queue.native);
    gpu_utils::check_gpu_call(
        cudaMemsetAsync(raw(m_counts_d), 0, sizeof(CandidateCounts), stream));
    gpu_utils::check_gpu_call(
        cudaMemsetAsync(raw(m_slots_d), 0, sizeof(uint64_t), stream));
    gpu_utils::check_gpu_call(cudaMemsetAsync(
        raw(m_per_dm_d), 0, m_per_dm_d.size() * sizeof(uint64_t), stream));
    m_result_begin   = m_position;
    m_result_samples = samples;
}

CandidateDeviceResultView FDMTBoxcarCudaEngine::device_view() {
    const Device device{kGPUBackend, m_device};
    return {DeviceSpan<const RawCandidate>(raw(m_candidates_d),
                                           m_candidates_d.size(), device),
            DeviceSpan<const CandidateCounts>(raw(m_counts_d), 1, device),
            m_result_begin, m_result_samples};
}

} // namespace dmt::algorithms::detail::fdmt_boxcar_cuda

namespace dmt::algorithms::detail {

std::unique_ptr<FDMTBoxcarEngine>
make_fdmt_boxcar_gpu(const plans::FDMTPlan& plan,
                     const BoxcarConfig& config,
                     const std::vector<plans::BoxcarPlanePlan>& planes,
                     const FDMTEngineConfig& cfg,
                     std::unique_ptr<BoxcarDriver> driver,
                     std::vector<std::pair<SizeType, SizeType>> kill_runs) {
    return std::make_unique<fdmt_boxcar_cuda::FDMTBoxcarCudaEngine>(
        plan, config, planes, cfg, std::move(driver), std::move(kill_runs));
}

} // namespace dmt::algorithms::detail
