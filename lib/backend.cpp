#include "dmt/common/backend.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/common/types.hpp"
#include "dmt/engines.hpp"

namespace dmt {

std::string_view to_string(Backend backend) noexcept {
    switch (backend) {
    case Backend::kCPU:
        return "cpu";
    case Backend::kCUDA:
        return "cuda";
    case Backend::kHIP:
        return "hip";
    }
    return "unknown";
}

Backend parse_backend(std::string_view name) {
    for (const auto b : {Backend::kCPU, Backend::kCUDA, Backend::kHIP}) {
        if (name == to_string(b)) {
            return b;
        }
    }
    throw std::invalid_argument(std::format(
        "Unknown backend '{}'. Expected 'cpu', 'cuda' or 'hip'", name));
}

std::vector<Backend> available_backends() {
    std::vector<Backend> out{Backend::kCPU};
#ifdef DMT_ENABLE_GPU
    out.push_back(algorithms::detail::kGPUBackend);
#endif
    return out;
}

bool is_available(Backend backend) {
    const auto avail = available_backends();
    return std::ranges::find(avail, backend) != avail.end();
}

} // namespace dmt

namespace dmt::algorithms::detail {

void throw_no_device_memory(std::string_view what, Backend backend) {
    throw std::invalid_argument(
        std::format("{}: device memory is not supported by the {} backend; "
                    "pass host memory (std::span) instead",
                    what, to_string(backend)));
}

void throw_unavailable(std::string_view algorithm, Backend backend) {
    std::string names;
    for (const auto b : available_backends()) {
        names += names.empty() ? "" : ", ";
        names += to_string(b);
    }
    throw std::invalid_argument(
        std::format("{}: backend '{}' is not available in this build "
                    "(available: {})",
                    algorithm, to_string(backend), names));
}

void check_device(const Device& view,
                  Backend backend,
                  int device,
                  std::string_view what) {
    if (view.id < 0) {
        return;
    }
    if (view.backend != backend || view.id != device) {
        throw std::invalid_argument(std::format(
            "{}: memory is on {}:{} but this instance runs on {}:{}", what,
            to_string(view.backend), view.id, to_string(backend), device));
    }
}

namespace {
std::atomic<bool> g_sdmt_gpu_always_shared{false};
} // namespace

void set_sdmt_gpu_always_shared(bool on) noexcept {
    g_sdmt_gpu_always_shared.store(on, std::memory_order_relaxed);
}

bool sdmt_gpu_always_shared() noexcept {
    return g_sdmt_gpu_always_shared.load(std::memory_order_relaxed);
}

namespace {
std::atomic<SizeType> g_fft_segment_cap{0};
} // namespace

void set_fft_segment_cap(SizeType nsamps) noexcept {
    g_fft_segment_cap.store(nsamps, std::memory_order_relaxed);
}

SizeType fft_segment_cap() noexcept {
    return g_fft_segment_cap.load(std::memory_order_relaxed);
}

// Device-memory defaults: host-only engines inherit these.
void FDMTEngine::execute(DeviceSpan<const float> /*waterfall*/,
                         DeviceSpan<float> /*dmt*/,
                         Stream /*stream*/) {
    throw_no_device_memory("FDMT::execute", backend());
}
void FDMTEngine::execute(DeviceSpan<const uint8_t> /*waterfall_packed*/,
                         SizeType /*nbits*/,
                         DeviceSpan<float> /*dmt*/,
                         Stream /*stream*/) {
    throw_no_device_memory("FDMT::execute", backend());
}
void FDMTEngine::reset(DeviceSpan<const float> /*waterfall*/,
                       DeviceSpan<float> /*dmt*/,
                       Stream /*stream*/) {
    throw_no_device_memory("FDMT::reset", backend());
}
void FDMTEngine::reset(DeviceSpan<const uint8_t> /*waterfall_packed*/,
                       SizeType /*nbits*/,
                       DeviceSpan<float> /*dmt*/,
                       Stream /*stream*/) {
    throw_no_device_memory("FDMT::reset", backend());
}
DeviceSpan<const float> FDMTEngine::view_level_data_device() const {
    throw_no_device_memory("FDMT::view_level_data_device", backend());
}
algorithms::FDMTSubbandDeviceView
FDMTEngine::view_subband_device(SizeType /*subband_idx*/) const {
    throw_no_device_memory("FDMT::view_subband_device", backend());
}
void FDMTEngine::save_history(DeviceSpan<float> /*out*/,
                              Stream /*stream*/) const {
    throw_no_device_memory("FDMT::save_history", backend());
}
void FDMTEngine::load_history(DeviceSpan<const float> /*in*/,
                              Stream /*stream*/) {
    throw_no_device_memory("FDMT::load_history", backend());
}

void DDMTEngine::execute(DeviceSpan<const float> /*waterfall*/,
                         DeviceSpan<float> /*dmt*/,
                         Stream /*stream*/) {
    throw_no_device_memory("DDMT::execute", backend());
}
void DDMTEngine::execute(DeviceSpan<const uint8_t> /*waterfall_packed*/,
                         SizeType /*nsamps*/,
                         DeviceSpan<int32_t> /*dmt*/,
                         Stream /*stream*/) {
    throw_no_device_memory("DDMT::execute", backend());
}
void DDMTEngine::execute_time_major(
    DeviceSpan<const uint8_t> /*filterbank_packed*/,
    SizeType /*nsamps*/,
    DeviceSpan<int32_t> /*dmt*/,
    Stream /*stream*/) {
    throw_no_device_memory("DDMT::execute_time_major", backend());
}
void DDMTEngine::save_history(DeviceSpan<float> /*out*/,
                              Stream /*stream*/) const {
    throw_no_device_memory("DDMT::save_history", backend());
}
void DDMTEngine::save_history(DeviceSpan<uint8_t> /*out*/,
                              Stream /*stream*/) const {
    throw_no_device_memory("DDMT::save_history", backend());
}
void DDMTEngine::load_history(DeviceSpan<const float> /*in*/,
                              Stream /*stream*/) {
    throw_no_device_memory("DDMT::load_history", backend());
}
void DDMTEngine::load_history(DeviceSpan<const uint8_t> /*in*/,
                              Stream /*stream*/) {
    throw_no_device_memory("DDMT::load_history", backend());
}

void DDMTFFTEngine::execute(DeviceSpan<const float> /*waterfall*/,
                            DeviceSpan<float> /*dmt*/,
                            Stream /*stream*/) {
    throw_no_device_memory("DDMTFFT::execute", backend());
}
void DDMTFFTEngine::execute(DeviceSpan<const uint8_t> /*waterfall_packed*/,
                            SizeType /*nsamps*/,
                            DeviceSpan<float> /*dmt*/,
                            Stream /*stream*/) {
    throw_no_device_memory("DDMTFFT::execute", backend());
}
void DDMTFFTEngine::save_history(DeviceSpan<float> /*out*/,
                                 Stream /*stream*/) const {
    throw_no_device_memory("DDMTFFT::save_history", backend());
}
void DDMTFFTEngine::load_history(DeviceSpan<const float> /*in*/,
                                 Stream /*stream*/) {
    throw_no_device_memory("DDMTFFT::load_history", backend());
}

void FDMTFFTEngine::execute(DeviceSpan<const float> /*waterfall*/,
                            DeviceSpan<float> /*dmt*/,
                            Stream /*stream*/) {
    throw_no_device_memory("FDMTFFT::execute", backend());
}
void FDMTFFTEngine::reset(DeviceSpan<const float> /*waterfall*/,
                          DeviceSpan<float> /*dmt*/,
                          Stream /*stream*/) {
    throw_no_device_memory("FDMTFFT::reset", backend());
}
void FDMTFFTEngine::execute(DeviceSpan<const uint8_t> /*packed*/,
                            SizeType /*nbits*/,
                            DeviceSpan<float> /*dmt*/,
                            Stream /*stream*/) {
    throw_no_device_memory("FDMTFFT::execute", backend());
}
void FDMTFFTEngine::reset(DeviceSpan<const uint8_t> /*packed*/,
                          SizeType /*nbits*/,
                          DeviceSpan<float> /*dmt*/,
                          Stream /*stream*/) {
    throw_no_device_memory("FDMTFFT::reset", backend());
}
void FDMTFFTEngine::save_history(DeviceSpan<float> /*out*/,
                                 Stream /*stream*/) const {
    throw_no_device_memory("FDMTFFT::save_history", backend());
}
void FDMTFFTEngine::load_history(DeviceSpan<const float> /*in*/,
                                 Stream /*stream*/) {
    throw_no_device_memory("FDMTFFT::load_history", backend());
}
DeviceSpan<const float> FDMTFFTEngine::view_level_data_device() const {
    throw_no_device_memory("FDMTFFT::view_level_data_device", backend());
}
algorithms::FDMTSubbandDeviceView
FDMTFFTEngine::view_subband_device(SizeType /*subband_idx*/) const {
    throw_no_device_memory("FDMTFFT::view_subband_device", backend());
}

void CohFDMTEngine::execute(DeviceSpan<const uint8_t> /*data_in*/,
                            DeviceSpan<float> /*dmt*/,
                            Stream /*stream*/) {
    throw_no_device_memory("CohFDMT::execute", backend());
}
void CohFDMTEngine::execute(DeviceSpan<const int8_t> /*data_in*/,
                            DeviceSpan<float> /*dmt*/,
                            Stream /*stream*/) {
    throw_no_device_memory("CohFDMT::execute", backend());
}

} // namespace dmt::algorithms::detail
