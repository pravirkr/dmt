#include "dmt/common/backend.hpp"

#include <algorithm>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/common/plans.hpp"
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
    case Backend::kMetal:
        return "metal";
    }
    return "unknown";
}

Backend parse_backend(std::string_view name) {
    for (const auto b :
         {Backend::kCPU, Backend::kCUDA, Backend::kHIP, Backend::kMetal}) {
        if (name == to_string(b)) {
            return b;
        }
    }
    throw std::invalid_argument(std::format(
        "Unknown backend '{}'. Expected 'cpu', 'cuda', 'hip' or 'metal'",
        name));
}

std::vector<Backend> available_backends() {
    std::vector<Backend> out{Backend::kCPU};
#ifdef DMT_ENABLE_CUDA
    out.push_back(Backend::kCUDA);
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
