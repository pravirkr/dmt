#include "dmt/algorithms/cfdmt.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

#include "dmt/common/backend.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/engines.hpp"

namespace dmt::algorithms {

namespace {

std::unique_ptr<detail::CohFDMTEngine>
make_cfdmt_engine(const plans::CohFDMTPlan& plan,
                  const detail::CohFDMTEngineConfig& cfg) {
    if (cfg.exec.backend == Backend::kCPU) {
        return detail::make_cfdmt_cpu(plan, cfg);
    }
#ifdef DMT_ENABLE_GPU
    if (cfg.exec.backend == detail::kGPUBackend) {
        return detail::make_cfdmt_gpu(plan, cfg);
    }
#endif
    detail::throw_unavailable("CohFDMT", cfg.exec.backend);
}

// The engines take raw bytes; int8_t and uint8_t views of the same memory
// are interchangeable (the BasebandFormat decides the decoding).
template <typename DataType>
std::span<const uint8_t> as_bytes_view(std::span<const DataType> s) {
    static_assert(sizeof(DataType) == 1, "CohFDMT input must be byte data");
    return {reinterpret_cast<const uint8_t*>(s.data()), s.size()};
}

template <typename DataType>
DeviceSpan<const uint8_t> as_bytes_view(DeviceSpan<const DataType> s) {
    static_assert(sizeof(DataType) == 1, "CohFDMT input must be byte data");
    DeviceSpan<const uint8_t> out;
    out.handle      = s.handle;
    out.byte_offset = s.byte_offset;
    out.count       = s.count;
    out.device      = s.device;
    return out;
}

} // namespace

class CohFDMT::Impl {
public:
    Impl(const CohFDMTConfig& config, Exec exec)
        : m_plan(config),
          m_cfg{.exec = exec},
          m_engine(make_cfdmt_engine(m_plan, m_cfg)) {}

    // The engine holds a reference to m_plan; declaration order matters.
    plans::CohFDMTPlan m_plan;
    detail::CohFDMTEngineConfig m_cfg;
    std::unique_ptr<detail::CohFDMTEngine> m_engine;
    // execute() is const but runs on the engine's buffers: calls from
    // several threads take turns (device calls only while they enqueue;
    // the engine orders their device work across streams).
    mutable std::mutex m_mutex;
};

CohFDMT::CohFDMT(const CohFDMTConfig& config, Exec exec)
    : m_impl(std::make_unique<Impl>(config, exec)) {}

CohFDMT::~CohFDMT()                                   = default;
CohFDMT::CohFDMT(CohFDMT&& other) noexcept            = default;
CohFDMT& CohFDMT::operator=(CohFDMT&& other) noexcept = default;

const plans::CohFDMTPlan& CohFDMT::get_plan() const noexcept {
    return m_impl->m_plan;
}
Backend CohFDMT::backend() const noexcept { return m_impl->m_cfg.exec.backend; }
int CohFDMT::nthreads() const noexcept {
    return backend() == Backend::kCPU ? std::max(1, m_impl->m_cfg.exec.nthreads)
                                      : 1;
}
int CohFDMT::device() const noexcept {
    return backend() == Backend::kCPU ? -1 : m_impl->m_cfg.exec.device;
}
SizeType CohFDMT::get_block_nsamps() const noexcept {
    return m_impl->m_plan.get_block_nsamps();
}
SizeType CohFDMT::get_stride_nsamps() const noexcept {
    return m_impl->m_plan.get_stride_nsamps();
}
SizeType CohFDMT::get_output_nsamps() const noexcept {
    return m_impl->m_plan.get_output_nsamps();
}
SizeType CohFDMT::get_input_size(SizeType igroup) const {
    return m_impl->m_plan.get_input_size(igroup);
}
SizeType CohFDMT::get_dmt_size() const noexcept {
    return m_impl->m_plan.get_dmt_size();
}
SizeType CohFDMT::get_buffer_size() const noexcept {
    return m_impl->m_plan.get_buffer_size();
}
plans::CohFDMTMemoryUsage CohFDMT::get_memory_usage() const noexcept {
    return m_impl->m_engine->memory_usage();
}

template <IntegralDataType DataType>
void CohFDMT::execute(std::span<const DataType> data_in,
                      std::span<float> dmt) const {
    const std::span<const uint8_t> group = as_bytes_view(data_in);
    const std::lock_guard lock(m_impl->m_mutex);
    m_impl->m_engine->execute(std::span(&group, 1), dmt);
}
template <IntegralDataType DataType>
void CohFDMT::execute(std::span<const std::span<const DataType>> groups,
                      std::span<float> dmt) const {
    std::vector<std::span<const uint8_t>> bytes;
    bytes.reserve(groups.size());
    for (const auto& g : groups) {
        bytes.push_back(as_bytes_view(g));
    }
    const std::lock_guard lock(m_impl->m_mutex);
    m_impl->m_engine->execute(bytes, dmt);
}
template <IntegralDataType DataType>
void CohFDMT::execute(DeviceSpan<const DataType> d_data_in,
                      DeviceSpan<float> d_dmt,
                      Stream stream) const {
    detail::check_device(d_data_in.device, backend(), m_impl->m_cfg.exec.device,
                         "CohFDMT::execute");
    detail::check_device(d_dmt.device, backend(), m_impl->m_cfg.exec.device,
                         "CohFDMT::execute");
    const DeviceSpan<const uint8_t> group = as_bytes_view(d_data_in);
    const std::lock_guard lock(m_impl->m_mutex);
    m_impl->m_engine->execute(std::span(&group, 1), d_dmt, stream);
}
template <IntegralDataType DataType>
void CohFDMT::execute(std::span<const DeviceSpan<const DataType>> d_groups,
                      DeviceSpan<float> d_dmt,
                      Stream stream) const {
    std::vector<DeviceSpan<const uint8_t>> bytes;
    bytes.reserve(d_groups.size());
    for (const auto& g : d_groups) {
        detail::check_device(g.device, backend(), m_impl->m_cfg.exec.device,
                             "CohFDMT::execute");
        bytes.push_back(as_bytes_view(g));
    }
    detail::check_device(d_dmt.device, backend(), m_impl->m_cfg.exec.device,
                         "CohFDMT::execute");
    const std::lock_guard lock(m_impl->m_mutex);
    m_impl->m_engine->execute(bytes, d_dmt, stream);
}

// Instantiate the public execute methods for each supported DataType
template void CohFDMT::execute<int8_t>(std::span<const int8_t>,
                                       std::span<float>) const;
template void CohFDMT::execute<uint8_t>(std::span<const uint8_t>,
                                        std::span<float>) const;
template void
CohFDMT::execute<int8_t>(std::span<const std::span<const int8_t>>,
                         std::span<float>) const;
template void
CohFDMT::execute<uint8_t>(std::span<const std::span<const uint8_t>>,
                          std::span<float>) const;
template void CohFDMT::execute<int8_t>(DeviceSpan<const int8_t>,
                                       DeviceSpan<float>,
                                       Stream) const;
template void CohFDMT::execute<uint8_t>(DeviceSpan<const uint8_t>,
                                        DeviceSpan<float>,
                                        Stream) const;
template void
CohFDMT::execute<int8_t>(std::span<const DeviceSpan<const int8_t>>,
                         DeviceSpan<float>,
                         Stream) const;
template void
CohFDMT::execute<uint8_t>(std::span<const DeviceSpan<const uint8_t>>,
                          DeviceSpan<float>,
                          Stream) const;

} // namespace dmt::algorithms
