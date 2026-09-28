#include "dmt/algorithms/cfdmt.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <utility>

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

} // namespace

class CohFDMT::Impl {
public:
    Impl(plans::CohFDMTPlan plan, Exec exec)
        : m_plan(std::move(plan)),
          m_cfg{.exec = exec},
          m_engine(make_cfdmt_engine(m_plan, m_cfg)) {}

    // The engine holds a reference to m_plan; declaration order matters.
    plans::CohFDMTPlan m_plan;
    detail::CohFDMTEngineConfig m_cfg;
    std::unique_ptr<detail::CohFDMTEngine> m_engine;
};

CohFDMT::CohFDMT(float f_center,
                 float bw_sub,
                 SizeType nsub,
                 float tbin,
                 SizeType nbin,
                 SizeType nfft,
                 float t_p,
                 float dm_max,
                 float dm_min,
                 SizeType noverlap,
                 std::string_view data_order,
                 Exec exec)
    : m_impl(std::make_unique<Impl>(plans::CohFDMTPlan(f_center,
                                                       bw_sub,
                                                       nsub,
                                                       tbin,
                                                       nbin,
                                                       nfft,
                                                       t_p,
                                                       dm_max,
                                                       dm_min,
                                                       noverlap,
                                                       data_order),
                                    exec)) {}

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
SizeType CohFDMT::get_dmt_size() const noexcept {
    return m_impl->m_plan.get_dmt_size();
}
SizeType CohFDMT::get_buffer_size() const noexcept {
    return m_impl->m_plan.get_buffer_size();
}

template <IntegralDataType DataType>
void CohFDMT::execute(std::span<const DataType> data_in,
                      std::span<float> dmt) const {
    m_impl->m_engine->execute(data_in, dmt);
}
template <IntegralDataType DataType>
void CohFDMT::execute(DeviceSpan<const DataType> d_data_in,
                      DeviceSpan<float> d_dmt,
                      Stream stream) const {
    detail::check_device(d_data_in.device, backend(), m_impl->m_cfg.exec.device,
                         "CohFDMT::execute");
    detail::check_device(d_dmt.device, backend(), m_impl->m_cfg.exec.device,
                         "CohFDMT::execute");
    m_impl->m_engine->execute(d_data_in, d_dmt, stream);
}
void CohFDMT::reset_history() noexcept { m_impl->m_engine->reset_history(); }

// Instantiate the public execute methods for each supported DataType
template void CohFDMT::execute<int8_t>(std::span<const int8_t>,
                                       std::span<float>) const;
template void CohFDMT::execute<uint8_t>(std::span<const uint8_t>,
                                        std::span<float>) const;
template void CohFDMT::execute<int8_t>(DeviceSpan<const int8_t>,
                                       DeviceSpan<float>,
                                       Stream) const;
template void CohFDMT::execute<uint8_t>(DeviceSpan<const uint8_t>,
                                        DeviceSpan<float>,
                                        Stream) const;

} // namespace dmt::algorithms
