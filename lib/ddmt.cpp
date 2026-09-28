#include "dmt/algorithms/ddmt.hpp"

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

std::unique_ptr<detail::DDMTEngine>
make_ddmt_engine(const plans::DDMTPlan& plan,
                 const detail::DDMTEngineConfig& cfg) {
    if (cfg.exec.backend == Backend::kCPU) {
        return detail::make_ddmt_cpu(plan, cfg);
    }
#ifdef DMT_ENABLE_GPU
    if (cfg.exec.backend == detail::kGPUBackend) {
        return detail::make_ddmt_gpu(plan, cfg);
    }
#endif
    detail::throw_unavailable("DDMT", cfg.exec.backend);
}

} // namespace

class DDMT::Impl {
public:
    Impl(plans::DDMTPlan plan, Exec exec, SizeType nbeams)
        : m_plan(std::move(plan)),
          m_cfg{.nbeams = nbeams, .exec = exec},
          m_engine(make_ddmt_engine(m_plan, m_cfg)) {}

    // The engine holds a reference to m_plan; declaration order matters.
    plans::DDMTPlan m_plan;
    detail::DDMTEngineConfig m_cfg;
    std::unique_ptr<detail::DDMTEngine> m_engine;

    template <typename T>
    void check_device(const DeviceSpan<T>& span, std::string_view what) const {
        detail::check_device(span.device, m_cfg.exec.backend, m_cfg.exec.device,
                             what);
    }
};

DDMT::DDMT(float f_min,
           float f_max,
           SizeType nchans,
           float tsamp,
           float dm_max,
           float dm_step,
           float dm_min,
           Exec exec,
           SizeType nbits,
           std::span<const uint8_t> kill_mask,
           SizeType nbeams)
    : m_impl(std::make_unique<Impl>(plans::DDMTPlan(f_min,
                                                    f_max,
                                                    nchans,
                                                    tsamp,
                                                    dm_max,
                                                    dm_step,
                                                    dm_min,
                                                    nbits,
                                                    kill_mask),
                                    exec,
                                    nbeams)) {}

DDMT::DDMT(float f_min,
           float f_max,
           SizeType nchans,
           float tsamp,
           std::span<const float> dm_arr,
           Exec exec,
           SizeType nbits,
           std::span<const uint8_t> kill_mask,
           SizeType nbeams)
    : m_impl(std::make_unique<Impl>(
          plans::DDMTPlan(
              f_min, f_max, nchans, tsamp, dm_arr, nbits, kill_mask),
          exec,
          nbeams)) {}

DDMT::DDMT(float f_min,
           float f_max,
           SizeType nchans,
           float tsamp,
           const plans::LevinConfig& levin,
           Exec exec,
           SizeType nbits,
           std::span<const uint8_t> kill_mask,
           SizeType nbeams)
    : m_impl(std::make_unique<Impl>(
          plans::DDMTPlan(f_min, f_max, nchans, tsamp, levin, nbits, kill_mask),
          exec,
          nbeams)) {}

DDMT::DDMT(const plans::DDMTPlan& plan, Exec exec, SizeType nbeams)
    : m_impl(std::make_unique<Impl>(plan, exec, nbeams)) {}

DDMT::~DDMT()                                = default;
DDMT::DDMT(DDMT&& other) noexcept            = default;
DDMT& DDMT::operator=(DDMT&& other) noexcept = default;

const plans::DDMTPlan& DDMT::get_plan() const noexcept {
    return m_impl->m_plan;
}
SizeType DDMT::get_nbeams() const noexcept { return m_impl->m_cfg.nbeams; }
Backend DDMT::backend() const noexcept { return m_impl->m_cfg.exec.backend; }
int DDMT::nthreads() const noexcept {
    return backend() == Backend::kCPU ? std::max(1, m_impl->m_cfg.exec.nthreads)
                                      : 1;
}
int DDMT::device() const noexcept {
    return backend() == Backend::kCPU ? -1 : m_impl->m_cfg.exec.device;
}

void DDMT::execute(std::span<const float> waterfall, std::span<float> dmt) {
    m_impl->m_engine->execute(waterfall, dmt);
}
void DDMT::execute(DeviceSpan<const float> d_waterfall,
                   DeviceSpan<float> d_dmt,
                   Stream stream) {
    m_impl->check_device(d_waterfall, "DDMT::execute");
    m_impl->check_device(d_dmt, "DDMT::execute");
    m_impl->m_engine->execute(d_waterfall, d_dmt, stream);
}
void DDMT::execute(std::span<const uint8_t> waterfall_packed,
                   SizeType nsamps,
                   std::span<int32_t> dmt) {
    m_impl->m_engine->execute(waterfall_packed, nsamps, dmt);
}
void DDMT::execute(DeviceSpan<const uint8_t> d_waterfall_packed,
                   SizeType nsamps,
                   DeviceSpan<int32_t> d_dmt,
                   Stream stream) {
    m_impl->check_device(d_waterfall_packed, "DDMT::execute");
    m_impl->check_device(d_dmt, "DDMT::execute");
    m_impl->m_engine->execute(d_waterfall_packed, nsamps, d_dmt, stream);
}
void DDMT::execute_time_major(std::span<const uint8_t> filterbank_packed,
                              SizeType nsamps,
                              std::span<int32_t> dmt) {
    m_impl->m_engine->execute_time_major(filterbank_packed, nsamps, dmt);
}
void DDMT::execute_time_major(DeviceSpan<const uint8_t> d_filterbank_packed,
                              SizeType nsamps,
                              DeviceSpan<int32_t> d_dmt,
                              Stream stream) {
    m_impl->check_device(d_filterbank_packed, "DDMT::execute_time_major");
    m_impl->check_device(d_dmt, "DDMT::execute_time_major");
    m_impl->m_engine->execute_time_major(d_filterbank_packed, nsamps, d_dmt,
                                         stream);
}

SizeType DDMT::get_output_nsamps(SizeType input_nsamps) const noexcept {
    return m_impl->m_engine->get_output_nsamps(input_nsamps);
}
void DDMT::reset_history() noexcept { m_impl->m_engine->reset_history(); }
SizeType DDMT::history_state_size() const noexcept {
    return m_impl->m_engine->history_state_size();
}

void DDMT::save_history(std::span<float> out) const {
    m_impl->m_engine->save_history(out);
}
void DDMT::save_history(std::span<uint8_t> out) const {
    m_impl->m_engine->save_history(out);
}
void DDMT::load_history(std::span<const float> in) {
    m_impl->m_engine->load_history(in);
}
void DDMT::load_history(std::span<const uint8_t> in) {
    m_impl->m_engine->load_history(in);
}
void DDMT::save_history(DeviceSpan<float> d_out, Stream stream) const {
    m_impl->check_device(d_out, "DDMT::save_history");
    m_impl->m_engine->save_history(d_out, stream);
}
void DDMT::save_history(DeviceSpan<uint8_t> d_out, Stream stream) const {
    m_impl->check_device(d_out, "DDMT::save_history");
    m_impl->m_engine->save_history(d_out, stream);
}
void DDMT::load_history(DeviceSpan<const float> d_in, Stream stream) {
    m_impl->check_device(d_in, "DDMT::load_history");
    m_impl->m_engine->load_history(d_in, stream);
}
void DDMT::load_history(DeviceSpan<const uint8_t> d_in, Stream stream) {
    m_impl->check_device(d_in, "DDMT::load_history");
    m_impl->m_engine->load_history(d_in, stream);
}

} // namespace dmt::algorithms
