#include "dmt/algorithms/ddmt_fft.hpp"

#include <algorithm>
#include <cstdint>
#include <format>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "dmt/common/backend.hpp"
#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/ddmt_fft_common.hpp"
#include "dmt/dm_utils.hpp"
#include "dmt/engines.hpp"
#include "dmt/logging.hpp"

namespace dmt::algorithms {

DDMTFFTMethod parse_ddmt_fft_method(std::string_view name) {
    if (name == "auto") {
        return DDMTFFTMethod::kAuto;
    }
    if (name == "brute") {
        return DDMTFFTMethod::kBrute;
    }
    if (name == "nufft") {
        return DDMTFFTMethod::kNUFFT;
    }
    throw std::invalid_argument(std::format(
        "DDMTFFT: unknown method '{}' (expected auto, brute or nufft)", name));
}

std::string_view to_string(DDMTFFTMethod method) noexcept {
    switch (method) {
    case DDMTFFTMethod::kBrute:
        return "brute";
    case DDMTFFTMethod::kNUFFT:
        return "nufft";
    case DDMTFFTMethod::kAuto:
        break;
    }
    return "auto";
}

namespace {

DDMTFFTOptions resolve_options(const plans::DDMTPlan& plan,
                               DDMTFFTOptions options) {
    if (!utils::is_finite_bits(options.tolerance) ||
        options.tolerance < 1.0E-7 || options.tolerance > 1.0E-2) {
        throw std::invalid_argument(std::format(
            "DDMTFFT: tolerance {} outside [1e-7, 1e-2]", options.tolerance));
    }
    const ddmt_fft::DelayModel model(plan, options.guard);
    options.method = ddmt_fft::resolve_method(options.method, model);
    return options;
}

std::unique_ptr<detail::DDMTFFTEngine>
make_engine(const plans::DDMTPlan& plan,
            const detail::DDMTFFTEngineConfig& cfg) {
    if (cfg.exec.backend == Backend::kCPU) {
        return detail::make_ddmt_fft_cpu(plan, cfg);
    }
#ifdef DMT_ENABLE_GPU
    if (cfg.exec.backend == detail::kGPUBackend) {
        return detail::make_ddmt_fft_gpu(plan, cfg);
    }
#endif
    detail::throw_unavailable("DDMTFFT", cfg.exec.backend);
}

} // namespace

class DDMTFFT::Impl {
public:
    Impl(plans::DDMTPlan plan,
         Exec exec,
         SizeType nbeams,
         DDMTFFTOptions options)
        : m_plan(std::move(plan)),
          m_cfg{.nbeams  = nbeams,
                .exec    = exec,
                .options = resolve_options(m_plan, options)},
          m_engine(make_engine(m_plan, m_cfg)) {
        if (nbeams == 0) {
            throw std::invalid_argument("DDMTFFT: nbeams must be >= 1");
        }
    }

    // The engine holds a reference to m_plan; declaration order matters.
    plans::DDMTPlan m_plan;
    detail::DDMTFFTEngineConfig m_cfg;
    std::unique_ptr<detail::DDMTFFTEngine> m_engine;

    // Once per engine: blocks short against the context waste most of every
    // transform on overlap (the time-domain engines have no such cost).
    void note_block(SizeType nsamps) {
        if (m_noted) {
            return;
        }
        m_noted        = true;
        const auto ctx = m_engine->max_delay() + m_cfg.options.guard;
        const double eff =
            static_cast<double>(nsamps) / static_cast<double>(nsamps + ctx);
        if (eff < 0.6) {
            logging::debug("DDMTFFT: blocks of {} samples keep only {:.0f}% of "
                           "each transform (context {} samples); blocks of "
                           ">= {} samples (get_suggested_nsamps()) keep 80%",
                           nsamps, 100.0 * eff, ctx,
                           ddmt_fft::suggested_nsamps(ctx));
        }
    }
    bool m_noted{false};

    template <typename T>
    void check_device(const DeviceSpan<T>& span, std::string_view what) const {
        detail::check_device(span.device, m_cfg.exec.backend, m_cfg.exec.device,
                             what);
    }
};

DDMTFFT::DDMTFFT(float f_min,
                 float f_max,
                 SizeType nchans,
                 float tsamp,
                 float dm_max,
                 float dm_step,
                 float dm_min,
                 Exec exec,
                 SizeType nbits,
                 std::span<const uint8_t> kill_mask,
                 SizeType nbeams,
                 DDMTFFTOptions options)
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
                                    nbeams,
                                    options)) {}

DDMTFFT::DDMTFFT(float f_min,
                 float f_max,
                 SizeType nchans,
                 float tsamp,
                 std::span<const float> dm_arr,
                 Exec exec,
                 SizeType nbits,
                 std::span<const uint8_t> kill_mask,
                 SizeType nbeams,
                 DDMTFFTOptions options)
    : m_impl(std::make_unique<Impl>(
          plans::DDMTPlan(
              f_min, f_max, nchans, tsamp, dm_arr, nbits, kill_mask),
          exec,
          nbeams,
          options)) {}

DDMTFFT::DDMTFFT(float f_min,
                 float f_max,
                 SizeType nchans,
                 float tsamp,
                 const plans::LevinConfig& levin,
                 Exec exec,
                 SizeType nbits,
                 std::span<const uint8_t> kill_mask,
                 SizeType nbeams,
                 DDMTFFTOptions options)
    : m_impl(std::make_unique<Impl>(
          plans::DDMTPlan(f_min, f_max, nchans, tsamp, levin, nbits, kill_mask),
          exec,
          nbeams,
          options)) {}

DDMTFFT::DDMTFFT(const plans::DDMTPlan& plan,
                 Exec exec,
                 SizeType nbeams,
                 DDMTFFTOptions options)
    : m_impl(std::make_unique<Impl>(plan, exec, nbeams, options)) {}

DDMTFFT::~DDMTFFT()                                   = default;
DDMTFFT::DDMTFFT(DDMTFFT&& other) noexcept            = default;
DDMTFFT& DDMTFFT::operator=(DDMTFFT&& other) noexcept = default;

const plans::DDMTPlan& DDMTFFT::get_plan() const noexcept {
    return m_impl->m_plan;
}
SizeType DDMTFFT::get_nbeams() const noexcept { return m_impl->m_cfg.nbeams; }
Backend DDMTFFT::backend() const noexcept { return m_impl->m_cfg.exec.backend; }
int DDMTFFT::nthreads() const noexcept {
    return backend() == Backend::kCPU ? std::max(1, m_impl->m_cfg.exec.nthreads)
                                      : 1;
}
int DDMTFFT::device() const noexcept {
    return backend() == Backend::kCPU ? -1 : m_impl->m_cfg.exec.device;
}
DDMTFFTOptions DDMTFFT::get_options() const noexcept {
    return m_impl->m_cfg.options;
}
std::string_view DDMTFFT::method_used() const noexcept {
    return m_impl->m_engine->method_used();
}
SizeType DDMTFFT::get_max_delay() const noexcept {
    return m_impl->m_engine->max_delay();
}
SizeType DDMTFFT::get_suggested_nsamps() const noexcept {
    return ddmt_fft::suggested_nsamps(m_impl->m_engine->max_delay() +
                                      m_impl->m_cfg.options.guard);
}

void DDMTFFT::execute(std::span<const float> waterfall, std::span<float> dmt) {
    const auto rows = m_impl->m_cfg.nbeams * m_impl->m_plan.get_nchans();
    m_impl->note_block(rows > 0 ? waterfall.size() / rows : 0);
    m_impl->m_engine->execute(waterfall, dmt);
}
void DDMTFFT::execute(DeviceSpan<const float> d_waterfall,
                      DeviceSpan<float> d_dmt,
                      Stream stream) {
    m_impl->check_device(d_waterfall, "DDMTFFT::execute");
    m_impl->check_device(d_dmt, "DDMTFFT::execute");
    m_impl->m_engine->execute(d_waterfall, d_dmt, stream);
}
void DDMTFFT::execute(std::span<const uint8_t> waterfall_packed,
                      SizeType nsamps,
                      std::span<float> dmt) {
    m_impl->note_block(nsamps);
    m_impl->m_engine->execute(waterfall_packed, nsamps, dmt);
}
void DDMTFFT::execute(DeviceSpan<const uint8_t> d_waterfall_packed,
                      SizeType nsamps,
                      DeviceSpan<float> d_dmt,
                      Stream stream) {
    m_impl->check_device(d_waterfall_packed, "DDMTFFT::execute");
    m_impl->check_device(d_dmt, "DDMTFFT::execute");
    m_impl->m_engine->execute(d_waterfall_packed, nsamps, d_dmt, stream);
}
void DDMTFFT::execute_time_major(std::span<const uint8_t> filterbank_packed,
                                 SizeType nsamps,
                                 std::span<float> dmt) {
    m_impl->note_block(nsamps);
    m_impl->m_engine->execute_time_major(filterbank_packed, nsamps, dmt);
}

SizeType DDMTFFT::get_output_nsamps(SizeType input_nsamps) const noexcept {
    return m_impl->m_engine->get_output_nsamps(input_nsamps);
}
void DDMTFFT::reset_history() noexcept { m_impl->m_engine->reset_history(); }
void DDMTFFT::set_gulp_size(SizeType gulp_size) {
    m_impl->m_engine->set_gulp_size(gulp_size);
}
SizeType DDMTFFT::get_gulp_size() const noexcept {
    return m_impl->m_engine->get_gulp_size();
}
SizeType DDMTFFT::history_state_size() const noexcept {
    return m_impl->m_engine->history_state_size();
}
void DDMTFFT::save_history(std::span<float> out) const {
    m_impl->m_engine->save_history(out);
}
void DDMTFFT::load_history(std::span<const float> in) {
    m_impl->m_engine->load_history(in);
}
void DDMTFFT::save_history(DeviceSpan<float> d_out, Stream stream) const {
    m_impl->check_device(d_out, "DDMTFFT::save_history");
    m_impl->m_engine->save_history(d_out, stream);
}
void DDMTFFT::load_history(DeviceSpan<const float> d_in, Stream stream) {
    m_impl->check_device(d_in, "DDMTFFT::load_history");
    m_impl->m_engine->load_history(d_in, stream);
}

} // namespace dmt::algorithms
