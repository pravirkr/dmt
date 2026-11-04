#include "dmt/algorithms/fdmt_fft_boxcar.hpp"

#include <algorithm>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dmt/boxcar.hpp"
#include "dmt/fdmt_fft_common.hpp"

namespace dmt::algorithms {

class FDMTFFTBoxcar::Impl {
public:
    Impl(plans::FDMTPlan plan,
         BoxcarConfig config,
         bool smear,
         Exec exec,
         SizeType beams,
         bool frac,
         std::span<const uint8_t> kill)
        : plan(std::move(plan)),
          config(config),
          exec(exec),
          beams(beams) {
        if (this->plan.get_subband_depth() != 0 || config.input_nbits != 32) {
            throw std::invalid_argument(
                "FDMTFFTBoxcar: subband search and packed input are not "
                "supported (plan subband_depth 0, input_nbits 32)");
        }
        if (beams == 0 || beams > UINT32_MAX ||
            beams >
                SIZE_MAX / this->plan.get_nchans() / this->plan.get_nsamps()) {
            throw std::invalid_argument("FDMTFFTBoxcar: invalid nbeams");
        }
        planes = detail::make_boxcar_plans(this->plan, config, smear, kill);
        // Band-limited (fractional) shifts need context beyond the integer
        // tree's support: hold detections until the FFT support is warm.
        const auto geom =
            fdmt_fft::make_geometry(this->plan, FDMTMode::kValid, frac, smear);
        latency          = geom.latency;
        uint64_t integer = 0;
        for (const auto& p : planes) {
            for (const auto w : p.warmup_samples) {
                integer = std::max(integer, w);
            }
        }
        const uint64_t extra = std::max<uint64_t>(
            geom.guard, geom.support > integer ? geom.support - integer : 0);
        for (auto& p : planes) {
            for (auto& w : p.warmup_samples) {
                w += extra;
            }
        }
        const detail::FDMTFFTEngineConfig fft_cfg{
            .use_box_smearing  = smear,
            .mode              = FDMTMode::kValid,
            .nbeams            = beams,
            .exec              = exec,
            .fractional_delays = frac,
            .kill_mask         = std::vector<uint8_t>(kill.begin(), kill.end()),
        };
        const detail::FDMTEngineConfig cfg{
            .use_box_smearing = smear,
            .mode             = FDMTMode::kValid,
            .nbeams           = beams,
            .exec             = exec,
        };
        if (exec.backend == Backend::kCPU) {
            engine = detail::make_fdmt_boxcar_cpu(
                this->plan, config, planes, cfg,
                detail::make_boxcar_driver(
                    detail::make_fdmt_fft_cpu(this->plan, fft_cfg)));
        }
#ifdef DMT_ENABLE_GPU
        else if (exec.backend == detail::kGPUBackend) {
            engine = detail::make_fdmt_boxcar_gpu(
                this->plan, config, planes, cfg,
                detail::make_boxcar_driver(
                    detail::make_fdmt_fft_gpu(this->plan, fft_cfg)));
        }
#endif
        else {
            detail::throw_unavailable("FDMTFFTBoxcar", exec.backend);
        }
    }
    void check_stream(Stream stream) const {
        if (exec.backend == Backend::kCPU && stream.native) {
            throw std::invalid_argument(
                "FDMTFFTBoxcar: CPU backend has no stream");
        }
    }
    plans::FDMTPlan plan;
    BoxcarConfig config;
    Exec exec;
    SizeType beams;
    SizeType latency = 0;
    std::vector<plans::BoxcarPlanePlan> planes;
    std::unique_ptr<detail::FDMTBoxcarEngine> engine;
};

FDMTFFTBoxcar::FDMTFFTBoxcar(plans::FDMTPlan plan,
                             BoxcarConfig config,
                             bool smear,
                             Exec exec,
                             SizeType beams,
                             bool frac,
                             std::span<const uint8_t> kill_mask)
    : m_impl(std::make_unique<Impl>(
          std::move(plan), config, smear, exec, beams, frac, kill_mask)) {}
FDMTFFTBoxcar::~FDMTFFTBoxcar()                                   = default;
FDMTFFTBoxcar::FDMTFFTBoxcar(FDMTFFTBoxcar&&) noexcept            = default;
FDMTFFTBoxcar& FDMTFFTBoxcar::operator=(FDMTFFTBoxcar&&) noexcept = default;
CandidateResultView FDMTFFTBoxcar::execute(std::span<const float> waterfall) {
    return m_impl->engine->execute(waterfall);
}
CandidateDeviceResultView
FDMTFFTBoxcar::execute(DeviceSpan<const float> waterfall, Stream stream) {
    detail::check_device(waterfall.device, backend(), device(),
                         "FDMTFFTBoxcar::execute");
    return m_impl->engine->execute(waterfall, stream);
}
CandidateResultView FDMTFFTBoxcar::flush() { return m_impl->engine->flush(); }
CandidateDeviceResultView FDMTFFTBoxcar::flush_device(Stream stream) {
    m_impl->check_stream(stream);
    return m_impl->engine->flush_device(stream);
}
CandidateResultView FDMTFFTBoxcar::copy_result(Stream stream) {
    m_impl->check_stream(stream);
    return m_impl->engine->copy_result(stream);
}
void FDMTFFTBoxcar::reset_history(Stream stream) {
    m_impl->check_stream(stream);
    m_impl->engine->reset_history(stream);
}
const plans::FDMTPlan& FDMTFFTBoxcar::get_plan() const noexcept {
    return m_impl->plan;
}
const BoxcarConfig& FDMTFFTBoxcar::get_config() const noexcept {
    return m_impl->config;
}
std::span<const plans::BoxcarPlanePlan>
FDMTFFTBoxcar::get_search_plans() const noexcept {
    return m_impl->planes;
}
SizeType FDMTFFTBoxcar::get_nbeams() const noexcept { return m_impl->beams; }
SizeType FDMTFFTBoxcar::get_chunk_nsamps() const noexcept {
    return m_impl->engine->chunk_nsamps();
}
SizeType FDMTFFTBoxcar::get_output_latency() const noexcept {
    return m_impl->latency;
}
Backend FDMTFFTBoxcar::backend() const noexcept { return m_impl->exec.backend; }
int FDMTFFTBoxcar::device() const noexcept {
    return backend() == Backend::kCPU ? -1 : m_impl->exec.device;
}
int FDMTFFTBoxcar::nthreads() const noexcept {
    return backend() == Backend::kCPU ? std::max(1, m_impl->exec.nthreads) : 1;
}
FDMTBoxcarMemoryUsage FDMTFFTBoxcar::get_memory_usage() const noexcept {
    return m_impl->engine->memory_usage();
}
std::string FDMTFFTBoxcar::summary() const {
    return std::format(
        "{}*** FDMTFFTBoxcar ({}) ***\n{} beams, {} widths, segment {}, "
        "latency {}\nAllocated {:.2f} MiB; no DM-time output\n",
        get_plan().summary(), to_string(backend()), get_nbeams(),
        m_impl->planes.front().widths.size(), get_chunk_nsamps(),
        get_output_latency(),
        static_cast<double>(get_memory_usage().total()) / 1048576.0);
}

} // namespace dmt::algorithms
