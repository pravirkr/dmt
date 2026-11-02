#include "dmt/algorithms/fdmt_boxcar.hpp"

#include <algorithm>
#include <format>
#include <limits>
#include <stdexcept>
#include <utility>

#include "dmt/boxcar.hpp"

namespace dmt::algorithms {

class FDMTBoxcar::Impl {
public:
    Impl(plans::FDMTPlan plan,
         BoxcarConfig config,
         bool smear,
         Exec exec,
         SizeType beams,
         SizeType fuse,
         SizeType chunk)
        : plan(std::move(plan)),
          config(config),
          exec(exec),
          beams(beams),
          planes(detail::make_boxcar_plans(this->plan, config, smear)) {
        if (beams == 0 || beams > UINT32_MAX ||
            beams >
                SIZE_MAX / this->plan.get_nchans() / this->plan.get_nsamps()) {
            throw std::invalid_argument("FDMTBoxcar: invalid nbeams");
        }
        detail::FDMTEngineConfig cfg{
            .use_box_smearing = smear,
            .mode             = FDMTMode::kValid,
            .nbeams           = beams,
            .fuse_levels      = fuse,
            .int_tree         = false,
            .exec             = exec,
            .chunk_nsamps     = chunk,
            .max_fuse_levels =
                std::min(this->plan.get_niters() - 1,
                         this->plan.get_niters() - config.subband_levels),
            .search_upper_levels = config.subband_levels > 0,
            .search_nbits        = config.input_nbits,
        };
        if (exec.backend == Backend::kCPU) {
            engine = detail::make_fdmt_boxcar_cpu(
                this->plan, config, planes, cfg,
                detail::make_boxcar_driver(
                    detail::make_fdmt_cpu(this->plan, cfg)));
        }
#ifdef DMT_ENABLE_GPU
        else if (exec.backend == detail::kGPUBackend) {
            engine = detail::make_fdmt_boxcar_gpu(
                this->plan, config, planes, cfg,
                detail::make_boxcar_driver(
                    detail::make_fdmt_gpu(this->plan, cfg)));
        }
#endif
        else {
            detail::throw_unavailable("FDMTBoxcar", exec.backend);
        }
    }
    void check_stream(Stream stream) const {
        if (exec.backend == Backend::kCPU && stream.native) {
            throw std::invalid_argument(
                "FDMTBoxcar: CPU backend has no stream");
        }
    }
    plans::FDMTPlan plan;
    BoxcarConfig config;
    Exec exec;
    SizeType beams;
    std::vector<plans::BoxcarPlanePlan> planes;
    std::unique_ptr<detail::FDMTBoxcarEngine> engine;
};

FDMTBoxcar::FDMTBoxcar(plans::FDMTPlan plan,
                       BoxcarConfig config,
                       bool smear,
                       Exec exec,
                       SizeType beams,
                       SizeType fuse,
                       SizeType chunk)
    : m_impl(std::make_unique<Impl>(
          std::move(plan), config, smear, exec, beams, fuse, chunk)) {}
FDMTBoxcar::~FDMTBoxcar()                                = default;
FDMTBoxcar::FDMTBoxcar(FDMTBoxcar&&) noexcept            = default;
FDMTBoxcar& FDMTBoxcar::operator=(FDMTBoxcar&&) noexcept = default;
CandidateResultView FDMTBoxcar::execute(std::span<const float> waterfall) {
    return m_impl->engine->execute(waterfall);
}
CandidateDeviceResultView FDMTBoxcar::execute(DeviceSpan<const float> waterfall,
                                              Stream stream) {
    detail::check_device(waterfall.device, backend(), device(),
                         "FDMTBoxcar::execute");
    return m_impl->engine->execute(waterfall, stream);
}
CandidateResultView FDMTBoxcar::execute(std::span<const uint8_t> waterfall) {
    return m_impl->engine->execute(waterfall);
}
CandidateDeviceResultView
FDMTBoxcar::execute(DeviceSpan<const uint8_t> waterfall, Stream stream) {
    detail::check_device(waterfall.device, backend(), device(),
                         "FDMTBoxcar::execute");
    return m_impl->engine->execute(waterfall, stream);
}
CandidateResultView FDMTBoxcar::flush() { return m_impl->engine->flush(); }
CandidateDeviceResultView FDMTBoxcar::flush_device(Stream stream) {
    m_impl->check_stream(stream);
    return m_impl->engine->flush_device(stream);
}
CandidateResultView FDMTBoxcar::copy_result(Stream stream) {
    m_impl->check_stream(stream);
    return m_impl->engine->copy_result(stream);
}
void FDMTBoxcar::reset_history(Stream stream) {
    m_impl->check_stream(stream);
    m_impl->engine->reset_history(stream);
}
const plans::FDMTPlan& FDMTBoxcar::get_plan() const noexcept {
    return m_impl->plan;
}
const BoxcarConfig& FDMTBoxcar::get_config() const noexcept {
    return m_impl->config;
}
std::span<const plans::BoxcarPlanePlan>
FDMTBoxcar::get_search_plans() const noexcept {
    return m_impl->planes;
}
SizeType FDMTBoxcar::get_nbeams() const noexcept { return m_impl->beams; }
SizeType FDMTBoxcar::get_chunk_nsamps() const noexcept {
    return m_impl->engine->chunk_nsamps();
}
SizeType FDMTBoxcar::get_fuse_levels() const noexcept {
    return m_impl->engine->fuse_levels();
}
Backend FDMTBoxcar::backend() const noexcept { return m_impl->exec.backend; }
int FDMTBoxcar::device() const noexcept {
    return backend() == Backend::kCPU ? -1 : m_impl->exec.device;
}
int FDMTBoxcar::nthreads() const noexcept {
    return backend() == Backend::kCPU ? std::max(1, m_impl->exec.nthreads) : 1;
}
FDMTBoxcarMemoryUsage FDMTBoxcar::get_memory_usage() const noexcept {
    return m_impl->engine->memory_usage();
}
std::string FDMTBoxcar::summary() const {
    return std::format(
        "{}*** FDMTBoxcar ({}) ***\n{} beams, {} planes, {} widths, chunk {}, "
        "fuse_levels {}\nAllocated {:.2f} MiB; no root DM-time output\n",
        get_plan().summary(), to_string(backend()), get_nbeams(),
        m_impl->planes.size(), m_impl->planes.front().widths.size(),
        get_chunk_nsamps(), get_fuse_levels(),
        static_cast<double>(get_memory_usage().total()) / 1048576.0);
}

} // namespace dmt::algorithms
