#include "dmt/algorithms/fdmt_fft_boxcar.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "dmt/boxcar.hpp"
#include "dmt/boxcar_drain.hpp"
#include "dmt/dm_utils.hpp"
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
         std::span<const uint8_t> kill,
         SizeType max_memory,
         bool estimate_only = false)
        : plan(std::move(plan)),
          config(config),
          exec(exec),
          beams(beams) {
        if (this->plan.get_subband_depth() != 0 || config.input_nbits != 32) {
            throw std::invalid_argument(
                "FDMTFFTBoxcar: subband search and packed input are not "
                "supported (plan subband_depth 0, input_nbits 32)");
        }
        if (config.per_call_masks) {
            throw std::invalid_argument(
                "FDMTFFTBoxcar: per-call kill masks are not supported "
                "(BoxcarConfig::per_call_masks must be false)");
        }
        if (beams == 0 || beams > UINT32_MAX ||
            beams >
                SIZE_MAX / this->plan.get_nchans() / this->plan.get_nsamps()) {
            throw std::invalid_argument("FDMTFFTBoxcar: invalid nbeams");
        }
        // Band-limited (fractional) shifts need context beyond the integer
        // tree's support: hold detections until the FFT support is warm.
        const auto geom =
            fdmt_fft::make_geometry(this->plan, FDMTMode::kValid, frac, smear);
        latency                = geom.latency;
        const uint64_t integer = this->plan.get_fft_support();
        const uint64_t extra   = std::max<uint64_t>(
            geom.guard, geom.support > integer ? geom.support - integer : 0);
        planes =
            detail::make_boxcar_plans(this->plan, config, smear, kill, extra);
        detail::FDMTFFTEngineConfig fft_cfg{
            .use_box_smearing  = smear,
            .mode              = FDMTMode::kValid,
            .nbeams            = beams,
            .exec              = exec,
            .fractional_delays = frac,
            .kill_mask         = std::vector<uint8_t>(kill.begin(), kill.end()),
            .search            = true,
            // Candidates are sized, not allocated: only the chosen one is.
            .defer_allocation = true,
        };
        engine = make_engine(fft_cfg);
        // Budget: shorten the overlap-save segments (halving the transform
        // length) while the instance is over it. The boxcar buffers do not
        // depend on the segment, so fail at once when they alone are over.
        const SizeType discard = std::max(geom.skip, geom.support);
        const SizeType n_min =
            utils::next_fft_size((2 * (discard + geom.guard)) + 2);
        auto used = engine->memory_usage();
        // Budgeted bytes only: on a GPU the host result storage
        // (max_candidates) is reported, not budgeted.
        const SizeType host_cands =
            exec.backend == Backend::kCPU
                ? SizeType{0}
                : config.max_candidates * sizeof(RawCandidate);
        if (const SizeType fixed = used.peaks + used.candidates - host_cands;
            fixed >= max_memory) {
            throw std::invalid_argument(fmt::format(
                "FDMTFFTBoxcar: the boxcar buffers alone need {:.2f} GiB, "
                "over max_memory_bytes ({:.2f} GiB); lower max_boxcar or "
                "max_candidates, or raise the budget",
                static_cast<double>(fixed) / (1U << 30U),
                static_cast<double>(max_memory) / (1U << 30U)));
        }
        while (used.budgeted() > max_memory) {
            const SizeType n = std::min(geom.n_fft, engine->chunk_nsamps() +
                                                        discard + geom.guard);
            if (n / 2 < n_min) {
                throw std::invalid_argument(fmt::format(
                    "FDMTFFTBoxcar: this configuration needs {:.2f} GiB with "
                    "the shortest overlap-save segments tried ({} samples), "
                    "over max_memory_bytes ({:.2f} GiB)",
                    static_cast<double>(used.budgeted()) / (1U << 30U), n,
                    static_cast<double>(max_memory) / (1U << 30U)));
            }
            const SizeType hop  = engine->chunk_nsamps();
            fft_cfg.max_segment = n / 2;
            engine.reset();
            engine = make_engine(fft_cfg);
            used   = engine->memory_usage();
            if (used.budgeted() > max_memory && engine->chunk_nsamps() >= hop) {
                throw std::logic_error(
                    "FDMTFFTBoxcar: the segment cap did not shorten the "
                    "segments");
            }
        }
        if (!estimate_only) {
            engine->allocate();
        }
    }
    [[nodiscard]] std::unique_ptr<detail::FDMTBoxcarEngine>
    make_engine(const detail::FDMTFFTEngineConfig& fft_cfg) const {
        const detail::FDMTEngineConfig cfg{
            .use_box_smearing = fft_cfg.use_box_smearing,
            .mode             = FDMTMode::kValid,
            .nbeams           = beams,
            .exec             = exec,
            .defer_allocation = fft_cfg.defer_allocation,
        };
        if (exec.backend == Backend::kCPU) {
            return detail::make_fdmt_boxcar_cpu(
                plan, config, planes, cfg,
                detail::make_boxcar_driver(
                    detail::make_fdmt_fft_cpu(plan, fft_cfg)));
        }
#ifdef DMT_ENABLE_GPU
        if (exec.backend == detail::kGPUBackend) {
            return detail::make_fdmt_boxcar_gpu(
                plan, config, planes, cfg,
                detail::make_boxcar_driver(
                    detail::make_fdmt_fft_gpu(plan, fft_cfg)));
        }
#endif
        detail::throw_unavailable("FDMTFFTBoxcar", exec.backend);
    }
    void check_stream(Stream stream) const {
        if (exec.backend == Backend::kCPU && stream.native != nullptr) {
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
    detail::DrainState drain_state;
};

FDMTFFTBoxcar::FDMTFFTBoxcar(plans::FDMTPlan plan,
                             BoxcarConfig config,
                             bool use_box_smearing,
                             Exec exec,
                             SizeType nbeams,
                             bool fractional_delays,
                             std::span<const uint8_t> kill_mask,
                             SizeType max_memory_bytes)
    : m_impl(std::make_unique<Impl>(std::move(plan),
                                    config,
                                    use_box_smearing,
                                    exec,
                                    nbeams,
                                    fractional_delays,
                                    kill_mask,
                                    max_memory_bytes)) {}
FDMTBoxcarMemoryEstimate
FDMTFFTBoxcar::estimate_memory(const plans::FDMTPlan& plan,
                               BoxcarConfig config,
                               bool use_box_smearing,
                               Exec exec,
                               SizeType nbeams,
                               bool fractional_delays,
                               std::span<const uint8_t> kill_mask,
                               SizeType max_memory_bytes) {
    const Impl impl(plan, config, use_box_smearing, exec, nbeams,
                    fractional_delays, kill_mask, max_memory_bytes,
                    /*estimate_only=*/true);
    return {.usage            = impl.engine->memory_usage(),
            .chunk_nsamps     = impl.engine->chunk_nsamps(),
            .host_staging     = impl.engine->host_staging_bytes(),
            .search_path      = impl.engine->search_path(),
            .search_latency   = impl.latency,
            .max_memory_bytes = max_memory_bytes};
}
FDMTFFTBoxcar::~FDMTFFTBoxcar()                                   = default;
FDMTFFTBoxcar::FDMTFFTBoxcar(FDMTFFTBoxcar&&) noexcept            = default;
FDMTFFTBoxcar& FDMTFFTBoxcar::operator=(FDMTFFTBoxcar&&) noexcept = default;
CandidateResultView FDMTFFTBoxcar::execute(std::span<const float> waterfall) {
    m_impl->drain_state.check_open("FDMTFFTBoxcar::execute");
    return m_impl->engine->execute(waterfall);
}
CandidateDeviceResultView
FDMTFFTBoxcar::execute(DeviceSpan<const float> waterfall, Stream stream) {
    m_impl->drain_state.check_open("FDMTFFTBoxcar::execute");
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
    m_impl->drain_state.reset();
}
CandidateResultView FDMTFFTBoxcar::drain() {
    return detail::drain_stream(*m_impl->engine, m_impl->plan, m_impl->beams,
                                m_impl->config.input_nbits, m_impl->latency,
                                m_impl->drain_state);
}
uint64_t FDMTFFTBoxcar::get_searched_nsamps() const noexcept {
    return detail::searched_nsamps(*m_impl->engine, m_impl->drain_state,
                                   m_impl->latency);
}
uint64_t FDMTFFTBoxcar::get_candidate_frontier() const noexcept {
    // New excursions begin at endpoints not yet searched by every row;
    // candidates report them less the plan's level-0 causal delay.
    const uint64_t raw =
        std::min(get_searched_nsamps(), m_impl->engine->open_begin_min());
    const uint64_t lag = get_plan().get_level0_causal_delay();
    return raw > lag ? raw - lag : 0;
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
    return fmt::format(
        "{}*** FDMTFFTBoxcar ({}) ***\n{} beams, {} widths, segment {}, "
        "latency {}\nAllocated {:.2f} MiB; no DM-time output\n",
        get_plan().summary(), to_string(backend()), get_nbeams(),
        m_impl->planes.front().widths.size(), get_chunk_nsamps(),
        get_output_latency(),
        static_cast<double>(get_memory_usage().total()) / 1048576.0);
}

} // namespace dmt::algorithms
