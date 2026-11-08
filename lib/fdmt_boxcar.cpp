#include "dmt/algorithms/fdmt_boxcar.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

#include <fmt/format.h>

#include "dmt/boxcar.hpp"
#include "dmt/fdmt_int_tree.hpp"

namespace dmt::algorithms {

class FDMTBoxcar::Impl {
public:
    Impl(plans::FDMTPlan plan,
         BoxcarConfig config,
         bool smear,
         Exec exec,
         SizeType beams,
         SizeType fuse,
         SizeType chunk,
         SizeType max_memory)
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
            // The searched levels are materialised; the engines also keep
            // fusion below the plan's staggered subbands.
            .max_fuse_levels     = std::min(this->plan.get_niters() - 1,
                                            this->plan.get_niters() -
                                                this->plan.get_subband_depth()),
            .search_upper_levels = this->plan.get_subband_depth() > 0,
            .search_nbits        = config.input_nbits,
        };
        // The boxcar's own buffers are fixed by plan and config; the FDMT
        // takes its path within the rest of the budget.
        const SizeType fixed = boxcar_bytes(smear, exec);
        if (fixed >= max_memory) {
            throw std::invalid_argument(fmt::format(
                "FDMTBoxcar: the boxcar buffers alone need {:.2f} GiB, over "
                "max_memory_bytes ({:.2f} GiB); lower max_boxcar or "
                "max_candidates, or raise the budget",
                static_cast<double>(fixed) / (1 << 30),
                static_cast<double>(max_memory) / (1 << 30)));
        }
        cfg.max_memory_bytes = max_memory - fixed;
        engine               = make_engine(cfg, smear, exec);
        // The fixed part is an estimate: with an automatic chunk, step it
        // down while the measured total is over the budget.
        SizeType used = engine->memory_usage().total();
        while (used > max_memory && chunk == 0 &&
               cfg.mode == FDMTMode::kValid && engine->chunk_nsamps() > 256) {
            const SizeType over  = used - max_memory;
            cfg.chunk_nsamps     = engine->chunk_nsamps() / 2;
            cfg.max_memory_bytes = cfg.max_memory_bytes > over
                                       ? cfg.max_memory_bytes - over
                                       : cfg.max_memory_bytes;
            engine.reset();
            engine = make_engine(cfg, smear, exec);
            used   = engine->memory_usage().total();
        }
        if (used > max_memory) {
            throw std::invalid_argument(fmt::format(
                "FDMTBoxcar: this configuration needs {:.2f} GiB, over "
                "max_memory_bytes ({:.2f} GiB)",
                static_cast<double>(used) / (1 << 30),
                static_cast<double>(max_memory) / (1 << 30)));
        }
    }
    std::unique_ptr<detail::FDMTBoxcarEngine> make_engine(
        const detail::FDMTEngineConfig& cfg, bool /*smear*/, Exec ex) const {
        if (ex.backend == Backend::kCPU) {
            return detail::make_fdmt_boxcar_cpu(
                plan, config, planes, cfg,
                detail::make_boxcar_driver(detail::make_fdmt_cpu(plan, cfg)));
        }
#ifdef DMT_ENABLE_GPU
        if (ex.backend == detail::kGPUBackend) {
            return detail::make_fdmt_boxcar_gpu(
                plan, config, planes, cfg,
                detail::make_boxcar_driver(detail::make_fdmt_gpu(plan, cfg)));
        }
#endif
        detail::throw_unavailable("FDMTBoxcar", ex.backend);
    }
    /// Carry, peaks and candidate storage of the search (bytes).
    [[nodiscard]] SizeType boxcar_bytes(bool smear, Exec ex) const {
        SizeType rows = 0;
        for (const auto& p : planes) {
            rows += p.dm_grid.size();
        }
        rows *= beams;
        const SizeType width   = planes.front().widths.back();
        const SizeType filters = planes.front().widths.size();
        const SizeType carry_elem =
            config.input_nbits == 32 ? sizeof(float)
            : detail::int_search_fits_u32(plan, smear, config.input_nbits)
                ? sizeof(uint32_t)
                : sizeof(uint64_t);
        const SizeType peaks =
            config.peak_mode == PeakFindingMode::kContiguousPeakPerWidth
                ? filters
                : 1;
        const SizeType cands = config.max_candidates * sizeof(RawCandidate) *
                               (ex.backend == Backend::kCPU ? 1 : 2);
        // Per row and width: threshold and inverse sigma (float), plus the
        // integer mean and cutoff of packed input; the row table (host and,
        // on a GPU, device).
        const SizeType luts =
            rows * filters *
            ((2 * sizeof(float)) + (config.input_nbits == 32
                                        ? 0
                                        : sizeof(double) + sizeof(uint64_t)));
        const SizeType row_table = rows * sizeof(detail::BoxcarRow) *
                                   (ex.backend == Backend::kCPU ? 1 : 2);
        return (rows * width * carry_elem) +
               (rows * peaks * sizeof(detail::BoxcarPeak)) + cands + luts +
               row_table;
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
                       SizeType chunk,
                       SizeType max_memory)
    : m_impl(std::make_unique<Impl>(std::move(plan),
                                    config,
                                    smear,
                                    exec,
                                    beams,
                                    fuse,
                                    chunk,
                                    max_memory)) {}
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
    const auto path = m_impl->engine->search_path();
    return fmt::format(
        "{}*** FDMTBoxcar ({}) ***\n{} beams, {} planes, {} widths, chunk {}, "
        "fuse_levels {}\n{}Allocated {:.2f} MiB; no root DM-time output\n",
        get_plan().summary(), to_string(backend()), get_nbeams(),
        m_impl->planes.size(), m_impl->planes.front().widths.size(),
        get_chunk_nsamps(), get_fuse_levels(),
        path.empty() ? std::string{} : fmt::format("Transform: {}\n", path),
        static_cast<double>(get_memory_usage().total()) / 1048576.0);
}

} // namespace dmt::algorithms
