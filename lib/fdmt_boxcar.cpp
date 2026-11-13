#include "dmt/algorithms/fdmt_boxcar.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/boxcar.hpp"
#include "dmt/boxcar_drain.hpp"
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
         SizeType max_memory,
         std::span<const uint8_t> kill)
        : plan(std::move(plan)),
          config(config),
          exec(exec),
          beams(beams),
          smear(smear),
          max_memory(max_memory),
          // Validates the mask's size; killed channels leave the noise model.
          planes(detail::make_boxcar_plans(this->plan, config, smear, kill)),
          kill_runs(detail::boxcar_kill_runs(kill)),
          ctor_mask(this->plan.get_nchans(), 1) {
        for (SizeType c = 0; c < kill.size(); ++c) {
            ctor_mask[c] = kill[c] != 0 ? 1 : 0;
        }
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
        cfg.search_only = true;
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
        SizeType used = engine->memory_usage().budgeted();
        while (used > max_memory && chunk == 0 &&
               cfg.mode == FDMTMode::kValid && engine->chunk_nsamps() > 256) {
            const SizeType over  = used - max_memory;
            cfg.chunk_nsamps     = engine->chunk_nsamps() / 2;
            cfg.max_memory_bytes = cfg.max_memory_bytes > over
                                       ? cfg.max_memory_bytes - over
                                       : cfg.max_memory_bytes;
            engine.reset();
            engine = make_engine(cfg, smear, exec);
            used   = engine->memory_usage().budgeted();
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
                detail::make_boxcar_driver(detail::make_fdmt_cpu(plan, cfg)),
                kill_runs);
        }
#ifdef DMT_ENABLE_GPU
        if (ex.backend == detail::kGPUBackend) {
            return detail::make_fdmt_boxcar_gpu(
                plan, config, planes, cfg,
                detail::make_boxcar_driver(detail::make_fdmt_gpu(plan, cfg)),
                kill_runs);
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
        // Budgeted bytes only: on a GPU the device copies (the host result
        // storage and host row table are reported, not budgeted).
        const SizeType cands = config.max_candidates * sizeof(RawCandidate);
        // Per row and width: threshold and inverse sigma (float), plus the
        // integer mean and cutoff of packed input; the row table.
        const SizeType luts =
            rows * filters *
            ((2 * sizeof(float)) + (config.input_nbits == 32
                                        ? 0
                                        : sizeof(double) + sizeof(uint64_t)));
        const SizeType row_table = rows * sizeof(detail::BoxcarRow);
        // A kill mask: on the CPU its channels are zeroed in a staging copy
        // of each block; on a GPU the transform reads one byte per row.
        SizeType staging = 0;
        if (!kill_runs.empty()) {
            staging =
                ex.backend != Backend::kCPU
                    ? beams * plan.get_nchans()
                    : beams * plan.get_nchans() *
                          (config.input_nbits == 32
                               ? plan.get_nsamps() * sizeof(float)
                               : bit_pack_utils::packed_row_bytes(
                                     plan.get_nsamps(), config.input_nbits));
        }
        return (rows * width * carry_elem) +
               (rows * peaks * sizeof(detail::BoxcarPeak)) + cands + luts +
               row_table + staging;
    }
    /**
     * One block with a per-call kill mask (empty: the constructor mask):
     * validates, composes the effective mask (constructor AND per-call),
     * hands the engine the block's tables and kill runs, then runs `call`
     * (the engine's execute). Before any per-call mask, the old path.
     */
    template <typename Call>
    auto run_masked(std::span<const uint8_t> mask,
                    SizeType waterfall_size,
                    Call&& call) {
        if (!mask.empty() && mask.size() != plan.get_nchans()) {
            throw std::invalid_argument(fmt::format(
                "FDMTBoxcar: kill_mask must be empty or have nchans ({}) "
                "entries, got {}",
                plan.get_nchans(), mask.size()));
        }
        if (!noise) {
            if (mask.empty()) {
                return call();
            }
            if (config.input_nbits != 32) {
                throw std::invalid_argument(
                    "FDMTBoxcar: per-call kill masks need float32 input "
                    "(input_nbits 32)");
            }
            if (waterfall_size !=
                beams * plan.get_nchans() * plan.get_nsamps()) {
                throw std::invalid_argument("FDMTBoxcar: wrong waterfall size");
            }
            enable_block_noise();
        }
        effective = ctor_mask;
        for (SizeType c = 0; c < mask.size(); ++c) {
            effective[c] = (ctor_mask[c] != 0 && mask[c] != 0) ? 1 : 0;
        }
        detail::BlockNoise entry;
        entry.first_sample = engine->stream_position();
        entry.steady       = tables_for(effective);
        // The first block of a stream has no previous block to straddle.
        if (entry.first_sample > 0 && effective != prev_mask) {
            union_mask.resize(effective.size());
            for (SizeType c = 0; c < effective.size(); ++c) {
                union_mask[c] = effective[c] | prev_mask[c];
            }
            entry.transition        = true;
            entry.transition_tables = tables_for(union_mask);
        }
        entry.kill_runs = detail::boxcar_kill_runs(effective);
        engine->push_block_noise(std::move(entry));
        try {
            auto view = call();
            prev_mask.swap(effective);
            return view;
        } catch (...) {
            engine->drop_block_noise();
            throw;
        }
    }
    /// Tables of an effective mask: null for the constructor mask (the
    /// search planes), else from the cache of recent masks or computed.
    /// The cache keeps the last three masks, so the OR of a change that
    /// only kills or only revives channels (one of the two masks) is free.
    std::shared_ptr<const detail::NoiseTables>
    tables_for(const std::vector<uint8_t>& mask) {
        if (mask == ctor_mask) {
            return nullptr;
        }
        for (auto it = cache.begin(); it != cache.end(); ++it) {
            if (it->first == mask) {
                auto hit = *it;
                cache.erase(it);
                cache.push_back(hit);
                return hit.second;
            }
        }
        auto tables = noise->compute(mask);
        cache.emplace_back(mask, tables);
        if (cache.size() > kCacheMasks) {
            cache.erase(cache.begin());
        }
        return tables;
    }
    /// First per-call mask: block-size requirement and memory budget, then
    /// the noise model and the engine's block ring. Throws before any
    /// state change.
    void enable_block_noise() {
        auto model = std::make_unique<detail::BoxcarNoiseModel>(
            plan, config, smear, std::max(1, exec.nthreads));
        const SizeType max_width = planes.front().widths.back();
        const uint64_t need      = model->max_support() + max_width - 1;
        if (plan.get_nsamps() < need) {
            throw std::invalid_argument(fmt::format(
                "FDMTBoxcar: a per-call kill mask needs blocks of at least "
                "{} samples (largest row support {} + max_boxcar {} - 1), so "
                "that every endpoint's input spans at most two blocks; this "
                "plan has {}",
                need, model->max_support(), max_width, plan.get_nsamps()));
        }
        // Table sets alive at once: steady and transition of each ring
        // entry, plus the mask cache.
        const uint64_t latency = engine->search_latency();
        const SizeType ring =
            1 + static_cast<SizeType>((latency + plan.get_nsamps() - 1) /
                                      plan.get_nsamps());
        // The noise model and its table sets live in host memory: budgeted
        // on the CPU only (on a GPU, max_memory_bytes bounds device memory).
        const SizeType model_bytes =
            model->plan_bytes() + model->workspace_bytes() +
            ((2 * ring + kCacheMasks) * model->table_bytes());
        const SizeType extra =
            (exec.backend == Backend::kCPU ? model_bytes : 0) +
            engine->block_noise_bytes();
        const SizeType used = engine->memory_usage().budgeted();
        if (used + extra > max_memory) {
            throw std::invalid_argument(fmt::format(
                "FDMTBoxcar: per-call kill masks need {:.2f} GiB more, over "
                "max_memory_bytes ({:.2f} GiB, {:.2f} GiB in use)",
                static_cast<double>(extra) / (1 << 30),
                static_cast<double>(max_memory) / (1 << 30),
                static_cast<double>(used) / (1 << 30)));
        }
        engine->enable_block_noise(model->support());
        noise     = std::move(model);
        prev_mask = ctor_mask;
        cache.clear();
    }
    void check_stream(Stream stream) const {
        if (exec.backend == Backend::kCPU && stream.native != nullptr) {
            throw std::invalid_argument(
                "FDMTBoxcar: CPU backend has no stream");
        }
    }
    static constexpr SizeType kCacheMasks = 3;
    plans::FDMTPlan plan;
    BoxcarConfig config;
    Exec exec;
    SizeType beams;
    bool smear;
    SizeType max_memory;
    std::vector<plans::BoxcarPlanePlan> planes;
    std::vector<std::pair<SizeType, SizeType>> kill_runs;
    std::unique_ptr<detail::FDMTBoxcarEngine> engine;
    detail::DrainState drain_state;
    // Per-call kill masks (0/1 per channel): the constructor mask, the
    // previous block's effective mask, scratch for the next, and the model
    // with its recent tables (created by the first per-call mask).
    std::vector<uint8_t> ctor_mask;
    std::vector<uint8_t> prev_mask;
    std::vector<uint8_t> effective;
    std::vector<uint8_t> union_mask;
    std::unique_ptr<detail::BoxcarNoiseModel> noise;
    std::vector<std::pair<std::vector<uint8_t>,
                          std::shared_ptr<const detail::NoiseTables>>>
        cache;
};

FDMTBoxcar::FDMTBoxcar(plans::FDMTPlan plan,
                       BoxcarConfig config,
                       bool use_box_smearing,
                       Exec exec,
                       SizeType nbeams,
                       SizeType fuse_levels,
                       SizeType chunk_nsamps,
                       SizeType max_memory_bytes,
                       std::span<const uint8_t> kill_mask)
    : m_impl(std::make_unique<Impl>(std::move(plan),
                                    config,
                                    use_box_smearing,
                                    exec,
                                    nbeams,
                                    fuse_levels,
                                    chunk_nsamps,
                                    max_memory_bytes,
                                    kill_mask)) {}
FDMTBoxcar::~FDMTBoxcar()                                = default;
FDMTBoxcar::FDMTBoxcar(FDMTBoxcar&&) noexcept            = default;
FDMTBoxcar& FDMTBoxcar::operator=(FDMTBoxcar&&) noexcept = default;
CandidateResultView FDMTBoxcar::execute(std::span<const float> waterfall) {
    return execute(waterfall, std::span<const uint8_t>{});
}
CandidateDeviceResultView FDMTBoxcar::execute(DeviceSpan<const float> waterfall,
                                              Stream stream) {
    return execute(waterfall, std::span<const uint8_t>{}, stream);
}
CandidateResultView FDMTBoxcar::execute(std::span<const float> waterfall,
                                        std::span<const uint8_t> kill_mask) {
    m_impl->drain_state.check_open("FDMTBoxcar::execute");
    return m_impl->run_masked(kill_mask, waterfall.size(), [&] {
        return m_impl->engine->execute(waterfall);
    });
}
CandidateDeviceResultView
FDMTBoxcar::execute(DeviceSpan<const float> waterfall,
                    std::span<const uint8_t> kill_mask,
                    Stream stream) {
    m_impl->drain_state.check_open("FDMTBoxcar::execute");
    detail::check_device(waterfall.device, backend(), device(),
                         "FDMTBoxcar::execute");
    return m_impl->run_masked(kill_mask, waterfall.size(), [&] {
        return m_impl->engine->execute(waterfall, stream);
    });
}
CandidateResultView FDMTBoxcar::execute(std::span<const uint8_t> waterfall) {
    m_impl->drain_state.check_open("FDMTBoxcar::execute");
    return m_impl->engine->execute(waterfall);
}
CandidateDeviceResultView
FDMTBoxcar::execute(DeviceSpan<const uint8_t> waterfall, Stream stream) {
    m_impl->drain_state.check_open("FDMTBoxcar::execute");
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
    m_impl->drain_state.reset();
    if (m_impl->noise) {
        m_impl->prev_mask = m_impl->ctor_mask; // same size: no allocation
    }
}
CandidateResultView FDMTBoxcar::drain() {
    return detail::drain_stream(*m_impl->engine, m_impl->plan, m_impl->beams,
                                m_impl->config.input_nbits,
                                m_impl->engine->search_latency(),
                                m_impl->drain_state);
}
uint64_t FDMTBoxcar::get_searched_nsamps() const noexcept {
    return detail::searched_nsamps(*m_impl->engine, m_impl->drain_state,
                                   m_impl->engine->search_latency());
}
uint64_t FDMTBoxcar::get_candidate_frontier() const noexcept {
    // New excursions begin at endpoints not yet searched by every row.
    return std::min(get_searched_nsamps(), m_impl->engine->open_begin_min());
}
uint64_t FDMTBoxcar::get_search_latency() const noexcept {
    return m_impl->engine->search_latency();
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
    auto usage = m_impl->engine->memory_usage();
    if (m_impl->noise) {
        // The per-call mask's model and the tables alive (ring and cache).
        const SizeType plan_bytes =
            m_impl->noise->plan_bytes() + m_impl->noise->live_table_bytes();
        usage.plan += plan_bytes;
        usage.workspace += m_impl->noise->workspace_bytes();
        if (backend() != Backend::kCPU) {
            usage.host += plan_bytes + m_impl->noise->workspace_bytes();
        }
    }
    return usage;
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
