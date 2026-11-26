#include "dmt/algorithms/fdmt_boxcar.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "dmt/bit_pack_utils.hpp"
#include "dmt/boxcar.hpp"
#include "dmt/boxcar_drain.hpp"
#include "dmt/gpu_search_list.hpp"

namespace dmt::algorithms {

namespace {

/// Lightweight driver for deferred boxcar-only memory sizing (no FDMT built).
class SizingBoxcarDriver final : public detail::BoxcarDriver {
public:
    SizingBoxcarDriver(SizeType chunk, uint64_t latency)
        : m_chunk(chunk),
          m_latency(latency) {}

    void search(std::span<const float> /*wf*/,
                detail::FDMTSearchSink& /*sink*/) override {}
    void search(DeviceSpan<const float> /*wf*/,
                detail::FDMTSearchSink& /*sink*/,
                Stream /*stream*/) override {}
    void reset(Stream /*stream*/) override {}
    void allocate() override {}
    [[nodiscard]] bool stages_host_input() const noexcept override {
        return true;
    }
    bool set_search_keep(const uint8_t* /*keep*/, float /*fill*/) override {
        return true;
    }
    [[nodiscard]] FDMTMemoryUsage memory() const noexcept override {
        return {};
    }
    [[nodiscard]] SizeType chunk_nsamps() const noexcept override {
        return m_chunk;
    }
    [[nodiscard]] SizeType fuse_levels() const noexcept override { return 0; }
    [[nodiscard]] uint64_t search_latency() const noexcept override {
        return m_latency;
    }

private:
    SizeType m_chunk;
    uint64_t m_latency;
};

} // namespace

class FDMTBoxcar::Impl {
public:
    struct BuildResult {
        bool ok{false};
        SizeType needed{0};
        bool smaller_path{false};
        std::string error;
    };

    Impl(plans::FDMTPlan plan,
         BoxcarConfig config,
         bool smear,
         Exec exec,
         SizeType beams,
         SizeType fuse,
         SizeType chunk,
         SizeType max_memory,
         std::span<const uint8_t> kill,
         bool build_now     = true,
         bool estimate_only = false)
        : plan(std::move(plan)),
          config(config),
          exec(exec),
          beams(beams),
          smear(smear),
          fuse(fuse),
          chunk(chunk),
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
        // Declared per-call masks: their model is checked and its buffers
        // reserved before the FDMT chooses its path.
        if (config.per_call_masks) {
            per_call_model = make_noise_model();
        }
        // The boxcar's own buffers are sized directly by a deferred engine
        // (no mirrored formulas); the FDMT takes its path within the rest.
        fixed = compute_fixed_bytes();
        if (build_now) {
            const auto res = try_build(max_memory);
            if (!res.ok) {
                throw std::invalid_argument(res.error);
            }
            finalize_build(estimate_only);
        }
    }

    /// Tries to pick and build (deferred) an engine within `budget`. Never
    /// throws for a budget miss (non-budget errors still throw).
    BuildResult try_build(SizeType budget) {
        engine.reset();
        max_memory = budget;
        if (fixed >= budget) {
            return {
                .ok     = false,
                .needed = fixed + 1,
                .error  = fmt::format(
                    "FDMTBoxcar: the boxcar buffers alone need {:.2f} GiB, "
                    "over max_memory_bytes ({:.2f} GiB); lower max_boxcar or "
                    "max_candidates (or max_cands_per_dm with kKeepBrightest), "
                    "or raise the budget",
                    static_cast<double>(fixed) / (1 << 30),
                    static_cast<double>(budget) / (1 << 30)),
            };
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
            .max_fuse_levels =
                std::min(plan.get_niters() - 1,
                         plan.get_niters() - plan.get_subband_depth()),
            .search_upper_levels = plan.get_subband_depth() > 0,
            .search_nbits        = config.input_nbits,
            .max_memory_bytes    = budget - fixed,
            .search_only         = true,
            // Candidates are sized, not allocated: only the chosen one is.
            .defer_allocation = true,
        };
        const auto over = [&](SizeType needed, bool smaller_path) {
            return BuildResult{
                .ok           = false,
                .needed       = needed,
                .smaller_path = smaller_path,
                .error        = fmt::format(
                    "FDMTBoxcar: this configuration needs {:.2f} GiB, "
                    "over max_memory_bytes ({:.2f} GiB)",
                    static_cast<double>(needed) / (1 << 30),
                    static_cast<double>(budget) / (1 << 30)),
            };
        };
        // The engine, with the declared masks' buffers sized (deferred), and
        // the budgeted bytes of both.
        const auto build = [&](const detail::FDMTEngineConfig& ecfg,
                               SizeType& used_out,
                               SizeType* fdmt_bytes = nullptr) -> BuildResult {
            auto eng = try_make_engine(ecfg, fdmt_bytes);
            if (!eng.engine) {
                return {.ok     = false,
                        .needed = eng.needed,
                        .error  = std::move(eng.error)};
            }
            engine = std::move(eng.engine);
            if (per_call_model) {
                engine->enable_block_noise(per_call_model->support());
            }
            used_out =
                engine->memory_usage().budgeted() +
                (per_call_model && exec.backend == Backend::kCPU
                     ? model_bytes(*per_call_model, ring(*per_call_model))
                     : SizeType{0});
            return {.ok = true, .needed = used_out, .error = {}};
        };
        // The fixed part is an estimate: with an automatic chunk, step it
        // down while the measured total is over the budget. An FDMT over
        // its share reports what the whole engine would need.
        const auto select =
            [&](const detail::FDMTEngineConfig& start) -> BuildResult {
            cfg                 = start;
            SizeType used       = 0;
            SizeType fdmt_bytes = 0;
            SizeType boxcar_est = fixed;
            auto fail_fdmt      = [&](const BuildResult& r) {
                const SizeType need =
                    r.needed > std::numeric_limits<SizeType>::max() - boxcar_est
                        ? std::numeric_limits<SizeType>::max()
                        : boxcar_est + r.needed;
                return over(std::max(budget + 1, need), false);
            };
            auto res = build(cfg, used, &fdmt_bytes);
            if (!res.ok) {
                return fail_fdmt(res);
            }
            if (used > budget && chunk == 0 && cfg.mode == FDMTMode::kValid &&
                exec.backend == Backend::kCPU &&
                detail::is_skewed_path(engine->path())) {
                const SizeType boxcar_actual =
                    used > fdmt_bytes ? used - fdmt_bytes : SizeType{0};
                boxcar_est = std::max(boxcar_est, boxcar_actual);
                if (boxcar_actual < budget) {
                    const SizeType exact_budget = budget - boxcar_actual;
                    if (exact_budget < cfg.max_memory_bytes) {
                        cfg.max_memory_bytes = exact_budget;
                        engine.reset();
                        res = build(cfg, used, &fdmt_bytes);
                        if (!res.ok) {
                            return fail_fdmt(res);
                        }
                    }
                }
            }
            while (used > budget && chunk == 0 &&
                   cfg.mode == FDMTMode::kValid &&
                   engine->chunk_nsamps() > 256) {
                const SizeType excess = used - budget;
                cfg.chunk_nsamps      = engine->chunk_nsamps() / 2;
                cfg.max_memory_bytes  = cfg.max_memory_bytes > excess
                                            ? cfg.max_memory_bytes - excess
                                            : cfg.max_memory_bytes;
                engine.reset();
                res = build(cfg, used, &fdmt_bytes);
                if (!res.ok) {
                    return fail_fdmt(res);
                }
            }
            if (used > budget) {
                // The FDMT fit its share on its larger path (CPU tree, GPU
                // unskewed direct levels), the whole engine did not.
                return over(used, !detail::is_skewed_path(engine->path()));
            }
            return {.ok = true, .needed = used, .error = {}};
        };
        // GPU subband search with an automatic chunk: the first
        // configuration that fits of a simple ordered rule
        // (detail::gpu_search_list). Each candidate is built deferred with
        // its path and chunk fixed, so nothing it decides depends on the
        // budget; the facade compares the whole engine's bytes.
        const bool listed = exec.backend != Backend::kCPU && chunk == 0 &&
                            cfg.mode == FDMTMode::kValid &&
                            cfg.search_upper_levels;
        const auto select_listed = [&]() -> BuildResult {
            const auto base = cfg;
#ifdef DMT_ENABLE_GPU
            const SizeType cache = detail::fdmt_gpu_cache_chunk(plan, base);
#else
            const SizeType cache = 0; // no GPU engine (try_make_engine throws)
#endif
            const auto list = detail::gpu_search_list(cache, config.max_boxcar);
            std::optional<detail::GPUSearchEntry> last; // the engine built
            SizeType last_bytes = 0;
            const auto build_at = [&](const detail::GPUSearchEntry& e)
                -> std::optional<SizeType> {
                cfg                  = base;
                cfg.chunk_nsamps     = e.chunk;
                cfg.path             = e.path;
                cfg.max_memory_bytes = std::numeric_limits<SizeType>::max();
                engine.reset();
                last.reset();
                SizeType bytes = 0;
                const auto res = build(cfg, bytes);
                if (!res.ok) {
                    return std::nullopt;
                }
                last_bytes = bytes;
                last       = e;
                return last_bytes;
            };
            // An entry this plan cannot run (e.g. a 32-bit extent limit of
            // its path or chunk) is skipped, as on the automatic path.
            std::optional<std::invalid_argument> unavailable;
            const auto measure = [&](const detail::GPUSearchEntry& e)
                -> std::optional<SizeType> {
                try {
                    return build_at(e);
                } catch (const std::invalid_argument& err) {
                    engine.reset();
                    last.reset();
                    if (!unavailable) {
                        unavailable.emplace(err);
                    }
                    return std::nullopt;
                }
            };
            const auto choice = detail::choose_gpu_search(
                list, budget, plan.get_nsamps(), measure);
            if (!choice.index) {
                if (choice.min_needed == std::numeric_limits<SizeType>::max() &&
                    unavailable) {
                    throw *unavailable; // no listed entry can run
                }
                return over(std::max(budget + 1, choice.min_needed), false);
            }
            auto pick  = list[*choice.index];
            pick.chunk = std::min(pick.chunk, plan.get_nsamps());
            if (last != pick) {
                const auto rebuilt = build_at(pick);
                if (!rebuilt || *rebuilt > budget) {
                    throw std::logic_error(
                        "FDMTBoxcar: the chosen configuration grew on rebuild");
                }
            }
            return {.ok = true, .needed = last_bytes, .error = {}};
        };

        // Over the budget on the larger path: the smaller path may fit.
        const auto first_cfg = cfg;
        BuildResult first    = listed ? select_listed() : select(first_cfg);
        if (first.ok || listed || !first.smaller_path ||
            first_cfg.path != detail::FDMTPath::kAuto) {
            return first;
        }
        auto smaller = first_cfg;
        smaller.path = detail::FDMTPath::kSkewed;
        engine.reset();
        try {
            BuildResult second = select(smaller);
            if (second.ok) {
                return second;
            }
            return second.needed < first.needed ? second : first;
        } catch (const std::invalid_argument&) {
            return first; // the smaller path is not available here
        }
    }

    void finalize_build(bool estimate_only) {
        if (!estimate_only) {
            engine->allocate();
        }
        if (per_call_model) {
            noise           = std::move(per_call_model);
            reserved_tables = table_sets(ring(*noise));
            prev_mask       = ctor_mask;
            prev_runs       = kill_runs;
            live.reset(ctor_mask, 0);
        }
    }

    /// estimate_memory()'s result for this (estimate-only) instance.
    [[nodiscard]] FDMTBoxcarMemoryEstimate estimate() const {
        return {.usage            = usage(),
                .chunk_nsamps     = engine->chunk_nsamps(),
                .search_path      = engine->search_path(),
                .search_latency   = engine->search_latency(),
                .max_memory_bytes = max_memory};
    }
    /// get_memory_usage() (and the estimate): the engine's, plus the
    /// per-call masks' model. Declared masks report their reserved table
    /// sets, so the usage is fixed from construction; otherwise the table
    /// sets alive.
    [[nodiscard]] FDMTBoxcarMemoryUsage usage() const noexcept {
        auto u = engine->memory_usage();
        const detail::BoxcarNoiseModel* m =
            noise ? noise.get() : per_call_model.get();
        if (m != nullptr) {
            SizeType tables = 0;
            if (noise) {
                tables = reserved_tables > 0
                             ? reserved_tables * noise->table_bytes()
                             : noise->live_table_bytes();
            } else {
                tables = table_sets(ring(*m)) * m->table_bytes();
            }
            const SizeType plan_bytes = m->plan_bytes() + tables;
            u.plan += plan_bytes;
            u.workspace += m->workspace_bytes();
            if (exec.backend != Backend::kCPU) {
                u.host += plan_bytes + m->workspace_bytes();
            }
        }
        return u;
    }
    /// Samples a window reaches back over at most: the largest row support
    /// plus the widest box, less one.
    [[nodiscard]] SizeType reach(const detail::BoxcarNoiseModel& m) const {
        return m.max_support() + planes.front().widths.back() - 1;
    }
    /// Block entries the engine keeps for `latency`: the blocks a call's
    /// endpoints lie in (1 + ceil(latency / nsamps)), capped at their count
    /// for blocks of `reach` samples (a GPU: 2). Shorter blocks fold the
    /// oldest entries into one catch-all (BlockNoise::catch_all).
    [[nodiscard]] std::pair<SizeType, bool>
    ring(uint64_t latency, uint64_t reach, Exec ex) const noexcept {
        const uint64_t ns = plan.get_nsamps();
        const auto blocks = [&](uint64_t n) {
            return 1 + static_cast<SizeType>((latency + n - 1) / n);
        };
        const SizeType all = blocks(ns);
        const SizeType cap = ex.backend == Backend::kCPU
                                 ? blocks(std::max<uint64_t>(reach, 1))
                                 : SizeType{2};
        return {std::min(all, cap), all > cap};
    }
    [[nodiscard]] std::pair<SizeType, bool>
    ring(const detail::BoxcarNoiseModel& m) const noexcept {
        return ring(engine->search_latency(), reach(m), exec);
    }
    /// Table sets alive at once: steady and transition of each ring entry,
    /// the mask cache, and the latest catch-all.
    [[nodiscard]] static SizeType
    table_sets(std::pair<SizeType, bool> ring) noexcept {
        return (2 * ring.first) + kCacheMasks + (ring.second ? 1 : 0);
    }
    /// Host bytes of the noise model with its table sets.
    [[nodiscard]] static SizeType
    model_bytes(const detail::BoxcarNoiseModel& model,
                std::pair<SizeType, bool> ring) noexcept {
        return model.plan_bytes() + model.workspace_bytes() +
               (table_sets(ring) * model.table_bytes());
    }
    /// The per-call mask model, after the checks a first per-call mask
    /// makes. Packed input: channels a per-call mask kills read the code
    /// input_mean (an integer), so the model keeps the constructor mask's
    /// mean (`reference`).
    [[nodiscard]] std::unique_ptr<detail::BoxcarNoiseModel>
    make_noise_model() const {
        const bool packed = config.input_nbits != 32;
        return std::make_unique<detail::BoxcarNoiseModel>(
            plan, config, smear, std::max(1, exec.nthreads),
            packed ? std::span<const uint8_t>(ctor_mask)
                   : std::span<const uint8_t>{});
    }
    /// Budgeted bytes of the boxcar engine itself (and declared per-call
    /// masks at zero search latency: an exact lower bound for every path,
    /// refined with the chosen engine's actual bytes if a lagged path needs
    /// extra mask ring entries), obtained from a deferred engine instance
    /// without constructing FDMT.
    [[nodiscard]] SizeType compute_fixed_bytes() const {
        const SizeType sizing_chunk =
            chunk != 0 ? chunk : std::min<SizeType>(plan.get_nsamps(), 4096);
        const detail::FDMTEngineConfig cfg{
            .use_box_smearing = smear,
            .mode             = FDMTMode::kValid,
            .nbeams           = beams,
            .exec             = exec,
            .chunk_nsamps     = sizing_chunk,
            .search_nbits     = config.input_nbits,
            .search_only      = true,
            .defer_allocation = true,
        };
        auto driver = std::make_unique<SizingBoxcarDriver>(sizing_chunk, 0);
        std::unique_ptr<detail::FDMTBoxcarEngine> sizing;
        if (exec.backend == Backend::kCPU) {
            sizing = detail::make_fdmt_boxcar_cpu(plan, config, planes, cfg,
                                                  std::move(driver), kill_runs);
        }
#ifdef DMT_ENABLE_GPU
        else if (exec.backend == detail::kGPUBackend) {
            sizing = detail::make_fdmt_boxcar_gpu(plan, config, planes, cfg,
                                                  std::move(driver), kill_runs);
        }
#endif
        else {
            detail::throw_unavailable("FDMTBoxcar", exec.backend);
        }
        SizeType bytes = sizing->memory_usage().budgeted();
        if (per_call_model) {
            bytes += sizing->block_noise_bytes();
            if (exec.backend == Backend::kCPU) {
                bytes += model_bytes(*per_call_model,
                                     ring(0, reach(*per_call_model), exec));
            }
        }
        return bytes;
    }
    struct EngineAttempt {
        std::unique_ptr<detail::FDMTBoxcarEngine> engine;
        SizeType needed{0};
        std::string error;
    };
    EngineAttempt try_make_engine(const detail::FDMTEngineConfig& cfg,
                                  SizeType* fdmt_bytes = nullptr) const {
        if (exec.backend == Backend::kCPU) {
            auto fdmt = detail::try_make_fdmt_cpu(plan, cfg);
            if (!fdmt.engine) {
                return {.engine = nullptr,
                        .needed = fdmt.needed,
                        .error  = std::move(fdmt.error)};
            }
            if (fdmt_bytes != nullptr) {
                *fdmt_bytes = fdmt.engine->get_memory_usage().total();
            }
            return {.engine = detail::make_fdmt_boxcar_cpu(
                        plan, config, planes, cfg,
                        detail::make_boxcar_driver(std::move(fdmt.engine)),
                        kill_runs),
                    .needed = 0,
                    .error  = {}};
        }
#ifdef DMT_ENABLE_GPU
        if (exec.backend == detail::kGPUBackend) {
            auto fdmt = detail::try_make_fdmt_gpu(plan, cfg);
            if (!fdmt.engine) {
                return {.engine = nullptr,
                        .needed = fdmt.needed,
                        .error  = std::move(fdmt.error)};
            }
            if (fdmt_bytes != nullptr) {
                *fdmt_bytes = fdmt.engine->get_memory_usage().total();
            }
            return {.engine = detail::make_fdmt_boxcar_gpu(
                        plan, config, planes, cfg,
                        detail::make_boxcar_driver(std::move(fdmt.engine)),
                        kill_runs),
                    .needed = 0,
                    .error  = {}};
        }
#endif
        detail::throw_unavailable("FDMTBoxcar", exec.backend);
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
            const SizeType row =
                config.input_nbits == 32
                    ? plan.get_nsamps()
                    : bit_pack_utils::packed_row_bytes(plan.get_nsamps(),
                                                       config.input_nbits);
            if (waterfall_size != beams * plan.get_nchans() * row) {
                throw std::invalid_argument(
                    config.input_nbits == 32
                        ? "FDMTBoxcar: wrong waterfall size"
                        : "FDMTBoxcar: packed input size or bit depth "
                          "mismatch");
            }
            enable_block_noise();
        }
        effective = ctor_mask;
        for (SizeType c = 0; c < mask.size(); ++c) {
            effective[c] = (ctor_mask[c] != 0 && mask[c] != 0) ? 1 : 0;
        }
        detail::BlockNoise entry;
        const uint64_t first = engine->stream_position();
        entry.first_sample   = first;
        entry.steady         = tables_for(effective);
        const bool changed   = effective != prev_mask;
        // Windows that reach before the block: the history mask; blocks
        // shorter than the search latency needs: the catch-all of the
        // entries older than the engine keeps (detail::block_masks()).
        detail::block_masks(live, effective, prev_mask, first, reach(*noise),
                            engine->noise_catch_all_begin(), block);
        if (block.transition) {
            entry.transition        = true;
            entry.transition_tables = tables_for(block.history);
        }
        if (block.catch_all) {
            entry.catch_all     = tables_for(block.catch_mask);
            entry.has_catch_all = true;
        }
        // An unchanged mask reuses its kill runs instead of rebuilding them.
        if (changed) {
            next_runs = detail::boxcar_kill_runs(effective);
        }
        entry.kill_runs = changed ? next_runs : prev_runs;
        engine->push_block_noise(std::move(entry));
        try {
            auto view = call();
            live.update(effective, first + plan.get_nsamps());
            if (changed) {
                prev_mask.swap(effective);
                prev_runs.swap(next_runs);
            }
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
    /// First per-call mask: memory budget, then
    /// the noise model and the engine's block ring. Throws before any
    /// state change.
    void enable_block_noise() {
        auto model = make_noise_model();
        // The noise model and its table sets live in host memory: budgeted
        // on the CPU only (on a GPU, max_memory_bytes bounds device memory).
        const SizeType extra =
            (exec.backend == Backend::kCPU ? model_bytes(*model, ring(*model))
                                           : 0) +
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
        prev_runs = kill_runs;
        cache.clear();
        // Enabled mid-stream: the constructor mask's channels were live in
        // every earlier block.
        live.reset(ctor_mask, engine->stream_position());
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
    SizeType fuse;
    SizeType chunk;
    SizeType max_memory;
    SizeType fixed = 0;
    std::vector<plans::BoxcarPlanePlan> planes;
    std::vector<std::pair<SizeType, SizeType>> kill_runs;
    std::unique_ptr<detail::FDMTBoxcarEngine> engine;
    detail::DrainState drain_state;
    // Per-call kill masks (0/1 per channel): the constructor mask, the
    // previous block's effective mask, scratch for the next, and the model
    // with its recent tables (created by the first per-call mask).
    std::vector<uint8_t> ctor_mask;
    std::vector<uint8_t> prev_mask;
    /// Kill runs of prev_mask, and of a changed mask until its call ends.
    std::vector<std::pair<SizeType, SizeType>> prev_runs, next_runs;
    std::vector<uint8_t> effective;
    detail::BlockMasks block;
    detail::LiveHistory live;
    std::unique_ptr<detail::BoxcarNoiseModel> per_call_model;
    std::unique_ptr<detail::BoxcarNoiseModel> noise;
    /// BoxcarConfig::per_call_masks: the table sets reserved (0: undeclared,
    /// the live sets are reported).
    SizeType reserved_tables = 0;
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
FDMTBoxcarMemoryEstimate
FDMTBoxcar::estimate_memory(const plans::FDMTPlan& plan,
                            BoxcarConfig config,
                            bool use_box_smearing,
                            Exec exec,
                            SizeType nbeams,
                            SizeType fuse_levels,
                            SizeType chunk_nsamps,
                            SizeType max_memory_bytes,
                            std::span<const uint8_t> kill_mask) {
    const Impl impl(plan, config, use_box_smearing, exec, nbeams, fuse_levels,
                    chunk_nsamps, max_memory_bytes, kill_mask,
                    /*build_now=*/true, /*estimate_only=*/true);
    return impl.estimate();
}
FDMTBoxcarMemoryEstimate
FDMTBoxcar::estimate_minimum_memory(const plans::FDMTPlan& plan,
                                    BoxcarConfig config,
                                    bool use_box_smearing,
                                    Exec exec,
                                    SizeType nbeams,
                                    SizeType fuse_levels,
                                    SizeType chunk_nsamps,
                                    std::span<const uint8_t> kill_mask) {
    // Raise the budget to what each failed build says it needs (a lower
    // bound of the smallest budget that builds) until one builds: the
    // boxcar's fixed bytes first, then the FDMT's smallest path.
    // The row tables and declared noise model do not depend on the budget:
    // constructed and validated once on a single Impl.
    Impl impl(plan, config, use_box_smearing, exec, nbeams, fuse_levels,
              chunk_nsamps, /*max_memory=*/0, kill_mask,
              /*build_now=*/false, /*estimate_only=*/true);
    // Each failure's `needed` is exact for the path it tried, except when
    // the boxcar's chunk loop shrank the FDMT's share: then step up (at
    // least 1 MiB) and, once a budget builds, narrow it down below.
    SizeType budget = 1;
    SizeType failed = 0; // largest budget known not to build
    bool stepped    = false;
    std::optional<FDMTBoxcarMemoryEstimate> found;
    for (int attempt = 0; attempt < 64 && !found; ++attempt) {
        const auto res = impl.try_build(budget);
        if (res.ok) {
            found = impl.estimate();
        } else {
            failed = budget;
            if (res.needed > budget) {
                budget = res.needed;
            } else {
                stepped = true;
                budget += std::max<SizeType>(budget / 1024, 1 << 20);
            }
        }
    }
    if (!found) {
        throw std::logic_error("FDMTBoxcar::estimate_minimum_memory: the "
                               "budget search did not converge");
    }
    // A failure's `needed` adds the boxcar's estimated bytes to the FDMT's
    // need, so the budget found can exceed the minimum (GPU). Its own
    // configuration's usage is then usually the minimum; on the CPU the
    // path choice keeps headroom, so the usage does not build and the
    // budget found (exact there) stands, unless the search stepped: then
    // bisect between the last failure and the budget found.
    const SizeType usage = found->usage.budgeted();
    if (found->max_memory_bytes == usage ||
        (exec.backend == Backend::kCPU && !stepped)) {
        return *found;
    }
    if (impl.try_build(usage).ok) {
        return impl.estimate();
    }
    if (!stepped) {
        return *found;
    }
    SizeType hi = found->max_memory_bytes;
    SizeType lo = std::max(failed, usage);
    while (hi - lo > 1) {
        const SizeType mid = lo + ((hi - lo) / 2);
        if (impl.try_build(mid).ok) {
            found = impl.estimate();
            hi    = mid;
        } else {
            lo = mid;
        }
    }
    return *found;
}
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
    return execute(waterfall, std::span<const uint8_t>{});
}
CandidateResultView FDMTBoxcar::execute(std::span<const uint8_t> waterfall,
                                        std::span<const uint8_t> kill_mask) {
    m_impl->drain_state.check_open("FDMTBoxcar::execute");
    return m_impl->run_masked(kill_mask, waterfall.size(), [&] {
        return m_impl->engine->execute(waterfall);
    });
}
CandidateDeviceResultView
FDMTBoxcar::execute(DeviceSpan<const uint8_t> waterfall, Stream stream) {
    return execute(waterfall, std::span<const uint8_t>{}, stream);
}
CandidateDeviceResultView
FDMTBoxcar::execute(DeviceSpan<const uint8_t> waterfall,
                    std::span<const uint8_t> kill_mask,
                    Stream stream) {
    m_impl->drain_state.check_open("FDMTBoxcar::execute");
    detail::check_device(waterfall.device, backend(), device(),
                         "FDMTBoxcar::execute");
    return m_impl->run_masked(kill_mask, waterfall.size(), [&] {
        return m_impl->engine->execute(waterfall, stream);
    });
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
        m_impl->prev_runs = m_impl->kill_runs;
        m_impl->live.reset(m_impl->ctor_mask, 0);
    }
}
CandidateResultView FDMTBoxcar::drain() {
    return detail::drain_stream(
        *m_impl->engine, m_impl->plan, m_impl->beams, m_impl->config,
        m_impl->planes, m_impl->engine->search_latency(), m_impl->drain_state);
}

namespace {

void check_tail(const plans::FDMTPlan& plan, SizeType n_tail) {
    if (n_tail == 0 || n_tail > plan.get_nsamps()) {
        throw std::invalid_argument(fmt::format(
            "FDMTBoxcar::drain: n_tail must be 1..nsamps ({}), got {}",
            plan.get_nsamps(), n_tail));
    }
}

} // namespace

CandidateResultView FDMTBoxcar::drain(std::span<const float> tail,
                                      SizeType n_tail) {
    check_tail(m_impl->plan, n_tail);
    return detail::drain_stream(
        *m_impl->engine, m_impl->plan, m_impl->beams, m_impl->config,
        m_impl->planes, m_impl->engine->search_latency(), m_impl->drain_state,
        n_tail, [&] {
            return m_impl->run_masked(
                {}, tail.size(), [&] { return m_impl->engine->execute(tail); });
        });
}
CandidateResultView FDMTBoxcar::drain(std::span<const uint8_t> tail,
                                      SizeType n_tail) {
    check_tail(m_impl->plan, n_tail);
    return detail::drain_stream(
        *m_impl->engine, m_impl->plan, m_impl->beams, m_impl->config,
        m_impl->planes, m_impl->engine->search_latency(), m_impl->drain_state,
        n_tail, [&] {
            return m_impl->run_masked(
                {}, tail.size(), [&] { return m_impl->engine->execute(tail); });
        });
}
CandidateResultView FDMTBoxcar::drain(DeviceSpan<const float> tail,
                                      SizeType n_tail,
                                      Stream stream) {
    check_tail(m_impl->plan, n_tail);
    detail::check_device(tail.device, backend(), device(), "FDMTBoxcar::drain");
    return detail::drain_stream(
        *m_impl->engine, m_impl->plan, m_impl->beams, m_impl->config,
        m_impl->planes, m_impl->engine->search_latency(), m_impl->drain_state,
        n_tail, [&] {
            static_cast<void>(m_impl->run_masked({}, tail.size(), [&] {
                return m_impl->engine->execute(tail, stream);
            }));
            return m_impl->engine->copy_result(stream);
        });
}
CandidateResultView FDMTBoxcar::drain(DeviceSpan<const uint8_t> tail,
                                      SizeType n_tail,
                                      Stream stream) {
    check_tail(m_impl->plan, n_tail);
    detail::check_device(tail.device, backend(), device(), "FDMTBoxcar::drain");
    return detail::drain_stream(
        *m_impl->engine, m_impl->plan, m_impl->beams, m_impl->config,
        m_impl->planes, m_impl->engine->search_latency(), m_impl->drain_state,
        n_tail, [&] {
            static_cast<void>(m_impl->run_masked({}, tail.size(), [&] {
                return m_impl->engine->execute(tail, stream);
            }));
            return m_impl->engine->copy_result(stream);
        });
}
uint64_t FDMTBoxcar::get_searched_nsamps() const noexcept {
    return detail::searched_nsamps(*m_impl->engine, m_impl->drain_state,
                                   m_impl->engine->search_latency());
}
uint64_t FDMTBoxcar::get_candidate_frontier() const noexcept {
    // New excursions begin at endpoints not yet searched by every row;
    // candidates report them less the plan's level-0 causal delay.
    const uint64_t raw =
        std::min(get_searched_nsamps(), m_impl->engine->open_begin_min());
    const uint64_t lag = get_plan().get_level0_causal_delay();
    return raw > lag ? raw - lag : 0;
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
    return m_impl->usage();
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
