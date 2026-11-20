#include "dmt/algorithms/fdmt_boxcar.hpp"

#include <algorithm>
#include <bit>
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
#include "dmt/fdmt_int_tree.hpp"
#include "dmt/gpu_search_list.hpp"
#include "dmt/kadane_monoid.hpp"
#include "dmt/kadane_plan.hpp"
#include "dmt/kadane_simd.hpp"

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
         std::span<const uint8_t> kill,
         bool estimate_only                               = false,
         const std::vector<plans::BoxcarPlanePlan>* known = nullptr)
        : plan(std::move(plan)),
          config(config),
          exec(exec),
          beams(beams),
          smear(smear),
          max_memory(max_memory),
          // Validates the mask's size; killed channels leave the noise model.
          planes(known != nullptr ? *known
                                  : detail::make_boxcar_plans(
                                        this->plan, config, smear, kill)),
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
        // Candidates are sized, not allocated: only the chosen one is.
        cfg.defer_allocation = true;
        // Declared per-call masks: their model is checked and its buffers
        // reserved before the FDMT chooses its path.
        std::unique_ptr<detail::BoxcarNoiseModel> model;
        if (config.per_call_masks) {
            model = make_noise_model();
        }
        // The boxcar's own buffers are fixed by plan and config; the FDMT
        // takes its path within the rest of the budget.
        const SizeType fixed =
            boxcar_bytes(smear, exec) +
            (model ? mask_bytes_bound(*model, exec) : SizeType{0});
        if (fixed >= max_memory) {
            throw detail::BudgetError(
                fmt::format(
                    "FDMTBoxcar: the boxcar buffers alone need {:.2f} GiB, "
                    "over "
                    "max_memory_bytes ({:.2f} GiB); lower max_boxcar or "
                    "max_candidates (or max_cands_per_dm with kKeepBrightest), "
                    "or "
                    "raise the budget",
                    static_cast<double>(fixed) / (1 << 30),
                    static_cast<double>(max_memory) / (1 << 30)),
                fixed + 1);
        }
        cfg.max_memory_bytes = max_memory - fixed;
        // The engine, with the declared masks' buffers sized (deferred), and
        // the budgeted bytes of both.
        const auto build = [&] {
            engine = make_engine(cfg, smear, exec);
            if (model) {
                engine->enable_block_noise(model->support());
            }
            return engine->memory_usage().budgeted() +
                   (model && exec.backend == Backend::kCPU
                        ? model_bytes(*model, ring(*model))
                        : SizeType{0});
        };
        // The fixed part is an estimate: with an automatic chunk, step it
        // down while the measured total is over the budget. An FDMT over
        // its share reports what the whole engine would need.
        const auto over = [&](SizeType needed, bool smaller_path) {
            return detail::BudgetError(
                fmt::format("FDMTBoxcar: this configuration needs {:.2f} GiB, "
                            "over max_memory_bytes ({:.2f} GiB)",
                            static_cast<double>(needed) / (1 << 30),
                            static_cast<double>(max_memory) / (1 << 30)),
                needed, smaller_path);
        };
        const auto select = [&](const detail::FDMTEngineConfig& start) {
            cfg           = start;
            SizeType used = 0;
            try {
                used = build();
                while (used > max_memory && chunk == 0 &&
                       cfg.mode == FDMTMode::kValid &&
                       engine->chunk_nsamps() > 256) {
                    const SizeType excess = used - max_memory;
                    cfg.chunk_nsamps      = engine->chunk_nsamps() / 2;
                    cfg.max_memory_bytes  = cfg.max_memory_bytes > excess
                                                ? cfg.max_memory_bytes - excess
                                                : cfg.max_memory_bytes;
                    engine.reset();
                    used = build();
                }
            } catch (const detail::BudgetError& e) {
                // SIZE_MAX: no configuration of that path fits at all.
                const SizeType need =
                    e.needed() > std::numeric_limits<SizeType>::max() - fixed
                        ? std::numeric_limits<SizeType>::max()
                        : fixed + e.needed();
                throw over(std::max(max_memory + 1, need), false);
            }
            if (used > max_memory) {
                // The FDMT fit its share on its larger path (CPU tree, GPU
                // unskewed direct levels), the whole engine did not.
                throw over(used, engine->search_path().find("skewed") ==
                                     std::string::npos);
            }
            return used;
        };
        // GPU subband search with an automatic chunk: the first
        // configuration that fits of a simple ordered rule
        // (detail::gpu_search_list). Each candidate is built deferred with
        // its path and chunk fixed, so nothing it decides depends on the
        // budget; the facade compares the whole engine's bytes.
        const bool listed = exec.backend != Backend::kCPU && chunk == 0 &&
                            cfg.mode == FDMTMode::kValid &&
                            cfg.search_upper_levels;
        const auto select_listed = [&] {
            const auto base = cfg;
#ifdef DMT_ENABLE_GPU
            const SizeType cache =
                detail::fdmt_gpu_cache_chunk(this->plan, base);
#else
            const SizeType cache = 0; // no GPU engine (make_engine throws)
#endif
            const auto list = detail::gpu_search_list(cache, config.max_boxcar);
            std::optional<detail::GPUSearchEntry> last; // the engine built
            SizeType last_bytes = 0;
            const auto build_at = [&](const detail::GPUSearchEntry& e) {
                cfg                  = base;
                cfg.chunk_nsamps     = e.chunk;
                cfg.gpu_path         = e.path;
                cfg.max_memory_bytes = std::numeric_limits<SizeType>::max();
                engine.reset();
                last.reset();
                last_bytes = build();
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
                } catch (const detail::BudgetError&) {
                    throw;
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
                list, max_memory, this->plan.get_nsamps(), measure);
            if (!choice.index) {
                if (choice.min_needed == std::numeric_limits<SizeType>::max() &&
                    unavailable) {
                    throw *unavailable; // no listed entry can run
                }
                throw over(std::max(max_memory + 1, choice.min_needed), false);
            }
            auto pick  = list[*choice.index];
            pick.chunk = std::min(pick.chunk, this->plan.get_nsamps());
            if (last != pick && build_at(pick) > max_memory) {
                throw std::logic_error(
                    "FDMTBoxcar: the chosen configuration grew on rebuild");
            }
        };
        // Over the budget on the larger path: the smaller path may fit.
        const auto first_cfg = cfg;
        try {
            if (listed) {
                select_listed();
            } else {
                select(first_cfg);
            }
        } catch (const detail::BudgetError& first) {
            if (listed) {
                throw; // the list already holds the smaller path
            }
            auto smaller = first_cfg;
            if (exec.backend == Backend::kCPU) {
                smaller.skew_history_bytes =
                    detail::FDMTEngineConfig::kSkewHistoryNoTree;
            } else {
                smaller.gpu_path = detail::FDMTEngineConfig::GPUPath::kSkewed;
            }
            if (!first.smaller_path() ||
                first_cfg.skew_history_bytes !=
                    detail::FDMTEngineConfig::kSkewHistoryAuto ||
                first_cfg.gpu_path !=
                    detail::FDMTEngineConfig::GPUPath::kAuto) {
                throw;
            }
            engine.reset();
            try {
                select(smaller);
            } catch (const detail::BudgetError& e) {
                if (e.needed() < first.needed()) {
                    throw;
                }
                throw first;
            } catch (const std::invalid_argument&) {
                throw first; // the smaller path is not available here
            }
        }
        if (!estimate_only) {
            engine->allocate();
        }
        if (model) {
            noise           = std::move(model);
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
        if (noise) {
            const SizeType tables = reserved_tables > 0
                                        ? reserved_tables * noise->table_bytes()
                                        : noise->live_table_bytes();
            const SizeType plan_bytes = noise->plan_bytes() + tables;
            u.plan += plan_bytes;
            u.workspace += noise->workspace_bytes();
            if (exec.backend != Backend::kCPU) {
                u.host += plan_bytes + noise->workspace_bytes();
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
    /// Budgeted bytes declared per-call masks add, before the path (hence
    /// the search latency) is known: the latency is at most the plan's
    /// largest delay.
    [[nodiscard]] SizeType mask_bytes_bound(const detail::BoxcarNoiseModel& m,
                                            Exec ex) const {
        SizeType base_rows = 0;
        for (const auto& p : planes) {
            base_rows += p.dm_grid.size();
        }
        const SizeType rows    = base_rows * beams;
        const SizeType filters = planes.front().widths.size();
        const auto threads = static_cast<SizeType>(std::max(1, ex.nthreads));
        if (ex.backend == Backend::kCPU) {
            const auto ring = this->ring(
                static_cast<uint64_t>(plan.get_dt_max()), reach(m), ex);
            // The staging block (when the constructor mask has none);
            // packed: the fill row and the integer screen cutoffs.
            const bool packed = config.input_nbits != 32;
            const SizeType row =
                packed ? bit_pack_utils::packed_row_bytes(plan.get_nsamps(),
                                                          config.input_nbits)
                       : plan.get_nsamps() * sizeof(float);
            const SizeType staging =
                kill_runs.empty() ? beams * plan.get_nchans() * row : 0;
            // Packed Kadane: per-thread int32 cuts or float cutoffs (the
            // larger; the gate kind is known only to the engine).
            const bool kadane_gate =
                detail::resolve_screen_mode(config.screen_mode, false) ==
                BoxcarScreenMode::kKadane;
            const SizeType fill =
                packed ? row + (threads * filters * sizeof(uint64_t)) +
                             (kadane_gate
                                  ? threads * std::max<SizeType>(
                                                  kadane::kMaxGateBuckets *
                                                      kadane::kMaxBatchRows *
                                                      sizeof(int32_t),
                                                  filters * sizeof(float))
                                  : 0)
                       : 0;
            return staging + fill + model_bytes(m, ring) +
                   (threads * filters * sizeof(float)) +
                   (threads * kadane::kMaxGateBuckets * kadane::kMaxBatchRows *
                    sizeof(float));
        }
        // GPU: row supports and (2 * 2 + 2) table slots of 2 tables (4 for
        // packed input: the integer cutoffs), one more when blocks may be
        // short for the latency (the catch-all).
        const SizeType keep =
            (kill_runs.empty() && plan.get_subband_depth() == 0)
                ? beams * plan.get_nchans()
                : 0;
        const SizeType slots =
            6 +
            (ring(static_cast<uint64_t>(plan.get_dt_max()), reach(m), ex).second
                 ? 1
                 : 0);
        return keep + (rows * sizeof(uint64_t)) +
               (slots * (config.input_nbits != 32 ? 4 : 2) * rows * filters *
                sizeof(float));
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
        const BoxcarScreenMode mode = detail::resolve_screen_mode(
            config.screen_mode, ex.backend != Backend::kCPU);
        SizeType base_rows = 0;
        for (const auto& p : planes) {
            base_rows += p.dm_grid.size();
        }
        const SizeType rows    = base_rows * beams;
        const SizeType width   = planes.front().widths.back();
        const SizeType filters = planes.front().widths.size();
        const SizeType carry_elem =
            config.input_nbits == 32 ? sizeof(float)
            : detail::int_search_fits_u32(plan, smear, config.input_nbits)
                ? sizeof(uint32_t)
                : sizeof(uint64_t);
        const SizeType carry = rows * width * carry_elem;
        const SizeType peaks =
            config.peak_mode == PeakFindingMode::kContiguousPeakPerWidth
                ? filters
                : SizeType{1};
        // Budgeted bytes only: on a GPU the device copies (the host result
        // storage and host row table are reported, not budgeted).
        SizeType cands = config.max_candidates * sizeof(RawCandidate);
        if (config.overflow_policy == OverflowPolicy::kKeepBrightest) {
            // The per-row stores of the brightest max_cands_per_dm (compact
            // entries on a GPU).
            cands +=
                rows * config.max_cands_per_dm *
                (ex.backend == Backend::kCPU ? sizeof(RawCandidate)
                                             : sizeof(detail::BoxcarKeepEntry));
            cands += rows * sizeof(uint64_t); // detections per row
        }
        if (config.max_candidates_per_plane != 0) {
            detail::check_plane_share(config, planes.size());
            // Per-plane counters; under kDropNew the shares' staging.
            cands += planes.size() * sizeof(uint64_t);
            if (config.overflow_policy != OverflowPolicy::kKeepBrightest) {
                cands += planes.size() * config.max_candidates_per_plane *
                         sizeof(RawCandidate);
            }
        }
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
        if (ex.backend != Backend::kCPU && plan.get_subband_depth() > 0) {
            // The row bytes also serve drain()'s zero blocks (a subband
            // search may have a search latency).
            staging = beams * plan.get_nchans();
        }
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
        // Boxes wider than 4096 on a GPU: prefix rings in device memory, at
        // most 256 MiB (launched in batches of blocks).
        SizeType rings = 0;
        if (ex.backend != Backend::kCPU && width > 4096) {
            rings = std::min<SizeType>(detail::kGlobalRingBytes,
                                       rows * std::bit_ceil(width + 1024) *
                                           sizeof(uint64_t));
        }
        SizeType kadane_extra = 0;
        if (mode == BoxcarScreenMode::kKadane &&
            (ex.backend == Backend::kCPU || width <= 4096)) {
            kadane_extra =
                (rows * kadane::kMaxGateBuckets *
                 (sizeof(float) + (config.use_trapezoid
                                       ? sizeof(kadane::LagWindow<float>)
                                       : 0))) +
                (base_rows *
                 ((2 * kadane::kMaxGateBuckets * sizeof(float)) + 1));
        }
        return carry + (rows * peaks * sizeof(detail::BoxcarPeak)) + cands +
               luts + row_table + staging + rings + kadane_extra;
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
    /// Kill runs of prev_mask, and of a changed mask until its call ends.
    std::vector<std::pair<SizeType, SizeType>> prev_runs, next_runs;
    std::vector<uint8_t> effective;
    detail::BlockMasks block;
    detail::LiveHistory live;
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
                    /*estimate_only=*/true);
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
    // The row tables do not depend on the budget: made (and the
    // configuration validated) once.
    const auto planes =
        detail::make_boxcar_plans(plan, config, use_box_smearing, kill_mask);
    const auto build = [&](SizeType budget) {
        return Impl(plan, config, use_box_smearing, exec, nbeams, fuse_levels,
                    chunk_nsamps, budget, kill_mask, /*estimate_only=*/true,
                    &planes)
            .estimate();
    };
    // Each failure's `needed` is exact for the path it tried, except when
    // the boxcar's chunk loop shrank the FDMT's share: then step up (at
    // least 1 MiB) and, once a budget builds, narrow it down below.
    SizeType budget = 1;
    SizeType failed = 0; // largest budget known not to build
    bool stepped    = false;
    std::optional<FDMTBoxcarMemoryEstimate> found;
    for (int attempt = 0; attempt < 64 && !found; ++attempt) {
        try {
            found = build(budget);
        } catch (const detail::BudgetError& e) {
            failed = budget;
            if (e.needed() > budget) {
                budget = e.needed();
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
    try {
        return build(usage);
    } catch (const detail::BudgetError&) {
        if (!stepped) {
            return *found;
        }
    }
    SizeType hi = found->max_memory_bytes;
    SizeType lo = std::max(failed, usage);
    while (hi - lo > 1) {
        const SizeType mid = lo + ((hi - lo) / 2);
        try {
            found = build(mid);
            hi    = mid;
        } catch (const detail::BudgetError&) {
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
