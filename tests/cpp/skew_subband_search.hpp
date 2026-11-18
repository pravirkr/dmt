#pragma once

// Subband search (plan subband_depth > 0) on the skewed-window paths against
// the tree path, shared by the CPU and CUDA skew tests. The skewed paths
// emit each subband row with a fixed lag (compute_skew_windows), so a row's
// last `lag` samples are searched one or more blocks later than on the tree
// path; candidates are compared up to a cutoff both paths have searched.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/common/candidate.hpp"
#include "dmt/common/plans.hpp"

#include "dmt/boxcar.hpp"
#include "dmt/boxcar_drain.hpp"
#include "dmt/boxcar_noise.hpp"
#include "dmt/engines.hpp"

#include "test_random.hpp"

namespace dmt::skew_search_test {

struct SearchRun {
    std::vector<RawCandidate> candidates;
    std::string path;
    uint64_t latency{0};     ///< search_latency() of the engine
    uint64_t searched{0};    ///< searched samples at the end (drain: all)
    bool zero_blocks{false}; ///< the engine's zero_blocks()
    /// memory_usage() before and after flush() / drain().
    algorithms::FDMTBoxcarMemoryUsage usage_before{};
    algorithms::FDMTBoxcarMemoryUsage usage_after{};
};

/// Integer-valued input (exact on every path): float samples v - 128 for
/// bytes v, or the bytes themselves packed.
struct SearchBlock {
    std::vector<float> values;
    std::vector<uint8_t> packed;
};

inline SearchBlock search_block(SizeType n, unsigned seed) {
    SearchBlock b{.values = std::vector<float>(n),
                  .packed = test_random::uniform_int<uint8_t>(n, seed, 0, 255)};
    for (SizeType i = 0; i < n; ++i) {
        b.values[i] = static_cast<float>(b.packed[i]) - 128.0F;
    }
    return b;
}

/// Block `b` (`ns` samples of `nchans` rows) of one fixed byte stream, so
/// runs with different block lengths see the same input.
inline SearchBlock stream_block(SizeType nchans, SizeType ns, SizeType b) {
    SearchBlock out{.values = std::vector<float>(nchans * ns),
                    .packed = std::vector<uint8_t>(nchans * ns)};
    for (SizeType c = 0; c < nchans; ++c) {
        for (SizeType t = 0; t < ns; ++t) {
            uint64_t x =
                ((uint64_t{c} << 32) | (b * ns + t)) + 0x9E3779B97F4A7C15ULL;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
            x ^= x >> 31;
            const auto v             = static_cast<uint8_t>(x & 0xFF);
            out.packed[(c * ns) + t] = v;
            out.values[(c * ns) + t] = static_cast<float>(v) - 128.0F;
        }
    }
    return out;
}

inline BoxcarConfig search_config(bool packed, float wtsp = 1.5F) {
    // Uniform bytes: sigma 73.9 per sample. A low threshold gives
    // candidates from noise in every plane. Width spacing 1.5 leaves the
    // width screen off; spacing 1 (24 widths) turns it on.
    return BoxcarConfig(24, wtsp, 4.2F, 73.9F,
                        PeakFindingMode::kContiguousPeakPerWidth, 1U << 20, 0,
                        true, packed ? 8 : 32, packed ? 127.5F : 0.0F);
}

/// Copies `n` row lags from the device (set by GPU test files, which can
/// call the runtime; null: GPU lags are not recorded).
inline void (*g_copy_device_lags)(uint64_t* dst,
                                  const uint64_t* src,
                                  SizeType n) = nullptr;

/// Forwards to a sink, recording each lagged level's row lags into `lags`
/// per level (`device`: through g_copy_device_lags).
class LagRecorder final : public algorithms::detail::FDMTSearchSink {
public:
    LagRecorder(algorithms::detail::FDMTSearchSink& inner,
                std::vector<std::vector<uint64_t>>* lags,
                const std::vector<SizeType>* level_rows,
                bool device)
        : m_inner(inner),
          m_lags(lags),
          m_level_rows(level_rows),
          m_device(device) {}
    void consume(SizeType level,
                 const algorithms::detail::FDMTSearchInput& input) override {
        if (m_lags != nullptr && input.row_lag != nullptr &&
            level < m_level_rows->size()) {
            if (m_lags->size() <= level) {
                m_lags->resize(level + 1);
            }
            const SizeType n = (*m_level_rows)[level];
            if (m_device) {
                (*m_lags)[level].resize(n);
                if (g_copy_device_lags != nullptr) {
                    g_copy_device_lags((*m_lags)[level].data(), input.row_lag,
                                       n);
                }
            } else {
                (*m_lags)[level].assign(input.row_lag, input.row_lag + n);
            }
        }
        m_inner.consume(level, input);
    }

private:
    algorithms::detail::FDMTSearchSink& m_inner;
    std::vector<std::vector<uint64_t>>* m_lags;
    const std::vector<SizeType>* m_level_rows;
    bool m_device;
};

/// A driver that throws from the next search() once armed: a failure after
/// the engine committed the call's kill-mask entry (allocation, launch).
/// With `lags`, records the CPU skewed path's row lags (LagRecorder).
class ThrowingDriver final : public algorithms::detail::BoxcarDriver {
public:
    ThrowingDriver(std::unique_ptr<BoxcarDriver> inner,
                   bool* armed,
                   std::vector<std::vector<uint64_t>>* lags = nullptr,
                   const std::vector<SizeType>* level_rows  = nullptr)
        : m_inner(std::move(inner)),
          m_armed(armed),
          m_lags(lags),
          m_level_rows(level_rows) {}
    void search(std::span<const float> in,
                algorithms::detail::FDMTSearchSink& sink) override {
        trip();
        if (m_lags != nullptr) {
            LagRecorder rec(sink, m_lags, m_level_rows, false);
            m_inner->search(in, rec);
            return;
        }
        m_inner->search(in, sink);
    }
    void search(DeviceSpan<const float> in,
                algorithms::detail::FDMTSearchSink& sink,
                Stream stream) override {
        trip();
        if (m_lags != nullptr) {
            LagRecorder rec(sink, m_lags, m_level_rows, true);
            m_inner->search(in, rec, stream);
            return;
        }
        m_inner->search(in, sink, stream);
    }
    void search(std::span<const uint8_t> in,
                SizeType nbits,
                algorithms::detail::FDMTSearchSink& sink) override {
        trip();
        m_inner->search(in, nbits, sink);
    }
    void search(DeviceSpan<const uint8_t> in,
                SizeType nbits,
                algorithms::detail::FDMTSearchSink& sink,
                Stream stream) override {
        trip();
        m_inner->search(in, nbits, sink, stream);
    }
    void reset(Stream stream) override { m_inner->reset(stream); }
    void allocate() override { m_inner->allocate(); }
    bool set_search_keep(const uint8_t* keep, float fill) override {
        return m_inner->set_search_keep(keep, fill);
    }
    [[nodiscard]] bool search_zero_supported() const noexcept override {
        return m_inner->search_zero_supported();
    }
    void search_zero(algorithms::detail::FDMTSearchSink& sink) override {
        trip();
        m_inner->search_zero(sink);
    }
    [[nodiscard]] algorithms::FDMTMemoryUsage memory() const noexcept override {
        return m_inner->memory();
    }
    [[nodiscard]] SizeType chunk_nsamps() const noexcept override {
        return m_inner->chunk_nsamps();
    }
    [[nodiscard]] SizeType fuse_levels() const noexcept override {
        return m_inner->fuse_levels();
    }
    [[nodiscard]] std::string search_path() const override {
        return m_inner->search_path();
    }
    [[nodiscard]] uint64_t search_latency() const noexcept override {
        return m_inner->search_latency();
    }

private:
    void trip() {
        if (*m_armed) {
            *m_armed = false;
            throw std::runtime_error("injected search failure");
        }
    }
    std::unique_ptr<BoxcarDriver> m_inner;
    bool* m_armed;
    std::vector<std::vector<uint64_t>>* m_lags;
    const std::vector<SizeType>* m_level_rows;
};

/// Streams `nblocks` blocks through FDMTBoxcar's engine on `exec`, on the
/// skewed path (`skew`) or the tree path, then flushes (or drains). With
/// `masks` (float input), block b runs with per-call kill mask masks[b]
/// (empty: none), handed to the engine as FDMTBoxcar does. `wtsp`: the
/// boxcar width spacing (search_config()). `planes_in`: the result of
/// make_boxcar_plans(plan, search_config(packed, wtsp), true) when the
/// caller already has it (it dominates the setup on wide plans).
/// `fail_at` (with `masks`): before block fail_at (nblocks: after the last
/// block, before flush() or drain()), a call with a different
/// mask throws after the engine committed it; the engine's
/// drop_block_noise() must leave the stream as if it never happened.
/// `box_in`: the boxcar configuration instead of search_config(); with
/// `per_call`, each call's candidates are appended there (and overflow is
/// allowed). `zero_drain` false: drain() pushes host zero blocks through
/// execute() (the reference for the zero-block drain). `row_lags` (float;
/// a GPU needs g_copy_device_lags): each searched row's lag on the skewed
/// path (0: unlagged), in plane order. `ctor_kill`: a constructor kill mask
/// (CPU; `planes_in` must then be its planes). `one_stream`: the input is
/// stream_block()'s (the same stream for any block length) instead of
/// search_block()'s.
///
/// Masks are handed over as FDMTBoxcar does: the history mask (the block's
/// channels and every channel live in the largest reach before it) for
/// transitions, and the catch-all when the engine asks for one.
inline SearchRun run_subband_search(
    const plans::FDMTPlan& plan,
    Exec exec,
    bool skew,
    bool packed,
    SizeType nblocks,
    SizeType chunk                                       = 1024,
    SizeType leaf                                        = 0,
    bool drain                                           = false,
    const std::vector<std::vector<uint8_t>>* masks       = nullptr,
    float wtsp                                           = 1.5F,
    const std::vector<plans::BoxcarPlanePlan>* planes_in = nullptr,
    SizeType fail_at                                     = SIZE_MAX,
    const BoxcarConfig* box_in                           = nullptr,
    std::vector<std::vector<RawCandidate>>* per_call     = nullptr,
    bool zero_drain                                      = true,
    std::vector<uint64_t>* row_lags                      = nullptr,
    std::span<const uint8_t> ctor_kill                   = {},
    bool one_stream                                      = false) {
    using namespace algorithms::detail; // NOLINT
    const auto box = box_in != nullptr ? *box_in : search_config(packed, wtsp);
    const auto planes =
        planes_in != nullptr ? *planes_in : make_boxcar_plans(plan, box, true);
    const SizeType root = plan.get_niters();
    FDMTEngineConfig cfg{};
    cfg.use_box_smearing = true;
    cfg.int_tree         = false;
    cfg.exec             = exec;
    cfg.chunk_nsamps     = chunk;
    cfg.max_fuse_levels  = std::min(root - 1, root - plan.get_subband_depth());
    cfg.search_upper_levels = plan.get_subband_depth() > 0;
    cfg.search_nbits        = packed ? 8 : 32;
    cfg.skew_history_bytes  = skew ? 0 : SIZE_MAX;
    cfg.force_skewed_direct = skew;
    cfg.skew_leaf           = leaf;
    const bool cpu          = exec.backend == Backend::kCPU;
    bool armed              = false;
    // Rows per level (planes are in level order), for the lag recorder.
    std::vector<SizeType> level_rows(root + 1, 0);
    for (const auto& p : planes) {
        level_rows[p.level] += p.dm_grid.size();
    }
    std::vector<std::vector<uint64_t>> lags;
    const auto driver = [&](std::unique_ptr<FDMTEngine> fdmt) {
        return std::make_unique<ThrowingDriver>(
            make_boxcar_driver(std::move(fdmt)), &armed,
            row_lags != nullptr ? &lags : nullptr, &level_rows);
    };
    auto engine = cpu ? make_fdmt_boxcar_cpu(plan, box, planes, cfg,
                                             driver(make_fdmt_cpu(plan, cfg)),
                                             boxcar_kill_runs(ctor_kill))
#ifdef DMT_ENABLE_GPU
                      : make_fdmt_boxcar_gpu(plan, box, planes, cfg,
                                             driver(make_fdmt_gpu(plan, cfg)));
#else
                      : nullptr;
#endif
    SearchRun run;
    run.path         = engine->search_path();
    const SizeType n = plan.get_nchans() * plan.get_nsamps();
    std::unique_ptr<BoxcarNoiseModel> model;
    std::vector<uint8_t> prev(plan.get_nchans(), 1);
    LiveHistory live;
    uint64_t reach = 0;
    if (masks != nullptr) {
        model = std::make_unique<BoxcarNoiseModel>(plan, box, true, 4);
        engine->enable_block_noise(model->support());
        live.reset(prev, 0);
        reach = model->max_support() + planes.front().widths.back() - 1;
    }
    const auto push = [&](SizeType b, bool failing = false) {
        std::vector<uint8_t> eff(plan.get_nchans(), 1);
        for (SizeType c = 0; c < (*masks)[b].size(); ++c) {
            eff[c] = (*masks)[b][c] != 0 ? 1 : 0;
        }
        if (failing) {
            // A different mask: the low half of the band killed.
            for (SizeType c = 0; c < eff.size() / 2; ++c) {
                eff[c] = 0;
            }
        }
        const std::vector<uint8_t> ones(plan.get_nchans(), 1);
        const auto tables = [&](const std::vector<uint8_t>& m) {
            return m == ones ? nullptr : model->compute(m);
        };
        BlockNoise entry;
        const uint64_t first = engine->stream_position();
        entry.first_sample   = first;
        entry.steady         = tables(eff);
        BlockMasks bm;
        block_masks(live, eff, prev, first, reach,
                    engine->noise_catch_all_begin(), bm);
        if (bm.transition) {
            entry.transition        = true;
            entry.transition_tables = tables(bm.history);
        }
        if (bm.catch_all) {
            entry.catch_all     = tables(bm.catch_mask);
            entry.has_catch_all = true;
        }
        entry.kill_runs = boxcar_kill_runs(eff);
        engine->push_block_noise(std::move(entry));
        if (!failing) {
            prev = eff;
        }
        return eff;
    };
    // Candidate frontier (FDMTBoxcar::get_candidate_frontier()): every
    // later candidate begins at or after it, and it never decreases.
    const uint64_t latency  = engine->search_latency();
    const auto frontier_now = [&] {
        const uint64_t pos = engine->stream_position();
        return std::min(pos > latency ? pos - latency : 0,
                        engine->open_begin_min());
    };
    uint64_t frontier = 0;
    const auto add    = [&](const CandidateResultView& r) {
        if (per_call != nullptr) {
            per_call->emplace_back(r.candidates.begin(), r.candidates.end());
        } else {
            REQUIRE_FALSE(r.overflowed());
        }
        for (const auto& c : r.candidates) {
            REQUIRE(c.begin_sample >= frontier);
        }
        run.candidates.insert(run.candidates.end(), r.candidates.begin(),
                              r.candidates.end());
        const uint64_t next = frontier_now();
        REQUIRE(next >= frontier);
        frontier = next;
    };
    for (SizeType b = 0; b < nblocks; ++b) {
        const auto blk =
            one_stream ? stream_block(plan.get_nchans(), plan.get_nsamps(), b)
                       : search_block(n, 900 + static_cast<unsigned>(b));
        if (masks != nullptr && b == fail_at) {
            push(b, true);
            armed = true;
            REQUIRE_THROWS_AS(
                engine->execute(std::span<const float>(blk.values)),
                std::runtime_error);
            engine->drop_block_noise(); // as FDMTBoxcar does
        }
        std::vector<uint8_t> eff;
        if (masks != nullptr) {
            eff = push(b);
        }
        add(packed ? engine->execute(std::span<const uint8_t>(blk.packed))
                   : engine->execute(std::span<const float>(blk.values)));
        if (masks != nullptr) {
            live.update(eff, engine->stream_position());
        }
    }
    if (masks != nullptr && fail_at == nblocks && nblocks > 0) {
        // A failing call after the last block, before the end of stream.
        const auto blk = search_block(n, 7777);
        push(nblocks - 1, true);
        armed = true;
        REQUIRE_THROWS_AS(engine->execute(std::span<const float>(blk.values)),
                          std::runtime_error);
        engine->drop_block_noise();
    }
    if (row_lags != nullptr) {
        row_lags->clear();
        for (SizeType l = 0; l <= root; ++l) {
            for (SizeType r = 0; r < level_rows[l]; ++r) {
                row_lags->push_back(
                    l < lags.size() && !lags[l].empty() ? lags[l][r] : 0);
            }
        }
    }
    run.latency      = engine->search_latency();
    run.zero_blocks  = engine->zero_blocks();
    run.usage_before = engine->memory_usage();
    if (drain) {
        DrainState state;
        add(drain_stream(*engine, plan, 1, packed ? 8 : 32, run.latency, state,
                         zero_drain));
        run.searched = searched_nsamps(*engine, state, run.latency);
    } else {
        add(engine->flush());
        const uint64_t pos = engine->stream_position();
        run.searched       = pos > run.latency ? pos - run.latency : 0;
    }
    run.usage_after = engine->memory_usage();
    return run;
}

inline auto candidate_key(const RawCandidate& c) {
    return std::tuple(c.subband_level, c.subband_idx, c.dm_idx, c.filter_idx,
                      c.begin_sample, c.end_sample, c.peak_sample);
}

/// Candidates of both runs that end before `cutoff` are identical.
inline void require_same_before(std::vector<RawCandidate> actual,
                                std::vector<RawCandidate> expected,
                                uint64_t cutoff,
                                float snr_rtol = 0.0F) {
    const auto keep = [&](std::vector<RawCandidate>& v) {
        std::erase_if(v, [&](const RawCandidate& c) {
            return c.end_sample + 1 >= cutoff;
        });
        std::ranges::sort(v, {}, candidate_key);
    };
    keep(actual);
    keep(expected);
    REQUIRE(expected.size() > 100);
    REQUIRE(actual.size() == expected.size());
    SizeType sub  = 0;
    SizeType stag = 0;
    for (const auto& e : expected) {
        sub += e.subband_level > 0 ? 1 : 0;
        stag += e.staggered;
    }
    // Plain comparison first; assertions (with their INFO) only for a
    // mismatch, as the lists hold tens of thousands of candidates.
    const auto same = [&](const RawCandidate& a, const RawCandidate& e) {
        return candidate_key(a) == candidate_key(e) &&
               a.staggered == e.staggered &&
               (snr_rtol == 0.0F
                    ? a.snr == e.snr
                    : Catch::Matchers::WithinRel(e.snr, snr_rtol).match(a.snr));
    };
    for (SizeType i = 0; i < actual.size(); ++i) {
        const auto& a = actual[i];
        const auto& e = expected[i];
        if (same(a, e)) {
            continue;
        }
        INFO("candidate " << i << " level " << e.subband_level << " band "
                          << e.subband_idx << " dm " << e.dm_idx << " filter "
                          << e.filter_idx << " [" << e.begin_sample << ", "
                          << e.end_sample << "] snr " << e.snr);
        REQUIRE(candidate_key(a) == candidate_key(e));
        REQUIRE(a.staggered == e.staggered);
        if (snr_rtol == 0.0F) {
            REQUIRE(a.snr == e.snr);
        } else {
            REQUIRE_THAT(a.snr, Catch::Matchers::WithinRel(e.snr, snr_rtol));
        }
    }
    // Every kind of row took part.
    REQUIRE(sub > 0);
    REQUIRE(stag > 0);
    REQUIRE(sub < actual.size());
}

} // namespace dmt::skew_search_test
