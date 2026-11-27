#pragma once

// A stream ending in a short block: FDMT::execute_tail() and
// FDMTBoxcar::drain(tail, n_tail) on k whole blocks plus a tail of n_tail
// samples give what one pass over the same k * nsamps + n_tail samples
// gives, whatever the tail block's padding holds.

#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "dmt/algorithms/fdmt.hpp"
#include "dmt/algorithms/fdmt_boxcar.hpp"
#include "dmt/common/backend.hpp"
#include "dmt/common/plans.hpp"

#include "fdmt_boxcar_reference.hpp"
#include "test_helpers.hpp"
#include "test_random.hpp"

namespace boxcar_test {

/// Rows [0, n) of each channel of `whole` (nchans x total) starting at t0, as
/// a (nchans, nsamps) block padded with `pad`.
template <typename T>
std::vector<T> slice_block(const std::vector<T>& whole,
                           SizeType nchans,
                           SizeType total,
                           SizeType t0,
                           SizeType n,
                           SizeType nsamps,
                           T pad) {
    std::vector<T> block(nchans * nsamps, pad);
    for (SizeType c = 0; c < nchans; ++c) {
        for (SizeType t = 0; t < n; ++t) {
            block[(c * nsamps) + t] = whole[(c * total) + t0 + t];
        }
    }
    return block;
}

inline void check_fdmt_tail(const Exec& exec) {
    constexpr SizeType kNchans = 16;
    constexpr SizeType kNsamps = 64;
    constexpr SizeType kBlocks = 3;
    constexpr SizeType kTail   = 37;
    constexpr SizeType kTotal  = (kBlocks * kNsamps) + kTail;
    const std::vector<IndexType> dts{0, 3, 8, 15, 24};
    const auto make = [&](SizeType nsamps) {
        return FDMT(plans::FDMTPlan(1200.0F, 1600.0F, kNchans, nsamps, 0.001F,
                                    dts, "valid"),
                    true, exec);
    };
    for (const SizeType nbits : {SizeType{32}, SizeType{8}}) {
        CAPTURE(nbits);
        const auto values = test_random::uniform_int<uint8_t>(
            kNchans * kTotal, 17 + nbits, 0, 255);
        std::vector<float> fvalues(values.begin(), values.end());
        auto whole      = make(kTotal);
        const auto nrow = whole.get_plan().get_dmt_ndms();
        std::vector<float> ref(whole.get_plan().get_dmt_size());
        if (nbits == 32) {
            whole.execute(fvalues, ref);
        } else {
            whole.execute(std::span<const uint8_t>(values), 8, ref);
        }
        auto stream = make(kNsamps);
        std::vector<float> got(nrow * kTotal, 0.0F);
        std::vector<float> out(stream.get_plan().get_dmt_size());
        for (SizeType k = 0; k <= kBlocks; ++k) {
            const SizeType n = k < kBlocks ? kNsamps : kTail;
            // Padding that must not matter: large values.
            const auto fb = slice_block(fvalues, kNchans, kTotal, k * kNsamps,
                                        n, kNsamps, 1.0e4F);
            const auto pb = slice_block(values, kNchans, kTotal, k * kNsamps, n,
                                        kNsamps, uint8_t{255});
            if (k < kBlocks) {
                if (nbits == 32) {
                    stream.execute(fb, out);
                } else {
                    stream.execute(std::span<const uint8_t>(pb), 8, out);
                }
            } else if (nbits == 32) {
                stream.execute_tail(std::span<const float>(fb), kTail, out);
            } else {
                stream.execute_tail(std::span<const uint8_t>(pb), 8, kTail,
                                    out);
            }
            for (SizeType r = 0; r < nrow; ++r) {
                for (SizeType t = 0; t < n; ++t) {
                    got[(r * kTotal) + (k * kNsamps) + t] =
                        out[(r * kNsamps) + t];
                }
            }
        }
        for (SizeType r = 0; r < nrow; ++r) {
            for (SizeType t = 0; t < kTotal; ++t) {
                INFO("row " << r << " sample " << t);
                const float a = got[(r * kTotal) + t];
                const float b = ref[(r * kTotal) + t];
                if (nbits == 32) {
                    REQUIRE(std::abs(a - b) <= 1e-4F * (1.0F + std::abs(b)));
                } else {
                    REQUIRE(a == b);
                }
            }
        }
        // The stream has ended: execute, execute_tail, reset and save_history
        // throw until reset_history() or load_history().
        std::vector<float> hist(stream.history_state_size());
        CHECK_THROWS_AS(stream.save_history(hist), std::logic_error);
        CHECK_THROWS_AS(stream.execute(fvalues, out), std::logic_error);
        CHECK_THROWS_AS(
            stream.execute_tail(std::span<const float>(out), 1, out),
            std::logic_error);
        stream.reset_history();
        CHECK_NOTHROW(stream.save_history(hist));
        CHECK_THROWS_AS(
            stream.execute_tail(
                std::span<const float>(fvalues).first(kNchans * kNsamps), 0,
                out),
            std::invalid_argument);
        CHECK_THROWS_AS(
            stream.execute_tail(
                std::span<const float>(fvalues).first(kNchans * kNsamps),
                kNsamps + 1, out),
            std::invalid_argument);
    }

    if (exec.backend == Backend::kCPU) {
        // Skewed CPU path (forced below the tree path's budget): integer-valued
        // [0, n_tail) must stay on exact integer sums and never read NaN/Inf
        // or non-integer 0.5F padding in [n_tail, nsamps).
        constexpr SizeType kSkewChans  = 64;
        constexpr SizeType kSkewSamps  = 512;
        constexpr SizeType kSkewBlocks = 2;
        constexpr SizeType kSkewTail   = 173;
        constexpr SizeType kSkewTotal  = (kSkewBlocks * kSkewSamps) + kSkewTail;
        const auto skew_plan           = [&](SizeType ns) {
            return plans::FDMTPlan(1100.0F, 1500.0F, kSkewChans, ns, 0.001F,
                                   600, 0, 1, "valid", 1);
        };
        const auto splan = skew_plan(kSkewSamps);
        const SizeType tree_b =
            FDMT::estimate_memory(splan, true, exec).usage.total();
        const SizeType skew_b = tree_b - 1;
        const auto raw_u8     = test_random::uniform_int<uint8_t>(
            kSkewChans * kSkewTotal, 211, 0, 200);
        const std::vector<float> fvals(raw_u8.begin(), raw_u8.end());

        FDMT tree_ref(skew_plan(kSkewTotal), true, exec);
        const SizeType ndms = tree_ref.get_plan().get_dmt_ndms();
        std::vector<float> ref_out(tree_ref.get_plan().get_dmt_size());
        tree_ref.execute(fvals, ref_out);

        const float qnan = std::bit_cast<float>(uint32_t{0x7FC00000U});
        const float pinf = std::bit_cast<float>(uint32_t{0x7F800000U});
        std::vector<float> got_zero;
        for (const float pad : {0.0F, 0.5F, qnan, pinf}) {
            FDMT skew_fdmt(splan, true, exec, 1, algorithms::kFDMTAutoFuse,
                           true, 0, skew_b);
            REQUIRE(skew_fdmt.get_chunk_nsamps() < kSkewSamps);
            std::vector<float> out(splan.get_dmt_size());
            std::vector<float> got(ndms * kSkewTotal, 0.0F);
            for (SizeType k = 0; k <= kSkewBlocks; ++k) {
                const SizeType n = k < kSkewBlocks ? kSkewSamps : kSkewTail;
                const auto blk =
                    slice_block(fvals, kSkewChans, kSkewTotal, k * kSkewSamps,
                                n, kSkewSamps, k < kSkewBlocks ? 0.0F : pad);
                if (k < kSkewBlocks) {
                    skew_fdmt.execute(blk, out);
                } else {
                    skew_fdmt.execute_tail(std::span<const float>(blk),
                                           kSkewTail, out);
                }
                for (SizeType r = 0; r < ndms; ++r) {
                    for (SizeType t = 0; t < n; ++t) {
                        got[(r * kSkewTotal) + (k * kSkewSamps) + t] =
                            out[(r * kSkewSamps) + t];
                    }
                }
            }
            if (got_zero.empty()) {
                got_zero = got;
                for (SizeType i = 0; i < got.size(); ++i) {
                    REQUIRE(std::abs(got[i] - ref_out[i]) <=
                            1e-4F * (1.0F + std::abs(ref_out[i])));
                }
            } else {
                // Bitwise identical across 0.0F, 0.5F, NaN, and +Inf padding.
                REQUIRE(got == got_zero);
            }
        }
    }
}

inline void check_boxcar_tail(const Exec& exec) {
    constexpr SizeType kNchans = 16;
    constexpr SizeType kNsamps = 128;
    constexpr SizeType kBlocks = 3;
    constexpr SizeType kTail   = 45;
    constexpr SizeType kTotal  = (kBlocks * kNsamps) + kTail;
    const std::vector<IndexType> dts{0, 3, 8, 15, 24, 40};
    BoxcarConfig cfg;
    cfg.max_boxcar       = 16;
    cfg.wtsp             = 1.5F;
    cfg.detect_thresh    = 2.5F;
    cfg.max_candidates   = SizeType{1} << 20;
    cfg.max_cands_per_dm = 0;
    cfg.input_nbits      = 8;
    cfg.input_mean       = 128.0F;
    cfg.noise_sigma      = 30.0F;
    const auto values =
        test_random::uniform_int<uint8_t>(kNchans * kTotal, 91, 76, 180);
    const auto plan = [&](SizeType nsamps) {
        return plans::FDMTPlan(1200.0F, 1600.0F, kNchans, nsamps, 0.001F, dts,
                               "valid");
    };
    std::vector<RawCandidate> want;
    {
        FDMTBoxcar whole(plan(kTotal), cfg, true, exec);
        append(want, whole.execute(std::span<const uint8_t>(values)));
        append(want, whole.drain());
        CHECK(whole.get_searched_nsamps() == kTotal);
    }
    REQUIRE(want.size() > 20);
    FDMTBoxcar stream(plan(kNsamps), cfg, true, exec);
    std::vector<RawCandidate> got;
    for (SizeType k = 0; k < kBlocks; ++k) {
        const auto b = slice_block(values, kNchans, kTotal, k * kNsamps,
                                   kNsamps, kNsamps, uint8_t{0});
        append(got, stream.execute(std::span<const uint8_t>(b)));
        if (k == 0) {
            // A throwing drain(tail, n_tail) rolls back m_search_stop and
            // leaves the open stream usable for subsequent execute() + drain().
            const auto pos_before = stream.get_searched_nsamps();
            const auto bad_tail =
                std::span<const uint8_t>(b).first(b.size() - 1);
            CHECK_THROWS_AS(stream.drain(bad_tail, kTail),
                            std::invalid_argument);
            CHECK(stream.get_searched_nsamps() == pos_before);
        }
    }
    // Bright padding: never searched.
    const auto tail = slice_block(values, kNchans, kTotal, kBlocks * kNsamps,
                                  kTail, kNsamps, uint8_t{255});
    append(got, stream.drain(std::span<const uint8_t>(tail), kTail));
    CHECK(stream.get_searched_nsamps() == kTotal);
    for (const auto& c : got) {
        REQUIRE(c.end_sample < kTotal);
    }
    compare_exact(got, want);
    CHECK_THROWS_AS(stream.drain(std::span<const uint8_t>(tail), kTail),
                    std::logic_error);
    stream.reset_history();
    CHECK_THROWS_AS(stream.drain(std::span<const uint8_t>(tail), 0),
                    std::invalid_argument);

    // Per-call kill_mask on drain(tail, n_tail, kill_mask) with
    // per_call_masks = true: both constant mask vs constructor-masked one-pass
    // reference, and changing masks across the tail transition.
    {
        BoxcarConfig mcfg;
        mcfg.max_boxcar       = 16;
        mcfg.wtsp             = 1.5F;
        mcfg.detect_thresh    = 2.25F;
        mcfg.max_candidates   = SizeType{1} << 20;
        mcfg.max_cands_per_dm = 0;
        mcfg.per_call_masks   = true;

        std::vector<uint8_t> mask_a(kNchans, 1);
        mask_a[2]  = 0;
        mask_a[5]  = 0;
        mask_a[11] = 0;
        std::vector<uint8_t> mask_b(kNchans, 1);
        mask_b[1]  = 0;
        mask_b[5]  = 0;
        mask_b[14] = 0;

        std::vector<float> clean(kNchans * kTotal, 0.0F);
        for (SizeType i = 0; i < clean.size(); ++i) {
            clean[i] = (static_cast<float>(values[i]) - 128.0F) / 30.0F;
        }
        // Inject RFI into channels killed by mask_a so an unmasked drain fails.
        std::vector<float> dirty_a = clean;
        for (const SizeType ch : {SizeType{2}, SizeType{5}, SizeType{11}}) {
            for (SizeType t = 0; t < kTotal; ++t) {
                dirty_a[(ch * kTotal) + t] = 40.0F;
            }
        }
        std::vector<RawCandidate> want_const;
        {
            BoxcarConfig ref_cfg   = mcfg;
            ref_cfg.per_call_masks = false;
            FDMTBoxcar whole_m(plan(kTotal), ref_cfg, true, exec, 1,
                               algorithms::kFDMTAutoFuse, 0,
                               algorithms::kFDMTDefaultMaxMemory, mask_a);
            append(want_const,
                   whole_m.execute(std::span<const float>(dirty_a)));
            append(want_const, whole_m.drain());
        }
        REQUIRE(want_const.size() > 10);

        FDMTBoxcar masked_stream(plan(kNsamps), mcfg, true, exec);
        std::vector<RawCandidate> got_const;
        for (SizeType k = 0; k < kBlocks; ++k) {
            const auto fb = slice_block(dirty_a, kNchans, kTotal, k * kNsamps,
                                        kNsamps, kNsamps, 0.0F);
            append(got_const, masked_stream.execute(fb, mask_a));
            if (k == 0) {
                const std::vector<uint8_t> bad_mask(kNchans - 1, 1);
                CHECK_THROWS_AS(masked_stream.drain(fb, kTail, bad_mask),
                                std::invalid_argument);
                CHECK_THROWS_AS(
                    masked_stream.drain(
                        std::span<const float>(fb).first(fb.size() - 1), kTail,
                        mask_a),
                    std::invalid_argument);
            }
        }
        const auto ftail =
            slice_block(dirty_a, kNchans, kTotal, kBlocks * kNsamps, kTail,
                        kNsamps, 1.0e4F);
        append(got_const, masked_stream.drain(ftail, kTail, mask_a));
        CHECK(masked_stream.get_searched_nsamps() == kTotal);
        compare(got_const, want_const);

        // Changing mask on the final tail block (mask_a -> mask_b): under
        // kAllAboveThreshold, every endpoint < kTotal matches executing a
        // zero-padded final block with mask_b and filtering peak_sample <
        // kTotal.
        mcfg.peak_mode                  = PeakFindingMode::kAllAboveThreshold;
        std::vector<float> tail_b_dirty = slice_block(
            clean, kNchans, kTotal, kBlocks * kNsamps, kTail, kNsamps, 1.0e4F);
        std::vector<float> tail_b_zero = slice_block(
            clean, kNchans, kTotal, kBlocks * kNsamps, kTail, kNsamps, 0.0F);
        for (const SizeType ch : {SizeType{1}, SizeType{5}, SizeType{14}}) {
            for (SizeType t = 0; t < kTail; ++t) {
                tail_b_dirty[(ch * kNsamps) + t] = 50.0F;
                tail_b_zero[(ch * kNsamps) + t]  = 50.0F;
            }
        }
        FDMTBoxcar chg_tail(plan(kNsamps), mcfg, true, exec);
        FDMTBoxcar chg_ref(plan(kNsamps), mcfg, true, exec);
        std::vector<RawCandidate> got_chg;
        std::vector<RawCandidate> all_ref;
        for (SizeType k = 0; k < kBlocks; ++k) {
            const auto fb = slice_block(dirty_a, kNchans, kTotal, k * kNsamps,
                                        kNsamps, kNsamps, 0.0F);
            append(got_chg, chg_tail.execute(fb, mask_a));
            append(all_ref, chg_ref.execute(fb, mask_a));
        }
        append(got_chg, chg_tail.drain(tail_b_dirty, kTail, mask_b));
        append(all_ref, chg_ref.execute(tail_b_zero, mask_b));
        append(all_ref, chg_ref.drain());
        std::vector<RawCandidate> want_chg;
        for (const auto& c : all_ref) {
            if (c.peak_sample < kTotal) {
                want_chg.push_back(c);
            }
        }
        REQUIRE_FALSE(want_chg.empty());
        compare_exact(got_chg, want_chg);
    }

    if (exec.backend == Backend::kCPU) {
        // Skewed CPU boxcar path (subband_depth = 1, search_latency > 0) with
        // integer-valued [0, n_tail) float samples and 0.5F / NaN / +Inf in
        // [n_tail, nsamps).
        constexpr SizeType kSkewChans  = 64;
        constexpr SizeType kSkewSamps  = 512;
        constexpr SizeType kSkewBlocks = 2;
        constexpr SizeType kSkewTail   = 173;
        constexpr SizeType kSkewTotal  = (kSkewBlocks * kSkewSamps) + kSkewTail;
        const plans::FDMTPlan splan(1100.0F, 1500.0F, kSkewChans, kSkewSamps,
                                    0.001F, 600, 0, 1, "valid", 1);
        BoxcarConfig scfg;
        scfg.max_boxcar       = 16;
        scfg.wtsp             = 1.5F;
        scfg.detect_thresh    = 2.5F;
        scfg.noise_sigma      = 30.0F;
        scfg.max_candidates   = SizeType{1} << 20;
        scfg.max_cands_per_dm = 0;

        const auto min_est =
            FDMTBoxcar::estimate_minimum_memory(splan, scfg, true, exec);
        REQUIRE(min_est.search_path.find("skewed") != std::string::npos);
        REQUIRE(min_est.search_latency > 0);

        const auto raw_i = test_random::uniform_int<int>(
            kSkewChans * kSkewTotal, 307, -50, 50);
        std::vector<float> fvals(raw_i.begin(), raw_i.end());
        // Add an exact-integer pulse near the end of the short tail so lagged
        // subband rows must search it during drain(tail, kSkewTail).
        for (SizeType c = 32; c < kSkewChans; ++c) {
            for (SizeType t = kSkewTotal - 12; t < kSkewTotal - 6; ++t) {
                fvals[(c * kSkewTotal) + t] += 40.0F;
            }
        }

        const float qnan = std::bit_cast<float>(uint32_t{0x7FC00000U});
        const float pinf = std::bit_cast<float>(uint32_t{0x7F800000U});
        std::vector<RawCandidate> want_skew;
        for (const float pad : {0.0F, 0.5F, qnan, pinf}) {
            FDMTBoxcar skew_s(splan, scfg, true, exec, 1,
                              algorithms::kFDMTAutoFuse, 0,
                              min_est.max_memory_bytes);
            REQUIRE(skew_s.get_search_latency() > 0);
            std::vector<RawCandidate> got_skew;
            for (SizeType k = 0; k < kSkewBlocks; ++k) {
                const auto fb =
                    slice_block(fvals, kSkewChans, kSkewTotal, k * kSkewSamps,
                                kSkewSamps, kSkewSamps, 0.0F);
                append(got_skew, skew_s.execute(fb));
            }
            const auto ftail = slice_block(fvals, kSkewChans, kSkewTotal,
                                           kSkewBlocks * kSkewSamps, kSkewTail,
                                           kSkewSamps, pad);
            append(got_skew, skew_s.drain(ftail, kSkewTail));
            CHECK(skew_s.get_searched_nsamps() == kSkewTotal);
            for (const auto& c : got_skew) {
                REQUIRE(c.end_sample < kSkewTotal);
            }
            if (want_skew.empty()) {
                want_skew = got_skew;
                REQUIRE_FALSE(want_skew.empty());
            } else {
                compare_exact(got_skew, want_skew);
            }
        }
    }
}

} // namespace boxcar_test
