// The GPU subband search's ordered list and its first-fit chooser
// (lib/dmt/gpu_search_list.hpp), on synthetic memory: the chooser's
// bisection picks what building every entry picks, every budget from the
// minimum builds, and the pick's rank never gets worse as the budget grows.
// The device-built engines are covered in fdmt_boxcar_cuda_t.cu.

#include "dmt/gpu_search_list.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using dmt::SizeType;
using dmt::algorithms::detail::choose_gpu_search;
using dmt::algorithms::detail::FDMTEngineConfig;
using dmt::algorithms::detail::gpu_direct_min_chunk;
using dmt::algorithms::detail::gpu_search_list;
using dmt::algorithms::detail::GPUSearchEntry;
using Path = FDMTEngineConfig::GPUPath;

namespace {
// Bytes of a path: increasing in the chunk (base + per-sample slope, and
// `concave`: half the slope above chunk 2048, as when a long chunk moves the
// direct top's level); `skewed_available` false drops the skewed path.
struct Synthetic {
    SizeType direct_base;
    SizeType direct_slope;
    SizeType skewed_base;
    SizeType skewed_slope;
    bool skewed_available{true};
    SizeType nsamps{65536};
    bool concave{false};
    mutable SizeType builds{0};
    std::optional<SizeType> operator()(const GPUSearchEntry& e) const {
        ++builds;
        const SizeType c = std::min(e.chunk, nsamps);
        if (e.path == Path::kDirect) {
            if (concave && c > 2048) {
                return direct_base + (direct_slope * 2048) +
                       (direct_slope * (c - 2048) / 2);
            }
            return direct_base + (direct_slope * c);
        }
        if (!skewed_available) {
            return std::nullopt;
        }
        return skewed_base + (skewed_slope * c);
    }
};

void check_sweep(const Synthetic& m, SizeType cache, SizeType max_boxcar) {
    const auto list = gpu_search_list(cache, max_boxcar);
    std::optional<SizeType> prev_rank;
    SizeType first_fit_budget = 0;
    // Budgets from below the minimum to past the largest entry.
    const SizeType hi = m.direct_base + (m.direct_slope * 9000);
    for (SizeType b = m.skewed_base / 2; b <= hi; b += (hi / 997) + 1) {
        Synthetic brute = m;
        const auto all  = choose_gpu_search(list, b, m.nsamps, brute, false);
        Synthetic pred  = m;
        const auto got  = choose_gpu_search(list, b, m.nsamps, pred, true);
        INFO("budget " << b);
        REQUIRE(got.index == all.index);
        if (!got.index) {
            // Nothing fits: the reported minimum does.
            REQUIRE(prev_rank == std::nullopt);
            Synthetic again = m;
            REQUIRE(choose_gpu_search(list, got.min_needed, m.nsamps, again)
                        .index.has_value());
            continue;
        }
        if (first_fit_budget == 0) {
            first_fit_budget = b;
        }
        if (prev_rank) {
            CHECK(*got.index <= *prev_rank); // the rank never gets worse
        }
        prev_rank = got.index;
    }
    CHECK(first_fit_budget != 0);
    CHECK(prev_rank == SizeType{0}); // the head fits at the top budget
}
} // namespace

TEST_CASE("gpu_search_list order", "[budget][gpu_list]") {
    CHECK(gpu_direct_min_chunk(4096) == 512);
    CHECK(gpu_direct_min_chunk(1562) == 256);
    CHECK(gpu_direct_min_chunk(128) == 256);
    CHECK(gpu_direct_min_chunk(1U << 20) == 4096);
    // Wide boxcars: direct down to 512, skewed, then direct 256.
    const auto wide = gpu_search_list(0, 4096);
    REQUIRE(wide.size() == 10);
    CHECK(wide[0] == GPUSearchEntry{4096, Path::kDirect});
    CHECK(wide[3] == GPUSearchEntry{512, Path::kDirect});
    CHECK(wide[4] == GPUSearchEntry{4096, Path::kSkewed});
    CHECK(wide[8] == GPUSearchEntry{256, Path::kSkewed});
    CHECK(wide.back() == GPUSearchEntry{256, Path::kDirect});
    // Narrow boxcars: every direct chunk before the skewed path.
    const auto narrow = gpu_search_list(0, 1562);
    REQUIRE(narrow.size() == 10);
    CHECK(narrow[4] == GPUSearchEntry{256, Path::kDirect});
    CHECK(narrow[5] == GPUSearchEntry{4096, Path::kSkewed});
    CHECK(narrow.back() == GPUSearchEntry{256, Path::kSkewed});
    // The cache chunk goes first, once.
    const auto cached = gpu_search_list(1024, 128);
    REQUIRE(cached.size() == 10);
    CHECK(cached.front() == GPUSearchEntry{1024, Path::kDirect});
    CHECK(std::count(cached.begin(), cached.end(),
                     GPUSearchEntry{1024, Path::kDirect}) == 1);
}

TEST_CASE("choose_gpu_search: bisection matches building every entry",
          "[budget][gpu_list]") {
    SECTION("UWL-like: direct needs far more than skewed") {
        const Synthetic m{9'600'000'000ULL, 1'000'000ULL, 3'600'000'000ULL,
                          1'000'000ULL};
        check_sweep(m, 0, 4096);
        check_sweep(m, 0, 1562);
    }
    SECTION("small plan with a cache chunk") {
        const Synthetic m{180'000'000ULL, 40'000ULL, 150'000'000ULL, 45'000ULL};
        check_sweep(m, 1024, 4096);
        check_sweep(m, 1024, 128);
    }
    SECTION("no skewed path") {
        Synthetic m{100'000'000ULL, 50'000ULL, 0, 0};
        m.skewed_available = false;
        check_sweep(m, 0, 4096);
        check_sweep(m, 0, 1562);
    }
    SECTION("concave: chunk 4096 cheaper than a line predicts") {
        Synthetic m{5'500'000'000ULL, 1'000'000ULL, 4'000'000'000ULL,
                    1'200'000ULL};
        m.concave = true;
        check_sweep(m, 0, 4096);
        check_sweep(m, 0, 1562);
    }
    SECTION("short blocks: chunks capped at nsamps") {
        Synthetic m{100'000'000ULL, 50'000ULL, 80'000'000ULL, 60'000ULL};
        m.nsamps = 2048;
        check_sweep(m, 0, 4096);
        check_sweep(m, 0, 1562);
    }
}

TEST_CASE("choose_gpu_search builds few engines", "[budget][gpu_list]") {
    const Synthetic m{9'600'000'000ULL, 1'000'000ULL, 3'600'000'000ULL,
                      1'000'000ULL};
    const auto list = gpu_search_list(0, 4096);
    // Between skewed/4096 and direct/1024: one build rejects the direct run
    // (its shortest entry), then the skewed run's shortest and a bisection.
    Synthetic pred = m;
    const auto got = choose_gpu_search(list, 9'000'000'000ULL, m.nsamps, pred);
    REQUIRE(got.index == SizeType{4});
    CHECK(pred.builds <= 5);
    // A budget the head fits: the run's shortest entry and a bisection.
    Synthetic head = m;
    REQUIRE(choose_gpu_search(list, SIZE_MAX / 2, m.nsamps, head).index ==
            SizeType{0});
    CHECK(head.builds <= 3);
}
