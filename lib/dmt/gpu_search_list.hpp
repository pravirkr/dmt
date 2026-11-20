#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "dmt/engines.hpp"

namespace dmt::algorithms::detail {

/// One configuration of a GPU subband search: the FDMT path and chunk.
struct GPUSearchEntry {
    SizeType chunk;
    FDMTEngineConfig::GPUPath path;
    friend bool operator==(const GPUSearchEntry&,
                           const GPUSearchEntry&) = default;
};

/// Longest automatic chunk of a GPU subband search (8192 is 14-16% faster
/// on UWL above 18 GiB; not worth the memory).
inline constexpr SizeType kGPUSearchMaxChunk = 4096;

/// Shortest chunk at which a GPU subband search prefers the direct path to
/// the skewed one: max_boxcar / 8 as a power of two, in [256, 4096]. The
/// boxcar's per-chunk cost grows with max_boxcar, so short direct chunks
/// lose to long skewed ones on wide boxcars and win on narrow ones (L40,
/// 8-bit, x real time; UWL depth 4): max_boxcar 4096, DM 100-1100: direct
/// 256 0.92 vs skewed 4096 2.79; max_boxcar 1562, DM 0-2000, dmt grid:
/// direct 256 2.68 vs skewed 4096 1.85. On the UWL plans measured
/// (2026-11-19) the pick is the fastest that fits at 10 GiB and within
/// about 30% of it at any budget (worst in narrow windows where a larger
/// budget first fits a short direct chunk).
inline SizeType gpu_direct_min_chunk(SizeType max_boxcar) {
    const SizeType eighth = std::max<SizeType>(1, max_boxcar / 8);
    return std::clamp<SizeType>(std::bit_ceil(eighth), 256, kGPUSearchMaxChunk);
}

/**
 * @brief The GPU subband search's configurations in order of preference:
 * an FDMTBoxcar with subband_depth > 0 and an automatic chunk takes the
 * first that fits max_memory_bytes. A simple rule, not an optimum (the
 * crossovers move with the plan, max_boxcar and DM range):
 * 1. direct at `cache_chunk` (the chunk whose two state halves fit the L2,
 *    fdmt_gpu_cache_chunk(); 0: none): small plans, as before this list;
 * 2. direct, 4096 down to gpu_direct_min_chunk(max_boxcar);
 * 3. skewed, 4096 down to 256 (longer chunks are faster on either path);
 * 4. direct below that, down to 256.
 */
inline std::vector<GPUSearchEntry> gpu_search_list(SizeType cache_chunk,
                                                   SizeType max_boxcar) {
    using P             = FDMTEngineConfig::GPUPath;
    const SizeType tmin = gpu_direct_min_chunk(max_boxcar);
    std::vector<GPUSearchEntry> list;
    if (cache_chunk != 0) {
        list.push_back({cache_chunk, P::kDirect});
    }
    const auto add = [&](SizeType c, P path) {
        const GPUSearchEntry e{c, path};
        if (std::find(list.begin(), list.end(), e) == list.end()) {
            list.push_back(e);
        }
    };
    for (SizeType c = kGPUSearchMaxChunk; c >= tmin; c /= 2) {
        add(c, P::kDirect);
    }
    for (SizeType c = kGPUSearchMaxChunk; c >= 256; c /= 2) {
        add(c, P::kSkewed);
    }
    for (SizeType c = tmin / 2; c >= 256; c /= 2) {
        add(c, P::kDirect);
    }
    return list;
}

/// Result of choose_gpu_search().
struct GPUSearchChoice {
    std::optional<SizeType> index; ///< into the list; empty: none fits
    SizeType min_needed{std::numeric_limits<SizeType>::max()}; ///< built
};

/**
 * @brief First entry of `list` whose bytes fit `budget`.
 *
 * `measure(entry)` builds the configuration (deferred, at no budget) and
 * returns its budgeted bytes, or nullopt when the configuration is not
 * available for the plan. Chunks are capped at `nsamps`. With `search`,
 * each path's memory is taken as non-decreasing in the chunk (measured; not
 * affine: a longer chunk can move the direct top's level): the list is
 * walked in runs of one path with non-increasing chunks; a run whose last
 * (shortest) entry does not fit is skipped after that one build, else its
 * first fitting entry is found by bisection. Without `search` every entry
 * is built in order (tests). `min_needed` is the smallest bytes built; when
 * nothing fits, every run's shortest entry (so each path's chunk 256) was
 * built, so it is the minimum.
 */
template <typename Measure>
GPUSearchChoice choose_gpu_search(const std::vector<GPUSearchEntry>& list,
                                  SizeType budget,
                                  SizeType nsamps,
                                  Measure&& measure,
                                  bool search = true) {
    using P = FDMTEngineConfig::GPUPath;
    GPUSearchChoice out;
    std::map<std::pair<SizeType, P>, std::optional<SizeType>> built;
    const auto capped = [&](SizeType i) {
        return std::min(list[i].chunk, nsamps);
    };
    const auto fits = [&](SizeType i) {
        const auto key = std::make_pair(capped(i), list[i].path);
        auto it        = built.find(key);
        if (it == built.end()) {
            const auto bytes = measure(GPUSearchEntry{key.first, key.second});
            if (bytes) {
                out.min_needed = std::min(out.min_needed, *bytes);
            }
            it = built.emplace(key, bytes).first;
        }
        return it->second && *it->second <= budget;
    };
    if (!search) {
        for (SizeType i = 0; i < list.size(); ++i) {
            if (fits(i)) {
                out.index = i;
                return out;
            }
        }
        return out;
    }
    for (SizeType first = 0; first < list.size();) {
        SizeType end = first + 1; // the run [first, end)
        while (end < list.size() && list[end].path == list[first].path &&
               capped(end) <= capped(end - 1)) {
            ++end;
        }
        if (fits(end - 1)) {
            // Fitting entries are a suffix of the run: its first one.
            SizeType lo = first;
            SizeType hi = end - 1;
            while (lo < hi) {
                const SizeType mid = lo + ((hi - lo) / 2);
                if (fits(mid)) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            out.index = lo;
            return out;
        }
        first = end;
    }
    return out;
}

} // namespace dmt::algorithms::detail
