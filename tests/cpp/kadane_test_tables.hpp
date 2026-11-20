#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "dmt/common/types.hpp"

#include "dmt/fdmt_variance.hpp"
#include "dmt/kadane_plan.hpp"

namespace dmt::kadane_test {

struct SyntheticPlaneTables {
    std::vector<uint32_t> widths;
    std::vector<float> raw_thresh;
    std::vector<float> inv_sigma;
    std::vector<double> int_mean;
    std::vector<uint64_t> int_thresh;
    SizeType n_rows     = 0;
    double integer_unit = 1.0;
};

/// Build exact analytical row tables matching `make_boxcar_plans()` for rows
/// with specified `(leaf_len, n_active_chans)`.
inline SyntheticPlaneTables
make_synthetic_tables(std::span<const uint32_t> widths,
                      bool use_trapezoid,
                      float detect_thresh,
                      float noise_sigma,
                      std::span<const std::pair<SizeType, SizeType>> row_specs,
                      bool packed       = false,
                      double input_mean = 0.0,
                      int scale_bits    = 0) {
    SyntheticPlaneTables out;
    out.widths.assign(widths.begin(), widths.end());
    out.n_rows         = row_specs.size();
    out.integer_unit   = std::ldexp(1.0, -scale_bits);
    const double scale = std::ldexp(1.0, scale_bits);
    const SizeType nf  = widths.size();

    out.raw_thresh.resize(out.n_rows * nf);
    out.inv_sigma.resize(out.n_rows * nf);
    if (packed) {
        out.int_mean.resize(out.n_rows * nf);
        out.int_thresh.resize(out.n_rows * nf);
    }

    for (SizeType r = 0; r < out.n_rows; ++r) {
        const auto [leaf_len, n_chans] = row_specs[r];
        if (n_chans == 0) {
            for (SizeType f = 0; f < nf; ++f) {
                out.inv_sigma[(r * nf) + f] = 0.0F;
                out.raw_thresh[(r * nf) + f] =
                    std::numeric_limits<float>::max();
                if (packed) {
                    out.int_mean[(r * nf) + f]   = 0.0;
                    out.int_thresh[(r * nf) + f] = UINT64_MAX;
                }
            }
            continue;
        }
        const double weight      = 1.0 / static_cast<double>(leaf_len);
        const double mean_weight = static_cast<double>(n_chans) *
                                   static_cast<double>(leaf_len) * weight;
        for (SizeType f = 0; f < nf; ++f) {
            const uint32_t w = widths[f];
            const uint32_t we =
                use_trapezoid ? kadane::trapezoid_edge_width(w) : 0U;
            const double var = static_cast<double>(n_chans) *
                               algorithms::detail::fdmt_trapezoid_variance(
                                   leaf_len, w, we, weight);
            const double sigma =
                static_cast<double>(noise_sigma) * std::sqrt(var);
            out.inv_sigma[(r * nf) + f] = static_cast<float>(1.0 / sigma);
            out.raw_thresh[(r * nf) + f] =
                static_cast<float>(static_cast<double>(detect_thresh) * sigma);
            if (packed) {
                const double fw = we == 0U ? static_cast<double>(w)
                                           : static_cast<double>(2U * (w - we));
                const double mean = input_mean * mean_weight * fw * scale;
                const double cutoff =
                    std::ceil(mean + (static_cast<double>(detect_thresh) *
                                      sigma * scale));
                out.int_mean[(r * nf) + f] = mean;
                out.int_thresh[(r * nf) + f] =
                    static_cast<uint64_t>(std::max(1.0, cutoff));
            }
        }
    }
    return out;
}

inline bool exact_tile_has_hit_f32(std::span<const float> stream,
                                   SizeType t0,
                                   SizeType tile_len,
                                   std::span<const uint32_t> widths,
                                   bool use_trapezoid,
                                   std::span<const float> row_thr,
                                   SizeType warmup = 0) {
    std::vector<double> prefix(stream.size() + 1, 0.0);
    for (SizeType i = 0; i < stream.size(); ++i) {
        prefix[i + 1] = prefix[i] + static_cast<double>(stream[i]);
    }
    const SizeType t_end = std::min(stream.size(), t0 + tile_len);
    for (SizeType f = 0; f < widths.size(); ++f) {
        const uint32_t w = widths[f];
        const uint32_t we =
            use_trapezoid ? kadane::trapezoid_edge_width(w) : 0U;
        const auto th = static_cast<double>(row_thr[f]);
        for (SizeType t = t0; t < t_end; ++t) {
            if (t + 1 < static_cast<SizeType>(w) || t < warmup + w - 1) {
                continue;
            }
            double sum = prefix[t + 1] - prefix[t + 1 - w];
            if (we > 0U) {
                sum += prefix[t + 1 - we] - prefix[t + 1 - w + we];
            }
            if (sum >= th) {
                return true;
            }
        }
    }
    return false;
}

inline bool exact_tile_has_hit_int(std::span<const uint32_t> stream,
                                   SizeType t0,
                                   SizeType tile_len,
                                   std::span<const uint32_t> widths,
                                   bool use_trapezoid,
                                   std::span<const uint64_t> row_int_thr,
                                   SizeType warmup = 0) {
    std::vector<uint64_t> prefix(stream.size() + 1, 0ULL);
    for (SizeType i = 0; i < stream.size(); ++i) {
        prefix[i + 1] = prefix[i] + static_cast<uint64_t>(stream[i]);
    }
    const SizeType t_end = std::min(stream.size(), t0 + tile_len);
    for (SizeType f = 0; f < widths.size(); ++f) {
        const uint32_t w = widths[f];
        const uint32_t we =
            use_trapezoid ? kadane::trapezoid_edge_width(w) : 0U;
        const uint64_t thr = row_int_thr[f];
        for (SizeType t = t0; t < t_end; ++t) {
            if (t + 1 < static_cast<SizeType>(w) || t < warmup + w - 1) {
                continue;
            }
            uint64_t sum = prefix[t + 1] - prefix[t + 1 - w];
            if (we > 0U) {
                sum += prefix[t + 1 - we] - prefix[t + 1 - w + we];
            }
            if (sum >= thr) {
                return true;
            }
        }
    }
    return false;
}

} // namespace dmt::kadane_test
