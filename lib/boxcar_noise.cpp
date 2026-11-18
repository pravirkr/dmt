#include "dmt/boxcar_noise.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "dmt/fdmt_box_weights.hpp"
#include "dmt/fdmt_variance.hpp"

namespace dmt::algorithms::detail {

namespace {
// As in boxcar_plans.cpp: an opaque integer test, safe under -ffast-math.
bool finite(float x) {
    const volatile uint32_t bits = std::bit_cast<uint32_t>(x);
    return (bits & 0x7F800000U) != 0x7F800000U;
}
} // namespace

BoxcarNoiseModel::BoxcarNoiseModel(const plans::FDMTPlan& plan,
                                   const BoxcarConfig& config,
                                   bool smear,
                                   int nthreads,
                                   std::span<const uint8_t> reference)
    : m_plan(plan),
      m_config(config),
      m_smear(smear),
      m_packed(config.input_nbits != 32),
      m_threads(std::max(1, nthreads)),
      m_root(plan.get_niters()),
      m_first_searched(plan.get_niters() - plan.get_subband_depth()),
      m_widths(
          plans::generate_box_width_trials(config.max_boxcar, config.wtsp)),
      m_live(std::make_shared<std::atomic<SizeType>>(0)) {
    const auto& pc = plan.get_container();
    // Rows in make_boxcar_plans() order: searched levels, subbands, the
    // subband's searched trials (FDMTCoordGrid::search_rows; empty: all).
    for (SizeType l = m_first_searched; l <= m_root; ++l) {
        for (const auto& grid : pc.grids[l]) {
            const auto add = [&](SizeType d) {
                m_row_node.emplace_back(
                    static_cast<uint32_t>(l),
                    static_cast<uint32_t>(grid.coord_offset + d));
            };
            if (grid.search_rows.empty()) {
                for (SizeType d = 0; d < grid.ndt; ++d) {
                    add(d);
                }
            } else {
                for (const auto d : grid.search_rows) {
                    add(d);
                }
            }
        }
    }
    // Support of each coordinate with every channel active, level by level:
    // a leaf reaches |dt| samples back, a head shifted by the merge delay.
    std::vector<uint64_t> below;
    std::vector<uint64_t> level;
    std::vector<std::vector<uint64_t>> searched(m_root - m_first_searched + 1);
    for (SizeType l = 0; l <= m_root; ++l) {
        const auto& coords = pc.coordinates[l];
        level.assign(coords.size(), 0);
        for (SizeType i = 0; i < coords.size(); ++i) {
            const auto& c = coords[i];
            if (l == 0) {
                const auto dt = pc.grids[0][c.i_sub].dt_grid[c.i_dt];
                level[i]      = static_cast<uint64_t>(std::abs(dt));
                if (!pc.level0_boxes.empty()) {
                    // chan_dm plan: the farthest read of the level-0 row.
                    const auto& box = pc.level0_boxes[i];
                    level[i] = static_cast<uint64_t>(smear ? box.lag + box.s
                                                           : box.point);
                }
                continue;
            }
            const SizeType prev = pc.state_shape[l - 1].nsamps;
            uint64_t s          = below[c.tail_buf_offset / prev];
            if (c.i_coord_head != SIZE_MAX) {
                s = std::max(s, below[c.head_buf_offset / prev] + c.delay);
            }
            level[i] = s;
        }
        if (l >= m_first_searched) {
            searched[l - m_first_searched] = level;
        }
        below.swap(level);
    }
    auto support = std::make_shared<std::vector<uint64_t>>(m_row_node.size());
    for (SizeType r = 0; r < m_row_node.size(); ++r) {
        const auto [l, node] = m_row_node[r];
        (*support)[r]        = searched[l - m_first_searched][node];
        m_max_support        = std::max(m_max_support, (*support)[r]);
    }
    m_support = std::move(support);
    m_searched.resize(m_root - m_first_searched + 1);
    m_offsets.resize(m_root + 1);
    fill({}, true);
    if (m_packed) {
        // Killed channels read as the code input_mean: exact only for an
        // integer code (the dyadic leaf weights keep every mean an integer).
        if (!(std::floor(m_config.input_mean) == m_config.input_mean)) {
            throw std::invalid_argument(
                "FDMTBoxcar: per-call kill masks on packed input need an "
                "integer input_mean (killed channels read as that code)");
        }
        if (!reference.empty() && reference.size() != plan.get_nchans()) {
            throw std::invalid_argument("FDMTBoxcar: kill mask size mismatch");
        }
        m_reference.assign(plan.get_nchans(), 1);
        for (SizeType c = 0; c < reference.size(); ++c) {
            m_reference[c] = reference[c] != 0 ? 1 : 0;
        }
        fill(m_reference, false);
        m_reference_mean.resize(rows());
        for (SizeType r = 0; r < rows(); ++r) {
            m_reference_mean[r] = mean_weight(r);
        }
    }
}

double BoxcarNoiseModel::mean_weight(SizeType r) const noexcept {
    const auto [l, node] = m_row_node[r];
    const auto& b   = l >= m_first_searched ? m_searched[l - m_first_searched]
                                            : m_below[l % 2];
    const auto& off = m_offsets[l];
    const double unit =
        std::ldexp(1.0, -static_cast<int>(m_plan.get_box_fraction_bits()));
    double sum = 0;
    for (SizeType i = off[node]; i < off[node + 1]; ++i) {
        const Bin& e = b[i];
        if (e.count == 0) {
            continue;
        }
        const SizeType len = e.len;
        const double weight =
            m_smear ? box_weight(len, m_plan.get_box_fraction_bits()) * unit
                    : 1.0;
        sum += static_cast<double>(e.count) * static_cast<double>(len) * weight;
    }
    return sum;
}

std::vector<BoxcarNoiseModel::Bin>&
BoxcarNoiseModel::bins(SizeType level) noexcept {
    return level >= m_first_searched ? m_searched[level - m_first_searched]
                                     : m_below[level % 2];
}

void BoxcarNoiseModel::fill(std::span<const uint8_t> mask, bool layout) {
    const auto& pc = m_plan.get_container();
    SizeType below_max[2]{0, 0};
    for (SizeType l = 0; l <= m_root; ++l) {
        const auto& coords = pc.coordinates[l];
        const auto n       = static_cast<std::ptrdiff_t>(coords.size());
        auto& off          = m_offsets[l];
        auto& out          = bins(l);
        if (!layout) {
            // Same size as the layout (within the reserved capacity).
            out.resize(off.back());
        }
        if (l == 0) {
            if (layout) {
                off.resize(coords.size() + 1);
                for (SizeType i = 0; i <= coords.size(); ++i) {
                    off[i] = i;
                }
                out.resize(coords.size());
            }
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                const auto& c = coords[i];
                const auto dt = pc.grids[0][c.i_sub].dt_grid[c.i_dt];
                const auto s =
                    pc.level0_boxes.empty()
                        ? static_cast<uint32_t>(std::abs(dt))
                        : static_cast<uint32_t>(
                              pc.level0_boxes[static_cast<SizeType>(i)].s);
                const auto len = m_smear ? s + 1U : 1U;
                const bool on  = mask.empty() || mask[c.i_sub] != 0;
                out[i]         = {len, on ? 1U : 0U};
            }
        } else {
            const SizeType prev = pc.state_shape[l - 1].nsamps;
            const auto& in      = bins(l - 1);
            const auto& in_off  = m_offsets[l - 1];
            // Merge of the (ascending) tail and head histograms.
            const auto merge = [&](SizeType i, Bin* dst) -> SizeType {
                const auto& c    = coords[i];
                const SizeType a = c.tail_buf_offset / prev;
                const Bin* x     = in.data() + in_off[a];
                const Bin* xe    = in.data() + in_off[a + 1];
                const Bin* y     = nullptr;
                const Bin* ye    = nullptr;
                if (c.i_coord_head != SIZE_MAX) {
                    const SizeType b = c.head_buf_offset / prev;
                    y                = in.data() + in_off[b];
                    ye               = in.data() + in_off[b + 1];
                }
                SizeType k = 0;
                while (x != xe || y != ye) {
                    Bin v{};
                    if (y == ye || (x != xe && x->len < y->len)) {
                        v = *x++;
                    } else if (x == xe || y->len < x->len) {
                        v = *y++;
                    } else {
                        v = {x->len, x->count + y->count};
                        ++x;
                        ++y;
                    }
                    if (dst != nullptr) {
                        dst[k] = v;
                    }
                    ++k;
                }
                return k;
            };
            if (layout) {
                off.assign(coords.size() + 1, 0);
#pragma omp parallel for num_threads(m_threads) schedule(dynamic, 256)
                for (std::ptrdiff_t i = 0; i < n; ++i) {
                    off[i + 1] = merge(static_cast<SizeType>(i), nullptr);
                }
                for (SizeType i = 0; i < coords.size(); ++i) {
                    off[i + 1] += off[i];
                }
                out.resize(off.back());
            }
#pragma omp parallel for num_threads(m_threads) schedule(dynamic, 256)
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                merge(static_cast<SizeType>(i), out.data() + off[i]);
            }
        }
        if (layout && l < m_first_searched) {
            below_max[l % 2] = std::max(below_max[l % 2], out.size());
        }
    }
    if (layout) {
        // Keep the alternating buffers at their largest level, so later
        // fills never reallocate (and the usage stays what was reported).
        for (int k = 0; k < 2; ++k) {
            m_below[k].reserve(below_max[k]);
        }
    }
}

std::shared_ptr<const NoiseTables>
BoxcarNoiseModel::compute(std::span<const uint8_t> mask) {
    if (mask.size() != m_plan.get_nchans()) {
        throw std::invalid_argument("FDMTBoxcar: kill mask size mismatch");
    }
    if (m_packed) {
        for (SizeType c = 0; c < mask.size(); ++c) {
            if (mask[c] != 0 && m_reference[c] == 0) {
                throw std::invalid_argument(
                    "FDMTBoxcar: a packed per-call mask keeps a channel the "
                    "constructor mask kills");
            }
        }
    }
    // The bins below the searched levels are rebuilt level by level into
    // the two alternating buffers: resize within their reserved capacity.
    fill(mask, false);
    const SizeType nf = m_widths.size();
    auto tables       = std::make_shared<NoiseTables>();
    tables->thresh.resize(rows() * nf);
    tables->inv_sigma.resize(rows() * nf);
    if (m_packed) {
        tables->integer_thresh.resize(rows() * nf);
    }
    const double unit =
        std::ldexp(1.0, -static_cast<int>(m_plan.get_box_fraction_bits()));
    bool ok          = true;
    const auto nrows = static_cast<std::ptrdiff_t>(rows());
#pragma omp parallel for num_threads(m_threads) schedule(dynamic, 64)          \
    reduction(&& : ok)
    for (std::ptrdiff_t r = 0; r < nrows; ++r) {
        const auto [l, node] = m_row_node[r];
        const auto& b        = bins(l);
        const auto& off      = m_offsets[l];
        const Bin* h         = b.data() + off[node];
        const Bin* he        = b.data() + off[node + 1];
        // Packed: the row's mean weight under this mask and the reference.
        const double mw     = m_packed ? mean_weight(r) : 0.0;
        const double mw_ref = m_packed ? m_reference_mean[r] : 0.0;
        // The arithmetic and order of make_boxcar_plans(): ascending
        // length over the lengths present (nonzero counts).
        for (SizeType f = 0; f < nf; ++f) {
            const SizeType width = m_widths[f];
            const SizeType w_edge =
                m_config.use_trapezoid ? trapezoid_edge_width(m_widths[f]) : 0;
            double variance = 0;
            for (const Bin* e = h; e != he; ++e) {
                if (e->count == 0) {
                    continue;
                }
                const SizeType len = e->len;
                const double weight =
                    m_smear
                        ? box_weight(len, m_plan.get_box_fraction_bits()) * unit
                        : 1.0;
                const auto cnt = static_cast<double>(e->count);
                variance +=
                    cnt * fdmt_trapezoid_variance(len, width, w_edge, weight);
            }
            const SizeType k = (static_cast<SizeType>(r) * nf) + f;
            if (variance == 0) {
                tables->inv_sigma[k] = 0;
                tables->thresh[k]    = std::numeric_limits<float>::max();
                if (m_packed) {
                    tables->integer_thresh[k] = UINT64_MAX;
                }
                continue;
            }
            const double sigma    = m_config.noise_sigma * std::sqrt(variance);
            const float inv_sigma = static_cast<float>(1.0 / sigma);
            const float threshold =
                static_cast<float>(m_config.detect_thresh * sigma);
            ok = ok && finite(inv_sigma) && finite(threshold) &&
                 inv_sigma > 0 && threshold > 0;
            tables->inv_sigma[k] = inv_sigma;
            tables->thresh[k]    = threshold;
            if (m_packed) {
                // make_boxcar_plans()'s cutoff for this mask, then moved by
                // the exact difference of the reference mean and this
                // mask's (killed channels add input_mean each).
                const double scale = m_smear ? 1.0 / unit : 1.0;
                const double filter_weight =
                    w_edge == 0 ? static_cast<double>(width)
                                : static_cast<double>(2 * (width - w_edge));
                const double mean =
                    m_config.input_mean * mw * filter_weight * scale;
                const double mean_ref =
                    m_config.input_mean * mw_ref * filter_weight * scale;
                const double cutoff =
                    std::ceil(mean + (m_config.detect_thresh * sigma * scale));
                const double shift = mean_ref - mean;
                // A whole number (integer mean code, dyadic weights).
                const bool fits = cutoff < std::ldexp(1.0, 64) && shift >= 0 &&
                                  shift < std::ldexp(1.0, 63) &&
                                  shift == std::floor(shift);
                ok              = ok && fits;
                if (fits) {
                    const auto c0             = static_cast<uint64_t>(cutoff);
                    const auto sh             = static_cast<uint64_t>(shift);
                    ok                        = ok && c0 <= UINT64_MAX - sh;
                    tables->integer_thresh[k] = c0 + sh;
                }
            }
        }
    }
    if (!ok) {
        throw std::invalid_argument(
            m_packed ? "FDMTBoxcar: sigma/threshold is outside float32 range "
                       "or the integer threshold exceeds uint64"
                     : "FDMTBoxcar: sigma/threshold is outside float32 range");
    }
    tables->live = m_live;
    tables->id   = m_next_id++;
    m_live->fetch_add(tables->bytes(), std::memory_order_relaxed);
    return tables;
}

SizeType BoxcarNoiseModel::plan_bytes() const noexcept {
    return (m_row_node.size() * sizeof(m_row_node[0])) +
           (m_support->size() * sizeof(uint64_t)) +
           (m_reference_mean.size() * sizeof(double)) + m_reference.size() +
           // FDMTBoxcar's LiveHistory (one timestamp per channel).
           (m_plan.get_nchans() * sizeof(uint64_t));
}

SizeType BoxcarNoiseModel::workspace_bytes() const noexcept {
    SizeType bytes = 0;
    for (const auto& o : m_offsets) {
        bytes += o.capacity() * sizeof(uint64_t);
    }
    for (const auto& s : m_searched) {
        bytes += s.capacity() * sizeof(Bin);
    }
    bytes += (m_below[0].capacity() + m_below[1].capacity()) * sizeof(Bin);
    return bytes;
}

} // namespace dmt::algorithms::detail
