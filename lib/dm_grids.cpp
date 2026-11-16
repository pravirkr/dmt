#include "dmt/common/dm_grids.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>

#include "dmt/dm_utils.hpp"

namespace dmt::plans {

namespace {

// The dispersion constant of dmtlib (Manchester & Taylor), in double.
constexpr double kDMConstant = 1.0 / 2.41e-4;

// Boxcar widths (samples) the S/N model searches over.
constexpr std::array<double, 19> kWidths{
    1.0,  2.0,  3.0,  4.0,   6.0,   8.0,   12.0,   16.0,   24.0,  32.0,
    48.0, 64.0, 96.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0};
constexpr SizeType kNumWidths = kWidths.size();
// Channels are pooled into at most this many contiguous groups.
constexpr SizeType kMaxGroups = 64;
// Pulse arrival times tried across the band's delay span.
constexpr SizeType kNumTaus = 21;
// Scale of the modelled per-channel pulse width (1: second-moment match).
constexpr double kWidthScale = 1.0;
// Log-spaced nodes of the half step over |DM - chan_dm| (plus 0).
constexpr SizeType kNumNodes = 36;
constexpr int kBisections    = 22;

void validate_band(const DMGridConfig& c) {
    // Bit tests: -ffast-math may fold std::isfinite (dm_utils.hpp).
    const auto finite = [](double x) { return utils::is_finite_bits(x); };
    if (!finite(c.f_min) || !finite(c.f_max) || !finite(c.tsamp) ||
        !finite(c.pulse_width) || !finite(c.chan_dm)) {
        throw std::invalid_argument("DM grid band parameters must be finite");
    }
    if (!(c.f_min > 0.0) || !(c.f_max > c.f_min)) {
        throw std::invalid_argument(
            fmt::format("DM grid: invalid frequency band f_min={}, f_max={}",
                        c.f_min, c.f_max));
    }
    if (c.nchans < 2 || !(c.tsamp > 0.0)) {
        throw std::invalid_argument(
            fmt::format("DM grid: invalid nchans={} (must be >= 2) or tsamp={}",
                        c.nchans, c.tsamp));
    }
    if (c.pulse_width < 0.0) {
        throw std::invalid_argument("DM grid: pulse_width must be >= 0");
    }
    if (!c.kill_mask.empty()) {
        if (c.kill_mask.size() != c.nchans) {
            throw std::invalid_argument(
                fmt::format("DM grid: kill_mask has {} entries, expected {}",
                            c.kill_mask.size(), c.nchans));
        }
        const auto live = std::ranges::count_if(
            c.kill_mask, [](uint8_t k) { return k != 0; });
        if (live < 2) {
            throw std::invalid_argument(fmt::format(
                "DM grid: need at least 2 active channels (got {})", live));
        }
    }
}

void validate_range(double dm_min, double dm_max) {
    if (!utils::is_finite_bits(dm_min) || !utils::is_finite_bits(dm_max)) {
        throw std::invalid_argument(
            "DM grid: dm_min and dm_max must be finite");
    }
    if (dm_max < dm_min) {
        throw std::invalid_argument(fmt::format(
            "DM grid: dm_max ({}) must be >= dm_min ({})", dm_max, dm_min));
    }
}

/**
 * The S/N a pulse keeps after the boxcar search when the trial DM is off by
 * delta, modelled over channel groups.
 *
 * Channel group g (n_g channels) delays a pulse by r_g samples per unit DM
 * relative to the band top and smears it over s_g = u * s_per_dm_g samples
 * at u = |DM - chan_dm|. FDMT's level-0 box (L_g = 1 + s_g samples) adds
 * the same again, so each channel's pulse is a top hat of second moment
 * (w0^2 + 1 + 2 s_g^2) / 12, modelled as a Gaussian of that width, with
 * w0 = pulse_width / tsamp. A boxcar of W samples at arrival tau then
 * collects n_g / 2 [erf((delta r_g - tau + W/2) / D_g) -
 * erf((delta r_g - tau - W/2) / D_g)], D_g = sqrt(2) sigma_g, over noise
 * of variance sum_g n_g V_g(W), V_g the exact variance of a box of L_g
 * samples summed over W. The S/N is the best over W and tau, relative to
 * delta = 0.
 */
class LossModel {
public:
    explicit LossModel(const DMGridConfig& c) : m_w0(c.pulse_width / c.tsamp) {
        const double df = (c.f_max - c.f_min) / static_cast<double>(c.nchans);
        std::vector<double> r_all;
        std::vector<double> s_all;
        for (SizeType ch = 0; ch < c.nchans; ++ch) {
            if (!c.kill_mask.empty() && c.kill_mask[ch] == 0) {
                continue;
            }
            const double f_lo  = c.f_min + (static_cast<double>(ch) * df);
            const double f_hi  = f_lo + df;
            const double f_mid = 0.5 * (f_lo + f_hi);
            r_all.push_back(
                kDMConstant *
                ((1.0 / (f_mid * f_mid)) - (1.0 / (c.f_max * c.f_max))) /
                c.tsamp);
            s_all.push_back(kDMConstant *
                            ((1.0 / (f_lo * f_lo)) - (1.0 / (f_hi * f_hi))) /
                            c.tsamp);
        }
        // Contiguous groups of active channels, the first n % m one larger.
        const SizeType n = r_all.size();
        if (n < 2) {
            throw std::invalid_argument(fmt::format(
                "DM grid: need at least 2 active channels (got {})", n));
        }
        const SizeType m = std::min(n, kMaxGroups);
        SizeType start   = 0;
        for (SizeType g = 0; g < m; ++g) {
            const SizeType len = (n / m) + (g < n % m ? 1 : 0);
            double r_sum       = 0.0;
            double s_sum       = 0.0;
            for (SizeType i = start; i < start + len; ++i) {
                r_sum += r_all[i];
                s_sum += s_all[i];
            }
            const auto cnt = static_cast<double>(len);
            m_n.push_back(cnt);
            m_r.push_back(r_sum / cnt);
            m_s.push_back(s_sum / cnt);
            start += len;
        }
        m_r_min    = *std::ranges::min_element(m_r);
        m_r_max    = *std::ranges::max_element(m_r);
        m_span_r   = std::max(m_r_max - m_r_min, 1e-12);
        m_s_max_ch = *std::ranges::max_element(s_all);
    }

    /// Largest single-channel smearing per unit DM (samples).
    double max_channel_smearing() const noexcept { return m_s_max_ch; }

    /// Per-u state: the groups' pulse widths and the noise of each boxcar.
    struct AtU {
        std::vector<double> width; // D_g
        std::array<double, kNumWidths> noise{};
        std::array<double, kNumWidths> snr_w{}; // S/N of each width at 0
        double snr0{};
        double min_width{};
    };

    AtU at(double u) const {
        AtU a;
        const SizeType m = m_n.size();
        a.width.resize(m);
        std::array<double, kNumWidths> var{};
        for (SizeType g = 0; g < m; ++g) {
            const double s = u * m_s[g];
            const double l = 1.0 + s;
            a.width[g] = kWidthScale *
                         std::sqrt(((m_w0 * m_w0) + 1.0 + (2.0 * s * s)) / 6.0);
            for (SizeType w = 0; w < kNumWidths; ++w) {
                const double mn = std::min(l, kWidths[w]);
                const double mx = std::max(l, kWidths[w]);
                var[w] += m_n[g] *
                          (((mx - mn + 1.0) * mn * mn) +
                           ((mn - 1.0) * mn * ((2.0 * mn) - 1.0) / 3.0)) /
                          (l * l);
            }
        }
        a.min_width = *std::ranges::min_element(a.width);
        a.snr0      = 0.0;
        for (SizeType w = 0; w < kNumWidths; ++w) {
            a.noise[w] = std::sqrt(var[w]);
            double sig = 0.0;
            for (SizeType g = 0; g < m; ++g) {
                sig += m_n[g] * std::erf(0.5 * kWidths[w] / a.width[g]);
            }
            a.snr_w[w] = sig / a.noise[w];
            a.snr0     = std::max(a.snr0, a.snr_w[w]);
        }
        return a;
    }

    /// S/N kept at a trial DM delta away, relative to delta = 0.
    double ratio(const AtU& a, double delta) const {
        const SizeType m = m_n.size();
        double best      = 0.0;
        for (SizeType t = 0; t < kNumTaus; ++t) {
            const double unit =
                static_cast<double>(t) / static_cast<double>(kNumTaus - 1);
            const double tau = delta * (m_r_min + (unit * m_span_r));
            std::array<double, kNumWidths> sig{};
            for (SizeType g = 0; g < m; ++g) {
                const double err = (delta * m_r[g]) - tau;
                const double inv = 1.0 / a.width[g];
                const double hn  = 0.5 * m_n[g];
                for (SizeType w = 0; w < kNumWidths; ++w) {
                    const double hw = 0.5 * kWidths[w];
                    sig[w] += hn * (std::erf((err + hw) * inv) -
                                    std::erf((err - hw) * inv));
                }
            }
            for (SizeType w = 0; w < kNumWidths; ++w) {
                best = std::max(best, sig[w] / a.noise[w]);
            }
        }
        return best / a.snr0;
    }

    /**
     * ratio(a, delta) >= target (> target when @p strict), without the
     * full maximum: a width whose S/N at delta = 0 is below the target
     * cannot reach it at any tau (a centred window holds the most of a
     * Gaussian), and the search stops at the first window that does.
     */
    bool reaches(const AtU& a, double delta, double target, bool strict) const {
        std::array<double, kNumWidths> hw{};
        std::array<double, kNumWidths> noise{};
        SizeType nw = 0;
        for (SizeType w = 0; w < kNumWidths; ++w) {
            // Margin for rounding of the bound.
            if (a.snr_w[w] >= target * a.snr0 * (1.0 - 1e-9)) {
                hw[nw]    = 0.5 * kWidths[w];
                noise[nw] = a.noise[w];
                ++nw;
            }
        }
        const SizeType m = m_n.size();
        std::array<double, kMaxGroups> inv{};
        std::array<double, kMaxGroups> hn{};
        std::array<double, kMaxGroups> err{};
        for (SizeType g = 0; g < m; ++g) {
            inv[g] = 1.0 / a.width[g];
            hn[g]  = 0.5 * m_n[g];
        }
        for (SizeType t = 0; t < kNumTaus; ++t) {
            const double unit =
                static_cast<double>(t) / static_cast<double>(kNumTaus - 1);
            const double tau = delta * (m_r_min + (unit * m_span_r));
            for (SizeType g = 0; g < m; ++g) {
                err[g] = (delta * m_r[g]) - tau;
            }
            for (SizeType k = 0; k < nw; ++k) {
                double sig = 0.0;
                for (SizeType g = 0; g < m; ++g) {
                    sig += hn[g] * (std::erf((err[g] + hw[k]) * inv[g]) -
                                    std::erf((err[g] - hw[k]) * inv[g]));
                }
                const double r = (sig / noise[k]) / a.snr0;
                if (strict ? r > target : r >= target) {
                    return true;
                }
            }
        }
        return false;
    }

    /// DM offset at which ratio() falls to @p target (bisection), at most
    /// @p cap.
    double half_step(double u, double target, double cap) const {
        const auto a = at(u);
        double lo    = 0.0;
        double hi    = std::max(1e-5, 6.0 * a.min_width / m_span_r);
        while (reaches(a, hi, target, true)) {
            hi *= 2.0;
            if (hi >= cap) {
                return cap; // the pulse keeps its S/N across the range
            }
        }
        for (int i = 0; i < kBisections; ++i) {
            const double mid = 0.5 * (lo + hi);
            if (reaches(a, mid, target, false)) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return 0.5 * (lo + hi);
    }

private:
    double m_w0;
    std::vector<double> m_n;
    std::vector<double> m_r;
    std::vector<double> m_s;
    double m_r_min{};
    double m_r_max{};
    double m_span_r{};
    double m_s_max_ch{};
};

/// The half step h(u) of a grid, sampled at nodes over u = |DM - chan_dm|
/// and interpolated linearly.
class HalfStep {
public:
    HalfStep(const LossModel& model, double target, double u_max, double cap) {
        u_max = std::max(u_max, 1e-6);
        const double u_scale =
            std::max(1.0 / std::max(model.max_channel_smearing(), 1e-12), 1e-4);
        const double lo_node = std::min(u_scale * 0.05, 0.1 * u_max);
        m_u.push_back(0.0);
        const double l0 = std::log(lo_node);
        const double l1 = std::log(u_max);
        for (SizeType i = 0; i < kNumNodes; ++i) {
            const double u =
                i == 0 ? lo_node
                : i + 1 == kNumNodes
                    ? u_max
                    : std::exp(l0 + ((l1 - l0) * static_cast<double>(i) /
                                     static_cast<double>(kNumNodes - 1)));
            if (u > m_u.back()) {
                m_u.push_back(u);
            }
        }
        m_h.reserve(m_u.size());
        for (const double u : m_u) {
            m_h.push_back(model.half_step(u, target, cap));
        }
    }

    double operator()(double u) const {
        if (u <= m_u.front()) {
            return m_h.front();
        }
        if (u >= m_u.back()) {
            return m_h.back();
        }
        const auto it  = std::ranges::upper_bound(m_u, u);
        const auto i   = static_cast<SizeType>(it - m_u.begin());
        const double f = (u - m_u[i - 1]) / (m_u[i] - m_u[i - 1]);
        return m_h[i - 1] + (f * (m_h[i] - m_h[i - 1]));
    }

private:
    std::vector<double> m_u;
    std::vector<double> m_h;
};

constexpr SizeType kMaxGridTrials = 10'000'000;

/// Trials from a to b (a < b) by midpoint steps of h(|DM - chan_dm|).
std::vector<double> step_grid(
    const HalfStep& h, double a, double b, double chan_dm, bool clamp_end) {
    std::vector<double> dms{a};
    double cur = a;
    while (cur < b) {
        const double h0  = h(std::abs(cur - chan_dm));
        const double hm  = h(std::abs(std::min(b, cur + h0) - chan_dm));
        const double nxt = cur + h0 + hm;
        if (!(nxt > cur)) {
            throw std::invalid_argument(
                "DM grid: step underflowed floating-point precision");
        }
        if (dms.size() >= kMaxGridTrials) {
            throw std::invalid_argument(fmt::format(
                "DM grid: trial count exceeds maximum ({})", kMaxGridTrials));
        }
        if (nxt >= b) {
            dms.push_back(clamp_end ? b : nxt);
            break;
        }
        dms.push_back(nxt);
        cur = nxt;
    }
    return dms;
}

/// Uniform segments over a grid, split where its step halves or doubles,
/// each at the smallest step inside it and at least min_run trials long.
std::vector<double> piecewise_wrap(const std::vector<double>& dms,
                                   SizeType min_run) {
    const SizeType n = dms.size();
    if (n < 2) {
        return dms;
    }
    min_run          = std::max<SizeType>(min_run, 1);
    const auto st    = [&](SizeType i) { return dms[i + 1] - dms[i]; };
    const double end = dms.back();
    std::vector<double> out;
    double a   = dms.front();
    SizeType i = 0;
    while (a < end) {
        while (i + 2 < n && dms[i + 1] <= a) {
            ++i;
        }
        const double anchor = st(i);
        double step         = anchor;
        SizeType j          = i;
        while (j + 2 < n) {
            const double next = st(j + 1);
            if (!(0.5 * anchor < next && next < 2.0 * anchor)) {
                break;
            }
            ++j;
            step = std::min(step, next);
        }
        const double b = std::min(end, dms[j + 1]);
        auto count     = static_cast<SizeType>(std::ceil((b - a) / step));
        if (count < min_run) {
            step  = (b - a) / static_cast<double>(min_run);
            count = min_run;
        }
        count = std::max<SizeType>(count, 1);
        while (a + (step * static_cast<double>(count)) < b) {
            ++count;
        }
        for (SizeType q = 0; q < count; ++q) {
            out.push_back(a + (step * static_cast<double>(q)));
        }
        a += step * static_cast<double>(count);
    }
    out.push_back(a);
    return out;
}

double target_loss(const DMGridConfig& c) {
    if (c.tol != 0.0) {
        if (!utils::is_finite_bits(c.tol) || !(c.tol > 1.0)) {
            throw std::invalid_argument(
                fmt::format("DM grid: tol must be > 1 (or 0), got {}", c.tol));
        }
        const double loss = 1.0 - (1.0 / c.tol);
        if (!(loss > 0.0 && loss < 1.0 && 1.0 - loss < 1.0)) {
            throw std::invalid_argument(fmt::format(
                "DM grid: tol ({}) produces an unusable target loss", c.tol));
        }
        return loss;
    }
    if (!utils::is_finite_bits(c.max_snr_loss) ||
        !(c.max_snr_loss > 0.0 && c.max_snr_loss < 1.0) ||
        !(1.0 - c.max_snr_loss < 1.0)) {
        throw std::invalid_argument(fmt::format(
            "DM grid: max_snr_loss must be in (0, 1) and resolve in float64, "
            "got {}",
            c.max_snr_loss));
    }
    return c.max_snr_loss;
}

} // namespace

std::vector<double>
generate_dm_grid(const DMGridConfig& config, double dm_min, double dm_max) {
    validate_band(config);
    validate_range(dm_min, dm_max);
    const double target = 1.0 - target_loss(config);
    if (dm_min == dm_max) {
        return {dm_min};
    }
    const double cd   = config.chan_dm;
    const double span = dm_max - dm_min;
    const double d_lo = dm_min - cd;
    const double d_hi = dm_max - cd;
    if (!utils::is_finite_bits(span) || !utils::is_finite_bits(d_lo) ||
        !utils::is_finite_bits(d_hi)) {
        throw std::invalid_argument("DM grid: DM range overflows double");
    }
    const LossModel model(config);
    // One step table for the whole range; a step never needs to exceed it.
    const double u_max       = std::max(std::abs(d_lo), std::abs(d_hi));
    constexpr double kMaxCap = 0.25 * std::numeric_limits<double>::max();
    const double cap         = span > 0.5 * kMaxCap ? kMaxCap : 2.0 * span;
    const HalfStep h(model, target, u_max, cap);
    const bool clamp_end = !config.piecewise;
    std::vector<double> out;
    if (dm_max <= 0.0) {
        // Mirror (u = |DM - chan_dm| is unchanged with chan_dm mirrored).
        const auto pos = step_grid(h, -dm_max, -dm_min, -cd, clamp_end);
        out.reserve(pos.size());
        for (auto it = pos.rbegin(); it != pos.rend(); ++it) {
            out.push_back(-*it);
        }
        // dm_max = 0 negates to -0.0; store +0.0 (fast-math folds a
        // conditional negation back into -x, so assign the literal).
        if (dm_max == 0.0) {
            out.back() = 0.0;
        }
    } else if (dm_min < 0.0 && cd == 0.0) {
        // Negative trials mirror the positive ones exactly.
        const auto neg = step_grid(h, 0.0, -dm_min, 0.0, clamp_end);
        const auto pos = step_grid(h, 0.0, dm_max, 0.0, clamp_end);
        out.reserve(neg.size() + pos.size() - 1);
        for (auto it = neg.rbegin(); it + 1 != neg.rend(); ++it) {
            out.push_back(-*it);
        }
        out.insert(out.end(), pos.begin(), pos.end());
    } else {
        out = step_grid(h, dm_min, dm_max, cd, clamp_end);
    }
    if (config.piecewise) {
        return piecewise_wrap(out, config.min_run);
    }
    return out;
}

std::vector<double> generate_levin_dm_grid(const DMGridConfig& config,
                                           double dm_min,
                                           double dm_max) {
    validate_band(config);
    validate_range(dm_min, dm_max);
    if (!utils::is_finite_bits(config.tol) ||
        (config.tol != 0.0 && !(config.tol > 1.0))) {
        throw std::invalid_argument(
            fmt::format("DM grid: tol must be > 1 (or 0), got {}", config.tol));
    }
    if (config.chan_dm != 0.0) {
        throw std::invalid_argument(
            "DM grid: the Levin rule does not take chan_dm (use "
            "generate_dm_grid)");
    }
    if (!config.kill_mask.empty()) {
        throw std::invalid_argument(
            "DM grid: the Levin rule does not take kill_mask (use "
            "generate_dm_grid)");
    }
    if (dm_min < 0.0) {
        throw std::invalid_argument(fmt::format(
            "DM grid: the Levin rule does not support negative dm_min ({})",
            dm_min));
    }
    // The Levin rule works in float (DDMTPlan::generate_levin_dm_grid).
    const auto f   = [](double x) { return static_cast<float>(x); };
    const auto tol = f(config.tol != 0.0 ? config.tol : 1.25);
    const auto levin =
        config.piecewise
            ? utils::generate_levin_dm_grid_piecewise(
                  f(dm_min), f(dm_max), f(config.tsamp), f(config.pulse_width),
                  f(config.f_min), f(config.f_max), config.nchans, tol,
                  config.min_run)
            : utils::generate_levin_dm_grid(
                  f(dm_min), f(dm_max), f(config.tsamp), f(config.pulse_width),
                  f(config.f_min), f(config.f_max), config.nchans, tol);
    return {levin.begin(), levin.end()};
}

std::vector<double> calculate_snr_loss(const DMGridConfig& config,
                                       std::span<const double> trial_dms,
                                       std::span<const double> dm_test) {
    validate_band(config);
    if (trial_dms.empty()) {
        throw std::invalid_argument("calculate_snr_loss: no trial DMs");
    }
    for (const double x : trial_dms) {
        if (!utils::is_finite_bits(x)) {
            throw std::invalid_argument(
                "calculate_snr_loss: trial DMs must be finite");
        }
    }
    if (!std::ranges::is_sorted(trial_dms)) {
        throw std::invalid_argument(
            "calculate_snr_loss: trial_dms must be sorted ascending");
    }
    const LossModel model(config);
    std::vector<double> loss;
    loss.reserve(dm_test.size());
    for (const double dm : dm_test) {
        if (!utils::is_finite_bits(dm)) {
            throw std::invalid_argument(
                "calculate_snr_loss: test DMs must be finite");
        }
        const auto it = std::ranges::lower_bound(trial_dms, dm);
        double delta =
            std::abs(it == trial_dms.end() ? dm - trial_dms.back() : *it - dm);
        if (it != trial_dms.begin()) {
            delta = std::min(delta, std::abs(dm - *(it - 1)));
        }
        if (delta == 0.0) {
            loss.push_back(0.0);
            continue;
        }
        const auto a = model.at(std::abs(dm - config.chan_dm));
        loss.push_back(std::max(0.0, 1.0 - model.ratio(a, delta)));
    }
    return loss;
}

} // namespace dmt::plans
