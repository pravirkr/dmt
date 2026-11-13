#include "dmt/common/dm_grids.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>

#include "dmt/dm_utils.hpp"

namespace dmt::plans {

namespace {

// The dispersion constant of dmtlib.grid (Manchester & Taylor), in double:
// the grids are the ones the Python module returned before this port.
constexpr double kDMConstant = 1.0 / 2.41e-4;

void validate(const DMGridConfig& c, double dm_min, double dm_max) {
    // Bit tests: -ffast-math may fold std::isfinite (dm_utils.hpp).
    const auto finite = [](double x) { return utils::is_finite_bits(x); };
    if (!finite(c.f_min) || !finite(c.f_max) || !finite(c.tsamp) ||
        !finite(c.pulse_width) || !finite(c.chan_dm) || !finite(dm_min) ||
        !finite(dm_max) || !finite(c.max_snr_loss) || !finite(c.tol)) {
        throw std::invalid_argument("DM grid parameters must be finite");
    }
    if (!(c.f_min > 0.0) || !(c.f_max > c.f_min)) {
        throw std::invalid_argument(
            fmt::format("DM grid: invalid frequency band f_min={}, f_max={}",
                        c.f_min, c.f_max));
    }
    if (c.nchans == 0 || !(c.tsamp > 0.0)) {
        throw std::invalid_argument(fmt::format(
            "DM grid: invalid nchans={} or tsamp={}", c.nchans, c.tsamp));
    }
    if (c.pulse_width < 0.0) {
        throw std::invalid_argument("DM grid: pulse_width must be >= 0");
    }
    if (c.tol != 0.0 && !(c.tol > 1.0)) {
        throw std::invalid_argument(
            fmt::format("DM grid: tol must be > 1 (or 0), got {}", c.tol));
    }
    if (dm_max < dm_min) {
        throw std::invalid_argument(fmt::format(
            "DM grid: dm_max ({}) must be >= dm_min ({})", dm_max, dm_min));
    }
}

/// kSnrLoss on [dm_min, dm_max] with 0 <= dm_min: the step keeps the delay
/// error across the band within 2 eta W_eff (in samples, at least 1), with
/// the intra-channel smearing of the bottom channel at |DM - chan_dm|.
std::vector<double> snr_loss_positive(const DMGridConfig& c,
                                      double max_snr_loss,
                                      double dm_min,
                                      double dm_max,
                                      double chan_dm) {
    if (dm_max == dm_min) {
        return {dm_min};
    }
    const double f_min = c.f_min;
    const double f_max = c.f_max;
    const double tsamp = c.tsamp;
    const double width = c.pulse_width / tsamp;
    const double df    = (f_max - f_min) / static_cast<double>(c.nchans);
    const double k_disp =
        kDMConstant * (std::pow(f_min, -2.0) - std::pow(f_max, -2.0)) / tsamp;
    const double eta =
        std::sqrt((1.0 / std::pow(1.0 - max_snr_loss, 2.0)) - 1.0);
    std::vector<double> dms{dm_min};
    double cur = dm_min;
    while (cur < dm_max) {
        const double t_chan = (2.0 * kDMConstant * std::abs(cur - chan_dm) *
                               df / std::pow(f_min, 3.0)) /
                              tsamp;
        const double w_eff =
            std::sqrt(std::pow(width, 2.0) + 1.0 + std::pow(t_chan, 2.0));
        const double dt_step = std::max(1.0, 2.0 * eta * w_eff);
        cur += dt_step / k_disp;
        if (cur >= dm_max) {
            dms.push_back(dm_max);
            break;
        }
        dms.push_back(cur);
    }
    return dms;
}

std::vector<double>
snr_loss_grid(const DMGridConfig& c, double dm_min, double dm_max) {
    // tol = 1 / (1 - loss): the same knob as the Levin rules.
    const double loss = c.tol != 0.0 ? 1.0 - (1.0 / c.tol) : c.max_snr_loss;
    if (!(loss > 0.0 && loss < 1.0)) {
        throw std::invalid_argument(fmt::format(
            "DM grid: max_snr_loss must be in (0, 1), got {}", loss));
    }
    const double cd = c.chan_dm;
    // Negative DMs mirror positive ones (chan_dm mirrored with them).
    if (dm_min < 0.0 && dm_max > 0.0) {
        const auto neg = snr_loss_positive(c, loss, 0.0, -dm_min, -cd);
        const auto pos = snr_loss_positive(c, loss, 0.0, dm_max, cd);
        std::vector<double> out;
        out.reserve(neg.size() + pos.size() - 1);
        for (auto it = neg.rbegin(); it + 1 != neg.rend(); ++it) {
            out.push_back(-*it);
        }
        out.insert(out.end(), pos.begin(), pos.end());
        return out;
    }
    if (dm_max <= 0.0) {
        const auto pos = snr_loss_positive(c, loss, -dm_max, -dm_min, -cd);
        std::vector<double> out;
        out.reserve(pos.size());
        for (auto it = pos.rbegin(); it != pos.rend(); ++it) {
            out.push_back(-*it);
        }
        return out;
    }
    return snr_loss_positive(c, loss, dm_min, dm_max, cd);
}

} // namespace

std::vector<double>
generate_dm_grid(const DMGridConfig& config, double dm_min, double dm_max) {
    validate(config, dm_min, dm_max);
    if (config.method == DMGridMethod::kSnrLoss) {
        return snr_loss_grid(config, dm_min, dm_max);
    }
    if (config.chan_dm != 0.0) {
        throw std::invalid_argument(
            "DM grid: chan_dm needs the S/N-loss rule (the Levin rules keep "
            "the classic formula, without coherent dedispersion)");
    }
    if (dm_min < 0.0) {
        throw std::invalid_argument(fmt::format(
            "DM grid: the Levin rules do not support negative dm_min ({})",
            dm_min));
    }
    // The Levin rules work in float (DDMTPlan::generate_levin_dm_grid).
    const auto f   = [](double x) { return static_cast<float>(x); };
    const auto tol = f(config.tol != 0.0 ? config.tol : 1.25);
    const auto levin =
        config.method == DMGridMethod::kLevinPiecewise
            ? utils::generate_levin_dm_grid_piecewise(
                  f(dm_min), f(dm_max), f(config.tsamp), f(config.pulse_width),
                  f(config.f_min), f(config.f_max), config.nchans, tol,
                  config.min_run)
            : utils::generate_levin_dm_grid(
                  f(dm_min), f(dm_max), f(config.tsamp), f(config.pulse_width),
                  f(config.f_min), f(config.f_max), config.nchans, tol);
    return {levin.begin(), levin.end()};
}

std::vector<IndexType>
generate_dt_grid(const DMGridConfig& config, double dt_min, double dt_max) {
    const double k_disp =
        kDMConstant *
        (std::pow(config.f_min, -2.0) - std::pow(config.f_max, -2.0)) /
        config.tsamp;
    if (!utils::is_finite_bits(dt_min) || !utils::is_finite_bits(dt_max) ||
        !utils::is_finite_bits(k_disp) || !(k_disp > 0.0)) {
        throw std::invalid_argument("delay grid parameters must be finite");
    }
    const auto dms = generate_dm_grid(config, dt_min / k_disp, dt_max / k_disp);
    std::vector<IndexType> dts;
    dts.reserve(dms.size());
    for (const double dm : dms) {
        dts.push_back(static_cast<IndexType>(std::nearbyint(dm * k_disp)));
    }
    std::ranges::sort(dts);
    dts.erase(std::unique(dts.begin(), dts.end()), dts.end());
    // Both ends of the requested range are trials.
    dts.front() = static_cast<IndexType>(std::nearbyint(dt_min));
    dts.back()  = static_cast<IndexType>(std::nearbyint(dt_max));
    std::ranges::sort(dts);
    dts.erase(std::unique(dts.begin(), dts.end()), dts.end());
    return dts;
}

} // namespace dmt::plans
