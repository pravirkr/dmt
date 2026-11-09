#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

#include "dmt/dm_utils.hpp"
#include "dmt/logging.hpp"

// DDMTPlan and the DM grid generators.

namespace dmt::plans {

class DDMTPlan::Impl {
public:
    Impl(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         float dm_max,
         float dm_step,
         float dm_min                       = 0.0F,
         SizeType nbits                     = 32,
         std::span<const uint8_t> kill_mask = {},
         std::optional<float> f_ref         = std::nullopt)
        : m_f_min(f_min),
          m_f_max(f_max),
          m_f_ref(f_ref.value_or(f_min)),
          m_nchans(nchans),
          m_tsamp(tsamp),
          m_dm_arr(generate_dm_arr(dm_max, dm_step, dm_min)),
          m_nbits(nbits),
          m_kill_mask(kill_mask.begin(), kill_mask.end()) {
        validate_inputs();
        configure_plan();
        logging::debug("DDMT: dm_max={}, dm_min={}, dm_step={}", dm_max, dm_min,
                       dm_step);
    }

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         std::span<const float> dm_arr,
         SizeType nbits                     = 32,
         std::span<const uint8_t> kill_mask = {},
         std::optional<float> f_ref         = std::nullopt)
        : m_f_min(f_min),
          m_f_max(f_max),
          m_f_ref(f_ref.value_or(f_min)),
          m_nchans(nchans),
          m_tsamp(tsamp),
          m_dm_arr(dm_arr.begin(), dm_arr.end()),
          m_nbits(nbits),
          m_kill_mask(kill_mask.begin(), kill_mask.end()) {
        validate_inputs();
        configure_plan();
        logging::debug("DDMT: dm_count={}", m_dm_arr.size());
    }

    Impl(float f_min,
         float f_max,
         SizeType nchans,
         float tsamp,
         const LevinConfig& levin,
         SizeType nbits                     = 32,
         std::span<const uint8_t> kill_mask = {},
         std::optional<float> f_ref         = std::nullopt)
        : m_f_min(f_min),
          m_f_max(f_max),
          m_f_ref(f_ref.value_or(f_min)),
          m_nchans(nchans),
          m_tsamp(tsamp),
          m_dm_arr(
              levin.piecewise_uniform
                  ? utils::generate_levin_dm_grid_piecewise(levin.dm_start,
                                                            levin.dm_end,
                                                            tsamp,
                                                            levin.pulse_width,
                                                            f_min,
                                                            f_max,
                                                            nchans,
                                                            levin.tol)
                  : utils::generate_levin_dm_grid(levin.dm_start,
                                                  levin.dm_end,
                                                  tsamp,
                                                  levin.pulse_width,
                                                  f_min,
                                                  f_max,
                                                  nchans,
                                                  levin.tol)),
          m_nbits(nbits),
          m_kill_mask(kill_mask.begin(), kill_mask.end()) {
        validate_inputs();
        configure_plan();
        logging::debug("DDMT Levin: dm_start={}, dm_end={}, count={}",
                       levin.dm_start, levin.dm_end, m_dm_arr.size());
    }

    ~Impl()                                = default;
    Impl(Impl&& other) noexcept            = default;
    Impl& operator=(Impl&& other) noexcept = default;
    Impl(const Impl& other)                = default;
    Impl& operator=(const Impl& other)     = default;

    float get_f_min() const noexcept { return m_f_min; }
    float get_f_max() const noexcept { return m_f_max; }
    float get_f_ref() const noexcept { return m_f_ref; }
    SizeType get_time_offset() const noexcept {
        return m_container.time_offset;
    }
    SizeType get_nchans() const noexcept { return m_nchans; }
    float get_tsamp() const noexcept { return m_tsamp; }
    std::vector<float> get_dm_arr() const noexcept { return m_dm_arr; }
    const DDMTPlanContainer& get_container() const noexcept {
        return m_container;
    }
    [[nodiscard]] std::vector<float> get_dm_grid() const noexcept {
        return m_container.dm_arr;
    }
    [[nodiscard]] std::vector<float>
    get_fractional_delay_table() const noexcept {
        return m_container.fractional_delay_table;
    }
    SizeType get_nbits() const noexcept { return m_nbits; }
    [[nodiscard]] std::vector<uint8_t> get_kill_mask() const noexcept {
        return m_container.kill_mask;
    }
    void seal() noexcept { m_sealed = true; }
    void set_kill_mask(std::span<const uint8_t> kill_mask) {
        if (m_sealed) {
            throw std::invalid_argument(
                "DDMTPlan: kill_mask cannot be changed after the plan is "
                "bound to an engine");
        }
        if (kill_mask.size() != m_nchans) {
            throw std::invalid_argument(
                fmt::format("DDMT: kill_mask size={} must equal nchans={}",
                            kill_mask.size(), m_nchans));
        }
        m_container.kill_mask.resize(kill_mask.size());
        for (SizeType i = 0; i < kill_mask.size(); ++i) {
            m_container.kill_mask[i] = kill_mask[i] == 0 ? 0 : 1;
        }
    }
    [[nodiscard]] float get_effective_variance() const noexcept {
        return static_cast<float>(count_active_chans());
    }
    [[nodiscard]] float get_effective_sigma() const noexcept {
        return std::sqrt(get_effective_variance());
    }
    [[nodiscard]] std::vector<float>
    get_effective_variance_grid() const noexcept {
        return std::vector<float>(m_dm_arr.size(), get_effective_variance());
    }
    [[nodiscard]] std::vector<float> get_effective_sigma_grid() const noexcept {
        return std::vector<float>(m_dm_arr.size(), get_effective_sigma());
    }

private:
    float m_f_min;
    float m_f_max;
    float m_f_ref;
    SizeType m_nchans;
    float m_tsamp;
    std::vector<float> m_dm_arr;
    SizeType m_nbits;
    std::vector<uint8_t> m_kill_mask;

    DDMTPlanContainer m_container;
    bool m_sealed{false};

    static void canonicalize_mask(std::vector<uint8_t>& mask) {
        for (auto& value : mask) {
            value = value == 0 ? uint8_t{0} : uint8_t{1};
        }
    }

    [[nodiscard]] SizeType count_active_chans() const noexcept {
        return static_cast<SizeType>(
            std::ranges::count(m_container.kill_mask, static_cast<uint8_t>(1)));
    }

    void validate_inputs() const {
        if (!utils::is_finite_bits(m_f_min) ||
            !utils::is_finite_bits(m_f_max) ||
            !utils::is_finite_bits(m_tsamp)) {
            throw std::invalid_argument(
                "DDMT: f_min, f_max and tsamp must be finite");
        }
        if (!(m_f_min > 0.0F)) {
            throw std::invalid_argument(
                fmt::format("DDMT: f_min={} must be positive", m_f_min));
        }
        if (m_f_min >= m_f_max) {
            throw std::invalid_argument("f_min must be less than f_max");
        }
        if (!utils::is_finite_bits(m_f_ref) ||
            (m_f_ref != m_f_min && m_f_ref != m_f_max)) {
            throw std::invalid_argument(fmt::format(
                "DDMT: f_ref={} MHz must be either f_min={} or f_max={}",
                m_f_ref, m_f_min, m_f_max));
        }
        if (m_tsamp <= 0) {
            throw std::invalid_argument("tsamp must be greater than 0");
        }
        if (m_nchans <= 0) {
            throw std::invalid_argument("nchans must be greater than 0");
        }
        if (m_dm_arr.empty()) {
            throw std::invalid_argument("dm_arr must not be empty");
        }
        for (const float dm : m_dm_arr) {
            if (!utils::is_finite_bits(dm) || !(dm >= 0.0F)) {
                throw std::invalid_argument(
                    "DDMT: DM trials must be finite and >= 0");
            }
        }
        if (m_nbits != 1 && m_nbits != 2 && m_nbits != 4 && m_nbits != 8 &&
            m_nbits != 16 && m_nbits != 32) {
            throw std::invalid_argument(fmt::format(
                "DDMT: nbits={} must be one of 1, 2, 4, 8, 16, 32", m_nbits));
        }
        if (!m_kill_mask.empty() && m_kill_mask.size() != m_nchans) {
            throw std::invalid_argument(
                fmt::format("DDMT: kill_mask size={} must equal nchans={}",
                            m_kill_mask.size(), m_nchans));
        }
        if (m_nbits < 32) {
            // Worst case (no kill_mask applied yet): every channel at the
            // maximum representable value for nbits. The int32_t
            // accumulator used by the packed execute() path must not
            // overflow.
            const auto max_val = static_cast<double>((1U << m_nbits) - 1U);
            const auto max_sum = max_val * static_cast<double>(m_nchans);
            if (max_sum >
                static_cast<double>(std::numeric_limits<int32_t>::max())) {
                throw std::invalid_argument(fmt::format(
                    "DDMT: nbits={} with nchans={} can overflow the "
                    "int32_t accumulator (worst-case sum {})",
                    m_nbits, m_nchans, max_sum));
            }
        }
    }
    void configure_plan() {
        m_container.nchans    = m_nchans;
        m_container.dm_arr    = m_dm_arr;
        m_container.nbits     = m_nbits;
        m_container.kill_mask = m_kill_mask.empty()
                                    ? std::vector<uint8_t>(m_nchans, 1)
                                    : m_kill_mask;
        canonicalize_mask(m_container.kill_mask);
        // Channel c is centred at f_min + (c + 1/2) df; output times refer
        // to f_ref (get_f_ref(), get_time_offset()).
        const auto df    = (m_f_max - m_f_min) / static_cast<float>(m_nchans);
        const auto f_ref = static_cast<double>(m_f_ref);
        m_container.time_offset = utils::generate_delay_offset(
            m_dm_arr, m_nchans, m_f_min, df, m_tsamp, f_ref);
        m_container.delay_table = utils::generate_delay_table(
            m_dm_arr, m_nchans, m_f_min, df, m_tsamp, f_ref);
        m_container.fractional_delay_table =
            utils::generate_fractional_delay_table(m_nchans, m_f_min, df,
                                                   m_tsamp, f_ref);
    }

    static std::vector<float>
    generate_dm_arr(float dm_max, float dm_step, float dm_min) {
        if (!utils::is_finite_bits(dm_min) || !utils::is_finite_bits(dm_max) ||
            !utils::is_finite_bits(dm_step)) {
            throw std::invalid_argument(
                "DDMT: dm_min, dm_max and dm_step must be finite");
        }
        if (dm_step <= 0.0F || dm_min > dm_max) {
            return {};
        }

        // Number of steps: floor((max - min) / step) + 1. The span is
        // computed in double so a non-finite or overflowing count is rejected
        // before it is cast to SizeType.
        const double steps = std::floor(
            (static_cast<double>(dm_max) - static_cast<double>(dm_min)) /
            static_cast<double>(dm_step));
        if (!(steps >= 0.0) ||
            steps >=
                static_cast<double>(std::numeric_limits<SizeType>::max())) {
            throw std::invalid_argument("DDMT: DM grid is too large");
        }
        const auto count = static_cast<SizeType>(steps) + 1;

        std::vector<float> dm_arr;
        dm_arr.reserve(count);

        for (SizeType i = 0; i < count; ++i) {
            dm_arr.push_back(dm_min + (static_cast<float>(i) * dm_step));
        }

        return dm_arr;
    }
};

// --- Definitions for FDMTPlan ---

DDMTPlan::DDMTPlan(float f_min,
                   float f_max,
                   SizeType nchans,
                   float tsamp,
                   float dm_max,
                   float dm_step,
                   float dm_min,
                   SizeType nbits,
                   std::span<const uint8_t> kill_mask,
                   std::optional<float> f_ref)
    : m_impl(std::make_unique<Impl>(f_min,
                                    f_max,
                                    nchans,
                                    tsamp,
                                    dm_max,
                                    dm_step,
                                    dm_min,
                                    nbits,
                                    kill_mask,
                                    f_ref)) {}

DDMTPlan::DDMTPlan(float f_min,
                   float f_max,
                   SizeType nchans,
                   float tsamp,
                   std::span<const float> dm_arr,
                   SizeType nbits,
                   std::span<const uint8_t> kill_mask,
                   std::optional<float> f_ref)
    : m_impl(std::make_unique<Impl>(
          f_min, f_max, nchans, tsamp, dm_arr, nbits, kill_mask, f_ref)) {}

DDMTPlan::DDMTPlan(float f_min,
                   float f_max,
                   SizeType nchans,
                   float tsamp,
                   const LevinConfig& levin,
                   SizeType nbits,
                   std::span<const uint8_t> kill_mask,
                   std::optional<float> f_ref)
    : m_impl(std::make_unique<Impl>(
          f_min, f_max, nchans, tsamp, levin, nbits, kill_mask, f_ref)) {}

DDMTPlan::~DDMTPlan()                              = default;
DDMTPlan::DDMTPlan(DDMTPlan&&) noexcept            = default;
DDMTPlan& DDMTPlan::operator=(DDMTPlan&&) noexcept = default;
DDMTPlan::DDMTPlan(const DDMTPlan& other)
    : m_impl(std::make_unique<Impl>(*other.m_impl)) {}

DDMTPlan& DDMTPlan::operator=(const DDMTPlan& other) {
    if (this != &other) {
        m_impl = std::make_unique<Impl>(*other.m_impl);
    }
    return *this;
}
float DDMTPlan::get_f_min() const noexcept { return m_impl->get_f_min(); }
float DDMTPlan::get_f_max() const noexcept { return m_impl->get_f_max(); }
float DDMTPlan::get_f_ref() const noexcept { return m_impl->get_f_ref(); }
SizeType DDMTPlan::get_nchans() const noexcept { return m_impl->get_nchans(); }
float DDMTPlan::get_tsamp() const noexcept { return m_impl->get_tsamp(); }
std::vector<float> DDMTPlan::get_dm_arr() const noexcept {
    return m_impl->get_dm_arr();
}
const DDMTPlanContainer& DDMTPlan::get_container() const noexcept {
    return m_impl->get_container();
}
std::vector<float> DDMTPlan::get_dm_grid() const noexcept {
    return m_impl->get_dm_grid();
}
SizeType DDMTPlan::get_time_offset() const noexcept {
    return m_impl->get_time_offset();
}
std::vector<float> DDMTPlan::get_fractional_delay_table() const noexcept {
    return m_impl->get_fractional_delay_table();
}
SizeType DDMTPlan::get_nbits() const noexcept { return m_impl->get_nbits(); }
std::vector<uint8_t> DDMTPlan::get_kill_mask() const noexcept {
    return m_impl->get_kill_mask();
}
std::string DDMTPlan::summary() const {
    const auto dms     = get_dm_arr();
    const auto& delays = get_container().delay_table;
    const auto max_delay =
        delays.empty() ? SizeType{0} : *std::ranges::max_element(delays);
    const auto kill_mask = get_kill_mask();
    const auto n_masked  = static_cast<SizeType>(
        std::ranges::count(kill_mask, static_cast<uint8_t>(0)));
    const auto [dm_lo, dm_hi] =
        dms.empty() ? std::pair{0.0F, 0.0F}
                    : std::pair{std::ranges::min(dms), std::ranges::max(dms)};
    return fmt::format(
        "*** DDMT Plan Summary ***\n"
        "Band: {} - {} MHz, {} channels ({} masked), tsamp {} s\n"
        "Delays from channel centres, reference f_ref {} MHz, "
        "time offset {} samples\n"
        "DM trials: {} over [{}, {}] pc cm^-3, max delay {} samples\n"
        "Input: {}\n",
        get_f_min(), get_f_max(), get_nchans(), n_masked, get_tsamp(),
        get_f_ref(), get_time_offset(), dms.size(), dm_lo, dm_hi, max_delay,
        get_nbits() == 32 ? std::string("float32")
                          : fmt::format("packed {}-bit", get_nbits()));
}
void DDMTPlan::set_kill_mask(std::span<const uint8_t> kill_mask) {
    m_impl->set_kill_mask(kill_mask);
}
void DDMTPlan::seal() noexcept { m_impl->seal(); }
float DDMTPlan::get_effective_variance() const noexcept {
    return m_impl->get_effective_variance();
}
float DDMTPlan::get_effective_sigma() const noexcept {
    return m_impl->get_effective_sigma();
}
std::vector<float> DDMTPlan::get_effective_variance_grid() const noexcept {
    return m_impl->get_effective_variance_grid();
}
std::vector<float> DDMTPlan::get_effective_sigma_grid() const noexcept {
    return m_impl->get_effective_sigma_grid();
}
std::vector<float> DDMTPlan::generate_levin_dm_grid(float dm_start,
                                                    float dm_end,
                                                    float tsamp,
                                                    float pulse_width,
                                                    float f_min,
                                                    float f_max,
                                                    SizeType nchans,
                                                    float tol) {
    return utils::generate_levin_dm_grid(dm_start, dm_end, tsamp, pulse_width,
                                         f_min, f_max, nchans, tol);
}
std::vector<float>
DDMTPlan::generate_levin_dm_grid_piecewise(float dm_start,
                                           float dm_end,
                                           float tsamp,
                                           float pulse_width,
                                           float f_min,
                                           float f_max,
                                           SizeType nchans,
                                           float tol,
                                           SizeType min_run) {
    return utils::generate_levin_dm_grid_piecewise(dm_start, dm_end, tsamp,
                                                   pulse_width, f_min, f_max,
                                                   nchans, tol, min_run);
}

} // namespace dmt::plans
