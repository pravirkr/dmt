#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <numbers>
#include <random>
#include <stdexcept>
#include <utility>

#include "dmt/utils/simulate.hpp"

#include "dmt/baseband_layout.hpp"
#include "dmt/fft.hpp"

#include "dmt/common/types.hpp"
#include "dmt/dm_utils.hpp"

namespace dmt::utils {
std::tuple<std::vector<float>, SizeType> generate_pure_frb(SizeType nchans,
                                                           SizeType nsamps,
                                                           float f_min,
                                                           float f_max,
                                                           SizeType dt,
                                                           float pulse_toa,
                                                           float amplitude) {
    if (nchans == 0 || nsamps == 0) {
        return std::make_tuple(std::vector<float>{}, SizeType{0});
    }
    std::vector<float> arr(nchans * nsamps, 0.0F);
    SizeType nsamps_dispersed = 0;

    const float foff      = (f_max - f_min) / static_cast<float>(nchans);
    const float foff_half = foff / 2.0F;

    for (SizeType ichan = 0; ichan < nchans; ++ichan) {
        const auto freq =
            f_min + (static_cast<float>(ichan) * foff) + foff_half;
        const auto freq_min = freq - foff_half;
        const auto freq_max = freq + foff_half;
        const auto dt_start =
            static_cast<float>(dt) * utils::cff(f_min, freq_min, f_min, f_max);
        const auto tstart      = pulse_toa - dt_start;
        const auto tstart_int  = static_cast<IndexType>(tstart);
        const auto tstart_frac = tstart - static_cast<float>(tstart_int);

        const auto dt_sub    = static_cast<float>(dt) *
                               utils::cff(freq_min, freq_max, f_min, f_max);
        const auto tend      = tstart - dt_sub;
        const auto tend_int  = static_cast<IndexType>(tend);
        const auto tend_frac = 1.0F - (tend - static_cast<float>(tend_int));

        float* arr_chan_start = &arr[ichan * nsamps];

        if (tstart_int < 0 || std::cmp_greater_equal(tend_int, nsamps)) {
            continue;
        }

        if (tend_int >= 0 && tend_int <= tstart_int &&
            std::cmp_less(tstart_int, nsamps)) {
            if (tend_int == tstart_int) {
                arr_chan_start[tend_int] = amplitude;
                nsamps_dispersed += 1;
            } else {
                const float amp_per_sample = amplitude / dt_sub;
                std::fill(arr_chan_start + tend_int,
                          arr_chan_start + tstart_int + 1, amp_per_sample);
                arr_chan_start[tend_int] *= tend_frac;
                arr_chan_start[tstart_int] *= tstart_frac;
                nsamps_dispersed +=
                    static_cast<SizeType>(tstart_int - tend_int + 1);
            }
        } else if (tend_int < 0 && 0 <= tstart_int &&
                   std::cmp_less(tstart_int, nsamps)) {
            const float amp_per_sample = amplitude / dt_sub;
            std::fill(arr_chan_start, arr_chan_start + tstart_int + 1,
                      amp_per_sample);
            arr_chan_start[tstart_int] *= tstart_frac;
            nsamps_dispersed += static_cast<SizeType>(tstart_int + 1);
        } else if (tend_int >= 0 && std::cmp_less(tend_int, nsamps) &&
                   std::cmp_less_equal(nsamps, tstart_int)) {
            const float amp_per_sample = amplitude / dt_sub;
            std::fill(arr_chan_start + tend_int, arr_chan_start + nsamps,
                      amp_per_sample);
            arr_chan_start[tend_int] *= tend_frac;
            nsamps_dispersed += (nsamps - tend_int);
        }
    }
    return {arr, nsamps_dispersed};
}

std::vector<ComplexType> simulate_baseband(float f_center,
                                           float bw_sub,
                                           SizeType nsub,
                                           SizeType nsamps,
                                           std::span<const BasebandPulse> pulses,
                                           float noise_sigma,
                                           uint64_t seed,
                                           int nthreads) {
    if (nsub == 0 || nsamps == 0 || bw_sub <= 0.0F) {
        throw std::invalid_argument(
            "simulate_baseband: need nsub > 0, nsamps > 0, bw_sub > 0");
    }
    const double bw    = static_cast<double>(bw_sub) *
                         static_cast<double>(nsub);
    const double f_min = static_cast<double>(f_center) - (bw / 2.0);
    if (f_min <= 0.0) {
        throw std::invalid_argument("simulate_baseband: band below 0 MHz");
    }
    const SizeType nrows = 2 * nsub;
    std::vector<ComplexType> v(nrows * nsamps, ComplexType{0.0F, 0.0F});
    if (!pulses.empty()) {
        const auto n  = static_cast<double>(nsamps);
        const double k_disp = static_cast<double>(kDispConst);
#pragma omp parallel for num_threads(std::max(1, nthreads)) schedule(static)
        for (SizeType row = 0; row < nrows; ++row) {
            const SizeType isub = row % nsub;
            const double f_sub  = f_min + ((static_cast<double>(isub) + 0.5) *
                                          static_cast<double>(bw_sub));
            ComplexType* x = v.data() + (row * nsamps);
            for (SizeType k = 0; k < nsamps; ++k) {
                const double kk =
                    (k < (nsamps + 1) / 2) ? static_cast<double>(k)
                                           : static_cast<double>(k) - n;
                const double f_bb = kk * static_cast<double>(bw_sub) / n; // MHz
                const double f_rf = f_sub + f_bb;
                std::complex<double> acc{0.0, 0.0};
                for (const auto& p : pulses) {
                    // Delay t_arrival at every frequency plus the dispersion
                    // delay K DM / f^2: phase (turns) -f t + 1e6 K DM / f.
                    // The carrier term f_sub * t is a constant per subband.
                    double turns = (-f_bb * 1.0E6 * p.t_arrival) +
                                   (1.0E6 * k_disp * p.dm / f_rf);
                    turns -= std::floor(turns);
                    const double env =
                        p.width > 0.0
                            ? std::exp(-2.0 * std::numbers::pi *
                                       std::numbers::pi * p.width * p.width *
                                       (f_bb * 1.0E6) * (f_bb * 1.0E6))
                            : 1.0;
                    const double amp = std::sqrt(p.fluence) / n * env;
                    const double ang = 2.0 * std::numbers::pi * turns;
                    acc += amp * std::complex<double>(std::cos(ang),
                                                      std::sin(ang));
                }
                x[k] = {static_cast<float>(acc.real()),
                        static_cast<float>(acc.imag())};
            }
        }
        // Unnormalised inverse transform: sum_t |x|^2 = n * sum_k |X|^2.
        FFTWManager inv(FFTKind::kC2CBackward, nsamps, nrows,
                        std::max(1, nthreads));
        inv.execute(v);
        // Scale so that each pulse has its fluence: n * (sqrt(E) / n)^2 * n
        // = E after the n-point sum; nothing more to do.
    }
    if (noise_sigma > 0.0F) {
        std::mt19937_64 rng(seed);
        std::normal_distribution<float> gauss(0.0F, noise_sigma);
        for (auto& x : v) {
            const float re = gauss(rng);
            const float im = gauss(rng);
            x += ComplexType{re, im};
        }
    }
    return v;
}

std::vector<uint8_t> pack_baseband(std::span<const ComplexType> voltages,
                                   SizeType nsub,
                                   SizeType nsamps,
                                   const BasebandFormat& format,
                                   float scale,
                                   SizeType sub_begin,
                                   SizeType sub_count,
                                   SizeType t_begin,
                                   SizeType t_count) {
    validate_baseband_format(format);
    if (voltages.size() != 2 * nsub * nsamps) {
        throw std::invalid_argument(std::format(
            "pack_baseband: expected {} voltages, got {}", 2 * nsub * nsamps,
            voltages.size()));
    }
    if (sub_count == 0) {
        sub_count = nsub - sub_begin;
    }
    if (t_count == 0) {
        t_count = nsamps - t_begin;
    }
    if (sub_begin + sub_count > nsub || t_begin + t_count > nsamps) {
        throw std::invalid_argument("pack_baseband: range out of bounds");
    }
    const auto st      = baseband_strides(format, t_count, sub_count);
    const auto nbits   = static_cast<unsigned>(format.nbits);
    const auto per     = 8U / nbits;
    const auto mask    = (1U << nbits) - 1U;
    const auto nelem   = SizeType{4} * t_count * sub_count;
    std::vector<uint8_t> out((nelem * nbits) / 8, 0);
    const auto code_of = [&](float x) -> unsigned {
        const float q = x * scale;
        if (nbits == 2) {
            unsigned best = 0;
            for (unsigned c = 1; c < 4; ++c) {
                if (std::abs(q - format.levels_2bit[c]) <
                    std::abs(q - format.levels_2bit[best])) {
                    best = c;
                }
            }
            return best;
        }
        const int half = 1 << (nbits - 1U);
        const int vi   = std::clamp(static_cast<int>(std::lround(q)), -half,
                                    half - 1);
        return format.is_signed ? static_cast<unsigned>(vi) & mask
                                : static_cast<unsigned>(vi + half);
    };
    for (SizeType p = 0; p < 2; ++p) {
        for (SizeType s = 0; s < sub_count; ++s) {
            const ComplexType* x =
                voltages.data() + (((p * nsub) + sub_begin + s) * nsamps) +
                t_begin;
            for (SizeType t = 0; t < t_count; ++t) {
                const SizeType e =
                    (p * st.pol) + (s * st.freq) + (t * st.time);
                for (SizeType ri = 0; ri < 2; ++ri) {
                    const SizeType ei   = e + (ri * st.ri);
                    const unsigned code = code_of(ri == 0 ? x[t].real()
                                                          : x[t].imag());
                    const auto k        = static_cast<unsigned>(ei % per);
                    const unsigned shift =
                        format.msb_first ? 8U - (nbits * (k + 1U)) : nbits * k;
                    out[ei / per] = static_cast<uint8_t>(
                        out[ei / per] | ((code & mask) << shift));
                }
            }
        }
    }
    return out;
}

} // namespace dmt::utils
