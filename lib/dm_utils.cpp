#include "dmt/dm_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace dmt::utils {

float cff(float f1_start, float f1_end, float f2_start, float f2_end) {
    return (std::pow(f1_start, kDispCoeff) - std::pow(f1_end, kDispCoeff)) /
           (std::pow(f2_start, kDispCoeff) - std::pow(f2_end, kDispCoeff));
}

SizeType calculate_dt_sub(
    float f_start, float f_end, float f_min, float f_max, SizeType dt) {
    const float ratio = cff(f_start, f_end, f_min, f_max);
    return static_cast<SizeType>(std::round(static_cast<float>(dt) * ratio));
}

float get_dmconv(float f_min, float f_max, float tsamp) {
    const float dm_conv = kDispConst * (std::pow(f_min, kDispCoeff) -
                                        std::pow(f_max, kDispCoeff));
    return tsamp / dm_conv;
}

SizeType find_closest_index(const std::vector<SizeType>& arr_sorted,
                            SizeType val) {
    if (arr_sorted.empty()) {
        throw std::runtime_error("Array is empty");
    }
    auto it      = std::ranges::lower_bound(arr_sorted, val);
    SizeType idx = std::distance(arr_sorted.begin(), it);

    if (it != arr_sorted.end()) {
        if (it != arr_sorted.begin() && val - *(it - 1) < *it - val) {
            idx--;
        }
    } else {
        idx = arr_sorted.size() - 1;
    }
    return idx;
}

std::vector<SizeType> generate_delay_table(const float* dm_arr,
                                           SizeType dm_count,
                                           float f0,
                                           float df,
                                           SizeType nchans,
                                           float tsamp) {
    std::vector<SizeType> delay_table(nchans * dm_count);
    for (SizeType idm = 0; idm < dm_count; ++idm) {
        for (SizeType ichan = 0; ichan < nchans; ++ichan) {
            const auto a = 1.F / (f0 + static_cast<float>(ichan) * df);
            const auto b = 1.F / f0;
            const auto delay =
                kDispConst / tsamp * (a * a - b * b) * dm_arr[idm];
            delay_table[(idm * nchans) + ichan] =
                static_cast<SizeType>(std::round(delay));
        }
    }
    return delay_table;
}

SizeType minimum_overlap(float dm_max,
                         float fcenter,
                         float bw,
                         float tbin,
                         SizeType nsub,
                         SizeType nchan) {
    float bw_chan         = bw / static_cast<float>(nsub * nchan);
    float fmin_bottom_sub = fcenter - (bw / 2);
    float fmax_bottom_sub = fmin_bottom_sub + bw_chan;
    float delay           = kDispConst * dm_max *
                  (std::pow(fmin_bottom_sub, kDispCoeff) -
                   std::pow(fmax_bottom_sub, kDispCoeff));
    if (delay < 0) {
        throw std::runtime_error("Negative dispersion delay is not allowed");
    }
    float delay_samples = std::round(delay / tbin);
    return static_cast<SizeType>(delay_samples);
}

std::vector<float> generate_coherent_dms(
    float dm_min, float dm_max, float fcenter, float bw, float tbin, float tp) {
    float f_min = fcenter - (bw / 2);
    float f_max = fcenter + (bw / 2);
    float delay = kDispConst * dm_max *
                  (std::pow(f_min, kDispCoeff) - std::pow(f_max, kDispCoeff));
    auto ncoherent = static_cast<size_t>(std::ceil(delay * tbin / (tp * tp)));
    float coh_dm_step = dm_max / static_cast<float>(ncoherent);
    std::vector<float> dm_grid;
    dm_grid.reserve(ncoherent);
    for (size_t i = 0; i < ncoherent; ++i) {
        dm_grid.push_back(dm_min + (static_cast<float>(i) * coh_dm_step));
    }
    return dm_grid;
}

void dedisperse(float* __restrict__ waterfall,
                SizeType waterfall_size,
                float dm,
                float f_min,
                float f_max,
                SizeType nchans,
                SizeType nsamps,
                float tsamp) {
    if (waterfall_size != nchans * nsamps) {
        throw std::runtime_error("Waterfall size mismatch");
    }
    const float foff = (f_max - f_min) / static_cast<float>(nchans);
    std::vector<int> shifts(nchans);
    for (SizeType ichan = 0; ichan < nchans; ++ichan) {
        const float fchan = f_min + (foff * static_cast<float>(ichan));
        const float delay =
            kDispConst * dm *
            (std::pow(f_min, kDispCoeff) - std::pow(fchan, kDispCoeff));
        shifts[ichan] = static_cast<int>(std::round(delay / tsamp));
    }
    for (SizeType ichan = 0; ichan < nchans; ++ichan) {
        const SizeType start = ichan * nsamps;
        const SizeType end   = start + nsamps;
        const int shift      = shifts[ichan];
        if (shift == 0) {
            continue;
        }
        if (shift > 0) {
            std::rotate(waterfall + start, waterfall + end - shift,
                        waterfall + end);
        } else {
            std::rotate(waterfall + start, waterfall + start - shift,
                        waterfall + end);
        }
    }
}

void compute_chirp(ComplexType* chirp_table,
                   SizeType chirp_table_size,
                   const float* dm_grid,
                   SizeType ndm,
                   float fcenter,
                   float bw,
                   SizeType nbin,
                   SizeType nsub,
                   SizeType nchan) {
    const SizeType mbin = nbin / nchan;
    const float bw_sub  = bw / static_cast<float>(nsub);
    const float bw_chan = bw_sub / static_cast<float>(nchan);
    const float bw_bin  = bw_chan / static_cast<float>(mbin);

    if (chirp_table_size != ndm * nsub * nbin) {
        throw std::runtime_error("Chirp table size mismatch");
    }

    std::vector<float> freqs_sub(nsub);
    for (SizeType i = 0; i < nsub; ++i) {
        freqs_sub[i] =
            fcenter - bw / 2 + (static_cast<float>(i) + 0.5F) * bw_sub;
    }
    std::vector<float> bin_freqs(mbin);
    for (SizeType i = 0; i < mbin; ++i) {
        bin_freqs[i] = -bw_chan / 2 + (static_cast<float>(i) + 0.5F) * bw_bin;
    }

    const float taper_const = 1.0F / (0.47F * bw_chan);
    const float taper_exp   = 80.0F;
    const float coeff_const =
        2.0F * std::numbers::pi_v<float> * kDispConst * 1.0E6F;

    for (SizeType idm = 0; idm < ndm; ++idm) {
        const float coeff = coeff_const * dm_grid[idm];
        for (SizeType isub = 0; isub < nsub; ++isub) {
            for (SizeType ichan = 0; ichan < nchan; ++ichan) {
                const float freq_chan =
                    freqs_sub[isub] + ((static_cast<float>(ichan) -
                                        static_cast<float>(nchan) / 2 + 0.5F) *
                                       bw_chan);
                for (SizeType ibin = 0; ibin < mbin; ++ibin) {
                    const float bin_freq    = bin_freqs[ibin];
                    const float freq_ratio  = bin_freq / freq_chan;
                    const float phase_delay = -coeff * freq_ratio * freq_ratio /
                                              (freq_chan + bin_freq);
                    const float taper =
                        1.0F / std::sqrt(1.0F + std::pow(bin_freq * taper_const,
                                                         taper_exp));
                    const SizeType idx = (idm * nsub * nchan * mbin) +
                                         (isub * nchan * mbin) +
                                         (ichan * mbin) + ibin;
                    chirp_table[idx] = std::polar(taper, phase_delay);
                }
            }
        }
    }
}

void add_offset_kernel(const float* __restrict arr1,
                       SizeType size_in1,
                       const float* __restrict arr2,
                       SizeType size_in2,
                       float* __restrict arr_out,
                       SizeType size_out,
                       SizeType offset) {
    // Debug checks using assert (only active when NDEBUG is not defined)
    assert(size_in1 == size_in2 && "Input sizes must be equal");
    assert(size_out >= size_in1 && "Output size must be >= input size");
    assert(offset < size_in1 && "Offset must be < input size");

    SizeType t          = 0;
    const SizeType nsum = size_in1 - offset;
    std::copy_n(arr1, offset, arr_out);
    t += offset;
#pragma omp simd
    for (SizeType i = 0; i < nsum; ++i) {
        arr_out[offset + i] = arr1[offset + i] + arr2[i];
    }
    t += nsum;
    const SizeType nrest = std::min(offset, size_out - size_in1);
    if (nrest > 0) {
        std::copy_n(arr2 + nsum, nrest, arr_out + size_in1);
        t += nrest;
    }
    if (t < size_out) {
        std::fill(arr_out + t, arr_out + size_out, 0.0F);
    }
}

} // namespace dmt::utils
