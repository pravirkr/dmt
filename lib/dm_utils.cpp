#include "dmt/dm_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace dmt::utils {

float cff(float f_start, float f_end, float f_min, float f_max) {
    return (std::pow(f_start, kDispCoeff) - std::pow(f_end, kDispCoeff)) /
           (std::pow(f_min, kDispCoeff) - std::pow(f_max, kDispCoeff));
}

SizeType calculate_dt_sub(
    float f_start, float f_end, float f_min, float f_max, SizeType dt) {
    const float ratio = cff(f_start, f_end, f_min, f_max);
    return static_cast<SizeType>(std::ceil(static_cast<float>(dt) * ratio));
}

float get_dmconv(float f_min, float f_max, float tsamp) {
    const float dm_conv = kDispConst * (std::pow(f_min, kDispCoeff) -
                                        std::pow(f_max, kDispCoeff));
    return tsamp / dm_conv;
}

template <typename T>
static SizeType find_nearest_sorted_idx_impl(std::span<const T> arr_sorted,
                                             T val) {
    if (arr_sorted.empty()) {
        throw std::invalid_argument("find_nearest_sorted_idx: array is empty");
    }
    const auto it = std::ranges::lower_bound(arr_sorted, val);
    auto idx = static_cast<SizeType>(std::distance(arr_sorted.begin(), it));

    // Handle case where val is larger than all elements
    if (it == arr_sorted.end()) {
        return arr_sorted.size() - 1;
    }
    // Check if previous element is closer
    if (it != arr_sorted.begin()) {
        const auto val_prev    = *(it - 1);
        const auto val_curr    = *it;
        const bool prev_closer = (val >= val_prev) && (val <= val_curr) &&
                                 ((val - val_prev) <= (val_curr - val));
        if (prev_closer) {
            --idx;
        }
    }
    return idx;
}

SizeType find_nearest_sorted_idx(std::span<const SizeType> arr_sorted,
                                 SizeType val) {
    return find_nearest_sorted_idx_impl(arr_sorted, val);
}

SizeType find_nearest_sorted_idx(std::span<const IndexType> arr_sorted,
                                 IndexType val) {
    return find_nearest_sorted_idx_impl(arr_sorted, val);
}

std::vector<SizeType> generate_delay_table(std::span<const float> dm_arr,
                                           SizeType nchans,
                                           float fch1,
                                           float foff,
                                           float tsamp) {
    const auto ndm = dm_arr.size();
    std::vector<SizeType> delay_table(nchans * ndm);
    for (SizeType idm = 0; idm < ndm; ++idm) {
        for (SizeType ichan = 0; ichan < nchans; ++ichan) {
            const auto a = 1.F / (fch1 + static_cast<float>(ichan) * foff);
            const auto b = 1.F / fch1;
            const auto delay =
                kDispConst / tsamp * (a * a - b * b) * dm_arr[idm];
            delay_table[(idm * nchans) + ichan] =
                static_cast<SizeType>(std::nearbyint(delay));
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
    float delay_samples = delay / tbin;
    return static_cast<SizeType>(std::nearbyint(delay_samples));
}

std::vector<float> generate_coherent_dms(
    float dm_min, float dm_max, float fcenter, float bw, float tbin, float tp) {
    float f_min = fcenter - (bw / 2);
    float f_max = fcenter + (bw / 2);
    // Total dispersion delay (s) across the requested [dm_min, dm_max] span
    // (not dm_max alone -- a trial dispersion measure of dm_min still needs
    // coherently dedispersing).
    float delay = kDispConst * (dm_max - dm_min) *
                  (std::pow(f_min, kDispCoeff) - std::pow(f_max, kDispCoeff));
    // The fine FDMT stage searches its residual symmetrically around each
    // coarse trial (dt in [-n_p, n_p) rather than the paper's one-sided
    // [0, n_p)), doubling the DM width covered per coarse trial -- hence the
    // factor of 2 in the denominator, halving the number of coherent trials
    // needed relative to the one-sided search.
    auto ncoherent = std::max<size_t>(
        1, static_cast<size_t>(std::ceil(delay * tbin / (2.0F * tp * tp))));
    float coh_dm_step = (dm_max - dm_min) / static_cast<float>(ncoherent);
    std::vector<float> dm_grid;
    dm_grid.reserve(ncoherent);
    for (size_t i = 0; i < ncoherent; ++i) {
        // Grid points are the *center* of each trial's symmetric residual
        // window, not a left edge.
        dm_grid.push_back(dm_min +
                          ((static_cast<float>(i) + 0.5F) * coh_dm_step));
    }
    return dm_grid;
}

namespace {
// Per-channel integer sample delay at DM `dm`, relative to f_min, for the
// inter-channel bulk shift applied after coherent per-channel dechirping.
std::vector<int> compute_dedisperse_shifts(
    float dm, float f_min, float f_max, SizeType nchans, float tsamp) {
    const float foff  = (f_max - f_min) / static_cast<float>(nchans);
    const float f_ref = f_min + (foff * 0.5F);
    std::vector<int> shifts(nchans);
    for (SizeType ichan = 0; ichan < nchans; ++ichan) {
        const float fchan = f_min + (foff * (static_cast<float>(ichan) + 0.5F));
        const float delay =
            kDispConst * dm *
            (std::pow(f_ref, kDispCoeff) - std::pow(fchan, kDispCoeff));
        shifts[ichan] =
            std::max(0, static_cast<int>(std::nearbyint(delay / tsamp)));
    }
    return shifts;
}
} // namespace

std::vector<int> generate_dedisperse_shift_table(std::span<const float> dm_grid,
                                                 float f_min,
                                                 float f_max,
                                                 SizeType nchans,
                                                 float tsamp) {
    std::vector<int> table(dm_grid.size() * nchans);
    for (SizeType idm = 0; idm < dm_grid.size(); ++idm) {
        const auto shifts = compute_dedisperse_shifts(dm_grid[idm], f_min,
                                                      f_max, nchans, tsamp);
        std::ranges::copy(shifts,
                          table.begin() + static_cast<IndexType>(idm * nchans));
    }
    return table;
}

std::vector<SizeType>
generate_dedisperse_offset_table(std::span<const float> dm_grid,
                                 float f_min,
                                 float f_max,
                                 SizeType nchans,
                                 float tsamp) {
    std::vector<SizeType> offsets(dm_grid.size() * nchans);
    for (SizeType idm = 0; idm < dm_grid.size(); ++idm) {
        const auto shifts = compute_dedisperse_shifts(dm_grid[idm], f_min,
                                                      f_max, nchans, tsamp);
        SizeType total    = 0;
        for (SizeType c = 0; c < nchans; ++c) {
            offsets[(idm * nchans) + c] = total;
            if (shifts[c] > 0) {
                total += static_cast<SizeType>(shifts[c]);
            }
        }
    }
    return offsets;
}

void ChannelDelayLineCPU::initialise(std::span<const float> dm_grid_coh,
                                     float f_min,
                                     float f_max,
                                     SizeType nchans,
                                     float tsamp) {
    const auto ndm = dm_grid_coh.size();
    m_shifts.resize(ndm);
    m_offsets.resize(ndm);
    m_histories.resize(ndm);

    for (SizeType idm = 0; idm < ndm; ++idm) {
        m_shifts[idm] = compute_dedisperse_shifts(dm_grid_coh[idm], f_min,
                                                  f_max, nchans, tsamp);
        m_offsets[idm].resize(nchans);
        SizeType total = 0;
        for (SizeType c = 0; c < nchans; ++c) {
            m_offsets[idm][c] = total;
            if (m_shifts[idm][c] > 0) {
                total += static_cast<SizeType>(m_shifts[idm][c]);
            }
        }
        m_histories[idm].assign(total, 0.0F);
    }
}

void ChannelDelayLineCPU::process(std::span<const float> in,
                                  std::span<float> out,
                                  SizeType idm,
                                  SizeType nchans,
                                  SizeType nsamps) {
    if (in.size() != nchans * nsamps || out.size() != in.size()) {
        throw std::runtime_error("ChannelDelayLineCPU: buffer size mismatch");
    }
    const auto& shifts  = m_shifts[idm];
    const auto& offsets = m_offsets[idm];
    auto& history       = m_histories[idm];

    for (SizeType ichan = 0; ichan < nchans; ++ichan) {
        const auto shift = shifts[ichan];
        const float* x   = in.data() + (ichan * nsamps);
        float* y         = out.data() + (ichan * nsamps);

        if (shift <= 0) {
            std::copy_n(x, nsamps, y);
            continue;
        }

        const auto s = static_cast<SizeType>(shift);
        float* h     = history.data() + offsets[ichan];

        if (nsamps >= s) {
            std::copy_n(h, s, y);
            std::copy_n(x, nsamps - s, y + s);
            std::copy_n(x + (nsamps - s), s, h);
        } else {
            std::copy_n(h, nsamps, y);
            std::copy(h + nsamps, h + s, h);
            std::copy_n(x, nsamps, h + (s - nsamps));
        }
    }
}

void ChannelDelayLineCPU::reset_history() noexcept {
    for (auto& hist : m_histories) {
        std::fill(hist.begin(), hist.end(), 0.0F);
    }
}

SizeType ChannelDelayLineCPU::history_size(SizeType idm) const noexcept {
    return (idm < m_histories.size()) ? m_histories[idm].size() : 0;
}

const std::vector<int>& ChannelDelayLineCPU::get_shifts(SizeType idm) const {
    return m_shifts.at(idm);
}

const std::vector<SizeType>&
ChannelDelayLineCPU::get_offsets(SizeType idm) const {
    return m_offsets.at(idm);
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
    const auto shifts =
        compute_dedisperse_shifts(dm, f_min, f_max, nchans, tsamp);
    for (SizeType ichan = 0; ichan < nchans; ++ichan) {
        const SizeType start = ichan * nsamps;
        const SizeType end   = start + nsamps;
        int shift            = shifts[ichan] % static_cast<int>(nsamps);
        if (shift < 0) {
            shift += static_cast<int>(nsamps);
        }
        if (shift == 0) {
            continue;
        }
        std::rotate(waterfall + start, waterfall + end - shift,
                    waterfall + end);
    }
}
} // namespace dmt::utils
