
#include <dmt/common/types.hpp>
#include <dmt/utils/simulate.hpp>

#include "dmt/dm_utils.hpp"

std::tuple<std::vector<float>, SizeType> generate_pure_frb(SizeType nchans,
                                                           SizeType nsamps,
                                                           float f_min,
                                                           float f_max,
                                                           SizeType dt,
                                                           float pulse_toa,
                                                           float amplitude) {
    std::vector<float> arr(nchans * nsamps, 0.0F);
    const float foff          = (f_max - f_min) / static_cast<float>(nchans);
    const float foff_half     = foff / 2.0F;
    SizeType nsamps_dispersed = 0;

    for (SizeType ichan = 0; ichan < nchans; ++ichan) {
        const auto freq =
            f_min + (static_cast<float>(ichan) * foff) + foff_half;
        const auto freq_min = freq - foff_half;
        const auto freq_max = freq + foff_half;
        const auto dt_start = static_cast<float>(dt) *
                              dm_utils::cff(f_min, freq_min, f_min, f_max);
        const auto tstart      = pulse_toa - dt_start;
        const auto tstart_int  = static_cast<SizeType>(std::floor(tstart));
        const auto tstart_frac = tstart - static_cast<float>(tstart_int);

        const auto dt_sub = static_cast<float>(dt) *
                            dm_utils::cff(freq_min, freq_max, f_min, f_max);
        const auto tend      = tstart - dt_sub;
        const auto tend_int  = static_cast<SizeType>(std::floor(tend));
        const auto tend_frac = tend - static_cast<float>(tend_int);

        float* arr_chan_start = &arr[ichan * nsamps];

        if (0 <= tend_int && tend_int <= tstart_int && tstart_int < nsamps) {
            if (tend_int == tstart_int) {
                arr_chan_start[tend_int] = amplitude;
                nsamps_dispersed += 1;
            } else {
                const float amp_per_sample = amplitude / dt_sub;
                std::fill(arr_chan_start + tend_int,
                          arr_chan_start + tstart_int + 1, amp_per_sample);
                arr_chan_start[tend_int] *= tend_frac;
                arr_chan_start[tstart_int] *= tstart_frac;
                nsamps_dispersed += tstart_int - tend_int + 1;
            }
        } else if (tend_int < 0 && 0 <= tstart_int && tstart_int < nsamps) {
            const float amp_per_sample = amplitude / dt_sub;
            std::fill(arr_chan_start, arr_chan_start + tstart_int + 1,
                      amp_per_sample);
            arr_chan_start[tstart_int] *= tstart_frac;
            nsamps_dispersed += tstart_int + 1;
        } else if (0 <= tend_int && tend_int < nsamps && nsamps <= tstart_int) {
            const float amp_per_sample = amplitude / dt_sub;
            std::fill(arr_chan_start + tend_int, arr_chan_start + nsamps,
                      amp_per_sample);
            arr_chan_start[tend_int] *= tend_frac;
            nsamps_dispersed += nsamps - tend_int;
        }
    }
    return {arr, nsamps_dispersed};
}