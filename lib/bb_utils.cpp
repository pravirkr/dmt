#include "dmt/bb_utils.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <format>
#include <numbers>

#include "dmt/common/types.hpp"

namespace dmt::bb_utils {

void compute_chirp(std::span<const float> dm_grid,
                   std::span<ComplexType> chirp_table,
                   float fcenter,
                   float bw,
                   SizeType nbin,
                   SizeType nsub,
                   SizeType nchan) {
    const auto ndm       = dm_grid.size();
    const SizeType mbin  = nbin / nchan;
    const double bw_sub  = static_cast<double>(bw) / static_cast<double>(nsub);
    const double bw_chan = bw_sub / static_cast<double>(nchan);
    const double bw_bin  = bw_chan / static_cast<double>(mbin);

    if (chirp_table.size() != ndm * nsub * nbin) {
        throw std::runtime_error("Chirp table size mismatch");
    }

    std::vector<double> freqs_sub(nsub);
    for (SizeType i = 0; i < nsub; ++i) {
        freqs_sub[i] =
            static_cast<double>(fcenter) - (static_cast<double>(bw) / 2.0) +
            ((static_cast<double>(i) + 0.5) * bw_sub);
    }
    std::vector<double> bin_freqs(mbin);
    for (SizeType i = 0; i < mbin; ++i) {
        bin_freqs[i] =
            (-bw_chan / 2.0) + ((static_cast<double>(i) + 0.5) * bw_bin);
    }

    const double taper_const = 1.0 / (0.47 * bw_chan);
    const double taper_exp   = 80.0;
    const double coeff_const =
        2.0 * std::numbers::pi_v<double> * static_cast<double>(kDispConst) * 1.0E6;

    for (SizeType idm = 0; idm < ndm; ++idm) {
        const double coeff = coeff_const * static_cast<double>(dm_grid[idm]);
        for (SizeType isub = 0; isub < nsub; ++isub) {
            for (SizeType ichan = 0; ichan < nchan; ++ichan) {
                const double freq_chan =
                    freqs_sub[isub] + ((static_cast<double>(ichan) -
                                        static_cast<double>(nchan) / 2.0 + 0.5) *
                                       bw_chan);
                for (SizeType ibin = 0; ibin < mbin; ++ibin) {
                    const double bin_freq    = bin_freqs[ibin];
                    const double freq_ratio  = bin_freq / freq_chan;
                    const double phase_delay = -coeff * freq_ratio * freq_ratio /
                                              (freq_chan + bin_freq);
                    const double taper =
                        1.0 / std::sqrt(1.0 + std::pow(bin_freq * taper_const,
                                                       taper_exp));
                    const SizeType idx = (idm * nsub * nchan * mbin) +
                                         (isub * nchan * mbin) +
                                         (ichan * mbin) + ibin;
                    chirp_table[idx] = std::polar(static_cast<float>(taper),
                                                  static_cast<float>(phase_delay));
                }
            }
        }
    }
}

void swap_spectrum(std::span<ComplexType> data1,
                   std::span<ComplexType> data2,
                   int n,
                   int batch_size) {
    if (n == 0 || batch_size == 0) {
        throw std::invalid_argument(std::format(
            "Invalid dimensions: n={}, batch_size={}", n, batch_size));
    }
    if (n % 2 != 0) {
        throw std::invalid_argument(std::format("n ({}) must be even", n));
    }
    const auto total_elements = n * batch_size;
    if (data1.size() != static_cast<SizeType>(total_elements) ||
        data2.size() != static_cast<SizeType>(total_elements)) {
        throw std::invalid_argument(
            std::format("Span size mismatch: data1={}, data2={}, expected={}",
                        data1.size(), data2.size(), total_elements));
    }

    ComplexType* data1_ptr = data1.data();
    ComplexType* data2_ptr = data2.data();
    // Swap the halves along the last dimension
    const auto mid_point = n / 2;
#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel for default(none)                                         \
    shared(data1_ptr, data2_ptr, n, batch_size, mid_point)
#endif
    for (int j = 0; j < batch_size; ++j) {
        const auto offset           = j * n;
        ComplexType* row_start1_ptr = data1_ptr + offset;
        std::rotate(row_start1_ptr, row_start1_ptr + mid_point,
                    row_start1_ptr + n);
        ComplexType* row_start2_ptr = data2_ptr + offset;
        std::rotate(row_start2_ptr, row_start2_ptr + mid_point,
                    row_start2_ptr + n);
    }
}

void apply_chirp(std::span<const ComplexType> data1_in,
                 std::span<const ComplexType> data2_in,
                 std::span<const ComplexType> chirp_table,
                 std::span<ComplexType> data1_out,
                 std::span<ComplexType> data2_out,
                 SizeType nsub,
                 SizeType nbin,
                 SizeType nfft,
                 SizeType idm,
                 float scale) {
    if (data1_in.size() != data1_out.size()) {
        throw std::runtime_error(
            "data1_in and data1_out must have the same size");
    }
    if (data2_in.size() != data2_out.size()) {
        throw std::runtime_error(
            "data2_in and data2_out must have the same size");
    }
    const auto nx = nsub * nbin;
    if (nx == 0 || chirp_table.size() % nx != 0) {
        throw std::runtime_error(
            "chirp_table size must be a multiple of nsub * nbin");
    }
    if (idm >= chirp_table.size() / nx) {
        throw std::runtime_error("idm is out of range for chirp_table");
    }
    const auto ny = nfft;
#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel for default(none)                                         \
    shared(data1_out, data2_out, data1_in, data2_in, chirp_table, nx, ny, idm, \
               scale)
#endif
    for (SizeType i = 0; i < nx; ++i) {
        for (SizeType j = 0; j < ny; ++j) {
            const auto idx       = i + (nx * j);
            const auto chirp_idx = i + (nx * idm);
            data1_out[idx] = data1_in[idx] * chirp_table[chirp_idx] * scale;
            data2_out[idx] = data2_in[idx] * chirp_table[chirp_idx] * scale;
        }
    }
}

void unpad_detect(std::span<const ComplexType> fft_p1,
                  std::span<const ComplexType> fft_p2,
                  std::span<float> intensity,
                  SizeType nchan,
                  SizeType nfft,
                  SizeType nsub,
                  SizeType mbin,
                  SizeType noverlap) {
    const SizeType noverlap_per_channel = noverlap / nchan;
    const SizeType mbin_adjusted        = mbin - (2 * noverlap_per_channel);
    const SizeType msamp                = nfft * mbin_adjusted;

    if (fft_p1.size() != (nfft * nsub * nchan * mbin)) {
        throw std::runtime_error("Invalid input size");
    }
    if (fft_p2.size() != (nfft * nsub * nchan * mbin)) {
        throw std::runtime_error("Invalid input size");
    }
    if (intensity.size() != (nsub * nchan * msamp)) {
        throw std::runtime_error("Invalid output size");
    }

#ifdef DMT_ENABLE_OPENMP
#pragma omp parallel for collapse(4) default(none)                             \
    shared(intensity, fft_p1, fft_p2, nsub, nchan, nfft, msamp, mbin,          \
               mbin_adjusted, noverlap_per_channel)
#endif
    for (SizeType ibin = 0; ibin < mbin_adjusted; ++ibin) {
        for (SizeType ichan = 0; ichan < nchan; ++ichan) {
            for (SizeType ifft = 0; ifft < nfft; ++ifft) {
                for (SizeType isub = 0; isub < nsub; ++isub) {
                    const SizeType isamp = ibin + (mbin_adjusted * ifft);
                    const SizeType ibin_adjusted = ibin + noverlap_per_channel;
                    const SizeType src_idx =
                        (ifft * nsub * nchan * mbin) +
                        (isub * nchan * mbin) + (ichan * mbin) +
                        ibin_adjusted;
                    const SizeType dst_idx =
                        (isub * nchan * msamp) + (ichan * msamp) + isamp;
                    intensity[dst_idx] =
                        std::norm(fft_p1[src_idx]) + std::norm(fft_p2[src_idx]);
                }
            }
        }
    }
}

} // namespace dmt::bb_utils
