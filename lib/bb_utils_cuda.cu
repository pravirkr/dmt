#include "dmt/bb_utils_cuda.cuh"

#include <cassert>

#include <cuda/std/span>
#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "dmt/common/types.hpp"
#include "dmt/cuda_utils.cuh"

namespace dmt::bb_utils {

namespace {

// Functor for in-place fftshift of two complex buffers
struct SwapSpectrumDual {
    ComplexTypeCUDA *buf1, *buf2;
    int n, batch_size, mid_point;

    __device__ void operator()(int idx) const {
        // Map idx to (row, col_pair) for swapping
        int row      = idx / mid_point;
        int col_pair = idx % mid_point;
        if (row < batch_size) {
            int col1 = col_pair;
            int col2 = col_pair + mid_point;
            int idx1 = (row * n) + col1;
            int idx2 = (row * n) + col2;
            // Swap buf1
            auto temp1 = buf1[idx1];
            buf1[idx1] = buf1[idx2];
            buf1[idx2] = temp1;
            // Swap buf2
            auto temp2 = buf2[idx1];
            buf2[idx1] = buf2[idx2];
            buf2[idx2] = temp2;
        }
    }
};

// Performs pointwise complex multiplication (and scaling) for two pairs:
// c[i + nx*j] = scale * (a[i + nx*j] * b[l*nx + i])
// e[i + nx*j] = scale * (d[i + nx*j] * b[l*nx + i])
// for i in [0, nx), j in [0, ny), where l is the row index of b.
struct ComplexMulScaleDual {
    const ComplexTypeCUDA *a, *d, *b;
    ComplexTypeCUDA *c, *e;
    int nx, ny, l;
    float scale;

    __device__ void operator()(int idx) const {
        int j = idx / nx;
        int i = idx - (j * nx);
        if (i < nx && j < ny) {
            // c[idx] = scale * (a[idx] * b[l * nx + i]);
            //  -> (ar + ai*i)(br + bi*i) = (ar*br - ai*bi) + (ar*bi + ai*br)*i
            float real1 = fmaf(a[idx].real(), b[(l * nx) + i].real(),
                               -a[idx].imag() * b[(l * nx) + i].imag());
            float imag1 = fmaf(a[idx].real(), b[(l * nx) + i].imag(),
                               a[idx].imag() * b[(l * nx) + i].real());
            c[idx]      = ComplexTypeCUDA(fmaf(scale, real1, 0.0F),
                                          fmaf(scale, imag1, 0.0F));
            float real2 = fmaf(d[idx].real(), b[(l * nx) + i].real(),
                               -d[idx].imag() * b[(l * nx) + i].imag());
            float imag2 = fmaf(d[idx].real(), b[(l * nx) + i].imag(),
                               d[idx].imag() * b[(l * nx) + i].real());
            e[idx]      = ComplexTypeCUDA(fmaf(scale, real2, 0.0F),
                                          fmaf(scale, imag2, 0.0F));
        }
    }
};

// Functor for the out-of-place causal streaming delay line.
// For channel c with shift S_c, if t >= S_c, y[t] = in[t - S_c].
// If t < S_c, y[t] is read from the causal history buffer.
struct ChannelDelayLineFunctor {
    const float* in;
    float* out;
    const float* history;
    const int* shift_table;
    const SizeType* offsets;
    int idm, nchans, nsamps;

    __device__ void operator()(int idx) const {
        int ichan = idx / nsamps;
        int t     = idx - (ichan * nsamps);
        int shift = shift_table[(idm * nchans) + ichan];
        if (shift <= 0) {
            out[idx] = in[idx];
            return;
        }
        if (t >= shift) {
            out[idx] = in[(ichan * nsamps) + (t - shift)];
        } else {
            SizeType offset = offsets[(idm * nchans) + ichan];
            out[idx]        = history[offset + t];
        }
    }
};

__global__ void update_delay_history_kernel(const float* in,
                                            float* history,
                                            const int* shift_table,
                                            const SizeType* offsets,
                                            int idm,
                                            int nchans,
                                            int nsamps) {
    int ichan = blockIdx.x;
    if (ichan >= nchans) {
        return;
    }
    int shift = shift_table[(idm * nchans) + ichan];
    if (shift <= 0) {
        return;
    }
    SizeType offset = offsets[(idm * nchans) + ichan];

    if (nsamps >= shift) {
        for (int i = threadIdx.x; i < shift; i += blockDim.x) {
            history[offset + i] = in[(ichan * nsamps) + (nsamps - shift + i)];
        }
    } else {
        for (int i = threadIdx.x; i < shift - nsamps; i += blockDim.x) {
            history[offset + i] = history[offset + i + nsamps];
        }
        __syncthreads();
        for (int i = threadIdx.x; i < nsamps; i += blockDim.x) {
            history[offset + (shift - nsamps) + i] = in[(ichan * nsamps) + i];
        }
    }
}

// Functor for Transposing and Unpadding the FFTs and calculating the intensity
struct TransposeUnpadDetect {
    const ComplexTypeCUDA *fft_p1, *fft_p2;
    float* intensity;
    int nchan, nfft, nsub, mbin, noverlap_per_channel, mbin_adjusted, msamp;

    __device__ void operator()(int idx) const {
        int ibin  = idx % mbin_adjusted;
        int tmp   = idx / mbin_adjusted;
        int ichan = tmp % nchan;
        int ifft  = tmp / nchan;
        for (int isub = 0; isub < nsub; ++isub) {
            int isamp         = ibin + (mbin_adjusted * ifft);
            int ibin_adjusted = ibin + noverlap_per_channel;
            int src_idx = (ifft * nsub * nchan * mbin) + (isub * nchan * mbin) +
                          (ichan * mbin) + ibin_adjusted;
            int dst_idx = (isub * nchan * msamp) + (ichan * msamp) + isamp;
            intensity[dst_idx] = norm(fft_p1[src_idx]) + norm(fft_p2[src_idx]);
        }
    }
};
} // namespace

void swap_spectrum(cuda::std::span<ComplexTypeCUDA> data1,
                   cuda::std::span<ComplexTypeCUDA> data2,
                   int n,
                   int batch_size,
                   cudaStream_t stream) {
    assert(n != 0 && batch_size != 0 && "Invalid dimensions in swap_spectrum");
    assert(n % 2 == 0 && "n must be even in swap_spectrum");
    assert(data1.size() == static_cast<SizeType>(n * batch_size) &&
           "Span size does not match dimensions for data1");
    assert(data2.size() == static_cast<SizeType>(n * batch_size) &&
           "Span size does not match dimensions for data2");

    const int mid_point = n / 2;
    auto first          = thrust::counting_iterator<int>(0);
    auto last = thrust::counting_iterator<int>(mid_point * batch_size);
    SwapSpectrumDual functor{.buf1       = data1.data(),
                             .buf2       = data2.data(),
                             .n          = n,
                             .batch_size = batch_size,
                             .mid_point  = mid_point};
    thrust::for_each(thrust::cuda::par.on(stream), first, last, functor);
    cuda_utils::check_last_cuda_error("thrust::for_each failed");
}

void apply_chirp(cuda::std::span<const ComplexTypeCUDA> data1_in,
                 cuda::std::span<const ComplexTypeCUDA> data2_in,
                 cuda::std::span<const ComplexTypeCUDA> chirp_table,
                 cuda::std::span<ComplexTypeCUDA> data1_out,
                 cuda::std::span<ComplexTypeCUDA> data2_out,
                 int nsub,
                 int nbin,
                 int nfft,
                 int idm,
                 float scale,
                 cudaStream_t stream) {
    assert(data1_in.size() == data1_out.size() &&
           "data1_in and data1_out must have the same size");
    assert(data2_in.size() == data2_out.size() &&
           "data2_in and data2_out must have the same size");
    [[maybe_unused]] const auto nx = static_cast<SizeType>(nsub * nbin);
    assert(nx > 0 && chirp_table.size() % nx == 0 &&
           "chirp_table size must be a multiple of nsub * nbin");
    assert(static_cast<SizeType>(idm) < chirp_table.size() / nx &&
           "idm is out of range for chirp_table");

    auto first = thrust::counting_iterator<int>(0);
    auto last  = thrust::counting_iterator<int>(nsub * nbin * nfft);
    ComplexMulScaleDual functor{.a     = data1_in.data(),
                                .d     = data2_in.data(),
                                .b     = chirp_table.data(),
                                .c     = data1_out.data(),
                                .e     = data2_out.data(),
                                .nx    = nsub * nbin,
                                .ny    = nfft,
                                .l     = idm,
                                .scale = scale};
    thrust::for_each(thrust::cuda::par.on(stream), first, last, functor);
    cuda_utils::check_last_cuda_error("thrust::for_each failed");
}

void unpad_detect(cuda::std::span<const ComplexTypeCUDA> fft_p1,
                  cuda::std::span<const ComplexTypeCUDA> fft_p2,
                  cuda::std::span<float> intensity,
                  int nchan,
                  int nfft,
                  int nsub,
                  int mbin,
                  int noverlap,
                  cudaStream_t stream) {
    assert(fft_p1.size() == static_cast<SizeType>(nfft * nsub * nchan * mbin));
    assert(fft_p2.size() == static_cast<SizeType>(nfft * nsub * nchan * mbin));

    const auto noverlap_per_channel = noverlap / nchan;
    const auto mbin_adjusted        = mbin - (2 * noverlap_per_channel);
    const auto msamp                = nfft * mbin_adjusted;

    assert(mbin_adjusted > 0 && "Invalid mbin or noverlap");
    assert(intensity.size() == static_cast<SizeType>(nsub * nchan * msamp));

    auto first = thrust::counting_iterator<int>(0);
    auto last  = thrust::counting_iterator<int>(mbin_adjusted * nchan * nfft);
    TransposeUnpadDetect functor{.fft_p1               = fft_p1.data(),
                                 .fft_p2               = fft_p2.data(),
                                 .intensity            = intensity.data(),
                                 .nchan                = nchan,
                                 .nfft                 = nfft,
                                 .nsub                 = nsub,
                                 .mbin                 = mbin,
                                 .noverlap_per_channel = noverlap_per_channel,
                                 .mbin_adjusted        = mbin_adjusted,
                                 .msamp                = msamp};
    thrust::for_each(thrust::cuda::par.on(stream), first, last, functor);
    cuda_utils::check_last_cuda_error("thrust::for_each failed");
}

void channel_delay_line(cuda::std::span<const float> in,
                        cuda::std::span<float> out,
                        cuda::std::span<float> history,
                        cuda::std::span<const int> shift_table,
                        cuda::std::span<const SizeType> offsets,
                        int idm,
                        int nchans,
                        int nsamps,
                        cudaStream_t stream) {
    assert(in.size() ==
           static_cast<SizeType>(nchans) * static_cast<SizeType>(nsamps));
    assert(out.size() == in.size());

    auto first = thrust::counting_iterator<int>(0);
    auto last  = thrust::counting_iterator<int>(nchans * nsamps);
    ChannelDelayLineFunctor functor{.in          = in.data(),
                                    .out         = out.data(),
                                    .history     = history.data(),
                                    .shift_table = shift_table.data(),
                                    .offsets     = offsets.data(),
                                    .idm         = idm,
                                    .nchans      = nchans,
                                    .nsamps      = nsamps};
    thrust::for_each(thrust::cuda::par.on(stream), first, last, functor);
    cuda_utils::check_last_cuda_error(
        "channel_delay_line: thrust::for_each failed");

    if (!history.empty()) {
        int threads_per_block = 256;
        int blocks            = nchans;
        update_delay_history_kernel<<<blocks, threads_per_block, 0, stream>>>(
            in.data(), history.data(), shift_table.data(), offsets.data(), idm,
            nchans, nsamps);
        cuda_utils::check_last_cuda_error(
            "channel_delay_line: update_delay_history_kernel failed");
    }
}

namespace {
__global__ void compute_chirp_kernel(const float* dm_grid,
                                     ComplexTypeCUDA* chirp_table,
                                     double fcenter,
                                     double bw,
                                     double bw_sub,
                                     double bw_chan,
                                     double bw_bin,
                                     double taper_const,
                                     double taper_exp,
                                     double coeff_const,
                                     SizeType ndm,
                                     SizeType nsub,
                                     SizeType nchan,
                                     SizeType mbin,
                                     SizeType total_elements) {
    SizeType idx = static_cast<SizeType>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_elements) {
        return;
    }

    SizeType ibin  = idx % mbin;
    SizeType tmp1  = idx / mbin;
    SizeType ichan = tmp1 % nchan;
    SizeType tmp2  = tmp1 / nchan;
    SizeType isub  = tmp2 % nsub;
    SizeType idm   = tmp2 / nsub;

    double freq_sub =
        fcenter - (bw / 2.0) + ((static_cast<double>(isub) + 0.5) * bw_sub);
    double bin_freq =
        (-bw_chan / 2.0) + ((static_cast<double>(ibin) + 0.5) * bw_bin);
    double freq_chan =
        freq_sub +
        ((static_cast<double>(ichan) - static_cast<double>(nchan) / 2.0 + 0.5) *
         bw_chan);
    double coeff      = coeff_const * static_cast<double>(dm_grid[idm]);
    double freq_ratio = bin_freq / freq_chan;
    double phase_delay =
        -coeff * freq_ratio * freq_ratio / (freq_chan + bin_freq);
    double taper = 1.0 / sqrt(1.0 + pow(bin_freq * taper_const, taper_exp));

    double s, c;
    sincos(phase_delay, &s, &c);
    chirp_table[idx] = ComplexTypeCUDA(static_cast<float>(taper * c),
                                       static_cast<float>(taper * s));
}
} // namespace

void compute_chirp(cuda::std::span<const float> dm_grid,
                   cuda::std::span<ComplexTypeCUDA> chirp_table,
                   float fcenter,
                   float bw,
                   SizeType nbin,
                   SizeType nsub,
                   SizeType nchan,
                   cudaStream_t stream) {
    const auto ndm       = dm_grid.size();
    const SizeType mbin  = nbin / nchan;
    const SizeType total = ndm * nsub * nchan * mbin;
    assert(chirp_table.size() == total);

    const double bw_sub  = static_cast<double>(bw) / static_cast<double>(nsub);
    const double bw_chan = bw_sub / static_cast<double>(nchan);
    const double bw_bin  = bw_chan / static_cast<double>(mbin);
    const double taper_const = 1.0 / (0.47 * bw_chan);
    const double taper_exp   = 80.0;
    const double coeff_const = 2.0 * std::numbers::pi_v<double> *
                               static_cast<double>(kDispConst) * 1.0E6;

    const int threads_per_block = 256;
    const int blocks            = static_cast<int>(
        (total + static_cast<SizeType>(threads_per_block) - 1) /
        static_cast<SizeType>(threads_per_block));

    compute_chirp_kernel<<<blocks, threads_per_block, 0, stream>>>(
        dm_grid.data(), chirp_table.data(), static_cast<double>(fcenter),
        static_cast<double>(bw), bw_sub, bw_chan, bw_bin, taper_const,
        taper_exp, coeff_const, ndm, nsub, nchan, mbin, total);
    cuda_utils::check_last_cuda_error("compute_chirp_kernel failed");
}

} // namespace dmt::bb_utils
