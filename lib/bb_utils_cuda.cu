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
            int src_idx       = (ifft * nsub * nchan * mbin) +
                          ((nsub - isub - 1) * nchan * mbin) + (ichan * mbin) +
                          ibin_adjusted;
            int dst_idx =
                (isub * nchan * msamp) + ((nchan - ichan - 1) * msamp) + isamp;
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
    assert(chirp_table.size() == static_cast<SizeType>(nsub * nbin) &&
           "chirp_table must have the same size as nsub * nbin");

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
    assert(fft_p1.size() == static_cast<SizeType>(nfft * nchan * mbin));
    assert(fft_p2.size() == static_cast<SizeType>(nfft * nchan * mbin));

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

} // namespace dmt::bb_utils
