#pragma once

#include <cuda/std/span>
#include <cuda_runtime_api.h>
#include <cufft.h>

#include "dmt/common/types.hpp"

namespace dmt::bb_utils_cu {

/**
 * @brief Swap halves of two spectrums (fftshift) along the last dimension (n).
 * Operates in-place on device memory. Since C2C FFTs put the center frequency
 * at bin zero in the frequency domain, the two halves of the spectrum need to
 * be swapped.
 *
 * For each row j in [0, batch_size), swaps elements i and i + n/2 for i < n/2
 * in both data1 and data2, effectively shifting the spectrum to center the zero
 * frequency.
 *
 * @param data1 Span of device memory holding complex data (batch_size rows, n
 * columns).
 * @param data2 Span of device memory holding complex data (batch_size rows, n
 * columns).
 * @param n Size of the dimension to swap (must be even).
 * @param batch_size Number of rows.
 * @param stream CUDA stream for asynchronous execution.
 */
void swap_spectrum(cuda::std::span<cufftComplex> data1,
                   cuda::std::span<cufftComplex> data2,
                   int n,
                   int batch_size,
                   cudaStream_t stream);

/**
 * @brief Apply a pre-computed chirp to a spectrum.
 *
 * @param data1_in Pointer to device memory holding complex data (batch_size
 * rows, n columns).
 * @param data2_in Pointer to device memory holding complex data (batch_size
 * rows, n columns).
 * @param chirp_table Pointer to device memory holding complex data (nsub rows,
 * nbin columns).
 * @param data1_out Pointer to device memory holding complex data (batch_size
 * rows, n columns).
 * @param data2_out Pointer to device memory holding complex data (batch_size
 * rows, n columns).
 * @param nsub Number of sub-bands.
 * @param nbin Number of frequency bins.
 * @param nfft Number of FFT points.
 * @param idm DM index.
 * @param scale Scaling factor.
 * @param stream CUDA stream for asynchronous execution.
 */
void apply_chirp(cuda::std::span<const cufftComplex> data1_in,
                 cuda::std::span<const cufftComplex> data2_in,
                 cuda::std::span<const cufftComplex> chirp_table,
                 cuda::std::span<cufftComplex> data1_out,
                 cuda::std::span<cufftComplex> data2_out,
                 int nsub,
                 int nbin,
                 int nfft,
                 int idm,
                 float scale,
                 cudaStream_t stream);

/**
 * Computes intensity from two FFT inputs, unpads overlap regions, and
 * transposes data.
 * @param fft_p1 First FFT input (complex, size nfft * nchan * mbin).
 * @param fft_p2 Second FFT input (complex, size nfft * nchan * mbin).
 * @param intensity Output intensity (size nsub * nchan * nfft * (mbin - 2 *
 * (noverlap / nchan))).
 * @param nchan Number of channels.
 * @param nfft Number of FFTs.
 * @param nsub Number of subbands.
 * @param mbin Number of bins per FFT.
 * @param noverlap Total overlap samples across channels.
 * @param stream CUDA stream for asynchronous execution.
 */
void unpad_detect(cuda::std::span<const ComplexTypeCUDA> fft_p1,
                  cuda::std::span<const ComplexTypeCUDA> fft_p2,
                  cuda::std::span<float> intensity,
                  int nchan,
                  int nfft,
                  int nsub,
                  int mbin,
                  int noverlap,
                  cudaStream_t stream);
} // namespace dmt::bb_utils_cu
