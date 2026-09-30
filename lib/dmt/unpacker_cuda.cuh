#pragma once

/**
 * @file unpacker_cuda.cuh
 * @brief Baseband block unpacker on the GPU (private): the device twin of
 * BasebandUnpackerCPU.
 */

#include <memory>
#include <span>
#include <vector>

#include "dmt/gpu_compat.cuh"

#include "dmt/common/baseband.hpp"
#include "dmt/common/types.hpp"
#include "dmt/gpu_utils.cuh"

namespace dmt::utils {

/**
 * @brief Decodes one device-resident baseband block (one device span per
 * subband group) into the forward-FFT input layout (sub, ifft, pol, nbin) of
 * complex float, like BasebandUnpackerCPU::execute().
 */
class BasebandUnpackerCUDA {
public:
    BasebandUnpackerCUDA(const BasebandFormat& format,
                         std::span<const SizeType> subband_groups,
                         SizeType nbin,
                         SizeType nfft,
                         SizeType noverlap,
                         int device_id = 0);
    ~BasebandUnpackerCUDA();
    BasebandUnpackerCUDA(BasebandUnpackerCUDA&&) noexcept;
    BasebandUnpackerCUDA& operator=(BasebandUnpackerCUDA&&) noexcept;
    BasebandUnpackerCUDA(const BasebandUnpackerCUDA&)            = delete;
    BasebandUnpackerCUDA& operator=(const BasebandUnpackerCUDA&) = delete;

    [[nodiscard]] SizeType block_nsamps() const noexcept;
    [[nodiscard]] SizeType input_size(SizeType igroup) const;
    [[nodiscard]] SizeType output_size() const noexcept;

    /**
     * @param groups One device span per subband group, input_size(g) bytes.
     * @param out output_size() complex values (device).
     * @throws std::invalid_argument on a size mismatch.
     */
    void execute(std::span<const cuda::std::span<const uint8_t>> groups,
                 cuda::std::span<ComplexTypeGPU> out,
                 cudaStream_t stream = nullptr);

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace dmt::utils
