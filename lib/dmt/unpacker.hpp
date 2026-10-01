#pragma once

/**
 * @file unpacker.hpp
 * @brief Baseband block unpacker: any BasebandFormat -> overlapping FFT
 * blocks of complex float per polarisation and subband.
 */

#include <memory>
#include <span>

#include "dmt/common/baseband.hpp"
#include "dmt/common/types.hpp"

namespace dmt::utils {

/**
 * @brief Decodes one baseband block (one span per subband group) into the
 * forward-FFT input layout (sub, ifft, pol, nbin) of complex float.
 *
 * FFT block j of a subband holds raw samples [j * L, j * L + nbin), with
 * L = nbin - 2 * noverlap, so consecutive blocks overlap by 2 * noverlap
 * (overlap-save). The input block holds nfft * L + 2 * noverlap samples per
 * subband; nothing is zero-padded.
 */
class BasebandUnpackerCPU {
public:
    BasebandUnpackerCPU(const BasebandFormat& format,
                        std::span<const SizeType> subband_groups,
                        SizeType nbin,
                        SizeType nfft,
                        SizeType noverlap,
                        int nthreads = 1);

    ~BasebandUnpackerCPU();
    BasebandUnpackerCPU(BasebandUnpackerCPU&&) noexcept;
    BasebandUnpackerCPU& operator=(BasebandUnpackerCPU&&) noexcept;
    BasebandUnpackerCPU(const BasebandUnpackerCPU&)            = delete;
    BasebandUnpackerCPU& operator=(const BasebandUnpackerCPU&) = delete;

    /// Raw samples per subband of one block (nfft * L + 2 * noverlap).
    [[nodiscard]] SizeType block_nsamps() const noexcept;
    /// Bytes of group @p igroup per block.
    [[nodiscard]] SizeType input_size(SizeType igroup) const;
    /// Complex elements of the output (nsub * nfft * 2 * nbin).
    [[nodiscard]] SizeType output_size() const noexcept;

    /// @throws std::invalid_argument unless there is one span of
    /// input_size(g) bytes per group.
    void validate(std::span<const std::span<const uint8_t>> groups) const;

    /**
     * @brief Decodes FFT block @p ifft of subband @p isub (both
     * polarisations, nbin samples each). No validation: call validate()
     * once per block of input first.
     */
    void unpack_block(std::span<const std::span<const uint8_t>> groups,
                      SizeType isub,
                      SizeType ifft,
                      ComplexType* pol0,
                      ComplexType* pol1) const noexcept;

    /**
     * @param groups One span per subband group, each input_size(g) bytes.
     * @param out output_size() complex values.
     * @throws std::invalid_argument on a size mismatch.
     */
    void execute(std::span<const std::span<const uint8_t>> groups,
                 std::span<ComplexType> out) const;

private:
    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace dmt::utils
