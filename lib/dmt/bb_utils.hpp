#pragma once

#include <array>
#include <optional>
#include <span>
#include <string_view>

#include "dmt/common/types.hpp"

namespace dmt::bb_utils {

inline constexpr std::array<std::pair<std::string_view, BasebandDataOrder>, 3>
    kBasebandDataOrders{
        {
            {"FTPRI", BasebandDataOrder::kFTPRI},
            {"PRITF", BasebandDataOrder::kPRITF},
            {"RITFP", BasebandDataOrder::kRITFP},
        },
};

[[nodiscard]] std::string supported_baseband_data_orders() noexcept;

[[nodiscard]] std::optional<BasebandDataOrder>
find_baseband_data_order(std::string_view name) noexcept;

[[nodiscard]] BasebandDataOrder parse_baseband_data_order(std::string_view name);

// Compute the chirp table for coherent dedispersion.
void compute_chirp(std::span<const float> dm_grid,
                   std::span<ComplexType> chirp_table,
                   float fcenter,
                   float bw,
                   SizeType nbin,
                   SizeType nsub,
                   SizeType nchan);

void swap_spectrum(std::span<ComplexType> data1,
                   std::span<ComplexType> data2,
                   int n,
                   int batch_size);

void apply_chirp(std::span<const ComplexType> data1_in,
                 std::span<const ComplexType> data2_in,
                 std::span<const ComplexType> chirp_table,
                 std::span<ComplexType> data1_out,
                 std::span<ComplexType> data2_out,
                 SizeType nsub,
                 SizeType nbin,
                 SizeType nfft,
                 SizeType idm,
                 float scale);

void unpad_detect(std::span<const ComplexType> fft_p1,
                  std::span<const ComplexType> fft_p2,
                  std::span<float> intensity,
                  SizeType nchan,
                  SizeType nfft,
                  SizeType nsub,
                  SizeType mbin,
                  SizeType noverlap);

} // namespace dmt::bb_utils
