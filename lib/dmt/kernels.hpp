#pragma once

#include "dmt/common/types.hpp"

namespace dmt::kernels {
/**
 * @brief Computes out[k] = data_tail[k] for k < offset,
 * out[k] = data_tail[k] + data_head[k - offset] for offset <= k < size_tail,
 * out[k] = data_head[k-offset] for size_tail <= k < size_tail + min(offset,
 * size_out-size_tail) out[k] = 0.0 otherwise up to size_out
 * @param data_tail Input tail data
 * @param size_tail Size of data_tail
 * @param data_head Input head data
 * @param size_head Size of data_head
 * @param out Output data
 * @param size_out Size of out data
 * @param offset Offset for summing data_head into out data
 */
void offset_add(const float* __restrict__ data_tail,
                SizeType size_tail,
                const float* __restrict__ data_head,
                SizeType size_head,
                float* __restrict__ out,
                SizeType size_out,
                SizeType offset) noexcept;
} // namespace dmt::kernels
