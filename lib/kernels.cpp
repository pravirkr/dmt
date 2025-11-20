#include "dmt/kernels.hpp"

namespace dmt::kernels {
void offset_add(const float* __restrict__ data_tail,
                SizeType size_tail,
                const float* __restrict__ data_head,
                SizeType size_head,
                float* __restrict__ out,
                SizeType size_out,
                SizeType offset) noexcept {
    // Debug checks using assert (only active when NDEBUG is not defined)
    assert(size_tail == size_head && "Input tail and head sizes must be equal");
    assert(size_out >= size_tail && "Output size must be >= Input tail size");
    assert(offset < size_tail && "Offset must be < input tail size");

    const SizeType nsum = size_tail - offset;
    // Part 1: Direct copy (first 'offset' elements from data_tail)
    std::copy_n(data_tail, offset, out);
// Part 2: Vectorized addition (overlap region)
#pragma omp simd
    for (SizeType i = 0; i < nsum; ++i) {
        out[offset + i] = data_tail[offset + i] + data_head[i];
    }
    // Part 3: Copy remaining from data_head if needed
    const SizeType nrest = std::min(offset, size_out - size_tail);
    if (nrest > 0) {
        std::copy_n(data_head + nsum, nrest, out + size_tail);
    }
    // Part 4: Zero-fill remaining output if needed
    const SizeType filled_so_far = size_tail + nrest;
    if (filled_so_far < size_out) {
        std::fill(out + filled_so_far, out + size_out, 0.0F);
    }
}
} // namespace dmt::kernels