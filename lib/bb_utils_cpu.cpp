#include "dmt/bb_utils_cpu.hpp"

void pointwise_complex_multiply(const ComplexType* __restrict__ a,
                                const ComplexType* __restrict__ b,
                                ComplexType* __restrict__ c,
                                SizeType nx,
                                SizeType ny,
                                SizeType idm,
                                float scale) {
    for (SizeType i = 0; i < nx; ++i) {
        for (SizeType j = 0; j < ny; ++j) {
            c[i + (nx * j)] = a[i + (nx * j)] * b[i + (nx * idm)] * scale;
        }
    }
}