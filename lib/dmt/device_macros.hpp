#pragma once

// Internal host/device qualifier macros for code compiled with both a C++
// compiler and nvcc / hip-clang.
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP__)
#define DMT_HD __host__ __device__
#define DMT_D __device__
#define DMT_H __host__
#else
#define DMT_HD
#define DMT_D
#define DMT_H
#endif
