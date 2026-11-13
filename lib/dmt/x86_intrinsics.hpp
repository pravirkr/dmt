#pragma once

/**
 * @file x86_intrinsics.hpp
 * @brief `<immintrin.h>` for the x86 CPU kernels; include it instead of
 * `<immintrin.h>` directly.
 *
 * GCC 12's AVX-512 headers build `_mm512_undefined_*()` from a
 * self-initialised variable (`__m512i __Y = __Y;`), so every inlined
 * intrinsic that takes one (`_mm512_cvtepu8_epi64`, `_mm512_srli_epi64`, ...)
 * raises a false `-Wmaybe-uninitialized` at -O2 and above. The warning is
 * located in the system header's lines, so it is silenced there only: our
 * own code keeps the warning.
 * This has to be the first inclusion of `<immintrin.h>` in the translation
 * unit, hence one header used everywhere.
 */

#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <immintrin.h>
#pragma GCC diagnostic pop
#else
#include <immintrin.h>
#endif
#endif
