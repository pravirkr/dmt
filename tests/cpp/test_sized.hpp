#pragma once

// Tests with a full and a small size.
//
// DMT_SIZED_TEST_CASE(name, tags) { ... } registers the body twice:
//   * `name`, tagged [slow]: the full problem sizes;
//   * `name (small)`: small sizes that take the same paths and checks.
// The body sees `bool full` and picks its sizes with it. ctest labels the
// [slow] tests `slow`; the CI test presets skip them, so CI runs the small
// versions and a plain `ctest` runs both.

#include <catch2/catch_test_macros.hpp>

#define DMT_SIZED_CAT2(a, b) a##b
#define DMT_SIZED_CAT(a, b) DMT_SIZED_CAT2(a, b)

#define DMT_SIZED_TEST_CASE_IMPL(fn, name, tags)                               \
    static void fn(bool full);                                                 \
    TEST_CASE(name, tags "[slow]") { fn(true); }                               \
    TEST_CASE(name " (small)", tags) { fn(false); }                            \
    static void fn([[maybe_unused]] bool full)

#define DMT_SIZED_TEST_CASE(name, tags)                                        \
    DMT_SIZED_TEST_CASE_IMPL(DMT_SIZED_CAT(dmt_sized_test_, __LINE__), name,   \
                             tags)
