#pragma once

// Import common types and constants
#include "common/plans.hpp" // IWYU pragma: export
#include "common/types.hpp" // IWYU pragma: export
#include "utils/simulate.hpp" // IWYU pragma: export

// Include backend-specific headers for each algorithm
#include "fdmt/fdmt_cpu.hpp"  // IWYU pragma: export
#include "fdmt/fdmt_cuda.hpp" // IWYU pragma: export

#include "ddmt/ddmt_cpu.hpp"  // IWYU pragma: export
#include "ddmt/ddmt_cuda.hpp" // IWYU pragma: export

#include "cfdmt/cfdmt_cpu.hpp"  // IWYU pragma: export
#include "cfdmt/cfdmt_cuda.hpp" // IWYU pragma: export
