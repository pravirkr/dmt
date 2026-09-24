#include <pybind11/pybind11.h>

#include "bindings/bind_cuda.hpp"

namespace py = pybind11;

PYBIND11_MODULE(libcudmt, mod) { // NOLINT
    mod.doc() = R"doc(
    CUDA Python bindings for the Dispersion Measure Transform library.

    This extension module is imported as ``dmtlib.libcudmt``.
    Classes: FDMTCUDA, FDMTFFTCUDA, CohFDMTCUDA, DDMTCUDA.

    See also
    --------
    dmtlib.libdmt : CPU counterparts.
    )doc";

    dmt::bind_fdmt_cuda(mod);
    dmt::bind_cfdmt_cuda(mod);
    dmt::bind_ddmt_cuda(mod);
}

