#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>

#include "bindings/bind.hpp"

namespace py = pybind11;

PYBIND11_MODULE(libdmt, mod) { // NOLINT
    mod.doc() = R"doc(
    CPU Python bindings for the Dispersion Measure Transform library.

    This extension module is imported as ``dmtlib.libdmt``. The public
    classes (``FDMTCPU``, ``FDMTFFTCPU``, ``CohFDMTCPU``, ``DDMTCPU`` and
    their plans) are also re-exported from the ``dmtlib`` package.

    See also
    --------
    dmtlib.libcudmt : CUDA counterparts, when the library is built with GPU support.
    )doc";

    py::add_ostream_redirect(mod, "ostream_redirect");
    dmt::bind_plans(mod);
    dmt::bind_fdmt(mod);
    dmt::bind_cfdmt(mod);
    dmt::bind_ddmt(mod);
}

