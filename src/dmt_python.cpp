#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings/bind.hpp"
#include "pybind_utils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(libdmt, mod) { // NOLINT
    mod.doc() = R"doc(
    Python bindings for the Dispersion Measure Transform library.

    This extension module is imported as ``dmtlib.libdmt``. The public
    classes (``FDMT``, ``FDMTFFT``, ``CohFDMT``, ``DDMT`` and their plans)
    are also re-exported from the ``dmtlib`` package. Each class runs on the
    backend given by its ``backend=`` keyword; :func:`available_backends`
    lists the backends in this build.
    )doc";

    dmt::bind_logging(mod);
    mod.def(
        "available_backends",
        [] {
            std::vector<std::string> names;
            for (const auto b : dmt::available_backends()) {
                names.emplace_back(dmt::to_string(b));
            }
            return names;
        },
        R"doc(
        Backends compiled into this build, e.g. ``['cpu']`` or
        ``['cpu', 'cuda']``. Pass one as ``backend=`` to an algorithm class.
        )doc");
    dmt::bind_plans(mod);
    dmt::bind_fdmt(mod);
    dmt::bind_cfdmt(mod);
    dmt::bind_ddmt(mod);
}
