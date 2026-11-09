#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "dmt/common/build_info.hpp"

#include "bindings/bind.hpp"
#include "pybind_utils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(libdmt, mod) { // NOLINT
    mod.doc() = R"doc(
    Python bindings for the Dispersion Measure Transform library.

    This extension module is imported as ``dmtlib.libdmt``. The public
    classes (``FDMT``, ``FDMTFFT``, ``CohFDMT``, ``DDMT``, ``DDMTFFT`` and
    their plans)
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
    mod.def(
        "build_info",
        [] {
            const auto& b = dmt::build_info();
            py::dict d;
            d["version"]    = std::string(b.version);
            d["git_sha"]    = std::string(b.git_sha);
            d["git_dirty"]  = b.git_dirty;
            d["compiler"]   = std::string(b.compiler);
            d["cxx_flags"]  = std::string(b.cxx_flags);
            d["build_type"] = std::string(b.build_type);
            d["gpu"]        = std::string(b.gpu);
            d["fftw"]       = std::string(b.fftw);
            d["openmp"]     = std::string(b.openmp);
            return d;
        },
        R"doc(
        Identity of the compiled extension: version, git commit (``git_sha``,
        ``git_dirty``), compiler and flags, build type, GPU toolkit, FFTW and
        OpenMP versions. Store it with any result produced by this build.
        )doc");
    dmt::bind_plans(mod);
    dmt::bind_fdmt(mod);
    dmt::bind_fdmt_boxcar(mod);
    dmt::bind_cfdmt(mod);
    dmt::bind_ddmt(mod);
    dmt::bind_ddmt_fft(mod);
}
