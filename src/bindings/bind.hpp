#pragma once

#include <pybind11/pybind11.h>

namespace dmt {

// CPU extension (libdmt). Each function registers one public surface.
void bind_plans(pybind11::module_& mod);
void bind_fdmt(pybind11::module_& mod);
void bind_cfdmt(pybind11::module_& mod);
void bind_ddmt(pybind11::module_& mod);

} // namespace dmt

