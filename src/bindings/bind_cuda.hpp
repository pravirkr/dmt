#pragma once

#include <pybind11/pybind11.h>

namespace dmt {

// CUDA extension (libcudmt).
void bind_fdmt_cuda(pybind11::module_& mod);
void bind_cfdmt_cuda(pybind11::module_& mod);
void bind_ddmt_cuda(pybind11::module_& mod);

} // namespace dmt

