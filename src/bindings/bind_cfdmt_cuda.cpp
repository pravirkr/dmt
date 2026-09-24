#include "bindings/bind_cuda.hpp"

#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include "dmt/dmt.hpp"
#include "pybind_utils.hpp"

namespace dmt {
using algorithms::CohFDMTCUDA;

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

void bind_cfdmt_cuda(py::module_& mod) {
    py::class_<CohFDMTCUDA>(
        mod, "CohFDMTCUDA",
        R"doc(
        Hybrid coherent FDMT on CUDA.

        Same constructor arguments as :class:`~dmtlib.libdmt.CohFDMTCPU`, with
        ``device_id`` instead of ``nthreads``.
        )doc")
        .def(py::init<float, float, SizeType, float, SizeType, SizeType, float,
                      float, float, SizeType, std::string_view, bool, int>(),
             "f_center"_a, "sub_bw"_a, "nsub"_a, "tbin"_a, "nbin"_a, "nfft"_a,
             "tp"_a, "dm_max"_a, "dm_min"_a = 0.0F, "noverlap"_a = 8192,
             "data_order"_a = "PRITF", "verbose"_a = false,
             "device_id"_a = 0)
        .def_property_readonly("plan", &CohFDMTCUDA::get_plan)
        .def("execute",
             [](CohFDMTCUDA& coh_fdmt,
                const py::array_t<uint8_t, py::array::c_style>& data_in) {
                 const auto& plan      = coh_fdmt.get_plan();
                 const auto ndm_total  = plan.get_ndm();
                 const auto nsamps_out = plan.get_dmt_nsamps();
                 py::array_t<float, py::array::c_style> dmt(
                     {static_cast<ssize_t>(ndm_total),
                      static_cast<ssize_t>(nsamps_out)});
                 coh_fdmt.execute(
                     std::span<const uint8_t>(data_in.data(), data_in.size()),
                     std::span<float>(dmt.mutable_data(), dmt.size()));
                 return dmt;
             })
        .def("execute",
             [](CohFDMTCUDA& coh_fdmt,
                const py::array_t<int8_t, py::array::c_style>& data_in) {
                 const auto& plan      = coh_fdmt.get_plan();
                 const auto ndm_total  = plan.get_ndm();
                 const auto nsamps_out = plan.get_dmt_nsamps();
                 py::array_t<float, py::array::c_style> dmt(
                     {static_cast<ssize_t>(ndm_total),
                      static_cast<ssize_t>(nsamps_out)});
                 coh_fdmt.execute(
                     std::span<const int8_t>(data_in.data(), data_in.size()),
                     std::span<float>(dmt.mutable_data(), dmt.size()));
                 return dmt;
             })
        .def("reset_history", &CohFDMTCUDA::reset_history);
}

} // namespace dmt
