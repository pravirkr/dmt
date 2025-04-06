#include <span>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include "dmt/fdmt.hpp"
#include "pybind_utils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(libcudmt, mod) { // NOLINT
    mod.doc() = "Python Bindings for dmt (CUDA Backend)";

    using dmt::FDMTCUDA;
    py::class_<FDMTCUDA>(mod, "FDMTCUDA", "FDMT CUDA Implementation Wrapper")
        .def(py::init<float, float, SizeType, SizeType, float, SizeType,
                      SizeType, SizeType, bool, bool, int>(),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("dt_max"),
             py::arg("dt_step") = 1, py::arg("dt_min") = 0,
             py::arg("use_history") = false, py::arg("verbose") = false,
             py::arg("device_id") = 0)
        .def_property_readonly(
            "plan", &FDMTCUDA::get_plan,
            "Get the FDMTPlan object containing transform details.")
        // execute take 2d array as input, and return 2d array as output
        .def(
            "execute",
            [](FDMTCUDA& fdmt,
               const py::array_t<float, py::array::c_style>& waterfall) {
                if (waterfall.ndim() != 2) {
                    throw std::runtime_error("Input waterfall must be a 2D "
                                             "NumPy array (nchans, nsamps).");
                }
                const auto& plan   = fdmt.get_plan();
                const auto& plan_c = plan.get_container();
                const auto niters  = plan.get_niters();
                py::array_t<float, py::array::c_style> dmt(
                    {plan_c.state_shape[niters].ncoords,
                     plan_c.state_shape[niters].nsamps});
                fdmt.execute(
                    std::span<const float>(waterfall.data(), waterfall.size()),
                    std::span<float>(dmt.mutable_data(), dmt.size()));
                return dmt;
            },
            py::arg("waterfall"),
            R"doc(
            Executes the FDMT transform on the GPU.

            This method handles copying data from the host (NumPy array) to the GPU,
            executing the transform, and copying the result back to the host.
            )doc");
}
