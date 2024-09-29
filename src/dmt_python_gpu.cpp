#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind_utils.hpp"

#include <dmt/fdmt/fdmt_cuda.hpp>

namespace py = pybind11;

PYBIND11_MODULE(libcudmt, mod) { // NOLINT
    mod.doc() = "Python Bindings for dmt";
    py::class_<FDMTCUDA>(mod, "FDMTGPU")
        .def(py::init<float, float, SizeType, SizeType, float, SizeType,
                      SizeType, SizeType, bool, int>(),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("dt_max"),
             py::arg("dt_step") = 1, py::arg("dt_min") = 0,
             py::arg("use_history") = false, py::arg("device_id") = 0)
        .def_static("set_log_level", &FDMTCUDA::set_log_level, py::arg("level"))
        // execute take 2d array as input, and return 2d array as output
        .def(
            "execute",
            [](FDMTCUDA& fdmt,
               const py::array_t<float, py::array::c_style>& waterfall) {
                const auto& plan   = fdmt.get_plan();
                const auto& plan_c = plan.get_container();
                const auto niters  = plan.get_niters();
                py::array_t<float, py::array::c_style> dmt(
                    {plan_c.state_shape[niters].ncoords,
                     plan_c.state_shape[niters].nsamps});
                fdmt.execute(waterfall.data(), waterfall.size(),
                             dmt.mutable_data(), dmt.size());
                return dmt;
            },
            py::arg("waterfall"))
        .def("initialise",
             [](FDMTCUDA& fdmt,
                const py::array_t<float, py::array::c_style>& waterfall) {
                 const auto& plan   = fdmt.get_plan();
                 const auto& plan_c = plan.get_container();
                 py::array_t<float, py::array::c_style> state(
                     {plan_c.state_shape[0].ncoords,
                      plan_c.state_shape[0].nsamps});
                 std::fill(state.mutable_data(),
                           state.mutable_data() + state.size(), 0.0F);
                 fdmt.initialise(waterfall.data(), waterfall.size(),
                                 state.mutable_data(), state.size());
                 return state;
             });
}
