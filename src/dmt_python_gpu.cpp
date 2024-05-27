#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind_utils.hpp"
#include <dmt/fdmt_gpu.hpp>

namespace py = pybind11;

PYBIND11_MODULE(libcudmt, mod) {
    mod.doc() = "Python Bindings for dmt";
    py::class_<FDMTGPU>(mod, "FDMTGPU")
        .def(py::init<float, float, size_t, size_t, float, size_t, size_t,
                      size_t>(),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("dt_max"),
             py::arg("dt_step") = 1, py::arg("dt_min") = 0)
        .def_property_readonly("df", &FDMTGPU::get_df)
        .def_property_readonly("correction", &FDMTGPU::get_correction)
        .def_property_readonly("niters", &FDMTGPU::get_niters)
        .def_property_readonly("dt_grid_final",
                               [](FDMTGPU& fdmt) {
                                   return as_pyarray_ref(
                                       fdmt.get_dt_grid_final());
                               })
        .def_property_readonly(
            "dm_grid_final",
            [](FDMTGPU& fdmt) { return as_pyarray(fdmt.get_dm_grid_final()); })
        .def_static("set_log_level", &FDMTGPU::set_log_level, py::arg("level"))
        // execute take 2d array as input, and return 2d array as output
        .def("execute",
             [](FDMTGPU& fdmt,
                const py::array_t<float, py::array::c_style>& waterfall) {
                 const auto* shape = waterfall.shape();
                 const auto dt_final_size =
                     static_cast<ssize_t>(fdmt.get_dt_grid_final().size());
                 py::array_t<float, py::array::c_style> dmt(
                     {dt_final_size, shape[1]});
                 fdmt.execute(waterfall.data(), waterfall.size(),
                              dmt.mutable_data(), dmt.size());
                 return dmt;
             })
        .def("initialise",
             [](FDMTGPU& fdmt,
                const py::array_t<float, py::array::c_style>& waterfall) {
                 const auto* shape = waterfall.shape();
                 const auto& plan  = fdmt.get_plan();
                 const auto nchans_ndt =
                     static_cast<ssize_t>(plan.state_shape[0][3]);
                 py::array_t<float, py::array::c_style> state(
                     {nchans_ndt, shape[1]});
                 std::fill(state.mutable_data(),
                           state.mutable_data() + state.size(), 0.0F);
                 fdmt.initialise(waterfall.data(), waterfall.size(),
                                 state.mutable_data(), state.size());
                 return state;
             });
}
