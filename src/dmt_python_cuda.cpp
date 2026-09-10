#include <span>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda/std/span>
#include <cuda_runtime_api.h>

#include "dmt/dmt.hpp"
#include "pybind_utils.hpp"

namespace dmt {
using algorithms::FDMTCUDA;

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

PYBIND11_MODULE(libcudmt, mod) { // NOLINT
    mod.doc() = "Python Bindings for dmt (CUDA Backend)";

    py::class_<FDMTCUDA>(mod, "FDMTCUDA", "FDMT CUDA Implementation Wrapper")
        .def(py::init<float, float, SizeType, SizeType, float, SizeType,
                      SizeType, SizeType, bool, std::string_view, bool, int>(),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("dt_max"),
             py::arg("dt_min") = 0, py::arg("dt_step") = 1,
             py::arg("use_box_smearing") = true, py::arg("mode") = "full",
             py::arg("verbose") = false, py::arg("device_id") = 0)
        .def(
            py::init([](float f_min, float f_max, SizeType nchans,
                        SizeType nsamps, float tsamp, const py::object& dt_grid,
                        const py::object& dt_arr, const py::object& dm_grid,
                        const py::object& dm_arr, bool use_box_smearing,
                        std::string_view mode, bool verbose, int device_id) {
                const auto [type, obj] =
                    resolve_custom_grid(dt_grid, dt_arr, dm_grid, dm_arr);
                if (type == CustomGridType::kDt) {
                    return FDMTCUDA(f_min, f_max, nchans, nsamps, tsamp,
                                    extract_dt_grid(obj), use_box_smearing,
                                    mode, verbose, device_id);
                }
                return FDMTCUDA(f_min, f_max, nchans, nsamps, tsamp,
                                extract_dm_grid(obj), use_box_smearing, mode,
                                verbose, device_id);
            }),
            py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
            py::arg("nsamps"), py::arg("tsamp"), py::kw_only(),
            py::arg("dt_grid") = py::none(), py::arg("dt_arr") = py::none(),
            py::arg("dm_grid") = py::none(), py::arg("dm_arr") = py::none(),
            py::arg("use_box_smearing") = true, py::arg("mode") = "full",
            py::arg("verbose") = false, py::arg("device_id") = 0)
        .def_property_readonly(
            "plan", &FDMTCUDA::get_plan,
            "Get the FDMTPlan object containing transform details.")
        .def_property_readonly("dt_grid_final",
                               [](FDMTCUDA& fdmt) {
                                   return as_pyarray(
                                       fdmt.get_plan().get_dt_grid_final());
                               })
        .def_property_readonly("dm_grid_final",
                               [](FDMTCUDA& fdmt) {
                                   return as_pyarray(
                                       fdmt.get_plan().get_dm_grid_final());
                               })
        .def("get_dt_grid_final",
             [](FDMTCUDA& fdmt) {
                 return as_pyarray(fdmt.get_plan().get_dt_grid_final());
             })
        .def("get_dm_grid_final",
             [](FDMTCUDA& fdmt) {
                 return as_pyarray(fdmt.get_plan().get_dm_grid_final());
             })
        // execute takes 2d array as input, and returns 2d array as output
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
                const auto ncoords = plan_c.state_shape[niters].ncoords;
                const auto nsamps  = plan_c.state_shape[niters].nsamps;
                py::array_t<float, py::array::c_style> dmt_buf(
                    plan.get_buffer_size());
                fdmt.execute(
                    std::span<const float>(waterfall.data(), waterfall.size()),
                    std::span<float>(dmt_buf.mutable_data(), dmt_buf.size()));
                return py::array_t<float>(
                    {ncoords, nsamps}, {nsamps * sizeof(float), sizeof(float)},
                    dmt_buf.data(), dmt_buf);
            },
            py::arg("waterfall"),
            R"doc(
            Executes the FDMT transform on the GPU.

            This method handles copying data from the host (NumPy array) to the GPU,
            executing the transform, and copying the result back to the host.
            )doc")
        .def_property_readonly("current_level", &FDMTCUDA::current_level)
        .def_property_readonly("total_levels", &FDMTCUDA::total_levels)
        .def_property_readonly("remaining_levels", &FDMTCUDA::remaining_levels)
        .def_property_readonly("num_subbands", &FDMTCUDA::num_subbands)
        .def_property_readonly("is_finished", &FDMTCUDA::is_finished);
}

} // namespace dmt