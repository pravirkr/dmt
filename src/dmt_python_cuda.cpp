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
                      SizeType, SizeType, bool, std::string_view, bool, int,
                      SizeType>(),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("dt_max"),
             py::arg("dt_min") = 0, py::arg("dt_step") = 1,
             py::arg("use_box_smearing") = true, py::arg("mode") = "full",
             py::arg("verbose") = false, py::arg("device_id") = 0,
             py::arg("nbeams") = 1)
        .def(
            py::init([](float f_min, float f_max, SizeType nchans,
                        SizeType nsamps, float tsamp, const py::object& dt_grid,
                        const py::object& dt_arr, const py::object& dm_grid,
                        const py::object& dm_arr, bool use_box_smearing,
                        std::string_view mode, bool verbose, int device_id,
                        SizeType nbeams) {
                const auto [type, obj] =
                    resolve_custom_grid(dt_grid, dt_arr, dm_grid, dm_arr);
                if (type == CustomGridType::kDt) {
                    return FDMTCUDA(f_min, f_max, nchans, nsamps, tsamp,
                                    extract_dt_grid(obj), use_box_smearing,
                                    mode, verbose, device_id, nbeams);
                }
                return FDMTCUDA(f_min, f_max, nchans, nsamps, tsamp,
                                extract_dm_grid(obj), use_box_smearing, mode,
                                verbose, device_id, nbeams);
            }),
            py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
            py::arg("nsamps"), py::arg("tsamp"), py::kw_only(),
            py::arg("dt_grid") = py::none(), py::arg("dt_arr") = py::none(),
            py::arg("dm_grid") = py::none(), py::arg("dm_arr") = py::none(),
            py::arg("use_box_smearing") = true, py::arg("mode") = "full",
            py::arg("verbose") = false, py::arg("device_id") = 0,
            py::arg("nbeams") = 1)
        .def_property_readonly(
            "plan", &FDMTCUDA::get_plan,
            "Get the FDMTPlan object containing transform details.")
        .def_property_readonly("nbeams", &FDMTCUDA::get_nbeams)
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
        .def(
            "execute",
            [](FDMTCUDA& fdmt,
               const py::array_t<float, py::array::c_style>& waterfall)
                -> py::object {
                const auto nbeams   = fdmt.get_nbeams();
                const auto& plan    = fdmt.get_plan();
                const auto& plan_c  = plan.get_container();
                const auto niters   = plan.get_niters();
                const auto ncoords  = plan_c.state_shape[niters].ncoords;
                const auto nsamps   = plan_c.state_shape[niters].nsamps;
                const auto dmt_size = plan.get_dmt_size();
                const auto buf_size = plan.get_buffer_size();

                if (waterfall.ndim() == 2) {
                    if (nbeams != 1) {
                        throw std::invalid_argument(
                            std::format("FDMTCUDA: Invalid size of waterfall. "
                                        "Expected (nbeams={}, nchans, nsamps) "
                                        "but got 2D array.",
                                        nbeams));
                    }
                    py::array_t<float, py::array::c_style> dmt_buf(buf_size);
                    fdmt.execute(
                        std::span<const float>(waterfall.data(),
                                               waterfall.size()),
                        std::span<float>(dmt_buf.mutable_data(),
                                         dmt_buf.size()));
                    return py::array_t<float>(
                        {ncoords, nsamps},
                        {nsamps * sizeof(float), sizeof(float)}, dmt_buf.data(),
                        dmt_buf);
                }
                if (waterfall.ndim() == 3) {
                    if (static_cast<SizeType>(waterfall.shape(0)) != nbeams) {
                        throw std::invalid_argument(
                            std::format("FDMTCUDA: Invalid size of waterfall. "
                                        "Expected (nbeams={}, nchans, nsamps), "
                                        "got leading dim {}.",
                                        nbeams, waterfall.shape(0)));
                    }
                    std::vector<float> dmt_buf(nbeams * buf_size, 0.0F);
                    fdmt.execute(
                        std::span<const float>(waterfall.data(),
                                               waterfall.size()),
                        std::span<float>(dmt_buf.data(), dmt_buf.size()));

                    py::array_t<float, py::array::c_style> result(
                        {nbeams, ncoords, nsamps});
                    auto* res_ptr = result.mutable_data();
                    for (SizeType b = 0; b < nbeams; ++b) {
                        std::copy_n(dmt_buf.data() + (b * buf_size), dmt_size,
                                    res_ptr + (b * dmt_size));
                    }
                    return result;
                }
                throw std::runtime_error("Input waterfall must be a 2D "
                                         "(nchans, nsamps) or 3D (nbeams, "
                                         "nchans, nsamps) NumPy array.");
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
        .def_property_readonly("is_finished", &FDMTCUDA::is_finished)
        .def("reset_history", &FDMTCUDA::reset_history,
             "Reset the internal history buffers for valid-mode streaming.");

}

} // namespace dmt