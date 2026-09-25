#include "bindings/bind_cuda.hpp"

#include <algorithm>
#include <cstdint>
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
using algorithms::FDMTCUDA;
using algorithms::FDMTFFTCUDA;
using algorithms::kFDMTAutoFuse;

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

void bind_fdmt_cuda(py::module_& mod) {
    py::class_<FDMTCUDA>(mod, "FDMTCUDA",
                         R"doc(
        Incoherent Fast Dispersion Measure Transform on CUDA.

        Same constructor arguments as :class:`~dmtlib.libdmt.FDMTCPU`, with
        ``device_id`` instead of ``nthreads``. ``fuse_levels=None`` (default)
        picks the fused-kernel depth from the plan (shared-memory tile of at
        least 256 samples in 48 KiB); an explicit depth is reduced until it
        fits the device's shared memory.

        See also
        --------
        dmtlib.libdmt.FDMTCPU
        )doc")
        .def(py::init([](float f_min, float f_max, SizeType nchans,
                         SizeType nsamps, float tsamp, IndexType dt_max,
                         IndexType dt_min, SizeType dt_step,
                         bool use_box_smearing, std::string_view mode,
                         bool verbose, int device_id, SizeType nbeams,
                         std::optional<SizeType> fuse_levels, bool int_tree) {
                 return FDMTCUDA(f_min, f_max, nchans, nsamps, tsamp, dt_max,
                                 dt_min, dt_step, use_box_smearing, mode,
                                 verbose, device_id, nbeams,
                                 fuse_levels.value_or(kFDMTAutoFuse), int_tree);
             }),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("dt_max"),
             py::arg("dt_min") = 0, py::arg("dt_step") = 1,
             py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
             py::arg("verbose") = false, py::arg("device_id") = 0,
             py::arg("nbeams") = 1, py::arg("fuse_levels") = py::none(),
             py::arg("int_tree") = true)
        .def(py::init([](float f_min, float f_max, SizeType nchans,
                         SizeType nsamps, float tsamp,
                         const py::object& dt_grid, const py::object& dt_arr,
                         const py::object& dm_grid, const py::object& dm_arr,
                         bool use_box_smearing, std::string_view mode,
                         bool verbose, int device_id, SizeType nbeams,
                         std::optional<SizeType> fuse_levels, bool int_tree) {
                 const auto [type, obj] =
                     resolve_custom_grid(dt_grid, dt_arr, dm_grid, dm_arr);
                 const auto fuse = fuse_levels.value_or(kFDMTAutoFuse);
                 if (type == CustomGridType::kDt) {
                     return FDMTCUDA(f_min, f_max, nchans, nsamps, tsamp,
                                     extract_dt_grid(obj), use_box_smearing,
                                     mode, verbose, device_id, nbeams, fuse,
                                     int_tree);
                 }
                 return FDMTCUDA(f_min, f_max, nchans, nsamps, tsamp,
                                 extract_dm_grid(obj), use_box_smearing, mode,
                                 verbose, device_id, nbeams, fuse, int_tree);
             }),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::kw_only(),
             py::arg("dt_grid") = py::none(), py::arg("dt_arr") = py::none(),
             py::arg("dm_grid") = py::none(), py::arg("dm_arr") = py::none(),
             py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
             py::arg("verbose") = false, py::arg("device_id") = 0,
             py::arg("nbeams") = 1, py::arg("fuse_levels") = py::none(),
             py::arg("int_tree") = true)
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
            Run the FDMT transform on the GPU.

            Copies ``waterfall`` to the device, executes, and returns a host
            array of the same layout as :meth:`dmtlib.libdmt.FDMTCPU.execute`.
            )doc")
        .def(
            "execute",
            [](FDMTCUDA& fdmt, const py::array& waterfall_obj,
               SizeType nbits) -> py::object {
                if (waterfall_obj.dtype().kind() != 'u' ||
                    waterfall_obj.itemsize() != 1) {
                    throw py::type_error(
                        "FDMTCUDA.execute: nbits given, so waterfall must be "
                        "a packed uint8 array");
                }
                const auto packed =
                    py::array_t<uint8_t, py::array::c_style>::ensure(
                        waterfall_obj);
                if (!packed || (packed.ndim() != 2 && packed.ndim() != 3)) {
                    throw std::runtime_error(
                        "Packed waterfall must be a 2D (nchans, row_bytes) or "
                        "3D (nbeams, nchans, row_bytes) uint8 NumPy array.");
                }
                const auto nbeams   = fdmt.get_nbeams();
                const auto& plan    = fdmt.get_plan();
                const auto& plan_c  = plan.get_container();
                const auto niters   = plan.get_niters();
                const auto ncoords  = plan_c.state_shape[niters].ncoords;
                const auto nsamps   = plan_c.state_shape[niters].nsamps;
                const auto dmt_size = plan.get_dmt_size();
                const auto buf_size = plan.get_buffer_size();
                if (packed.ndim() == 2 && nbeams != 1) {
                    throw std::invalid_argument(std::format(
                        "FDMTCUDA: nbeams={} requires a 3D packed waterfall",
                        nbeams));
                }
                std::vector<float> dmt_buf(nbeams * buf_size, 0.0F);
                fdmt.execute(
                    std::span<const uint8_t>(packed.data(), packed.size()),
                    nbits, std::span<float>(dmt_buf.data(), dmt_buf.size()));
                if (packed.ndim() == 2) {
                    py::array_t<float, py::array::c_style> result(
                        {ncoords, nsamps});
                    std::copy_n(dmt_buf.data(), dmt_size,
                                result.mutable_data());
                    return result;
                }
                py::array_t<float, py::array::c_style> result(
                    {nbeams, ncoords, nsamps});
                auto* res_ptr = result.mutable_data();
                for (SizeType b = 0; b < nbeams; ++b) {
                    std::copy_n(dmt_buf.data() + (b * buf_size), dmt_size,
                                res_ptr + (b * dmt_size));
                }
                return result;
            },
            py::arg("waterfall_packed"), py::arg("nbits"),
            R"doc(
            Run the FDMT transform on packed low-bit input on the GPU. Only the
            packed bytes are copied to the device. Same layout and output as
            :meth:`dmtlib.libdmt.FDMTCPU.execute` with ``nbits``.
            )doc")
        .def_property_readonly(
            "fuse_levels", &FDMTCUDA::get_fuse_levels,
            "Fusion depth the device execute() uses (0 = unfused).")
        .def_property_readonly("int_tree", &FDMTCUDA::get_int_tree,
                               "Whether packed input uses the integer tree.")
        .def_property_readonly(
            "memory_usage", &FDMTCUDA::get_memory_usage,
            "FDMTMemoryUsage: device bytes allocated at construction.")
        .def_property_readonly("current_level", &FDMTCUDA::current_level)
        .def_property_readonly("total_levels", &FDMTCUDA::total_levels)
        .def_property_readonly("remaining_levels", &FDMTCUDA::remaining_levels)
        .def_property_readonly("num_subbands", &FDMTCUDA::num_subbands)
        .def_property_readonly("is_finished", &FDMTCUDA::is_finished)
        .def("reset_history", &FDMTCUDA::reset_history,
             "Reset the internal history buffers for valid-mode streaming.");

    py::class_<FDMTFFTCUDA>(
        mod, "FDMTFFTCUDA", py::dynamic_attr(),
        R"doc(
        FFT-domain FDMT on CUDA.

        Same constructor arguments as :class:`~dmtlib.libdmt.FDMTFFTCPU`, with
        ``device_id`` instead of ``nthreads``.
        )doc")
        .def(py::init<float, float, SizeType, SizeType, float, IndexType,
                      IndexType, SizeType, bool, std::string_view, bool, int,
                      SizeType>(),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("dt_max"),
             py::arg("dt_min") = 0, py::arg("dt_step") = 1,
             py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
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
                    return FDMTFFTCUDA(f_min, f_max, nchans, nsamps, tsamp,
                                       extract_dt_grid(obj), use_box_smearing,
                                       mode, verbose, device_id, nbeams);
                }
                return FDMTFFTCUDA(f_min, f_max, nchans, nsamps, tsamp,
                                   extract_dm_grid(obj), use_box_smearing, mode,
                                   verbose, device_id, nbeams);
            }),
            py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
            py::arg("nsamps"), py::arg("tsamp"), py::kw_only(),
            py::arg("dt_grid") = py::none(), py::arg("dt_arr") = py::none(),
            py::arg("dm_grid") = py::none(), py::arg("dm_arr") = py::none(),
            py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
            py::arg("verbose") = false, py::arg("device_id") = 0,
            py::arg("nbeams") = 1)
        .def_property_readonly("plan", &FDMTFFTCUDA::get_plan)
        .def_property_readonly("nbeams", &FDMTFFTCUDA::get_nbeams)
        .def_property_readonly("dt_grid_final",
                               [](FDMTFFTCUDA& fdmt) {
                                   return as_pyarray(
                                       fdmt.get_plan().get_dt_grid_final());
                               })
        .def_property_readonly("dm_grid_final",
                               [](FDMTFFTCUDA& fdmt) {
                                   return as_pyarray(
                                       fdmt.get_plan().get_dm_grid_final());
                               })
        .def(
            "execute",
            [](FDMTFFTCUDA& fdmt,
               const py::array_t<float, py::array::c_style>& waterfall)
                -> py::object {
                const auto nbeams     = fdmt.get_nbeams();
                const auto& plan      = fdmt.get_plan();
                const auto ndms       = plan.get_dmt_ndms();
                const auto nsamps_out = plan.get_dmt_nsamps();
                if (waterfall.ndim() == 2) {
                    py::array_t<float, py::array::c_style> result(
                        {ndms, nsamps_out});
                    fdmt.execute(std::span<const float>(waterfall.data(),
                                                        waterfall.size()),
                                 std::span<float>(result.mutable_data(),
                                                  result.size()));
                    return result;
                }
                if (waterfall.ndim() == 3) {
                    py::array_t<float, py::array::c_style> result(
                        {nbeams, ndms, nsamps_out});
                    fdmt.execute(
                        std::span<const float>(waterfall.data(),
                                               waterfall.size()),
                        std::span<float>(result.mutable_data(), result.size()));
                    return result;
                }
                throw std::runtime_error("Input waterfall must be 2D or 3D.");
            },
            py::arg("waterfall"))
        .def(
            "reset",
            [](py::object self,
               const py::array_t<float, py::array::c_style>& waterfall,
               std::optional<py::array_t<float, py::array::c_style>> dmt_opt) {
                auto& fdmt = self.cast<FDMTFFTCUDA&>();
                py::array_t<float, py::array::c_style> dmt;
                if (dmt_opt.has_value()) {
                    dmt = *dmt_opt;
                } else {
                    dmt = py::array_t<float, py::array::c_style>(
                        fdmt.get_nbeams() * fdmt.get_plan().get_dmt_size());
                }
                self.attr("_waterfall_buffer") = waterfall;
                self.attr("_dmt_buffer")       = dmt;
                fdmt.reset(
                    std::span<const float>(waterfall.data(), waterfall.size()),
                    std::span<float>(dmt.mutable_data(), dmt.size()));
            },
            py::arg("waterfall"), py::arg("dmt") = py::none())
        .def(
            "advance",
            [](FDMTFFTCUDA& fdmt, SizeType levels) { fdmt.advance(levels); },
            py::arg("levels") = 1)
        .def(
            "advance_until_remaining",
            [](FDMTFFTCUDA& fdmt, SizeType remaining) {
                fdmt.advance_until_remaining(remaining);
            },
            py::arg("remaining_levels"))
        .def("view_level_data",
             [](py::object self) {
                 auto& fdmt = self.cast<FDMTFFTCUDA&>();
                 auto span  = fdmt.view_level_data();
                 py::array_t<float, py::array::c_style> host(
                     static_cast<py::ssize_t>(span.size()));
                 cudaMemcpy(host.mutable_data(), span.data(),
                            span.size() * sizeof(float),
                            cudaMemcpyDeviceToHost);
                 self.attr("_view_level_host") = host;
                 return host;
             })
        .def(
            "view_subband_data",
            [](py::object self, SizeType subband_idx) {
                auto& fdmt = self.cast<FDMTFFTCUDA&>();
                auto v     = fdmt.view_subband(subband_idx);
                py::array_t<float, py::array::c_style> host(
                    {v.ndt, v.nsamps});
                cudaMemcpy(host.mutable_data(), v.data.data(),
                           v.data.size() * sizeof(float),
                           cudaMemcpyDeviceToHost);
                return host;
            },
            py::arg("subband_idx"))
        .def(
            "view_subband",
            [](py::object self, SizeType subband_idx) {
                auto& fdmt = self.cast<FDMTFFTCUDA&>();
                auto v     = fdmt.view_subband(subband_idx);
                py::array_t<float, py::array::c_style> host(
                    {v.ndt, v.nsamps});
                cudaMemcpy(host.mutable_data(), v.data.data(),
                           v.data.size() * sizeof(float),
                           cudaMemcpyDeviceToHost);
                py::dict out;
                out["data"]        = host;
                out["subband_idx"] = v.subband_idx;
                out["ndt"]         = v.ndt;
                out["nsamps"]      = v.nsamps;
                out["f_start"]     = v.f_start;
                out["f_end"]       = v.f_end;
                return out;
            },
            py::arg("subband_idx"))
        .def("finalize",
             [](py::object self) {
                 auto& fdmt = self.cast<FDMTFFTCUDA&>();
                 fdmt.finalize();
                 py::array_t<float, py::array::c_style> dmt =
                     self.attr("_dmt_buffer")
                         .cast<py::array_t<float, py::array::c_style>>();
                 const auto& plan   = fdmt.get_plan();
                 const auto ncoords = plan.get_dmt_ndms();
                 const auto nsamps  = plan.get_dmt_nsamps();
                 return py::array_t<float>(
                     {ncoords, nsamps}, {nsamps * sizeof(float), sizeof(float)},
                     dmt.data(), self);
             })
        .def("reset_history", &FDMTFFTCUDA::reset_history)
        .def("get_effective_variance", &FDMTFFTCUDA::get_effective_variance,
             py::arg("dm_idx"), py::arg("boxcar_width") = 1)
        .def("get_effective_sigma", &FDMTFFTCUDA::get_effective_sigma,
             py::arg("dm_idx"), py::arg("boxcar_width") = 1)
        .def(
            "get_effective_variance_grid",
            [](const FDMTFFTCUDA& fdmt, SizeType boxcar_width) {
                return as_pyarray(
                    fdmt.get_effective_variance_grid(boxcar_width));
            },
            py::arg("boxcar_width") = 1)
        .def(
            "get_effective_sigma_grid",
            [](const FDMTFFTCUDA& fdmt, SizeType boxcar_width) {
                return as_pyarray(fdmt.get_effective_sigma_grid(boxcar_width));
            },
            py::arg("boxcar_width") = 1)
        .def_property_readonly("current_level", &FDMTFFTCUDA::current_level)
        .def_property_readonly("total_levels", &FDMTFFTCUDA::total_levels)
        .def_property_readonly("remaining_levels",
                               &FDMTFFTCUDA::remaining_levels)
        .def_property_readonly("num_subbands", &FDMTFFTCUDA::num_subbands)
        .def_property_readonly("is_finished", &FDMTFFTCUDA::is_finished);
}

} // namespace dmt
