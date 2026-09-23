#include "bindings/bind.hpp"

#include <algorithm>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dmt/dmt.hpp"
#include "pybind_utils.hpp"

namespace dmt {
using algorithms::FDMTCPU;
using algorithms::FDMTFFTCPU;
using plans::FDMTPlan;

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

void bind_fdmt(py::module_& mod) {
    py::class_<FDMTCPU>(mod, "FDMTCPU", py::dynamic_attr(),
                        R"doc(
        Incoherent Fast Dispersion Measure Transform on the CPU.

        Transforms a frequency–time waterfall into a DM/delay–time plane.
        ``mode='valid'`` keeps a streaming history so consecutive blocks
        join without a wrap artefact.

        Parameters
        ----------
        f_min, f_max : float
            Band edges in MHz.
        nchans, nsamps : int
            Channels and samples per block.
        tsamp : float
            Sampling interval in seconds.
        dt_max : int
            Maximum delay trial in samples (regular grid constructor).
        dt_min, dt_step : int, optional
            Delay-grid range and stride.
        use_box_smearing : bool, optional
            Account for intra-channel smearing in the tree.
        mode : {'valid', 'full', 'roll'}, optional
            Output time alignment.
        verbose : bool, optional
            Print the plan summary.
        nthreads : int, optional
            OpenMP threads (default 1).
        nbeams : int, optional
            Independent beams packed as ``(nbeams, nchans, nsamps)``.
        dt_grid, dt_arr, dm_grid, dm_arr : array_like, optional
            Keyword-only custom trial grid. Provide exactly one of these.

        See also
        --------
        FDMTPlan, FDMTFFTCPU, compute_fdmt
        )doc")
        .def(py::init<float, float, SizeType, SizeType, float, IndexType,
                      IndexType, SizeType, bool, std::string_view, bool, int,
                      SizeType>(),
             "f_min"_a, "f_max"_a, "nchans"_a, "nsamps"_a, "tsamp"_a,
             "dt_max"_a, "dt_min"_a = 0, "dt_step"_a = 1,
             "use_box_smearing"_a = true, "mode"_a = "valid",
             "verbose"_a = false, "nthreads"_a = 1, "nbeams"_a = 1)
        .def(py::init([](float f_min, float f_max, SizeType nchans,
                         SizeType nsamps, float tsamp,
                         const py::object& dt_grid, const py::object& dt_arr,
                         const py::object& dm_grid, const py::object& dm_arr,
                         bool use_box_smearing, std::string_view mode,
                         bool verbose, int nthreads, SizeType nbeams) {
                 const auto [type, obj] =
                     resolve_custom_grid(dt_grid, dt_arr, dm_grid, dm_arr);
                 if (type == CustomGridType::kDt) {
                     return FDMTCPU(f_min, f_max, nchans, nsamps, tsamp,
                                    extract_dt_grid(obj), use_box_smearing,
                                    mode, verbose, nthreads, nbeams);
                 }
                 return FDMTCPU(f_min, f_max, nchans, nsamps, tsamp,
                                extract_dm_grid(obj), use_box_smearing, mode,
                                verbose, nthreads, nbeams);
             }),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::kw_only(),
             py::arg("dt_grid") = py::none(), py::arg("dt_arr") = py::none(),
             py::arg("dm_grid") = py::none(), py::arg("dm_arr") = py::none(),
             py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
             py::arg("verbose") = false, py::arg("nthreads") = 1,
             py::arg("nbeams") = 1)
        .def_property_readonly(
            "plan", &FDMTCPU::get_plan,
            "Get the FDMTPlan object containing transform details.")
        .def_property_readonly(
            "nbeams", &FDMTCPU::get_nbeams,
            "Number of beams this instance processes together (see the "
            "nbeams constructor argument). Supports 2D arrays (nchans, nsamps) "
            "when nbeams=1, and 3D arrays (nbeams, nchans, nsamps) when "
            "nbeams>=1.")
        .def_property_readonly("dt_grid_final",
                               [](FDMTCPU& fdmt) {
                                   return as_pyarray(
                                       fdmt.get_plan().get_dt_grid_final());
                               })
        .def_property_readonly("dm_grid_final",
                               [](FDMTCPU& fdmt) {
                                   return as_pyarray(
                                       fdmt.get_plan().get_dm_grid_final());
                               })
        .def("get_dt_grid_final",
             [](FDMTCPU& fdmt) {
                 return as_pyarray(fdmt.get_plan().get_dt_grid_final());
             })
        .def("get_dm_grid_final",
             [](FDMTCPU& fdmt) {
                 return as_pyarray(fdmt.get_plan().get_dm_grid_final());
             })
        .def("get_effective_variance", &FDMTCPU::get_effective_variance,
             py::arg("dm_idx"), py::arg("boxcar_width") = 1,
             R"doc(
             Compute theoretical noise variance for a DM trial and boxcar width.

             Parameters
             ----------
             dm_idx : int
                 Index into the final DM/dt trial grid.
             boxcar_width : int, optional
                 Width of subsequent boxcar filter in samples (default 1).

             Returns
             -------
             float
                 Predicted noise variance scaled from input noise.
             )doc")
        .def("get_effective_sigma", &FDMTCPU::get_effective_sigma,
             py::arg("dm_idx"), py::arg("boxcar_width") = 1,
             R"doc(
             Compute theoretical noise standard deviation (sigma) for a DM trial and boxcar width.

             Parameters
             ----------
             dm_idx : int
                 Index into the final DM/dt trial grid.
             boxcar_width : int, optional
                 Width of subsequent boxcar filter in samples (default 1).

             Returns
             -------
             float
                 Predicted noise standard deviation (sqrt(variance)).
             )doc")
        .def(
            "get_effective_variance_grid",
            [](const FDMTCPU& fdmt, SizeType boxcar_width) {
                return as_pyarray(
                    fdmt.get_effective_variance_grid(boxcar_width));
            },
            py::arg("boxcar_width") = 1,
            R"doc(
            Compute theoretical noise variance for all DM trials across the grid.

            Parameters
            ----------
            boxcar_width : int, optional
                Width of subsequent boxcar filter in samples (default 1).

            Returns
            -------
            numpy.ndarray
                1D float32 array of shape ``(n_dm,)`` of theoretical variances.
            )doc")
        .def(
            "get_effective_sigma_grid",
            [](const FDMTCPU& fdmt, SizeType boxcar_width) {
                return as_pyarray(fdmt.get_effective_sigma_grid(boxcar_width));
            },
            py::arg("boxcar_width") = 1,
            R"doc(
            Compute theoretical noise standard deviation (sigma) for all DM trials.

            Parameters
            ----------
            boxcar_width : int, optional
                Width of subsequent boxcar filter in samples (default 1).

            Returns
            -------
            numpy.ndarray
                1D float32 array of shape ``(n_dm,)`` of theoretical standard deviations.
            )doc")
        // execute takes 2d or 3d array as input, and returns 2d or 3d array as
        // output
        .def(
            "execute",
            [](FDMTCPU& fdmt,
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
                        throw std::invalid_argument(std::format(
                            "FDMTCPU: Invalid size of waterfall. Expected {}, "
                            "got {}",
                            nbeams * plan.get_nchans() * plan.get_nsamps(),
                            waterfall.size()));
                    }
                    py::array_t<float, py::array::c_style> dmt_buf(buf_size);
                    fdmt.execute(std::span<const float>(waterfall.data(),
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
                        throw std::invalid_argument(std::format(
                            "FDMTCPU: Invalid size of waterfall. Expected {}, "
                            "got {}",
                            nbeams * plan.get_nchans() * plan.get_nsamps(),
                            waterfall.size()));
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
            Run the FDMT transform.

            Parameters
            ----------
            waterfall : numpy.ndarray, dtype float32
                C-contiguous array of shape ``(nchans, nsamps)`` for
                ``nbeams=1``, or ``(nbeams, nchans, nsamps)``.

            Returns
            -------
            numpy.ndarray
                ``(n_delays, n_times)`` or ``(nbeams, n_delays, n_times)``.
                ``n_times`` follows ``plan.dmt_nsamps`` for the chosen mode.

            Notes
            -----
            In ``valid`` mode, successive calls keep inter-block history.
            Call :meth:`reset_history` to start a new stream.
            )doc")
        .def(
            "reset",
            [](py::object self,
               const py::array_t<float, py::array::c_style>& waterfall,
               std::optional<py::array_t<float, py::array::c_style>> dmt_opt) {
                auto& fdmt = self.cast<FDMTCPU&>();
                if (waterfall.ndim() != 2) {
                    throw std::runtime_error("Input waterfall must be a 2D "
                                             "NumPy array (nchans, nsamps).");
                }
                py::array_t<float, py::array::c_style> dmt;
                if (dmt_opt.has_value()) {
                    dmt = *dmt_opt;
                } else {
                    dmt = py::array_t<float, py::array::c_style>(
                        fdmt.get_plan().get_buffer_size());
                }
                self.attr("_waterfall_buffer") = waterfall;
                self.attr("_dmt_buffer")       = dmt;
                fdmt.reset(
                    std::span<const float>(waterfall.data(), waterfall.size()),
                    std::span<float>(dmt.mutable_data(), dmt.size()));
            },
            py::arg("waterfall"), py::arg("dmt") = py::none(),
            "Reset and initialize the stepper with waterfall and optional dmt "
            "buffer.")
        .def("advance", &FDMTCPU::advance, py::arg("levels") = 1,
             "Advance execution by a given number of levels.")
        .def("advance_until_remaining", &FDMTCPU::advance_until_remaining,
             py::arg("remaining_levels"),
             "Advance execution until N levels remain before root.")
        .def("view_level_data",
             [](py::object self) {
                 auto& fdmt = self.cast<FDMTCPU&>();
                 auto span  = fdmt.view_level_data();
                 return py::array_t<float>(span.size(), span.data(), self);
             })
        .def(
            "view_subband_data",
            [](py::object self, SizeType subband_idx) {
                auto& fdmt    = self.cast<FDMTCPU&>();
                auto sub_view = fdmt.view_subband(subband_idx);
                return py::array_t<float>(
                    {sub_view.ndt, sub_view.nsamps},
                    {sub_view.nsamps * sizeof(float), sizeof(float)},
                    sub_view.data.data(), self);
            },
            py::arg("subband_idx"))
        .def(
            "view_subband",
            [](py::object self, SizeType subband_idx) {
                auto& fdmt = self.cast<FDMTCPU&>();
                return fdmt.view_subband(subband_idx);
            },
            py::arg("subband_idx"))
        .def("finalize",
             [](py::object self) {
                 auto& fdmt = self.cast<FDMTCPU&>();
                 fdmt.finalize();
                 py::array_t<float, py::array::c_style> dmt =
                     self.attr("_dmt_buffer")
                         .cast<py::array_t<float, py::array::c_style>>();
                 const auto& plan   = fdmt.get_plan();
                 const auto& plan_c = plan.get_container();
                 const auto niters  = plan.get_niters();
                 const auto ncoords = plan_c.state_shape[niters].ncoords;
                 const auto nsamps  = plan_c.state_shape[niters].nsamps;
                 return py::array_t<float>(
                     {ncoords, nsamps}, {nsamps * sizeof(float), sizeof(float)},
                     dmt.data(), self);
             })
        .def_property_readonly("current_level", &FDMTCPU::current_level)
        .def_property_readonly("total_levels", &FDMTCPU::total_levels)
        .def_property_readonly("remaining_levels", &FDMTCPU::remaining_levels)
        .def_property_readonly("num_subbands", &FDMTCPU::num_subbands)
        .def_property_readonly("is_finished", &FDMTCPU::is_finished)
        .def("reset_history", &FDMTCPU::reset_history,
             "Reset the internal history buffers for valid-mode streaming.")
        .def("history_state_size", &FDMTCPU::history_state_size,
             R"doc(
             Size (in float32 elements) of this instance's streaming history state.
             Zero for "full" or "roll" modes.
             )doc")
        .def("save_history", [](const FDMTCPU& fdmt) -> py::object {
            const auto sz = fdmt.history_state_size();
            py::array_t<float, py::array::c_style> hist(sz);
            fdmt.save_history(std::span<float>(hist.mutable_data(), hist.size()));
            return hist;
        },
        R"doc(
        Save the internal streaming history state into a 1D float32 NumPy array.
        Enables time-multiplexing multiple beams or sub-streams on a single FDMTCPU instance.
        )doc")
        .def("load_history", [](FDMTCPU& fdmt, const py::array& hist_obj) {
            if (hist_obj.dtype().kind() != 'f' || hist_obj.itemsize() != 4) {
                throw std::invalid_argument(
                    "FDMTCPU.load_history: expected a float32 array");
            }
            auto hist = hist_obj.cast<py::array_t<float, py::array::c_style>>();
            if (static_cast<SizeType>(hist.size()) != fdmt.history_state_size()) {
                throw std::invalid_argument(std::format(
                    "FDMTCPU.load_history: expected buffer of size {}, got {}",
                    fdmt.history_state_size(), hist.size()));
            }
            fdmt.load_history(std::span<const float>(hist.data(), hist.size()));
        },
        py::arg("history"),
        R"doc(
        Restore previously saved streaming history state from a 1D float32 NumPy array.
        )doc");

    py::class_<FDMTFFTCPU>(mod, "FDMTFFTCPU", py::dynamic_attr(),
                           R"doc(
        FFT-domain FDMT on the CPU.

        Same interface as :class:`FDMTCPU`, but delay shifts are applied
        with FFTs. Numerically close to ``mode='roll'`` on :class:`FDMTCPU`.

        Parameters
        ----------
        f_min, f_max : float
            Band edges in MHz.
        nchans, nsamps : int
            Channels and samples per block.
        tsamp : float
            Sampling interval in seconds.
        dt_max : int
            Maximum delay trial in samples (regular grid constructor).
        dt_min, dt_step : int, optional
            Delay-grid range and stride.
        use_box_smearing : bool, optional
            Account for intra-channel smearing.
        mode : {'valid', 'full', 'roll'}, optional
            Output time alignment.
        verbose : bool, optional
            Print the plan summary.
        nthreads : int, optional
            OpenMP / FFTW threads.
        nbeams : int, optional
            Packed independent beams.
        dt_grid, dt_arr, dm_grid, dm_arr : array_like, optional
            Keyword-only custom trial grid.

        See also
        --------
        FDMTCPU, compute_fdmt_fft
        )doc")
        .def(py::init<float, float, SizeType, SizeType, float, IndexType,
                      IndexType, SizeType, bool, std::string_view, bool, int,
                      SizeType>(),
             "f_min"_a, "f_max"_a, "nchans"_a, "nsamps"_a, "tsamp"_a,
             "dt_max"_a, "dt_min"_a = 0, "dt_step"_a = 1,
             "use_box_smearing"_a = true, "mode"_a = "valid",
             "verbose"_a = false, "nthreads"_a = 1, "nbeams"_a = 1)
        .def(py::init([](float f_min, float f_max, SizeType nchans,
                         SizeType nsamps, float tsamp,
                         const py::object& dt_grid, const py::object& dt_arr,
                         const py::object& dm_grid, const py::object& dm_arr,
                         bool use_box_smearing, std::string_view mode,
                         bool verbose, int nthreads, SizeType nbeams) {
                 const auto [type, obj] =
                     resolve_custom_grid(dt_grid, dt_arr, dm_grid, dm_arr);
                 if (type == CustomGridType::kDt) {
                     return std::make_unique<FDMTFFTCPU>(
                         f_min, f_max, nchans, nsamps, tsamp,
                         extract_dt_grid(obj), use_box_smearing, mode, verbose,
                         nthreads, nbeams);
                 }
                 return std::make_unique<FDMTFFTCPU>(
                     f_min, f_max, nchans, nsamps, tsamp, extract_dm_grid(obj),
                     use_box_smearing, mode, verbose, nthreads, nbeams);
             }),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::kw_only(),
             py::arg("dt_grid") = py::none(), py::arg("dt_arr") = py::none(),
             py::arg("dm_grid") = py::none(), py::arg("dm_arr") = py::none(),
             py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
             py::arg("verbose") = false, py::arg("nthreads") = 1,
             py::arg("nbeams") = 1)
        .def_property_readonly("plan", &FDMTFFTCPU::get_plan)
        .def_property_readonly("nbeams", &FDMTFFTCPU::get_nbeams)
        .def_property_readonly("dt_grid_final",
                               [](FDMTFFTCPU& fdmt) {
                                   return as_pyarray(
                                       fdmt.get_plan().get_dt_grid_final());
                               })
        .def_property_readonly("dm_grid_final",
                               [](FDMTFFTCPU& fdmt) {
                                   return as_pyarray(
                                       fdmt.get_plan().get_dm_grid_final());
                               })
        .def("get_dt_grid_final",
             [](FDMTFFTCPU& fdmt) {
                 return as_pyarray(fdmt.get_plan().get_dt_grid_final());
             })
        .def("get_dm_grid_final",
             [](FDMTFFTCPU& fdmt) {
                 return as_pyarray(fdmt.get_plan().get_dm_grid_final());
             })
        .def("get_effective_variance", &FDMTFFTCPU::get_effective_variance,
             py::arg("dm_idx"), py::arg("boxcar_width") = 1)
        .def("get_effective_sigma", &FDMTFFTCPU::get_effective_sigma,
             py::arg("dm_idx"), py::arg("boxcar_width") = 1)
        .def(
            "get_effective_variance_grid",
            [](const FDMTFFTCPU& fdmt, SizeType boxcar_width) {
                return as_pyarray(
                    fdmt.get_effective_variance_grid(boxcar_width));
            },
            py::arg("boxcar_width") = 1)
        .def(
            "get_effective_sigma_grid",
            [](const FDMTFFTCPU& fdmt, SizeType boxcar_width) {
                return as_pyarray(fdmt.get_effective_sigma_grid(boxcar_width));
            },
            py::arg("boxcar_width") = 1)
        .def(
            "execute",
            [](FDMTFFTCPU& fdmt,
               const py::array_t<float, py::array::c_style>& waterfall)
                -> py::object {
                const auto nbeams     = fdmt.get_nbeams();
                const auto& plan      = fdmt.get_plan();
                const auto ndms       = plan.get_dmt_ndms();
                const auto nsamps_out = plan.get_dmt_nsamps();

                if (waterfall.ndim() == 2) {
                    if (nbeams != 1) {
                        throw std::invalid_argument(std::format(
                            "FDMTFFTCPU: Invalid size of waterfall. Expected "
                            "{}, "
                            "got {}",
                            nbeams * plan.get_nchans() * plan.get_nsamps(),
                            waterfall.size()));
                    }
                    py::array_t<float, py::array::c_style> result(
                        {ndms, nsamps_out});
                    fdmt.execute(
                        std::span<const float>(waterfall.data(),
                                               waterfall.size()),
                        std::span<float>(result.mutable_data(), result.size()));
                    return result;
                }
                if (waterfall.ndim() == 3) {
                    if (static_cast<SizeType>(waterfall.shape(0)) != nbeams) {
                        throw std::invalid_argument(std::format(
                            "FDMTFFTCPU: Invalid size of waterfall. Expected "
                            "{}, "
                            "got {}",
                            nbeams * plan.get_nchans() * plan.get_nsamps(),
                            waterfall.size()));
                    }
                    py::array_t<float, py::array::c_style> result(
                        {nbeams, ndms, nsamps_out});
                    fdmt.execute(
                        std::span<const float>(waterfall.data(),
                                               waterfall.size()),
                        std::span<float>(result.mutable_data(), result.size()));
                    return result;
                }
                throw std::runtime_error("Input waterfall must be a 2D "
                                         "(nchans, nsamps) or 3D (nbeams, "
                                         "nchans, nsamps) NumPy array.");
            },
            py::arg("waterfall"),
            R"doc(
            Run the FFT-domain FDMT transform.

            Parameters
            ----------
            waterfall : numpy.ndarray, dtype float32
                C-contiguous ``(nchans, nsamps)`` or
                ``(nbeams, nchans, nsamps)``.

            Returns
            -------
            numpy.ndarray
                ``(n_delays, n_times)`` or ``(nbeams, n_delays, n_times)``.
            )doc")
        .def(
            "reset",
            [](py::object self,
               const py::array_t<float, py::array::c_style>& waterfall,
               std::optional<py::array_t<float, py::array::c_style>> dmt_opt) {
                auto& fdmt = self.cast<FDMTFFTCPU&>();
                py::array_t<float, py::array::c_style> dmt;
                if (dmt_opt.has_value()) {
                    dmt = *dmt_opt;
                } else {
                    const auto nbeams = fdmt.get_nbeams();
                    dmt               = py::array_t<float, py::array::c_style>(
                        nbeams * fdmt.get_plan().get_dmt_size());
                }
                self.attr("_waterfall_buffer") = waterfall;
                self.attr("_dmt_buffer")       = dmt;
                fdmt.reset(
                    std::span<const float>(waterfall.data(), waterfall.size()),
                    std::span<float>(dmt.mutable_data(), dmt.size()));
            },
            py::arg("waterfall"), py::arg("dmt") = py::none())
        .def("advance", &FDMTFFTCPU::advance, py::arg("levels") = 1)
        .def("advance_until_remaining", &FDMTFFTCPU::advance_until_remaining,
             py::arg("remaining_levels"))
        .def("view_level_data",
             [](py::object self) {
                 auto& fdmt = self.cast<FDMTFFTCPU&>();
                 auto span  = fdmt.view_level_data();
                 return py::array_t<float>(span.size(), span.data(), self);
             })
        .def(
            "view_subband_data",
            [](py::object self, SizeType subband_idx) {
                auto& fdmt    = self.cast<FDMTFFTCPU&>();
                auto sub_view = fdmt.view_subband(subband_idx);
                return py::array_t<float>(
                    {sub_view.ndt, sub_view.nsamps},
                    {sub_view.nsamps * sizeof(float), sizeof(float)},
                    sub_view.data.data(), self);
            },
            py::arg("subband_idx"))
        .def(
            "view_subband",
            [](py::object self, SizeType subband_idx) {
                auto& fdmt = self.cast<FDMTFFTCPU&>();
                return fdmt.view_subband(subband_idx);
            },
            py::arg("subband_idx"))
        .def("finalize",
             [](py::object self) {
                 auto& fdmt = self.cast<FDMTFFTCPU&>();
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
        .def_property_readonly("current_level", &FDMTFFTCPU::current_level)
        .def_property_readonly("total_levels", &FDMTFFTCPU::total_levels)
        .def_property_readonly("remaining_levels",
                               &FDMTFFTCPU::remaining_levels)
        .def_property_readonly("num_subbands", &FDMTFFTCPU::num_subbands)
        .def_property_readonly("is_finished", &FDMTFFTCPU::is_finished)
        .def("reset_history", &FDMTFFTCPU::reset_history);

    mod.def(
        "compute_fdmt_fft",
        [](const py::array_t<float, py::array::c_style>& waterfall, float f_min,
           float f_max, SizeType nchans, SizeType nsamps, float tsamp,
           IndexType dt_max, IndexType dt_min, SizeType dt_step,
           bool use_box_smearing, std::string_view mode, bool verbose,
           int nthreads, SizeType nbeams) {
            auto [dmt, fdmt_plan] = algorithms::compute_fdmt_fft(
                std::span<const float>(waterfall.data(), waterfall.size()),
                f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, dt_step,
                use_box_smearing, mode, verbose, nthreads, nbeams);
            return std::make_tuple(as_pyarray(std::move(dmt)), fdmt_plan);
        },
        py::arg("waterfall"), py::arg("f_min"), py::arg("f_max"),
        py::arg("nchans"), py::arg("nsamps"), py::arg("tsamp"),
        py::arg("dt_max"), py::arg("dt_min") = 0, py::arg("dt_step") = 1,
        py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
        py::arg("verbose") = false, py::arg("nthreads") = 1,
        py::arg("nbeams") = 1,
        R"doc(
        One-shot FFT-domain FDMT.

        Runs the FFT-domain FDMT on an input waterfall array of shape (nchans, nsamps),
        returning a tuple (dmt_matrix, plan).
        )doc");

    mod.def(
        "compute_fdmt_fft",
        [](const py::array_t<float, py::array::c_style>& waterfall, float f_min,
           float f_max, SizeType nchans, SizeType nsamps, float tsamp,
           const py::object& dt_grid, const py::object& dt_arr,
           const py::object& dm_grid, const py::object& dm_arr,
           bool use_box_smearing, std::string_view mode, bool verbose,
           int nthreads, SizeType nbeams) {
            const auto [type, obj] =
                resolve_custom_grid(dt_grid, dt_arr, dm_grid, dm_arr);
            if (type == CustomGridType::kDt) {
                auto [dmt, fdmt_plan] = algorithms::compute_fdmt_fft(
                    std::span<const float>(waterfall.data(), waterfall.size()),
                    f_min, f_max, nchans, nsamps, tsamp, extract_dt_grid(obj),
                    use_box_smearing, mode, verbose, nthreads, nbeams);
                return std::make_tuple(as_pyarray(std::move(dmt)), fdmt_plan);
            }
            auto [dmt, fdmt_plan] = algorithms::compute_fdmt_fft(
                std::span<const float>(waterfall.data(), waterfall.size()),
                f_min, f_max, nchans, nsamps, tsamp, extract_dm_grid(obj),
                use_box_smearing, mode, verbose, nthreads, nbeams);
            return std::make_tuple(as_pyarray(std::move(dmt)), fdmt_plan);
        },
        py::arg("waterfall"), py::arg("f_min"), py::arg("f_max"),
        py::arg("nchans"), py::arg("nsamps"), py::arg("tsamp"), py::kw_only(),
        py::arg("dt_grid") = py::none(), py::arg("dt_arr") = py::none(),
        py::arg("dm_grid") = py::none(), py::arg("dm_arr") = py::none(),
        py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
        py::arg("verbose") = false, py::arg("nthreads") = 1,
        py::arg("nbeams") = 1);

    mod.def(

        "compute_fdmt",
        [](const py::array_t<float, py::array::c_style>& waterfall, float f_min,
           float f_max, SizeType nchans, SizeType nsamps, float tsamp,
           IndexType dt_max, IndexType dt_min, SizeType dt_step,
           bool use_box_smearing, std::string_view mode, bool verbose,
           int nthreads, SizeType nbeams) {
            auto [dmt, fdmt_plan] = algorithms::compute_fdmt(
                std::span<const float>(waterfall.data(), waterfall.size()),
                f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, dt_step,
                use_box_smearing, mode, verbose, nthreads, nbeams);
            return std::make_tuple(as_pyarray(std::move(dmt)), fdmt_plan);
        },
        py::arg("waterfall"), py::arg("f_min"), py::arg("f_max"),
        py::arg("nchans"), py::arg("nsamps"), py::arg("tsamp"),
        py::arg("dt_max"), py::arg("dt_min") = 0, py::arg("dt_step") = 1,
        py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
        py::arg("verbose") = false, py::arg("nthreads") = 1,
        py::arg("nbeams") = 1,
        R"doc(
        One-shot incoherent FDMT.

        Runs the time-domain FDMT on an input waterfall array of shape (nchans, nsamps),
        returning a tuple (dmt_matrix, plan).
        )doc");

    mod.def(
        "compute_fdmt",
        [](const py::array_t<float, py::array::c_style>& waterfall, float f_min,
           float f_max, SizeType nchans, SizeType nsamps, float tsamp,
           const py::object& dt_grid, const py::object& dt_arr,
           const py::object& dm_grid, const py::object& dm_arr,
           bool use_box_smearing, std::string_view mode, bool verbose,
           int nthreads, SizeType nbeams) {
            const auto [type, obj] =
                resolve_custom_grid(dt_grid, dt_arr, dm_grid, dm_arr);
            if (type == CustomGridType::kDt) {
                auto [dmt, fdmt_plan] = algorithms::compute_fdmt(
                    std::span<const float>(waterfall.data(), waterfall.size()),
                    f_min, f_max, nchans, nsamps, tsamp, extract_dt_grid(obj),
                    use_box_smearing, mode, verbose, nthreads, nbeams);
                return std::make_tuple(as_pyarray(std::move(dmt)), fdmt_plan);
            }
            auto [dmt, fdmt_plan] = algorithms::compute_fdmt(
                std::span<const float>(waterfall.data(), waterfall.size()),
                f_min, f_max, nchans, nsamps, tsamp, extract_dm_grid(obj),
                use_box_smearing, mode, verbose, nthreads, nbeams);
            return std::make_tuple(as_pyarray(std::move(dmt)), fdmt_plan);
        },
        py::arg("waterfall"), py::arg("f_min"), py::arg("f_max"),
        py::arg("nchans"), py::arg("nsamps"), py::arg("tsamp"), py::kw_only(),
        py::arg("dt_grid") = py::none(), py::arg("dt_arr") = py::none(),
        py::arg("dm_grid") = py::none(), py::arg("dm_arr") = py::none(),
        py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
        py::arg("verbose") = false, py::arg("nthreads") = 1,
        py::arg("nbeams") = 1);

    mod.def(
        "add_frb_track",
        [](py::array_t<float, py::array::c_style>& waterfall,
           const FDMTPlan& plan, SizeType dm_idx, float amplitude,
           IndexType toffset, SizeType width) {
            algorithms::add_frb_track(
                std::span<float>(waterfall.mutable_data(), waterfall.size()),
                plan, dm_idx, amplitude, toffset, width);
        },
        py::arg("waterfall"), py::arg("plan"), py::arg("dm_idx"),
        py::arg("amplitude") = 1.0F, py::arg("toffset") = 0,
        py::arg("width") = 1,
        R"doc(
        Injects a synthetic dispersed pulse ("FRB track") into a waterfall
        array in place, so that it lands exactly on a given final DM/dt
        trial at a chosen time sample after running the FDMT transform.

        Parameters
        ----------
        waterfall : np.ndarray, shape (nchans, nsamps)
            Modified in place (amplitude is added, not overwritten).
        plan : FDMTPlan
            The plan whose trial grid and channel layout to target.
        dm_idx : int
            Index into the plan's final DM/dt trial grid.
        amplitude : float
            Amplitude added per channel, per injected sample.
        toffset : int
            Reference-channel time sample at which the pulse should peak.
        width : int
            Number of consecutive samples per channel to inject (>=1).
        )doc");
}

} // namespace dmt
