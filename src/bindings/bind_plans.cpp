#include "bindings/bind.hpp"

#include <span>
#include <string_view>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dmt/dmt.hpp"
#include "pybind_utils.hpp"

namespace dmt {
using algorithms::FDMTSubbandView;
using plans::CohFDMTPlan;
using plans::DDMTPlan;
using plans::FDMTComplexity;
using plans::FDMTCoord;
using plans::FDMTCoordGrid;
using plans::FDMTPlan;
using plans::FDMTPlanContainer;
using plans::FDMTShape;

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

void bind_plans(py::module_& mod) {
    PYBIND11_NUMPY_DTYPE(FDMTShape, nchans, ndt_min, ndt_max, ncoords,
                         ncoords_sum, ncoords_copy, nsamps, nelements, dt_max);
    PYBIND11_NUMPY_DTYPE(FDMTCoord, i_sub, i_dt, nsamps, buf_offset,
                         i_coord_tail, i_coord_head, delay, tail_buf_offset,
                         tail_nsamps, head_buf_offset, head_nsamps,
                         hist_offset);
    py::class_<FDMTComplexity>(mod, "FDMTComplexity")
        .def_readonly("n_dt", &FDMTComplexity::n_dt)
        .def_readonly("n_chans", &FDMTComplexity::n_chans)
        .def_readonly("brute_force_ops", &FDMTComplexity::brute_force_ops)
        .def_readonly("total_tree_nodes", &FDMTComplexity::total_tree_nodes)
        .def_readonly("sum_additions", &FDMTComplexity::sum_additions)
        .def_readonly("copy_nodes", &FDMTComplexity::copy_nodes)
        .def_readonly("ops_ratio", &FDMTComplexity::ops_ratio)
        .def("to_string", &FDMTComplexity::to_string)
        .def("__repr__", &FDMTComplexity::to_string);
    py::class_<FDMTCoordGrid>(mod, "FDMTSubDTGrid")
        .def_readonly("dt_grid", &FDMTCoordGrid::dt_grid)
        .def_readonly("ndt", &FDMTCoordGrid::ndt)
        .def_readonly("grid_offset", &FDMTCoordGrid::coord_offset)
        .def_readonly("f_start", &FDMTCoordGrid::f_start)
        .def_readonly("f_end", &FDMTCoordGrid::f_end);
    py::class_<FDMTSubbandView>(mod, "FDMTSubbandView")
        .def_property_readonly(
            "data",
            [](const py::object& self) {
                const auto& v = self.cast<const FDMTSubbandView&>();
                return py::array_t<float>(
                    {v.ndt, v.nsamps},
                    {v.nsamps * sizeof(float), sizeof(float)}, v.data.data(),
                    self);
            })
        .def_readonly("subband_idx", &FDMTSubbandView::subband_idx)
        .def_readonly("ndt", &FDMTSubbandView::ndt)
        .def_readonly("nsamps", &FDMTSubbandView::nsamps)
        .def_readonly("f_start", &FDMTSubbandView::f_start)
        .def_readonly("f_end", &FDMTSubbandView::f_end)
        .def_property_readonly("dt_grid", [](const FDMTSubbandView& v) {
            return std::vector<IndexType>(v.dt_grid.begin(), v.dt_grid.end());
        });
    py::class_<FDMTPlanContainer>(mod, "FDMTPlanContainer")
        .def_readonly("df_top", &FDMTPlanContainer::df_top)
        .def_readonly("df_bot", &FDMTPlanContainer::df_bot)
        .def_property_readonly("state_shape",
                               [](const FDMTPlanContainer& plan_c) {
                                   return as_pyarray_ref(plan_c.state_shape);
                               })
        .def_property_readonly("coordinates",
                               [](const FDMTPlanContainer& plan_c) {
                                   return as_listof_pyarray(plan_c.coordinates);
                               })
        .def_property_readonly("coordinates_sum",
                               [](const FDMTPlanContainer& plan_c) {
                                   return as_listof_pyarray(
                                       plan_c.coordinates_sum);
                               })
        .def_property_readonly("coordinates_copy",
                               [](const FDMTPlanContainer& plan_c) {
                                   return as_listof_pyarray(
                                       plan_c.coordinates_copy);
                               })
        .def_readonly("dt_grids", &FDMTPlanContainer::grids)
        .def_property_readonly("dt_grid_sub_top",
                               [](const FDMTPlanContainer& plan_c) {
                                   return as_listof_pyarray(
                                       plan_c.dt_grid_sub_top);
                               })
        .def_property_readonly("memory_usage",
                               &FDMTPlanContainer::get_memory_usage);
    py::class_<FDMTPlan>(mod, "FDMTPlan",
                         R"doc(
        Plan for an incoherent Fast Dispersion Measure Transform (FDMT).

        Encodes the delay/DM trial grid, tree geometry, buffer sizes, and
        per-trial variance used by :class:`FDMTCPU` and :class:`FDMTFFTCPU`.

        Parameters
        ----------
        f_min, f_max : float
            Band edges in MHz.
        nchans, nsamps : int
            Number of frequency channels and time samples in each block.
        tsamp : float
            Sampling interval in seconds.
        dt_max, dt_min : int, optional
            Inclusive delay-trial range in samples. ``dt_min`` may be negative.
        dt_step : int, optional
            Spacing of the regular delay grid (default 1).
        mode : {'valid', 'full', 'roll'}, optional
            Output time alignment. ``valid`` is streaming-safe.
        verbose : bool, optional
            Print a plan summary during construction.
        dt_grid, dt_arr, dm_grid, dm_arr : array_like, optional
            Keyword-only custom trial grid. Provide exactly one of these.

        Notes
        -----
        Custom grids are keyword-only so a float DM array is not mistaken
        for ``dt_max``. Unsorted or duplicate trials are sorted and uniqued.
        )doc")
        .def(py::init<float, float, SizeType, SizeType, float, IndexType,
                      IndexType, SizeType, std::string_view, bool>(),
             "f_min"_a, "f_max"_a, "nchans"_a, "nsamps"_a, "tsamp"_a,
             "dt_max"_a, "dt_min"_a = 0, "dt_step"_a = 1, "mode"_a = "valid",
             "verbose"_a = false)
        .def(py::init([](float f_min, float f_max, SizeType nchans,
                         SizeType nsamps, float tsamp,
                         const py::object& dt_grid, const py::object& dt_arr,
                         const py::object& dm_grid, const py::object& dm_arr,
                         std::string_view mode, bool verbose) {
                 const auto [type, obj] =
                     resolve_custom_grid(dt_grid, dt_arr, dm_grid, dm_arr);
                 if (type == CustomGridType::kDt) {
                     return FDMTPlan(f_min, f_max, nchans, nsamps, tsamp,
                                     extract_dt_grid(obj), mode, verbose);
                 }
                 return FDMTPlan(f_min, f_max, nchans, nsamps, tsamp,
                                 extract_dm_grid(obj), mode, verbose);
             }),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::kw_only(),
             py::arg("dt_grid") = py::none(), py::arg("dt_arr") = py::none(),
             py::arg("dm_grid") = py::none(), py::arg("dm_arr") = py::none(),
             py::arg("mode") = "valid", py::arg("verbose") = false)
        .def_property_readonly("f_min", &FDMTPlan::get_f_min)
        .def_property_readonly("f_max", &FDMTPlan::get_f_max)
        .def_property_readonly("nchans", &FDMTPlan::get_nchans)
        .def_property_readonly("nsamps", &FDMTPlan::get_nsamps)
        .def_property_readonly("tsamp", &FDMTPlan::get_tsamp)
        .def_property_readonly("dt_max", &FDMTPlan::get_dt_max)
        .def_property_readonly("dt_min", &FDMTPlan::get_dt_min)
        .def_property_readonly("dt_step", &FDMTPlan::get_dt_step)
        .def_property_readonly("mode", &FDMTPlan::get_mode)
        .def_property_readonly("is_custom_grid", &FDMTPlan::is_custom_grid)
        .def_property_readonly("df", &FDMTPlan::get_df)
        .def_property_readonly("niters", &FDMTPlan::get_niters)
        .def_property_readonly("container", &FDMTPlan::get_container)
        .def_property_readonly(
            "dt_grid_final",
            [](FDMTPlan& plan) { return as_pyarray(plan.get_dt_grid_final()); })
        .def_property_readonly(
            "dm_grid_final",
            [](FDMTPlan& plan) { return as_pyarray(plan.get_dm_grid_final()); })
        .def(
            "get_dt_grid_final",
            [](FDMTPlan& plan) { return as_pyarray(plan.get_dt_grid_final()); })
        .def(
            "get_dm_grid_final",
            [](FDMTPlan& plan) { return as_pyarray(plan.get_dm_grid_final()); })
        .def_property_readonly("smearing_grid_final",
                               [](FDMTPlan& plan) {
                                   return as_pyarray(
                                       plan.get_smearing_grid_final());
                               })
        .def(
            "trace_dm",
            [](const FDMTPlan& plan, SizeType dm_idx) {
                return as_pyarray(plan.trace_dm(dm_idx));
            },
            py::arg("dm_idx"),
            R"doc(
            Per-channel time shifts (samples) for one DM/dt trial.

            Parameters
            ----------
            dm_idx : int
                Index into the final DM/dt trial grid.

            Returns
            -------
            numpy.ndarray
                Integer shifts of length ``nchans``, relative to the
                unshifted reference channel. Used by :func:`add_frb_track`.
            )doc")
        .def("get_effective_variance", &FDMTPlan::get_effective_variance,
             py::arg("dm_idx"), py::arg("boxcar_width") = 1,
             py::arg("use_box_smearing") = true,
             R"doc(
             Effective noise variance for one DM trial after a boxcar of width W.

             Parameters
             ----------
             dm_idx : int
                 Index into the final trial grid.
             boxcar_width : int, optional
                 Boxcar width in samples (default 1).
             use_box_smearing : bool, optional
                 Include intra-channel smearing in the variance.

             Returns
             -------
             float
                 Variance of the FDMT output at that trial, for unit-variance
                 input noise.
             )doc")
        .def("get_effective_sigma", &FDMTPlan::get_effective_sigma,
             py::arg("dm_idx"), py::arg("boxcar_width") = 1,
             py::arg("use_box_smearing") = true)
        .def(
            "get_effective_variance_grid",
            [](const FDMTPlan& plan, SizeType boxcar_width,
               bool use_box_smearing) {
                return as_pyarray(plan.get_effective_variance_grid(
                    boxcar_width, use_box_smearing));
            },
            py::arg("boxcar_width") = 1, py::arg("use_box_smearing") = true)
        .def(
            "get_effective_sigma_grid",
            [](const FDMTPlan& plan, SizeType boxcar_width,
               bool use_box_smearing) {
                return as_pyarray(plan.get_effective_sigma_grid(
                    boxcar_width, use_box_smearing));
            },
            py::arg("boxcar_width") = 1, py::arg("use_box_smearing") = true)
        .def_property_readonly("complexity", &FDMTPlan::get_complexity)
        .def_property_readonly("dmt_ndms", &FDMTPlan::get_dmt_ndms)
        .def_property_readonly("dmt_nsamps", &FDMTPlan::get_dmt_nsamps)
        .def_property_readonly("dmt_size", &FDMTPlan::get_dmt_size)
        .def_property_readonly("buffer_size", &FDMTPlan::get_buffer_size)
        .def_property_readonly("history_size", &FDMTPlan::get_history_size)
        .def_property_readonly("history_init_size",
                               &FDMTPlan::get_history_init_size)
        .def_property_readonly("tree_history_size",
                               &FDMTPlan::get_tree_history_size)
        .def(
            "print_summary",
            [](FDMTPlan& plan, std::string_view prefix) {
                plan.print_summary(prefix);
            },
            "prefix"_a = "")
        .def("print_complexity_summary", &FDMTPlan::print_complexity_summary);
    py::class_<CohFDMTPlan>(mod, "CohFDMTPlan",
                            R"doc(
        Plan for the hybrid coherent Fast Dispersion Measure Transform.

        Combines coarse coherent dedispersion trials with a fine FDMT tree
        around each trial. Used by :class:`CohFDMTCPU` and :class:`CohFDMTGPU`.

        Parameters
        ----------
        fcenter : float
            Band centre frequency in MHz.
        bwsub : float
            Subband bandwidth in MHz.
        nsub : int
            Number of subbands.
        tbin : float
            Raw voltage sampling interval in seconds.
        nbin, nfft : int
            FFT length and number of FFT blocks per coherent segment.
        t_p : float
            Detected-time resolution in seconds after channelisation.
        dm_max, dm_min : float
            Coherent DM search range in pc cm^-3.
        noverlap_inp : int, optional
            Overlap samples for the convolution (must be < ``nbin``).
        data_order : {'PRITF', 'FTPRI', 'RITFP'}, optional
            Packed baseband layout.
        verbose : bool, optional
            Print a plan summary during construction.
        )doc")
        .def(py::init<float, float, SizeType, float, SizeType, SizeType, float,
                      float, float, SizeType, std::string_view, bool>(),
             "fcenter"_a, "bwsub"_a, "nsub"_a, "tbin"_a, "nbin"_a, "nfft"_a,
             "t_p"_a, "dm_max"_a, "dm_min"_a = 0.0F, "noverlap_inp"_a = 8192,
             "data_order"_a = "PRITF", "verbose"_a = false)
        .def_property_readonly("f_center", &CohFDMTPlan::get_f_center)
        .def_property_readonly("bw_sub", &CohFDMTPlan::get_bw_sub)
        .def_property_readonly("nsub", &CohFDMTPlan::get_nsub)
        .def_property_readonly("tbin", &CohFDMTPlan::get_tbin)
        .def_property_readonly("nbin", &CohFDMTPlan::get_nbin)
        .def_property_readonly("nfft", &CohFDMTPlan::get_nfft)
        .def_property_readonly("t_p", &CohFDMTPlan::get_t_p)
        .def_property_readonly("dm_max", &CohFDMTPlan::get_dm_max)
        .def_property_readonly("dm_min", &CohFDMTPlan::get_dm_min)
        .def_property_readonly("bw", &CohFDMTPlan::get_bw)
        .def_property_readonly("f_min", &CohFDMTPlan::get_f_min)
        .def_property_readonly("f_max", &CohFDMTPlan::get_f_max)
        .def_property_readonly("n_p", &CohFDMTPlan::get_n_p)
        .def_property_readonly("nchan", &CohFDMTPlan::get_nchan)
        .def_property_readonly("dm_grid_coh", &CohFDMTPlan::get_dm_grid_coh)
        .def_property_readonly("dm_grid_final", &CohFDMTPlan::get_dm_grid_final)
        .def_property_readonly("noverlap", &CohFDMTPlan::get_noverlap)
        .def_property_readonly("nsamp", &CohFDMTPlan::get_nsamp)
        .def_property_readonly("mbin", &CohFDMTPlan::get_mbin)
        .def_property_readonly("mchan", &CohFDMTPlan::get_mchan)
        .def_property_readonly("msamp", &CohFDMTPlan::get_msamp)
        .def_property_readonly("tsamp", &CohFDMTPlan::get_tsamp)
        .def_property_readonly("dt_max", &CohFDMTPlan::get_dt_max)
        .def_property_readonly("ndm", &CohFDMTPlan::get_ndm)
        .def_property_readonly("dmt_ndms", &CohFDMTPlan::get_dmt_ndms)
        .def_property_readonly("dmt_nsamps", &CohFDMTPlan::get_dmt_nsamps)
        .def_property_readonly("dmt_size", &CohFDMTPlan::get_dmt_size)
        .def_property_readonly("chirp_scale", &CohFDMTPlan::get_chirp_scale)
        .def_property_readonly("fdmt_plan", &CohFDMTPlan::get_fdmt_plan)
        .def(
            "get_effective_variance_grid",
            [](const CohFDMTPlan& plan, SizeType boxcar_width,
               bool use_box_smearing) {
                return as_pyarray(plan.get_effective_variance_grid(
                    boxcar_width, use_box_smearing));
            },
            py::arg("boxcar_width") = 1, py::arg("use_box_smearing") = true)
        .def(
            "get_effective_sigma_grid",
            [](const CohFDMTPlan& plan, SizeType boxcar_width,
               bool use_box_smearing) {
                return as_pyarray(plan.get_effective_sigma_grid(
                    boxcar_width, use_box_smearing));
            },
            py::arg("boxcar_width") = 1, py::arg("use_box_smearing") = true)
        .def("get_cumulative_count_grid",
             [](const CohFDMTPlan& plan) {
                 return as_pyarray(plan.get_cumulative_count_grid());
             })
        .def("print_summary", &CohFDMTPlan::print_summary);

    py::class_<DDMTPlan>(mod, "DDMTPlan",
                         R"doc(
        Plan for brute-force incoherent dedispersion (DDMT).

        Stores the DM trial list and the per-channel delay table used by
        :class:`DDMTCPU`.

        Parameters
        ----------
        f_min, f_max : float
            Band edges in MHz.
        nchans : int
            Number of frequency channels.
        tsamp : float
            Sampling interval in seconds.
        dm_max, dm_step : float
            Regular DM grid (pc cm^-3). Ignored when ``dm_arr`` is given.
        dm_min : float, optional
            Lowest DM trial (default 0).
        dm_arr : numpy.ndarray, optional
            Explicit DM trial list. Uses the second constructor.
        )doc")
        .def(py::init<float, float, SizeType, float, float, float, float>(),
             "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "dm_max"_a,
             "dm_step"_a, "dm_min"_a = 0.0F)
        .def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                         const py::array_t<float>& dm_arr) {
                 return DDMTPlan(
                     f_min, f_max, nchans, tsamp,
                     std::vector<float>(dm_arr.data(),
                                        dm_arr.data() + dm_arr.size()));
             }),
             "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "dm_arr"_a)
        .def_property_readonly("f_min", &DDMTPlan::get_f_min)
        .def_property_readonly("f_max", &DDMTPlan::get_f_max)
        .def_property_readonly("nchans", &DDMTPlan::get_nchans)
        .def_property_readonly("tsamp", &DDMTPlan::get_tsamp);
}

} // namespace dmt
