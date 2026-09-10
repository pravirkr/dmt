#include <span>

#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind_utils.hpp"

#include "dmt/dmt.hpp"

namespace dmt {
using algorithms::CohFDMTCPU;
using algorithms::DDMTCPU;
using algorithms::FDMTCPU;
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

PYBIND11_MODULE(libdmt, mod) { // NOLINT
    mod.doc() = "Python Bindings for dmt";

    py::add_ostream_redirect(mod, "ostream_redirect");
    mod.def(
        "generate_pure_frb",
        [](SizeType nchans, SizeType nsamps, float f_min, float f_max,
           SizeType dt, float pulse_toa, float amplitude = 1.0F) {
            const auto [arr, nsamps_dispersed] = utils::generate_pure_frb(
                nchans, nsamps, f_min, f_max, dt, pulse_toa, amplitude);
            return std::make_tuple(as_pyarray_ref(arr), nsamps_dispersed);
        },
        "nchans"_a, "nsamps"_a, "f_min"_a, "f_max"_a, "dt"_a, "pulse_toa"_a,
        "amplitude"_a = 1.0F);

    PYBIND11_NUMPY_DTYPE(FDMTShape, nchans, ndt_min, ndt_max, ncoords,
                         ncoords_sum, ncoords_copy, nsamps, nelements, dt_max);
    PYBIND11_NUMPY_DTYPE(FDMTCoord, i_sub, i_dt, nsamps, buf_offset,
                         i_coord_tail, i_coord_head, delay, tail_buf_offset,
                         tail_nsamps, head_buf_offset, head_nsamps);
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
            [](py::object self) {
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
            return std::vector<SizeType>(v.dt_grid.begin(), v.dt_grid.end());
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
    py::class_<FDMTPlan>(mod, "FDMTPlan")
        .def(py::init<float, float, SizeType, SizeType, float, SizeType,
                      SizeType, SizeType, std::string_view, bool>(),
             "f_min"_a, "f_max"_a, "nchans"_a, "nsamps"_a, "tsamp"_a,
             "dt_max"_a, "dt_min"_a = 0, "dt_step"_a = 1, "mode"_a = "full",
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
             py::arg("mode") = "full", py::arg("verbose") = false)
        .def_property_readonly("f_min", &FDMTPlan::get_f_min)
        .def_property_readonly("f_max", &FDMTPlan::get_f_max)
        .def_property_readonly("nchans", &FDMTPlan::get_nchans)
        .def_property_readonly("nsamps", &FDMTPlan::get_nsamps)
        .def_property_readonly("tsamp", &FDMTPlan::get_tsamp)
        .def_property_readonly("dt_max", &FDMTPlan::get_dt_max)
        .def_property_readonly("dt_min", &FDMTPlan::get_dt_min)
        .def_property_readonly("dt_step", &FDMTPlan::get_dt_step)
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
        .def_property_readonly("complexity", &FDMTPlan::get_complexity)
        .def_property_readonly("dmt_ndms", &FDMTPlan::get_dmt_ndms)
        .def_property_readonly("dmt_nsamps", &FDMTPlan::get_dmt_nsamps)
        .def_property_readonly("dmt_size", &FDMTPlan::get_dmt_size)
        .def_property_readonly("buffer_size", &FDMTPlan::get_buffer_size)
        .def_property_readonly("history_size", &FDMTPlan::get_history_size)
        .def_property_readonly("history_init_size",
                               &FDMTPlan::get_history_init_size)
        .def(
            "print_summary",
            [](FDMTPlan& plan, std::string_view prefix) {
                plan.print_summary(prefix);
            },
            "prefix"_a = "")
        .def("print_complexity_summary", &FDMTPlan::print_complexity_summary);
    py::class_<CohFDMTPlan>(mod, "CohFDMTPlan")
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
        .def_property_readonly("chirp_scale", &CohFDMTPlan::get_chirp_scale)
        .def_property_readonly("fdmt_plan", &CohFDMTPlan::get_fdmt_plan)
        .def("print_summary", &CohFDMTPlan::print_summary);

    py::class_<DDMTPlan>(mod, "DDMTPlan")
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

    py::class_<FDMTCPU>(mod, "FDMTCPU", py::dynamic_attr(),
                        "FDMT CPU Implementation Wrapper")
        .def(py::init<float, float, SizeType, SizeType, float, SizeType,
                      SizeType, SizeType, bool, std::string_view, bool, int>(),
             "f_min"_a, "f_max"_a, "nchans"_a, "nsamps"_a, "tsamp"_a,
             "dt_max"_a, "dt_min"_a = 0, "dt_step"_a = 1,
             "use_box_smearing"_a = true, "mode"_a = "full",
             "verbose"_a = false, "nthreads"_a = 1)
        .def(
            py::init([](float f_min, float f_max, SizeType nchans,
                        SizeType nsamps, float tsamp, const py::object& dt_grid,
                        const py::object& dt_arr, const py::object& dm_grid,
                        const py::object& dm_arr, bool use_box_smearing,
                        std::string_view mode, bool verbose, int nthreads) {
                const auto [type, obj] =
                    resolve_custom_grid(dt_grid, dt_arr, dm_grid, dm_arr);
                if (type == CustomGridType::kDt) {
                    return FDMTCPU(f_min, f_max, nchans, nsamps, tsamp,
                                   extract_dt_grid(obj), use_box_smearing, mode,
                                   verbose, nthreads);
                }
                return FDMTCPU(f_min, f_max, nchans, nsamps, tsamp,
                               extract_dm_grid(obj), use_box_smearing, mode,
                               verbose, nthreads);
            }),
            py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
            py::arg("nsamps"), py::arg("tsamp"), py::kw_only(),
            py::arg("dt_grid") = py::none(), py::arg("dt_arr") = py::none(),
            py::arg("dm_grid") = py::none(), py::arg("dm_arr") = py::none(),
            py::arg("use_box_smearing") = true, py::arg("mode") = "full",
            py::arg("verbose") = false, py::arg("nthreads") = 1)
        .def_property_readonly(
            "plan", &FDMTCPU::get_plan,
            "Get the FDMTPlan object containing transform details.")
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
        // execute take 2d array as input, and return 2d array as output
        .def(
            "execute",
            [](FDMTCPU& fdmt,
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
            Executes the FDMT transform on the CPU.

            Parameters
            ----------
            waterfall : numpy.ndarray
                A 2D NumPy array of shape (nchans, nsamps) containing the input data.
                Must be C-contiguous and of type float32.

            Returns
            -------
            numpy.ndarray
                A 2D NumPy array containing the transformed data.
                Shape is (n_delays, n_times) where n_delays depends on dt_max/dt_step
                and n_times is reduced from input nsamps based on maximum delay.

            Notes
            -----
            The input array must be properly sized according to the FDMT parameters
            specified during initialization (nchans, nsamps).
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
        .def_property_readonly("is_finished", &FDMTCPU::is_finished);

    mod.def(
        "compute_fdmt",
        [](const py::array_t<float, py::array::c_style>& waterfall, float f_min,
           float f_max, SizeType nchans, SizeType nsamps, float tsamp,
           SizeType dt_max, SizeType dt_min, SizeType dt_step,
           bool use_box_smearing, std::string_view mode, bool verbose,
           int nthreads) {
            auto [dmt, fdmt_plan] = algorithms::compute_fdmt(
                std::span<const float>(waterfall.data(), waterfall.size()),
                f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, dt_step,
                use_box_smearing, mode, verbose, nthreads);
            return std::make_tuple(as_pyarray(std::move(dmt)), fdmt_plan);
        },
        py::arg("waterfall"), py::arg("f_min"), py::arg("f_max"),
        py::arg("nchans"), py::arg("nsamps"), py::arg("tsamp"),
        py::arg("dt_max"), py::arg("dt_min") = 0, py::arg("dt_step") = 1,
        py::arg("use_box_smearing") = true, py::arg("mode") = "full",
        py::arg("verbose") = false, py::arg("nthreads") = 1);

    mod.def(
        "compute_fdmt",
        [](const py::array_t<float, py::array::c_style>& waterfall, float f_min,
           float f_max, SizeType nchans, SizeType nsamps, float tsamp,
           const py::object& dt_grid, const py::object& dt_arr,
           const py::object& dm_grid, const py::object& dm_arr,
           bool use_box_smearing, std::string_view mode, bool verbose,
           int nthreads) {
            const auto [type, obj] =
                resolve_custom_grid(dt_grid, dt_arr, dm_grid, dm_arr);
            if (type == CustomGridType::kDt) {
                auto [dmt, fdmt_plan] = algorithms::compute_fdmt(
                    std::span<const float>(waterfall.data(), waterfall.size()),
                    f_min, f_max, nchans, nsamps, tsamp, extract_dt_grid(obj),
                    use_box_smearing, mode, verbose, nthreads);
                return std::make_tuple(as_pyarray(std::move(dmt)), fdmt_plan);
            }
            auto [dmt, fdmt_plan] = algorithms::compute_fdmt(
                std::span<const float>(waterfall.data(), waterfall.size()),
                f_min, f_max, nchans, nsamps, tsamp, extract_dm_grid(obj),
                use_box_smearing, mode, verbose, nthreads);
            return std::make_tuple(as_pyarray(std::move(dmt)), fdmt_plan);
        },
        py::arg("waterfall"), py::arg("f_min"), py::arg("f_max"),
        py::arg("nchans"), py::arg("nsamps"), py::arg("tsamp"), py::kw_only(),
        py::arg("dt_grid") = py::none(), py::arg("dt_arr") = py::none(),
        py::arg("dm_grid") = py::none(), py::arg("dm_arr") = py::none(),
        py::arg("use_box_smearing") = true, py::arg("mode") = "full",
        py::arg("verbose") = false, py::arg("nthreads") = 1);

    py::class_<DDMTCPU>(mod, "DDMTCPU")
        .def(py::init<float, float, SizeType, float, float, float, float>(),
             "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "dm_max"_a,
             "dm_step"_a, "dm_min"_a = 0.0F)
        .def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                         const py::array_t<float>& dm_arr) {
                 return DDMTCPU(
                     f_min, f_max, nchans, tsamp,
                     std::vector<float>(dm_arr.data(),
                                        dm_arr.data() + dm_arr.size()));
             }),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("tsamp"), py::arg("dm_arr"))
        .def("execute",
             [](DDMTCPU& ddmt,
                const py::array_t<float, py::array::c_style>& waterfall) {
                 const auto& plan          = ddmt.get_plan();
                 const auto& plan_c        = plan.get_container();
                 const auto* shape         = waterfall.shape();
                 const auto nsamps         = static_cast<SizeType>(shape[1]);
                 const auto max_delay      = plan_c.delay_table.back();
                 const auto nsamps_reduced = nsamps - max_delay;
                 const auto dm_count       = plan_c.dm_arr.size();
                 py::array_t<float, py::array::c_style> dmt(
                     {dm_count, nsamps_reduced});
                 ddmt.execute(
                     std::span<const float>(waterfall.data(), waterfall.size()),
                     std::span<float>(dmt.mutable_data(), dmt.size()));
                 return dmt;
             });
    py::class_<CohFDMTCPU>(mod, "CohFDMTCPU")
        .def(py::init<float, float, SizeType, float, SizeType, SizeType, float,
                      float, float, SizeType, std::string_view, int, bool>(),
             "f_center"_a, "sub_bw"_a, "nsub"_a, "tbin"_a, "nbin"_a, "nfft"_a,
             "tp"_a, "dm_max"_a, "dm_min"_a = 0.0F, "noverlap"_a = 8192,
             "data_order"_a = "PRITF", "nthreads"_a = 1, "verbose"_a = false)
        .def_property_readonly("plan", &CohFDMTCPU::get_plan)
        // Bind each data type to execute method
        .def("execute",
             [](CohFDMTCPU& coh_fdmt,
                const py::array_t<uint8_t, py::array::c_style>& data_in) {
                 const auto* shape = data_in.shape();
                 py::array_t<float, py::array::c_style> dmt(
                     {static_cast<ssize_t>(coh_fdmt.get_plan().get_dmt_size()),
                      shape[1]});
                 coh_fdmt.execute(
                     std::span<const uint8_t>(data_in.data(), data_in.size()),
                     std::span<float>(dmt.mutable_data(), dmt.size()));
                 return dmt;
             });
}

} // namespace dmt