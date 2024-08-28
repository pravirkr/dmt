#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind_utils.hpp"
#include <dmt/cfdmt_cpu.hpp>
#include <dmt/ddmt_cpu.hpp>
#include <dmt/dmt_plans.hpp>
#include <dmt/dmt_simulate.hpp>
#include <dmt/fdmt_cpu.hpp>

namespace py = pybind11;

PYBIND11_MODULE(libdmt, mod) { // NOLINT
    mod.doc() = "Python Bindings for dmt";
    mod.def(
        "generate_pure_frb",
        [](SizeType nchans, SizeType nsamps, float f_min, float f_max,
           SizeType dt, float pulse_toa, float amplitude = 1.0F) {
            const auto [arr, nsamps_dispersed] = generate_pure_frb(
                nchans, nsamps, f_min, f_max, dt, pulse_toa, amplitude);
            return std::make_tuple(as_pyarray_ref(arr), nsamps_dispersed);
        },
        py::arg("nchans"), py::arg("nsamps"), py::arg("f_min"),
        py::arg("f_max"), py::arg("dt"), py::arg("pulse_toa"),
        py::arg("amplitude") = 1.0F);
    PYBIND11_NUMPY_DTYPE(FDMTShape, nchans, ndt_min, ndt_max, ncoords,
                         ncoords_sum, ncoords_copy, nsamps, nelements, dt_max);
    PYBIND11_NUMPY_DTYPE(FDMTCoord, i_sub, i_dt, nsamps, buf_offset,
                         i_coord_tail, i_coord_head, offset, tail_buf_offset,
                         tail_nsamps, head_buf_offset, head_nsamps);
    py::class_<FDMTCoordGrid>(mod, "FDMTSubDTGrid")
        .def_readonly("dt_grid", &FDMTCoordGrid::dt_grid)
        .def_readonly("ndt", &FDMTCoordGrid::ndt)
        .def_readonly("grid_offset", &FDMTCoordGrid::coord_offset)
        .def_readonly("f_start", &FDMTCoordGrid::f_start)
        .def_readonly("f_end", &FDMTCoordGrid::f_end);
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
                      SizeType, SizeType>(),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("dt_max"),
             py::arg("dt_step") = 1, py::arg("dt_min") = 0)
        .def_property_readonly("f_min", &FDMTPlan::get_f_min)
        .def_property_readonly("f_max", &FDMTPlan::get_f_max)
        .def_property_readonly("nchans", &FDMTPlan::get_nchans)
        .def_property_readonly("nsamps", &FDMTPlan::get_nsamps)
        .def_property_readonly("tsamp", &FDMTPlan::get_tsamp)
        .def_property_readonly("dt_max", &FDMTPlan::get_dt_max)
        .def_property_readonly("dt_step", &FDMTPlan::get_dt_step)
        .def_property_readonly("dt_min", &FDMTPlan::get_dt_min)
        .def_property_readonly("df", &FDMTPlan::get_df)
        .def_property_readonly("correction", &FDMTPlan::get_correction)
        .def_property_readonly("niters", &FDMTPlan::get_niters)
        .def_property_readonly("container", &FDMTPlan::get_container)
        .def_property_readonly("dt_grid_final",
                               [](FDMTPlan& plan) {
                                   return as_pyarray_ref(
                                       plan.get_dt_grid_final());
                               })
        .def_property_readonly(
            "dm_grid_final",
            [](FDMTPlan& plan) { return as_pyarray(plan.get_dm_grid_final()); })
        .def_property_readonly("dmt_size", &FDMTPlan::get_dmt_size)
        .def_property_readonly("buffer_size", &FDMTPlan::get_buffer_size)
        .def("print_summary", &FDMTPlan::print_summary)
        .def_static("set_log_level", &FDMTPlan::set_log_level,
                    py::arg("level"));
    py::class_<CohFDMTPlan>(mod, "CohFDMTPlan")
        .def(py::init<float, float, SizeType, float, SizeType, SizeType, float,
                      float, float, SizeType>(),
             py::arg("fcenter"), py::arg("bwsub"), py::arg("nsub"),
             py::arg("tbin"), py::arg("nbin"), py::arg("nfft"), py::arg("t_p"),
             py::arg("dm_max"), py::arg("dm_min") = 0.0F,
             py::arg("noverlap_inp") = 8192);

    py::class_<DDMTPlan>(mod, "DDMTPlan")
        .def(py::init<float, float, SizeType, float, float, float, float>(),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("tsamp"), py::arg("dm_max"), py::arg("dm_step"),
             py::arg("dm_min") = 0)
        .def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                         const py::array_t<float>& dm_arr) {
                 return DDMTPlan(
                     f_min, f_max, nchans, tsamp,
                     std::vector<float>(dm_arr.data(),
                                        dm_arr.data() + dm_arr.size()));
             }),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("tsamp"), py::arg("dm_arr"))
        .def_property_readonly("f_min", &DDMTPlan::get_f_min)
        .def_property_readonly("f_max", &DDMTPlan::get_f_max)
        .def_property_readonly("nchans", &DDMTPlan::get_nchans)
        .def_property_readonly("tsamp", &DDMTPlan::get_tsamp);

    py::class_<FDMTCPU>(mod, "FDMTCPU")
        .def(py::init<float, float, SizeType, SizeType, float, SizeType,
                      SizeType, SizeType, bool>(),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("dt_max"),
             py::arg("dt_step") = 1, py::arg("dt_min") = 0,
             py::arg("use_history") = false)
        .def_property_readonly("plan", &FDMTCPU::get_plan)
        .def_static("set_log_level", &FDMTCPU::set_log_level, py::arg("level"))
        .def_static("set_num_threads", &FDMTCPU::set_num_threads,
                    py::arg("nthreads"))
        // execute take 2d array as input, and return 2d array as output
        .def(
            "execute",
            [](FDMTCPU& fdmt,
               const py::array_t<float, py::array::c_style>& waterfall,
               bool normalize) {
                const auto& plan   = fdmt.get_plan();
                const auto& plan_c = plan.get_container();
                const auto niters  = plan.get_niters();
                py::array_t<float, py::array::c_style> dmt(
                    {plan_c.state_shape[niters].ncoords,
                     plan_c.state_shape[niters].nsamps});
                fdmt.execute(waterfall.data(), waterfall.size(),
                             dmt.mutable_data(), dmt.size(), normalize);
                return dmt;
            },
            py::arg("waterfall"), py::arg("normalize") = true)
        .def(
            "initialise",
            [](FDMTCPU& fdmt,
               const py::array_t<float, py::array::c_style>& waterfall,
               bool normalize) {
                const auto& plan   = fdmt.get_plan();
                const auto& plan_c = plan.get_container();
                py::array_t<float, py::array::c_style> state(
                    {plan_c.state_shape[0].ncoords,
                     plan_c.state_shape[0].nsamps});
                std::fill(state.mutable_data(),
                          state.mutable_data() + state.size(), 0.0F);
                fdmt.initialise(waterfall.data(), waterfall.size(),
                                state.mutable_data(), state.size(), normalize);
                return state;
            },
            py::arg("waterfall"), py::arg("normalize") = true);
    py::class_<DDMTCPU>(mod, "DDMTCPU")
        .def(py::init<float, float, SizeType, float, float, float, float>(),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("tsamp"), py::arg("dm_max"), py::arg("dm_step"),
             py::arg("dm_min") = 0)
        .def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                         const py::array_t<float>& dm_arr) {
                 return DDMTCPU(
                     f_min, f_max, nchans, tsamp,
                     std::vector<float>(dm_arr.data(),
                                        dm_arr.data() + dm_arr.size()));
             }),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("tsamp"), py::arg("dm_arr"))
        .def_static("set_num_threads", &DDMTCPU::set_num_threads,
                    py::arg("nthreads"))
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
                 ddmt.execute(waterfall.data(), waterfall.size(),
                              dmt.mutable_data(), dmt.size());
                 return dmt;
             });
    py::class_<CohFDMTCPU>(mod, "CohFDMTCPU")
        .def(py::init<float, float, SizeType, float, SizeType, SizeType, float,
                      float, float, SizeType>(),
             py::arg("f_center"), py::arg("sub_bw"), py::arg("nsub"),
             py::arg("tbin"), py::arg("nbin"), py::arg("nfft"), py::arg("tp"),
             py::arg("dm_max"), py::arg("dm_min") = 0.0F,
             py::arg("noverlap") = 8192)
        .def_static("set_num_threads", &CohFDMTCPU::set_num_threads,
                    py::arg("nthreads"))
        .def_property_readonly("plan", &CohFDMTCPU::get_plan)
        .def_property_readonly("dmt_size", &CohFDMTCPU::get_dmt_size)
        .def("execute",
             [](CohFDMTCPU& coh_fdmt,
                const py::array_t<uint8_t, py::array::c_style>& data_in,
                std::string in_order = "PRITF") {
                 const auto* shape = data_in.shape();
                 py::array_t<float, py::array::c_style> dmt(
                     {static_cast<ssize_t>(coh_fdmt.get_dmt_size()), shape[1]});
                 coh_fdmt.execute(data_in.data(), data_in.size(),
                                  std::move(in_order), dmt.mutable_data(),
                                  dmt.size());
                 return dmt;
             });
}
