#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind_utils.hpp"
#include <dmt/cfdmt_cpu.hpp>
#include <dmt/ddmt_cpu.hpp>
#include <dmt/dmt_simulate.hpp>
#include <dmt/fdmt_cpu.hpp>

namespace py = pybind11;

PYBIND11_MODULE(libdmt, mod) {
    mod.doc() = "Python Bindings for dmt";
    PYBIND11_NUMPY_DTYPE(FDMTCoord, i_sub, i_dt, nsamps, buffer_offset,
                         i_coord_tail, i_coord_head, offset);

    py::class_<FDMTSubDTGrid>(mod, "FDMTSubDTGrid")
        .def_readonly("dt_grid", &FDMTSubDTGrid::dt_grid)
        .def_readonly("ndt", &FDMTSubDTGrid::ndt)
        .def_readonly("sub_offset", &FDMTSubDTGrid::sub_offset);
    py::class_<FDMTPlan>(mod, "FDMTPlan")
        .def_readonly("df_top", &FDMTPlan::df_top)
        .def_readonly("df_bot", &FDMTPlan::df_bot)
        .def_readonly("state_shape", &FDMTPlan::state_shape)
        .def_property_readonly(
            "coordinates",
            [](const FDMTPlan& plan) {
                py::list res_list;
                for (const auto& inner : plan.coordinates) {
                    res_list.append(py::array_t<FDMTCoord>(
                        static_cast<ssize_t>(inner.size()), inner.data()));
                }
                return res_list;
            })
        .def_property_readonly(
            "coordinates_sum",
            [](const FDMTPlan& plan) {
                py::list res_list;
                for (const auto& inner : plan.coordinates_to_sum) {
                    res_list.append(py::array_t<FDMTCoord>(
                        static_cast<ssize_t>(inner.size()), inner.data()));
                }
                return res_list;
            })
        .def_property_readonly(
            "coordinates_copy",
            [](const FDMTPlan& plan) {
                py::list res_list;
                for (const auto& inner : plan.coordinates_to_copy) {
                    res_list.append(py::array_t<FDMTCoord>(
                        static_cast<ssize_t>(inner.size()), inner.data()));
                }
                return res_list;
            })
        .def_readonly("dt_grids", &FDMTPlan::dt_grids)
        .def_property_readonly(
            "dt_grid_sub_top",
            [](const FDMTPlan& plan) {
                py::list res_list;
                for (const auto& inner_vec : plan.dt_grid_sub_top) {
                    res_list.append(
                        as_pyarray(static_cast<DtGridType>(inner_vec)));
                }
                return res_list;
            })
        .def("get_memory_usage", &FDMTPlan::get_memory_usage)
        .def("get_memory_usage", &FDMTPlan::get_memory_usage)
        .def("print_summary", &FDMTPlan::print_summary);

    py::class_<FDMTCPU>(mod, "FDMTCPU")
        .def(py::init<float, float, size_t, size_t, float, size_t, size_t,
                      size_t>(),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("dt_max"),
             py::arg("dt_step") = 1, py::arg("dt_min") = 0)
        .def_property_readonly("df", &FDMTCPU::get_df)
        .def_property_readonly("correction", &FDMTCPU::get_correction)
        .def_property_readonly("niters", &FDMTCPU::get_niters)
        .def_property_readonly("plan", &FDMTCPU::get_plan)
        .def_property_readonly("dt_grid_final",
                               [](FDMTCPU& fdmt) {
                                   return as_pyarray_ref(
                                       fdmt.get_dt_grid_final());
                               })
        .def_property_readonly(
            "dm_grid_final",
            [](FDMTCPU& fdmt) { return as_pyarray(fdmt.get_dm_grid_final()); })
        .def_property_readonly("dmt_size", &FDMTCPU::get_dmt_size)
        .def_static("set_log_level", &FDMTCPU::set_log_level, py::arg("level"))
        .def_static("set_num_threads", &FDMTCPU::set_num_threads,
                    py::arg("nthreads"))
        // execute take 2d array as input, and return 2d array as output
        .def(
            "execute",
            [](FDMTCPU& fdmt,
               const py::array_t<float, py::array::c_style>& waterfall) {
                const auto& plan  = fdmt.get_plan();
                const auto niters = fdmt.get_niters();
                py::array_t<float, py::array::c_style> dmt(
                    {plan.state_shape[niters][3], plan.state_shape[niters][4]});
                fdmt.execute(waterfall.data(), waterfall.size(),
                             dmt.mutable_data(), dmt.size());
                return dmt;
            })
        .def("initialise",
             [](FDMTCPU& fdmt,
                const py::array_t<float, py::array::c_style>& waterfall) {
                 const auto& plan = fdmt.get_plan();
                 py::array_t<float, py::array::c_style> state(
                     {plan.state_shape[0][3], plan.state_shape[0][4]});
                 std::fill(state.mutable_data(),
                           state.mutable_data() + state.size(), 0.0F);
                 fdmt.initialise(waterfall.data(), waterfall.size(),
                                 state.mutable_data(), state.size());
                 return state;
             });
    py::class_<DDMTCPU>(mod, "DDMTCPU")
        .def(py::init<float, float, size_t, float, float, float, float>(),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("tsamp"), py::arg("dm_max"), py::arg("dm_step"),
             py::arg("dm_min") = 0)
        .def(py::init([](float f_min, float f_max, size_t nchans, float tsamp,
                         py::array_t<float> dm_arr) {
            return new DDMTCPU(f_min, f_max, nchans, tsamp, dm_arr.data(),
                               dm_arr.size());
        }))
        .def_static("set_num_threads", &DDMTCPU::set_num_threads,
                    py::arg("nthreads"))
        .def("execute",
             [](DDMTCPU& ddmt,
                const py::array_t<float, py::array::c_style>& waterfall) {
                 const auto* shape         = waterfall.shape();
                 const auto nsamps         = static_cast<size_t>(shape[1]);
                 const auto max_delay      = ddmt.get_plan().delay_table.back();
                 const auto nsamps_reduced = nsamps - max_delay;
                 const auto dm_count       = ddmt.get_plan().dm_arr.size();
                 py::array_t<float, py::array::c_style> dmt(
                     {dm_count, nsamps_reduced});
                 ddmt.execute(waterfall.data(), waterfall.size(),
                              dmt.mutable_data(), dmt.size());
                 return dmt;
             });
    py::class_<CohFDMTPlan>(mod, "CohFDMTPlan")
        .def(py::init<float, float, size_t, float, size_t, size_t, float, float,
                      float, size_t>(),
             py::arg("f_center"), py::arg("sub_bw"), py::arg("nsub"),
             py::arg("tbin"), py::arg("nbin"), py::arg("nfft"), py::arg("tp"),
             py::arg("dm_max"), py::arg("dm_min") = 0.0F,
             py::arg("noverlap") = 8192);
    py::class_<CohFDMTCPU>(mod, "CohFDMTCPU")
        .def(py::init<float, float, size_t, float, size_t, size_t, float, float,
                      float, size_t>(),
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
    mod.def("generate_pure_frb", [](size_t nchans, size_t nsamps, float f_min,
                                    float f_max, size_t dt, float pulse_toa,
                                    float amplitude = 1.0F) {
        auto [arr, nsamps_dispersed] = generate_pure_frb(
            nchans, nsamps, f_min, f_max, dt, pulse_toa, amplitude);
        return std::make_tuple(as_pyarray_ref(arr), nsamps_dispersed);
    });
}
