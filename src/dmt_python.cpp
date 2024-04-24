#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <dmt/fdmt.hpp>

namespace py = pybind11;

// helper function to avoid making a copy when returning a py::array_t
// source: https://github.com/pybind/pybind11/issues/1042#issuecomment-642215028
template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence&& seq) {
    auto size = seq.size();
    auto data = seq.data();
    std::unique_ptr<Sequence> seq_ptr
        = std::make_unique<Sequence>(std::move(seq));
    auto capsule = py::capsule(seq_ptr.get(), [](void* p) {
        std::unique_ptr<Sequence>(reinterpret_cast<Sequence*>(p));  // NOLINT
    });
    seq_ptr.release();
    return py::array(size, data, capsule);
}

PYBIND11_MODULE(libdmt, mod) {
    mod.doc() = "Python Bindings for dmt";
    py::class_<SubbandPlan>(mod, "SubbandPlan")
        .def_readonly("f_start", &SubbandPlan::f_start)
        .def_readonly("f_end", &SubbandPlan::f_end)
        .def_readonly("f_mid1", &SubbandPlan::f_mid1)
        .def_readonly("f_mid2", &SubbandPlan::f_mid2)
        .def_property_readonly("dt_grid",
                               [](const SubbandPlan& plan) {
                                   return as_pyarray(
                                       static_cast<DtGrid>(plan.dt_grid));
                               })
        .def_readonly("dt_plan", &SubbandPlan::dt_plan);

    py::class_<FDMTPlan>(mod, "FDMTPlan")
        .def_readonly("df_top", &FDMTPlan::df_top)
        .def_readonly("df_bot", &FDMTPlan::df_bot)
        .def_property_readonly(
            "dt_grid_sub_top",
            [](const FDMTPlan& plan) {
                py::list res_list;
                for (const auto& inner_vec : plan.dt_grid_sub_top) {
                    res_list.append(as_pyarray(static_cast<DtGrid>(inner_vec)));
                }
                return res_list;
            })
        .def_readonly("state_shape", &FDMTPlan::state_shape)
        .def_readonly("sub_plan", &FDMTPlan::sub_plan);

    py::class_<FDMT> clsFDMT(mod, "FDMT");
    clsFDMT.def(
        py::init<float, float, size_t, size_t, float, size_t, size_t, size_t>(),
        py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
        py::arg("nsamps"), py::arg("tsamp"), py::arg("dt_max"),
        py::arg("dt_step") = 1, py::arg("dt_min") = 0);
    clsFDMT.def_property_readonly("fdmt_plan", &FDMT::get_plan);
    clsFDMT.def_property_readonly("df", &FDMT::get_df);
    clsFDMT.def_property_readonly("correction", &FDMT::get_correction);
    clsFDMT.def_property_readonly("dt_grid_init", [](FDMT& fdmt) {
        return as_pyarray(fdmt.get_dt_grid_init());
    });
    clsFDMT.def_property_readonly("dt_grid_final", [](FDMT& fdmt) {
        return as_pyarray(fdmt.get_dt_grid_final());
    });
    clsFDMT.def_property_readonly("niters", &FDMT::get_niters);
    clsFDMT.def_property_readonly(
        "dm_arr", [](FDMT& fdmt) { return as_pyarray(fdmt.get_dm_arr()); });
    // execute take 2d array as input, and return 2d array as output
    clsFDMT.def("execute",
                [](FDMT& fdmt,
                   const py::array_t<float, py::array::c_style>& waterfall) {
                    const auto* shape = waterfall.shape();
                    const auto dt_final_size
                        = static_cast<ssize_t>(fdmt.get_dt_grid_final().size());
                    py::array_t<float, py::array::c_style> dmt(
                        {dt_final_size, shape[1]});
                    fdmt.execute(waterfall.data(), waterfall.size(),
                                 dmt.mutable_data(), dmt.size());
                    return dmt;
                });
    // initialise take 2d array as input, and return 3d array as output
    clsFDMT.def("initialise",
                [](FDMT& fdmt,
                   const py::array_t<float, py::array::c_style>& waterfall) {
                    const auto* shape = waterfall.shape();
                    const auto dt_init_size
                        = static_cast<ssize_t>(fdmt.get_dt_grid_init().size());
                    py::array_t<float, py::array::c_style> state(
                        {shape[0], dt_init_size, shape[1]});
                    std::fill(state.mutable_data(),
                              state.mutable_data() + state.size(), 0.0F);
                    fdmt.initialise(waterfall.data(), state.mutable_data());
                    return state;
                });
}
