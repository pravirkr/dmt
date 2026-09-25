#include "bindings/bind.hpp"

#include <algorithm>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dmt/dmt.hpp"
#include "pybind_utils.hpp"

namespace dmt {
using algorithms::CohFDMTCPU;

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

void bind_cfdmt(py::module_& mod) {
    mod.def(
        "generate_pure_frb",
        [](SizeType nchans, SizeType nsamps, float f_min, float f_max,
           SizeType dt, float pulse_toa, float amplitude = 1.0F) {
            auto [arr, nsamps_dispersed] = utils::generate_pure_frb(
                nchans, nsamps, f_min, f_max, dt, pulse_toa, amplitude);
            return std::make_tuple(as_pyarray(std::move(arr)),
                                   nsamps_dispersed);
        },
        "nchans"_a, "nsamps"_a, "f_min"_a, "f_max"_a, "dt"_a, "pulse_toa"_a,
        "amplitude"_a = 1.0F,
        R"doc(
        Inject a noise-free dispersed pulse into a zero waterfall.

        Parameters
        ----------
        nchans, nsamps : int
            Waterfall shape.
        f_min, f_max : float
            Band edges in MHz.
        dt : int
            Total delay across the band, in samples.
        pulse_toa : float
            Pulse time of arrival at the lowest channel, in samples.
        amplitude : float, optional
            Peak amplitude (default 1).

        Returns
        -------
        waterfall : numpy.ndarray
            Flattened ``float32`` array of length ``nchans * nsamps``.
        n_dispersed : int
            Number of samples that received energy.
        )doc");

    py::class_<CohFDMTCPU>(mod, "CohFDMTCPU",

                           R"doc(
        Hybrid coherent FDMT on the CPU.

        Unpacks packed baseband, applies coarse coherent chirps, detects
        Stokes I, and runs a fine FDMT around each coarse DM.

        Parameters
        ----------
        f_center, sub_bw : float
            Centre frequency and subband bandwidth in MHz.
        nsub : int
            Number of subbands.
        tbin : float
            Voltage sampling interval in seconds.
        nbin, nfft : int
            FFT length and blocks per coherent segment.
        tp : float
            Detected-time resolution in seconds.
        dm_max, dm_min : float
            Coherent DM search range in pc cm^-3.
        noverlap : int, optional
            Convolution overlap. Must be smaller than ``nbin``.
        data_order : {'PRITF', 'FTPRI', 'RITFP'}, optional
            Packed baseband layout.
        verbose : int, optional
            0 = silent, 1 = info, 2 = debug.
            Print the plan summary.
        nthreads : int, optional
            OpenMP / FFTW threads.

        See also
        --------
        CohFDMTPlan, CohFDMTCUDA
        )doc")
        .def(py::init<float, float, SizeType, float, SizeType, SizeType, float,
                      float, float, SizeType, std::string_view, bool, int>(),
             "f_center"_a, "sub_bw"_a, "nsub"_a, "tbin"_a, "nbin"_a, "nfft"_a,
             "tp"_a, "dm_max"_a, "dm_min"_a = 0.0F, "noverlap"_a = 8192,
             "data_order"_a = "PRITF", "verbose"_a = 0, "nthreads"_a = 1)
        .def_property_readonly("plan", &CohFDMTCPU::get_plan)
        // Bind each data type to execute method
        .def("execute",
             [](CohFDMTCPU& coh_fdmt,
                const py::array_t<uint8_t, py::array::c_style>& data_in) {
                 const auto& plan      = coh_fdmt.get_plan();
                 const auto ndm_total  = plan.get_ndm();
                 const auto nsamps_out = plan.get_dmt_nsamps();
                 py::array_t<float, py::array::c_style> dmt(
                     {static_cast<ssize_t>(ndm_total),
                      static_cast<ssize_t>(nsamps_out)});
                 coh_fdmt.execute(
                     std::span<const uint8_t>(data_in.data(), data_in.size()),
                     std::span<float>(dmt.mutable_data(), dmt.size()));
                 return dmt;
             })
        .def("execute",
             [](CohFDMTCPU& coh_fdmt,
                const py::array_t<int8_t, py::array::c_style>& data_in) {
                 const auto& plan      = coh_fdmt.get_plan();
                 const auto ndm_total  = plan.get_ndm();
                 const auto nsamps_out = plan.get_dmt_nsamps();
                 py::array_t<float, py::array::c_style> dmt(
                     {static_cast<ssize_t>(ndm_total),
                      static_cast<ssize_t>(nsamps_out)});
                 coh_fdmt.execute(
                     std::span<const int8_t>(data_in.data(), data_in.size()),
                     std::span<float>(dmt.mutable_data(), dmt.size()));
                 return dmt;
             })
        .def("reset_history", &CohFDMTCPU::reset_history,
             R"doc(
             Clear streaming delay-line history between independent observations.
             )doc");
}

} // namespace dmt
