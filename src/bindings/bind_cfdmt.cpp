#include "bindings/bind.hpp"

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
using algorithms::DDMTCPU;

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

    py::class_<DDMTCPU>(mod, "DDMTCPU",
                        R"doc(
        Brute-force incoherent dedispersion on the CPU.

        Parameters
        ----------
        f_min, f_max : float
            Band edges in MHz.
        nchans : int
            Number of frequency channels.
        tsamp : float
            Sampling interval in seconds.
        dm_max, dm_step : float
            Regular DM grid in pc cm^-3.
        dm_min : float, optional
            Lowest DM trial (default 0).
        dm_arr : numpy.ndarray, optional
            Explicit DM list (second constructor).

        See also
        --------
        DDMTPlan, FDMTCPU
        )doc")
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
        .def(
            "execute",
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
            },
            py::arg("waterfall"),
            R"doc(
             Dedisperse a waterfall onto the planned DM grid.

             Parameters
             ----------
             waterfall : numpy.ndarray, dtype float32
                 C-contiguous array of shape ``(nchans, nsamps)``.

             Returns
             -------
             numpy.ndarray
                 ``(n_dm, nsamps - max_delay)`` float32 array.
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
        verbose : bool, optional
            Print the plan summary.
        nthreads : int, optional
            OpenMP / FFTW threads.

        See also
        --------
        CohFDMTPlan, CohFDMTGPU
        )doc")
        .def(py::init<float, float, SizeType, float, SizeType, SizeType, float,
                      float, float, SizeType, std::string_view, bool, int>(),
             "f_center"_a, "sub_bw"_a, "nsub"_a, "tbin"_a, "nbin"_a, "nfft"_a,
             "tp"_a, "dm_max"_a, "dm_min"_a = 0.0F, "noverlap"_a = 8192,
             "data_order"_a = "PRITF", "verbose"_a = false, "nthreads"_a = 1)
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
