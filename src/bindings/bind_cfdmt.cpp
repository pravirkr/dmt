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
using algorithms::CohFDMT;

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

namespace {

// Runs CohFDMT into a get_buffer_size() arena and returns a zero-copy
// (ndm_total, nsamps) view of its leading get_dmt_size() result.
template <typename DataType>
py::array_t<float>
coh_fdmt_execute(CohFDMT& coh_fdmt,
                 const py::array_t<DataType, py::array::c_style>& data_in) {
    const auto& plan      = coh_fdmt.get_plan();
    const auto ndm_total  = static_cast<py::ssize_t>(plan.get_ndm());
    const auto nsamps_out = static_cast<py::ssize_t>(plan.get_dmt_nsamps());
    py::array_t<float, py::array::c_style> arena(
        static_cast<py::ssize_t>(plan.get_buffer_size()));
    coh_fdmt.execute(std::span<const DataType>(data_in.data(), data_in.size()),
                     std::span<float>(arena.mutable_data(), arena.size()));
    return {{ndm_total, nsamps_out},
            {nsamps_out * static_cast<py::ssize_t>(sizeof(float)),
             static_cast<py::ssize_t>(sizeof(float))},
            arena.data(),
            arena};
}

} // namespace

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

    py::class_<CohFDMT>(mod, "CohFDMT",
                        R"doc(
        Hybrid coherent FDMT.

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
        nthreads : int, optional
            OpenMP / FFTW threads on the CPU backend.
        backend : {'cpu', 'cuda', 'hip'}, optional
            Keyword-only. Where to run (default ``'cpu'``); see
            :func:`available_backends`.
        device : int, optional
            Keyword-only. Device ordinal on a GPU backend (default 0).

        See also
        --------
        CohFDMTPlan
        )doc")
        .def(py::init([](float f_center, float sub_bw, SizeType nsub,
                         float tbin, SizeType nbin, SizeType nfft, float tp,
                         float dm_max, float dm_min, SizeType noverlap,
                         std::string_view data_order, int nthreads,
                         std::string_view backend, int device) {
                 return CohFDMT(f_center, sub_bw, nsub, tbin, nbin, nfft, tp,
                                dm_max, dm_min, noverlap, data_order,
                                make_exec(backend, nthreads, device));
             }),
             "f_center"_a, "sub_bw"_a, "nsub"_a, "tbin"_a, "nbin"_a, "nfft"_a,
             "tp"_a, "dm_max"_a, "dm_min"_a = 0.0F, "noverlap"_a = 8192,
             "data_order"_a = "PRITF", "nthreads"_a = 1, py::kw_only(),
             "backend"_a = "cpu", "device"_a = 0)
        .def_property_readonly(
            "backend",
            [](const CohFDMT& coh) {
                return std::string(to_string(coh.backend()));
            },
            "Backend this instance runs on ('cpu', 'cuda', ...).")
        .def_property_readonly("nthreads", &CohFDMT::nthreads)
        .def_property_readonly("device", &CohFDMT::device)
        .def_property_readonly("plan", &CohFDMT::get_plan)
        // Bind each data type to execute method
        .def("execute", &coh_fdmt_execute<uint8_t>, py::arg("data_in"))
        .def("execute", &coh_fdmt_execute<int8_t>, py::arg("data_in"))
        .def("reset_history", &CohFDMT::reset_history,
             R"doc(
             Clear streaming delay-line history between independent observations.
             )doc");
}

} // namespace dmt
