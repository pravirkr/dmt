#include "bindings/bind.hpp"

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

#include "dmt/algorithms/ddmt_fft.hpp"
#include "dmt/common/fft_config.hpp"
#include "dmt/common/plans.hpp"
#include "pybind_utils.hpp"

namespace dmt {
using algorithms::DDMTFFT;
using algorithms::DDMTFFTMethod;
using algorithms::DDMTFFTOptions;
using plans::DDMTPlan;
using plans::LevinConfig;

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

namespace {

std::vector<uint8_t> mask_vec(const std::optional<py::array_t<uint8_t>>& m) {
    std::vector<uint8_t> v;
    if (m.has_value()) {
        v.assign(m->data(), m->data() + m->size());
    }
    return v;
}

DDMTFFTOptions
make_options(std::string_view method, double tolerance, SizeType guard) {
    return DDMTFFTOptions{
        .method    = algorithms::parse_ddmt_fft_method(method),
        .tolerance = tolerance,
        .guard     = guard,
    };
}

// (n_dm, n_out) for one beam, (nbeams, n_dm, n_out) otherwise.
py::array_t<float, py::array::c_style>
output_array(const DDMTFFT& eng, SizeType nsamps, bool three_d) {
    const auto ndm  = eng.get_plan().get_container().dm_arr.size();
    const auto nout = eng.get_output_nsamps(nsamps);
    if (three_d) {
        return py::array_t<float, py::array::c_style>(
            {static_cast<py::ssize_t>(eng.get_nbeams()),
             static_cast<py::ssize_t>(ndm), static_cast<py::ssize_t>(nout)});
    }
    return py::array_t<float, py::array::c_style>(
        {static_cast<py::ssize_t>(ndm), static_cast<py::ssize_t>(nout)});
}

} // namespace

void bind_ddmt_fft(py::module_& mod) {
    py::enum_<fft::Planner>(mod, "FFTPlanner",
                            "FFTW planner effort for CPU FFT plans.")
        .value("ESTIMATE", fft::Planner::kEstimate)
        .value("MEASURE", fft::Planner::kMeasure)
        .value("PATIENT", fft::Planner::kPatient)
        .value("EXHAUSTIVE", fft::Planner::kExhaustive);
    mod.def("set_fft_planner", &fft::set_planner, "planner"_a,
            R"doc(
        Set the FFTW planner effort for CPU FFT plans created from now on
        (engines already built keep theirs). ``MEASURE`` or ``PATIENT`` make
        construction slower and transforms often faster; save the result with
        :func:`export_fft_wisdom` to make later constructions instant.
        )doc");
    mod.def("get_fft_planner", &fft::get_planner);
    mod.def("import_fft_wisdom", &fft::import_wisdom, "path"_a,
            "Load FFTW wisdom from a file; returns True on success.");
    mod.def("export_fft_wisdom", &fft::export_wisdom, "path"_a,
            "Save the FFTW wisdom of this process; returns True on success.");
    mod.def("forget_fft_wisdom", &fft::forget_wisdom,
            "Discard all accumulated FFTW wisdom.");

    py::class_<DDMTFFT> cls(mod, "DDMTFFT",
                            R"doc(
        Fourier-domain direct dedispersion (FDD) with exact fractional delays.

        Every channel is delayed by its exact, unrounded dispersion delay as a
        phase ramp (band-limited interpolation), instead of DDMT's rounding
        to whole samples. Same plans, DM grids, kill mask, nbeams and
        streaming model as :class:`DDMT` (history across calls); the output is
        float32 for float and packed input alike.

        Parameters
        ----------
        f_min, f_max, nchans, tsamp, dm_max, dm_step, dm_min, dm_arr, levin, plan
            As for :class:`DDMT`.
        nthreads : int, optional
            OpenMP threads (CPU).
        nbits : int, optional
            Input precision: 32 (float) or 1, 2, 4, 8, 16 (packed).
        kill_mask : numpy.ndarray, optional
            Per-channel mask (1 = keep, 0 = drop).
        nbeams : int, optional
            Beams per call.
        method : {'auto', 'brute', 'nufft'}, optional
            Keyword-only. ``'nufft'`` (a type-1 non-uniform FFT over the DM
            axis, one per uniformly spaced run of the grid) is much faster
            than ``'brute'`` (exact per-channel phase rotation, any grid).
            ``'auto'`` runs the NUFFT on every uniform run of at least 32
            trials and brute force on the rest (:attr:`method_used`); for a
            wide DM range use a piecewise-uniform grid
            (``LevinConfig(..., piecewise_uniform=True)``).
        tolerance : float, optional
            Keyword-only. NUFFT relative accuracy (default 1e-6).
        guard : int, optional
            Keyword-only. Interpolation context kept around every segment
            (default 64); get_max_delay() = ceil(max delay) + guard.
        backend, device
            Keyword-only, as for :class:`DDMT`.
        )doc");
    cls.def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                        float dm_max, float dm_step, float dm_min, int nthreads,
                        SizeType nbits,
                        std::optional<py::array_t<uint8_t>> kill_mask,
                        SizeType nbeams, std::string_view method,
                        double tolerance, SizeType guard,
                        std::string_view backend, int device) {
                return DDMTFFT(f_min, f_max, nchans, tsamp, dm_max, dm_step,
                               dm_min, make_exec(backend, nthreads, device),
                               nbits, mask_vec(kill_mask), nbeams,
                               make_options(method, tolerance, guard));
            }),
            "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "dm_max"_a,
            "dm_step"_a, "dm_min"_a = 0.0F, "nthreads"_a = 1, "nbits"_a = 32,
            "kill_mask"_a = py::none(), "nbeams"_a = 1, py::kw_only(),
            "method"_a = "auto", "tolerance"_a = 1e-6, "guard"_a = 64,
            "backend"_a = "cpu", "device"_a = 0)
        .def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                         const py::array_t<float>& dm_arr, int nthreads,
                         SizeType nbits,
                         std::optional<py::array_t<uint8_t>> kill_mask,
                         SizeType nbeams, std::string_view method,
                         double tolerance, SizeType guard,
                         std::string_view backend, int device) {
                 return DDMTFFT(
                     f_min, f_max, nchans, tsamp,
                     std::span<const float>(dm_arr.data(), dm_arr.size()),
                     make_exec(backend, nthreads, device), nbits,
                     mask_vec(kill_mask), nbeams,
                     make_options(method, tolerance, guard));
             }),
             "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "dm_arr"_a,
             "nthreads"_a = 1, "nbits"_a = 32, "kill_mask"_a = py::none(),
             "nbeams"_a = 1, py::kw_only(), "method"_a = "auto",
             "tolerance"_a = 1e-6, "guard"_a = 64, "backend"_a = "cpu",
             "device"_a = 0)
        .def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                         const LevinConfig& levin, int nthreads, SizeType nbits,
                         std::optional<py::array_t<uint8_t>> kill_mask,
                         SizeType nbeams, std::string_view method,
                         double tolerance, SizeType guard,
                         std::string_view backend, int device) {
                 return DDMTFFT(f_min, f_max, nchans, tsamp, levin,
                                make_exec(backend, nthreads, device), nbits,
                                mask_vec(kill_mask), nbeams,
                                make_options(method, tolerance, guard));
             }),
             "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "levin"_a,
             "nthreads"_a = 1, "nbits"_a = 32, "kill_mask"_a = py::none(),
             "nbeams"_a = 1, py::kw_only(), "method"_a = "auto",
             "tolerance"_a = 1e-6, "guard"_a = 64, "backend"_a = "cpu",
             "device"_a = 0)
        .def(py::init([](const DDMTPlan& plan, int nthreads, SizeType nbeams,
                         std::string_view method, double tolerance,
                         SizeType guard, std::string_view backend, int device) {
                 return DDMTFFT(plan, make_exec(backend, nthreads, device),
                                nbeams, make_options(method, tolerance, guard));
             }),
             "plan"_a, "nthreads"_a = 1, "nbeams"_a = 1, py::kw_only(),
             "method"_a = "auto", "tolerance"_a = 1e-6, "guard"_a = 64,
             "backend"_a = "cpu", "device"_a = 0)
        .def_property_readonly("backend",
                               [](const DDMTFFT& e) {
                                   return std::string(to_string(e.backend()));
                               })
        .def_property_readonly("nthreads", &DDMTFFT::nthreads)
        .def_property_readonly("device", &DDMTFFT::device)
        .def_property_readonly("plan", &DDMTFFT::get_plan)
        .def_property_readonly("nbeams", &DDMTFFT::get_nbeams)
        .def_property_readonly(
            "method",
            [](const DDMTFFT& e) {
                return std::string(
                    algorithms::to_string(e.get_options().method));
            },
            "Method in effect ('brute' or 'nufft').")
        .def_property_readonly(
            "method_used",
            [](const DDMTFFT& e) { return std::string(e.method_used()); },
            "How the channel sum runs: 'nufft', 'piecewise_nufft' or "
            "'brute'.")
        .def_property_readonly(
            "guard", [](const DDMTFFT& e) { return e.get_options().guard; })
        .def_property_readonly(
            "tolerance",
            [](const DDMTFFT& e) { return e.get_options().tolerance; })
        .def_property_readonly("max_delay", &DDMTFFT::get_max_delay)
        .def_property_readonly(
            "suggested_nsamps", &DDMTFFT::get_suggested_nsamps,
            "Block length (samples per call) keeping >= 80% of every "
            "transform as output.")
        .def(
            "execute",
            [](DDMTFFT& e,
               const py::array_t<float, py::array::c_style>& waterfall) {
                const auto ndim = waterfall.ndim();
                if (ndim != 2 && ndim != 3) {
                    throw std::invalid_argument(
                        "DDMTFFT.execute: waterfall must be 2D (nchans, "
                        "nsamps) or 3D (nbeams, nchans, nsamps)");
                }
                const auto nbeams =
                    ndim == 3 ? static_cast<SizeType>(waterfall.shape(0)) : 1;
                if (nbeams != e.get_nbeams()) {
                    throw std::invalid_argument(std::format(
                        "DDMTFFT.execute: waterfall has {} beam(s), but this "
                        "instance was configured for {}",
                        nbeams, e.get_nbeams()));
                }
                const auto nsamps =
                    static_cast<SizeType>(waterfall.shape(ndim - 1));
                auto out = output_array(e, nsamps, ndim == 3);
                e.execute(
                    std::span<const float>(waterfall.data(), waterfall.size()),
                    std::span<float>(out.mutable_data(), out.size()));
                return out;
            },
            "waterfall"_a,
            R"doc(
            Dedisperse a float32 waterfall ``(nchans, nsamps)`` or
            ``(nbeams, nchans, nsamps)``; returns ``(n_dm, output_nsamps)`` or
            ``(nbeams, n_dm, output_nsamps)`` (float32).
            )doc")
        .def(
            "execute",
            [](DDMTFFT& e,
               const py::array_t<uint8_t, py::array::c_style>& packed,
               SizeType nsamps) {
                auto out = output_array(e, nsamps, e.get_nbeams() > 1);
                e.execute(
                    std::span<const uint8_t>(packed.data(), packed.size()),
                    nsamps, std::span<float>(out.mutable_data(), out.size()));
                return out;
            },
            "waterfall_packed"_a, "nsamps"_a,
            "Dedisperse a channel-major packed waterfall (float32 output).")
        .def(
            "execute_time_major",
            [](DDMTFFT& e,
               const py::array_t<uint8_t, py::array::c_style>& packed,
               SizeType nsamps) {
                auto out = output_array(e, nsamps, e.get_nbeams() > 1);
                e.execute_time_major(
                    std::span<const uint8_t>(packed.data(), packed.size()),
                    nsamps, std::span<float>(out.mutable_data(), out.size()));
                return out;
            },
            "filterbank_packed"_a, "nsamps"_a,
            "Dedisperse a time-major packed filterbank (float32 output).")
        .def("get_output_nsamps", &DDMTFFT::get_output_nsamps, "input_nsamps"_a)
        .def("reset_history", &DDMTFFT::reset_history)
        .def("history_state_size", &DDMTFFT::history_state_size)
        .def("set_gulp_size", &DDMTFFT::set_gulp_size, "gulp_size"_a)
        .def_property_readonly("gulp_size", &DDMTFFT::get_gulp_size)
        .def("save_history",
             [](const DDMTFFT& e) {
                 py::array_t<float, py::array::c_style> hist(
                     static_cast<py::ssize_t>(e.history_state_size()));
                 e.save_history(
                     std::span<float>(hist.mutable_data(), hist.size()));
                 return hist;
             })
        .def(
            "load_history",
            [](DDMTFFT& e, const py::array& obj) {
                if (obj.dtype().kind() != 'f' || obj.itemsize() != 4) {
                    throw std::invalid_argument(std::format(
                        "DDMTFFT.load_history: expects a float32 array, got "
                        "dtype '{}'",
                        std::string(py::str(obj.dtype()))));
                }
                auto hist = py::array_t<float, py::array::c_style>(obj);
                e.load_history(
                    std::span<const float>(hist.data(), hist.size()));
            },
            "history"_a);
}

} // namespace dmt
