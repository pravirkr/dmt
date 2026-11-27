#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dmt/algorithms/ddmt_fft.hpp"
#include "dmt/common/fft_config.hpp"
#include "dmt/common/plans.hpp"

#include "bindings/bind.hpp"
#include "dlpack_utils.hpp"
#include "pybind_utils.hpp"

namespace dmt {
using algorithms::DDMTFFT;
using algorithms::DDMTFFTMemoryEstimate;
using algorithms::DDMTFFTMemoryUsage;
using algorithms::DDMTFFTOptions;
using plans::DDMTPlan;
using plans::LevinConfig;

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

namespace {

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
output_array(const DDMTFFT& eng,
             SizeType nsamps,
             bool three_d,
             std::string_view what,
             const std::optional<py::array>& out) {
    const auto ndm =
        static_cast<py::ssize_t>(eng.get_plan().get_container().dm_arr.size());
    const auto nout = static_cast<py::ssize_t>(eng.get_output_nsamps(nsamps));
    if (three_d) {
        return shaped_output<float>(
            {static_cast<py::ssize_t>(eng.get_nbeams()), ndm, nout}, what, out);
    }
    return shaped_output<float>({ndm, nout}, what, out);
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

    py::class_<DDMTFFTMemoryUsage>(mod, "DDMTFFTMemoryUsage", R"doc(
        Memory a DDMTFFT instance holds, in bytes (host memory on the CPU
        backend, device memory on a GPU backend, ``pinned`` apart). Buffers
        are made by the calls that need them and kept (see
        ``DDMTFFT.estimate_memory``). Not counted: input and output arrays,
        the FFT libraries' plan objects (cuFFT work areas are counted), and
        per-call temporaries. ``pinned`` is not included in ``total``.
        )doc")
        .def_readonly("plan", &DDMTFFTMemoryUsage::plan)
        .def_readonly("history", &DDMTFFTMemoryUsage::history)
        .def_readonly("transform", &DDMTFFTMemoryUsage::transform)
        .def_readonly("workspace", &DDMTFFTMemoryUsage::workspace)
        .def_readonly("staging", &DDMTFFTMemoryUsage::staging)
        .def_readonly("pinned", &DDMTFFTMemoryUsage::pinned)
        .def_property_readonly("total", &DDMTFFTMemoryUsage::total)
        .def("__repr__", [](const DDMTFFTMemoryUsage& m) {
            return fmt::format("DDMTFFTMemoryUsage(plan={}, history={}, "
                               "transform={}, workspace={}, staging={}, "
                               "pinned={}, total={})",
                               m.plan, m.history, m.transform, m.workspace,
                               m.staging, m.pinned, m.total());
        });

    py::class_<DDMTFFTMemoryEstimate>(mod, "DDMTFFTMemoryEstimate", R"doc(
        What a DDMTFFT streaming calls of one length would hold, from
        ``DDMTFFT.estimate_memory`` (computed without allocating it).

        ``usage`` equals that instance's ``memory_usage`` once a cold stream
        of such calls is warm (``staging`` and ``pinned`` 0: CPU calls, GPU
        calls on float device arrays). The staging calls of another kind add:
        ``host_staging`` and ``host_pinned`` for host arrays (GPU; 0 on the
        CPU), ``time_major_staging`` for ``execute_time_major`` (packed plans)
        and ``device_staging`` for packed device arrays (GPU).
        )doc")
        .def_readonly("usage", &DDMTFFTMemoryEstimate::usage)
        .def_readonly("host_staging", &DDMTFFTMemoryEstimate::host_staging)
        .def_readonly("host_pinned", &DDMTFFTMemoryEstimate::host_pinned)
        .def_readonly("time_major_staging",
                      &DDMTFFTMemoryEstimate::time_major_staging)
        .def_readonly("device_staging", &DDMTFFTMemoryEstimate::device_staging)
        .def("__repr__", [](const DDMTFFTMemoryEstimate& e) {
            return fmt::format("DDMTFFTMemoryEstimate(total={}, "
                               "host_staging={}, host_pinned={}, "
                               "time_major_staging={}, device_staging={})",
                               e.usage.total(), e.host_staging, e.host_pinned,
                               e.time_major_staging, e.device_staging);
        });

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
            OpenMP threads (CPU) or host-staging worker cap (GPU; default 1).
        nbits : int, optional
            Input precision: 32 (float) or 1, 2, 4, 8, 16 (packed).
        kill_mask : array_like, optional
            Per-channel mask (1 = keep, 0 = drop), as for :class:`DDMT`.
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
            (default 64); ``max_delay`` is ``plan.time_offset`` plus the
            largest positive delay (rounded up) plus ``guard``.
        backend, device, f_ref
            Keyword-only, as for :class:`DDMT`.

        On a GPU backend :meth:`execute` and the history methods also take
        device arrays without a copy, asynchronously on ``stream``, as for
        :class:`DDMT`.

        Transform methods release the GIL during C++/GPU execution; a single
        instance holds mutable streaming history and workspace buffers and is
        not safe for concurrent calls from multiple threads.
        )doc");
    cls.def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                        float dm_max, float dm_step, float dm_min, int nthreads,
                        SizeType nbits, const py::object& kill_mask,
                        SizeType nbeams, std::string_view method,
                        double tolerance, SizeType guard,
                        std::string_view backend, int device,
                        std::optional<float> f_ref) {
                const auto kill_vec = extract_kill_mask(kill_mask);
                // Pure C++ from here: other Python threads run meanwhile.
                const py::gil_scoped_release release;
                return DDMTFFT(f_min, f_max, nchans, tsamp, dm_max, dm_step,
                               dm_min, make_exec(backend, nthreads, device),
                               nbits, kill_vec, nbeams,
                               make_options(method, tolerance, guard), f_ref);
            }),
            "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "dm_max"_a,
            "dm_step"_a, "dm_min"_a = 0.0F, "nthreads"_a = 1, "nbits"_a = 32,
            "kill_mask"_a = py::none(), "nbeams"_a = 1, py::kw_only(),
            "method"_a = "auto", "tolerance"_a = 1e-6, "guard"_a = 64,
            "backend"_a = "cpu", "device"_a = 0, "f_ref"_a = py::none())
        .def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                         const py::array_t<float>& dm_arr, int nthreads,
                         SizeType nbits, const py::object& kill_mask,
                         SizeType nbeams, std::string_view method,
                         double tolerance, SizeType guard,
                         std::string_view backend, int device,
                         std::optional<float> f_ref) {
                 const auto dms      = copy_native_1d<float>(dm_arr, "dm_arr");
                 const auto kill_vec = extract_kill_mask(kill_mask);
                 // Pure C++ from here: other Python threads run meanwhile.
                 const py::gil_scoped_release release;
                 return DDMTFFT(f_min, f_max, nchans, tsamp,
                                std::span<const float>(dms.data(), dms.size()),
                                make_exec(backend, nthreads, device), nbits,
                                kill_vec, nbeams,
                                make_options(method, tolerance, guard), f_ref);
             }),
             "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "dm_arr"_a,
             "nthreads"_a = 1, "nbits"_a = 32, "kill_mask"_a = py::none(),
             "nbeams"_a = 1, py::kw_only(), "method"_a = "auto",
             "tolerance"_a = 1e-6, "guard"_a = 64, "backend"_a = "cpu",
             "device"_a = 0, "f_ref"_a = py::none())
        .def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                         const LevinConfig& levin, int nthreads, SizeType nbits,
                         const py::object& kill_mask, SizeType nbeams,
                         std::string_view method, double tolerance,
                         SizeType guard, std::string_view backend, int device,
                         std::optional<float> f_ref) {
                 const auto kill_vec = extract_kill_mask(kill_mask);
                 // Pure C++ from here: other Python threads run meanwhile.
                 const py::gil_scoped_release release;
                 return DDMTFFT(f_min, f_max, nchans, tsamp, levin,
                                make_exec(backend, nthreads, device), nbits,
                                kill_vec, nbeams,
                                make_options(method, tolerance, guard), f_ref);
             }),
             "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "levin"_a,
             "nthreads"_a = 1, "nbits"_a = 32, "kill_mask"_a = py::none(),
             "nbeams"_a = 1, py::kw_only(), "method"_a = "auto",
             "tolerance"_a = 1e-6, "guard"_a = 64, "backend"_a = "cpu",
             "device"_a = 0, "f_ref"_a = py::none())
        .def(py::init([](const DDMTPlan& plan, int nthreads, SizeType nbeams,
                         std::string_view method, double tolerance,
                         SizeType guard, std::string_view backend, int device) {
                 // Pure C++ from here: other Python threads run meanwhile.
                 const py::gil_scoped_release release;
                 return DDMTFFT(plan, make_exec(backend, nthreads, device),
                                nbeams, make_options(method, tolerance, guard));
             }),
             "plan"_a, "nthreads"_a = 1, "nbeams"_a = 1, py::kw_only(),
             "method"_a = "auto", "tolerance"_a = 1e-6, "guard"_a = 64,
             "backend"_a = "cpu", "device"_a = 0)
        .def_static(
            "estimate_memory",
            [](const DDMTPlan& plan, SizeType nsamps, int nthreads,
               SizeType nbeams, std::string_view method, double tolerance,
               SizeType guard, std::string_view backend, int device) {
                // Pure C++ from here: other Python threads run meanwhile.
                const py::gil_scoped_release release;
                return DDMTFFT::estimate_memory(
                    plan, nsamps, make_exec(backend, nthreads, device), nbeams,
                    make_options(method, tolerance, guard));
            },
            "plan"_a, "nsamps"_a, "nthreads"_a = 1, "nbeams"_a = 1,
            py::kw_only(), "method"_a = "auto", "tolerance"_a = 1e-6,
            "guard"_a = 64, "backend"_a = "cpu", "device"_a = 0,
            R"doc(
            Memory a ``DDMTFFT(plan, nthreads, nbeams, ...)`` would hold once
            a stream of calls of ``nsamps`` input samples each is warm,
            without allocating it (:class:`DDMTFFTMemoryEstimate`).

            Buffers are made by the first calls that need them and kept: each
            transform length a call uses adds its spectra, up to four
            lengths. A stream started cold (or after ``reset_history``) uses
            the steady-state length and those of its shorter first outputs;
            the estimate counts exactly those. Exact: ``memory_usage`` of
            such an instance equals ``usage`` plus the staging of the calls
            made. GPU backends make their cuFFT plans (for their work-area
            sizes) but allocate nothing.
            )doc")
        .def_property_readonly(
            "memory_usage", &DDMTFFT::get_memory_usage,
            "DDMTFFTMemoryUsage: bytes held now (buffers are made by the "
            "calls that need them; see estimate_memory).")
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
            [](DDMTFFT& e, const py::array& waterfall_obj,
               const std::optional<py::array>& out_arr) {
                const auto waterfall = require_contiguous_array<float>(
                    waterfall_obj, "DDMTFFT.execute", "float32");
                const auto ndim = waterfall.ndim();
                check_input_shape("DDMTFFT.execute", "waterfall", waterfall,
                                  e.get_nbeams(), e.get_plan().get_nchans(), 0,
                                  "nchans, nsamps");
                const auto nsamps =
                    static_cast<SizeType>(waterfall.shape(ndim - 1));
                auto out = output_array(e, nsamps, ndim == 3, "DDMTFFT.execute",
                                        out_arr);
                const std::span<const float> in_span(waterfall.data(),
                                                     waterfall.size());
                const std::span<float> out_span(out.mutable_data(), out.size());
                {
                    const py::gil_scoped_release release;
                    e.execute(in_span, out_span);
                }
                return out;
            },
            "waterfall"_a.noconvert(), py::kw_only(), "out"_a = py::none(),
            R"doc(
            Dedisperse a float32 waterfall ``(nchans, nsamps)`` or
            ``(nbeams, nchans, nsamps)``; returns ``(n_dm, output_nsamps)`` or
            ``(nbeams, n_dm, output_nsamps)`` (float32). Keyword-only ``out``:
            a writeable, C-contiguous float32 array of at least that many
            elements, filled and returned as that shape (all host overloads).
            )doc")
        .def(
            "execute",
            [](DDMTFFT& e, const py::array& packed_obj, SizeType nsamps,
               const std::optional<py::array>& out_arr) {
                const auto packed = require_contiguous_array<uint8_t>(
                    packed_obj, "DDMTFFT.execute", "uint8");
                check_ddmt_packed_shape(
                    e, "DDMTFFT.execute", "waterfall_packed",
                    std::span<const py::ssize_t>(
                        packed.shape(), static_cast<SizeType>(packed.ndim())),
                    nsamps, false);
                auto out = output_array(e, nsamps, e.get_nbeams() > 1,
                                        "DDMTFFT.execute", out_arr);
                const std::span<const uint8_t> in_span(packed.data(),
                                                       packed.size());
                const std::span<float> out_span(out.mutable_data(), out.size());
                {
                    const py::gil_scoped_release release;
                    e.execute(in_span, nsamps, out_span);
                }
                return out;
            },
            "waterfall_packed"_a.noconvert(), py::kw_only(), "nsamps"_a,
            "out"_a = py::none(),
            "Dedisperse a channel-major packed waterfall (float32 output).")
        .def(
            "execute_time_major",
            [](DDMTFFT& e, const py::array& packed_obj, SizeType nsamps,
               const std::optional<py::array>& out_arr) {
                const auto packed = require_contiguous_array<uint8_t>(
                    packed_obj, "DDMTFFT.execute_time_major", "uint8");
                check_ddmt_packed_shape(
                    e, "DDMTFFT.execute_time_major", "filterbank_packed",
                    std::span<const py::ssize_t>(
                        packed.shape(), static_cast<SizeType>(packed.ndim())),
                    nsamps, true);
                auto out = output_array(e, nsamps, e.get_nbeams() > 1,
                                        "DDMTFFT.execute_time_major", out_arr);
                const std::span<const uint8_t> in_span(packed.data(),
                                                       packed.size());
                const std::span<float> out_span(out.mutable_data(), out.size());
                {
                    const py::gil_scoped_release release;
                    e.execute_time_major(in_span, nsamps, out_span);
                }
                return out;
            },
            "filterbank_packed"_a.noconvert(), py::kw_only(), "nsamps"_a,
            "out"_a = py::none(),
            "Dedisperse a time-major packed filterbank (float32 output).")
        .def(
            "execute",
            [](DDMTFFT& e, const DeviceArray& waterfall, const py::object& out,
               const py::object& stream) {
                const DeviceCall call =
                    device_call("DDMTFFT.execute", e.backend(), e.device(),
                                waterfall, stream);
                const auto input =
                    call.consume<float>(waterfall.obj, "waterfall", false);
                const auto nsamps = ddmt_waterfall_nsamps(e, call, input);
                return ddmt_execute_device<float, float>(
                    e, call, waterfall.obj, input, nsamps,
                    input.tensor().ndim == 3, out,
                    [&](DeviceSpan<const float> in, DeviceSpan<float> dmt,
                        Stream s) { e.execute(in, dmt, s); });
            },
            "waterfall"_a, py::kw_only(), "out"_a = py::none(),
            "stream"_a = py::none(),
            "Dedisperse a float32 device array (GPU backends) without "
            "copies; ``out`` and ``stream`` as for :meth:`DDMT.execute`.")
        .def(
            "execute",
            [](DDMTFFT& e, const DeviceArray& packed, SizeType nsamps,
               const py::object& out, const py::object& stream) {
                const DeviceCall call = device_call(
                    "DDMTFFT.execute", e.backend(), e.device(), packed, stream);
                const auto input =
                    call.consume<uint8_t>(packed.obj, "waterfall", false);
                const auto& dl = input.tensor();
                check_ddmt_packed_shape(
                    e, "DDMTFFT.execute", "waterfall_packed",
                    std::span<const int64_t>(dl.shape,
                                             static_cast<SizeType>(dl.ndim)),
                    nsamps, false);
                return ddmt_execute_device<uint8_t, float>(
                    e, call, packed.obj, input, nsamps, e.get_nbeams() > 1, out,
                    [&](DeviceSpan<const uint8_t> in, DeviceSpan<float> dmt,
                        Stream s) { e.execute(in, nsamps, dmt, s); });
            },
            "waterfall_packed"_a, py::kw_only(), "nsamps"_a,
            "out"_a = py::none(), "stream"_a = py::none(),
            "Dedisperse a channel-major packed uint8 device array (float32 "
            "output).")
        .def("get_output_nsamps", &DDMTFFT::get_output_nsamps, "input_nsamps"_a)
        .def("reset_history", &DDMTFFT::reset_history)
        .def("history_state_size", &DDMTFFT::history_state_size)
        .def("save_history",
             [](const DDMTFFT& e) {
                 py::array_t<float, py::array::c_style> hist(
                     static_cast<py::ssize_t>(e.history_state_size()));
                 const std::span<float> out(hist.mutable_data(), hist.size());
                 {
                     const py::gil_scoped_release release;
                     e.save_history(out);
                 }
                 return hist;
             })
        .def(
            "load_history",
            [](DDMTFFT& e, const py::array& obj) {
                const auto hist = require_contiguous_array<float>(
                    obj, "DDMTFFT.load_history", "float32");
                const std::span<const float> in(hist.data(), hist.size());
                const py::gil_scoped_release release;
                e.load_history(in);
            },
            py::arg("history").noconvert())
        .def(
            "save_history",
            [](const DDMTFFT& e, const DeviceArray& out,
               const py::object& stream) {
                return save_history_device<float>(e, "DDMTFFT.save_history",
                                                  out, stream);
            },
            "out"_a, py::kw_only(), "stream"_a = py::none(),
            "Save the streaming history into a float32 device array (GPU "
            "backends, asynchronous on ``stream``). Returns ``out``.")
        .def(
            "load_history",
            [](DDMTFFT& e, const DeviceArray& history,
               const py::object& stream) {
                load_history_device<float>(e, "DDMTFFT.load_history", history,
                                           stream);
            },
            "history"_a, py::kw_only(), "stream"_a = py::none(),
            "Restore the streaming history from a float32 device array (GPU "
            "backends, asynchronous on ``stream``).");
}

} // namespace dmt
