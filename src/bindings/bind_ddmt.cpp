#include <algorithm>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/algorithms/sdmt.hpp"
#include "dmt/common/plans.hpp"

#include "bindings/bind.hpp"
#include "dlpack_utils.hpp"
#include "pybind_utils.hpp"

namespace dmt {
using algorithms::DDMT;
using algorithms::DDMTMemoryEstimate;
using algorithms::DDMTMemoryUsage;
using algorithms::SDMT;
using plans::DDMTPlan;
using plans::LevinConfig;

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

namespace {

// Device-memory packed execute() / execute_time_major(): int32 output,
// beam-major 3D at nbeams > 1 like the host overloads.
template <typename Run>
py::object ddmt_packed_device(DDMT& ddmt,
                              std::string_view what,
                              const DeviceArray& packed,
                              SizeType nsamps,
                              const py::object& out,
                              const py::object& stream,
                              Run&& run) {
    const DeviceCall call =
        device_call(what, ddmt.backend(), ddmt.device(), packed, stream);
    const auto input = call.consume<uint8_t>(packed.obj, "waterfall", false);
    return ddmt_execute_device<uint8_t, int32_t>(
        ddmt, call, packed.obj, input, nsamps, ddmt.get_nbeams() > 1, out,
        [&](DeviceSpan<const uint8_t> in, DeviceSpan<int32_t> dmt, Stream s) {
            run(in, nsamps, dmt, s);
        });
}

} // namespace

/// Binds the constructor set shared by DDMT and SDMT.
template <typename T, typename Cls> void def_ddmt_inits(Cls& cls) {
    cls.def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                        float dm_max, float dm_step, float dm_min, int nthreads,
                        SizeType nbits, const py::object& kill_mask,
                        SizeType nbeams, std::string_view backend, int device,
                        std::optional<float> f_ref) {
                const auto km_vec = extract_kill_mask(kill_mask);
                return T(f_min, f_max, nchans, tsamp, dm_max, dm_step, dm_min,
                         make_exec(backend, nthreads, device), nbits, km_vec,
                         nbeams, f_ref);
            }),
            "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "dm_max"_a,
            "dm_step"_a, "dm_min"_a = 0.0F, "nthreads"_a = 1, "nbits"_a = 32,
            "kill_mask"_a = py::none(), "nbeams"_a = 1, py::kw_only(),
            "backend"_a = "cpu", "device"_a = 0, "f_ref"_a = py::none())
        .def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                         const py::array_t<float>& dm_arr, int nthreads,
                         SizeType nbits, const py::object& kill_mask,
                         SizeType nbeams, std::string_view backend, int device,
                         std::optional<float> f_ref) {
                 const auto km_vec = extract_kill_mask(kill_mask);
                 const auto dms    = copy_native_1d<float>(dm_arr, "dm_arr");
                 return T(f_min, f_max, nchans, tsamp,
                          std::span<const float>(dms.data(), dms.size()),
                          make_exec(backend, nthreads, device), nbits, km_vec,
                          nbeams, f_ref);
             }),
             "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "dm_arr"_a,
             "nthreads"_a = 1, "nbits"_a = 32, "kill_mask"_a = py::none(),
             "nbeams"_a = 1, py::kw_only(), "backend"_a = "cpu", "device"_a = 0,
             "f_ref"_a = py::none())
        .def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                         const LevinConfig& levin, int nthreads, SizeType nbits,
                         const py::object& kill_mask, SizeType nbeams,
                         std::string_view backend, int device,
                         std::optional<float> f_ref) {
                 const auto km_vec = extract_kill_mask(kill_mask);
                 return T(f_min, f_max, nchans, tsamp, levin,
                          make_exec(backend, nthreads, device), nbits, km_vec,
                          nbeams, f_ref);
             }),
             "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "levin"_a,
             "nthreads"_a = 1, "nbits"_a = 32, "kill_mask"_a = py::none(),
             "nbeams"_a = 1, py::kw_only(), "backend"_a = "cpu", "device"_a = 0,
             "f_ref"_a = py::none())
        .def(py::init([](const DDMTPlan& plan, int nthreads, SizeType nbeams,
                         std::string_view backend, int device) {
                 return T(plan, make_exec(backend, nthreads, device), nbeams);
             }),
             "plan"_a, "nthreads"_a = 1, "nbeams"_a = 1, py::kw_only(),
             "backend"_a = "cpu", "device"_a = 0);
    cls.def_static(
        "estimate_memory",
        [](const DDMTPlan& plan, SizeType nsamps, int nthreads, SizeType nbeams,
           SizeType gulp_size, std::string_view backend, int device) {
            return T::estimate_memory(plan, nsamps,
                                      make_exec(backend, nthreads, device),
                                      nbeams, gulp_size);
        },
        "plan"_a, "nsamps"_a, "nthreads"_a = 1, "nbeams"_a = 1,
        "gulp_size"_a = 0, py::kw_only(), "backend"_a = "cpu", "device"_a = 0,
        R"doc(
        Memory an instance built from ``plan`` with these arguments (and
        ``set_gulp_size(gulp_size)``) holds when making calls of ``nsamps``
        input samples, without allocating it (:class:`DDMTMemoryEstimate`).

        Exact: ``memory_usage`` of such an instance equals ``usage`` plus the
        staging fields of the kind of call made. On the CPU nothing depends
        on ``nsamps``; a GPU backend allocates nothing.
        )doc");
}

void bind_ddmt(py::module_& mod) {
    py::class_<DDMTMemoryUsage>(mod, "DDMTMemoryUsage", R"doc(
        Memory a DDMT or SDMT instance holds, in bytes (host memory on the
        CPU backend, device memory on a GPU backend; ``pinned`` is host
        memory and not in ``total``). Plan tables and history are made at
        construction; on a GPU the staging of host-memory and time-major
        calls is made by the first such call and kept. Not counted: input
        and output arrays and per-call temporaries.
        )doc")
        .def_readonly("plan", &DDMTMemoryUsage::plan)
        .def_readonly("history", &DDMTMemoryUsage::history)
        .def_readonly("staging", &DDMTMemoryUsage::staging)
        .def_readonly("pinned", &DDMTMemoryUsage::pinned)
        .def_property_readonly("total", &DDMTMemoryUsage::total)
        .def("__repr__", [](const DDMTMemoryUsage& m) {
            return fmt::format("DDMTMemoryUsage(plan={}, history={}, "
                               "staging={}, pinned={}, total={})",
                               m.plan, m.history, m.staging, m.pinned,
                               m.total());
        });

    py::class_<DDMTMemoryEstimate>(mod, "DDMTMemoryEstimate", R"doc(
        What a DDMT or SDMT making calls of one length would hold, from
        ``estimate_memory`` (computed without allocating it).

        ``usage`` equals that instance's ``memory_usage`` after
        construction, and after any calls on the CPU or with channel-major
        device arrays on a GPU. On a GPU, host-array calls add
        ``host_staging`` device and ``host_pinned`` pinned bytes;
        ``execute_time_major`` with host arrays adds ``time_major_staging``
        and ``time_major_pinned``, with device arrays
        ``device_time_major_staging`` (packed plans).
        )doc")
        .def_readonly("usage", &DDMTMemoryEstimate::usage)
        .def_readonly("host_staging", &DDMTMemoryEstimate::host_staging)
        .def_readonly("host_pinned", &DDMTMemoryEstimate::host_pinned)
        .def_readonly("time_major_staging",
                      &DDMTMemoryEstimate::time_major_staging)
        .def_readonly("time_major_pinned",
                      &DDMTMemoryEstimate::time_major_pinned)
        .def_readonly("device_time_major_staging",
                      &DDMTMemoryEstimate::device_time_major_staging)
        .def("__repr__", [](const DDMTMemoryEstimate& e) {
            return fmt::format("DDMTMemoryEstimate(total={}, host_staging={}, "
                               "host_pinned={}, time_major_staging={}, "
                               "time_major_pinned={}, "
                               "device_time_major_staging={})",
                               e.usage.total(), e.host_staging, e.host_pinned,
                               e.time_major_staging, e.time_major_pinned,
                               e.device_time_major_staging);
        });

    py::class_<DDMT> ddmt_cls(mod, "DDMT",
                              R"doc(
        Direct Dispersion Measure Transform (DDMT) incoherent dedispersion.

        Supports float32 waterfalls, packed low-bit integers (1, 2, 4, 8, 16 bits),
        channel kill masks, streaming history across consecutive chunks, and both
        channel-major and time-major filterbank layouts.

        Parameters
        ----------
        f_min, f_max : float
            Band edges in MHz. Channel ``c`` is centred at
            ``f_min + (c + 0.5) * df``, ``df = (f_max - f_min) / nchans``, and
            is sampled at that centre frequency. Output sample ``t`` (from
            the start of the stream) is the sum for a signal arriving at
            ``plan.f_ref`` at input sample ``t + plan.time_offset``. Rows must be ascending in
            frequency; for a SIGPROC header (``fch1`` a channel centre,
            usually ``foff < 0``) use :func:`dmtlib.sigproc_band` to get the
            edges and flip the waterfall.
        nchans : int
            Number of frequency channels.
        tsamp : float
            Sampling interval in seconds.
        dm_max, dm_step : float
            Linear DM trial grid specification.
        dm_min : float, optional
            Lowest DM trial (default 0).
        dm_arr : numpy.ndarray, optional
            Explicit DM trial array.
        levin : LevinConfig, optional
            Lina Levin pulse-broadening tolerance DM grid parameters.
        plan : DDMTPlan, optional
            Pre-configured DDMT plan.
        nthreads : int, optional
            OpenMP threads (CPU) or host-staging worker cap (GPU; default 1).
        nbits : int, optional
            Bit precision per sample (32 for float, or 1, 2, 4, 8, 16 for packed integer).
        kill_mask : array_like, optional
            Per-channel mask of bool or any integer dtype (1 = keep,
            0 = drop). Any other nonzero value is kept and stored as 1;
            floats raise ``TypeError``. A non-contiguous view is read in
            logical order.
        nbeams : int, optional
            Number of independent beams batched through this instance
            (default 1). At nbeams > 1, execute()/execute_time_major() take
            beam-major arrays (nbeams, nchans, nsamps) and return
            (nbeams, n_dm, output_nsamps); at nbeams == 1 the plain 2D
            shapes still work.
        backend : {'cpu', 'cuda', 'hip'}, optional
            Keyword-only. Where to run (default ``'cpu'``); see
            :func:`available_backends`.
        device : int, optional
            Keyword-only. Device ordinal on a GPU backend (default 0).
        f_ref : float, optional
            Keyword-only. Reference frequency of the output times in MHz
            (default ``None``: ``f_min``, the bottom band edge, as for FDMT;
            best for real-time triggering) or ``f_max``, the top band edge;
            any other value raises ``ValueError``. ``f_max`` gives
            ``plan.time_offset == 0`` (the SIGPROC/PRESTO convention of
            series that start at the file's first sample), so output sample
            ``t`` refers to input sample ``t`` and the incomplete samples are
            at the end. Only the delay table changes; speed and history are
            the same.

        NumPy (host) arrays work on every backend; a GPU backend pipelines
        the host transfers and blocks until the result is on the host. On a
        GPU backend the execute and history methods also take device arrays
        (PyTorch, CuPy, Numba, or any other ``__dlpack__`` or
        ``__cuda_array_interface__`` array on this instance's device)
        without a copy, asynchronously on ``stream``, as for
        :meth:`FDMT.execute`.

        Transform methods release the GIL during C++/GPU execution; a single
        instance holds mutable streaming history and staging buffers and is
        not safe for concurrent calls from multiple threads.
        )doc");
    def_ddmt_inits<DDMT>(ddmt_cls);
    ddmt_cls
        .def_property_readonly(
            "backend",
            [](const DDMT& ddmt) {
                return std::string(to_string(ddmt.backend()));
            },
            "Backend this instance runs on ('cpu', 'cuda', ...).")
        .def_property_readonly("nthreads", &DDMT::nthreads)
        .def_property_readonly("device", &DDMT::device)
        .def_property_readonly("plan", &DDMT::get_plan)
        .def_property_readonly("nbeams", &DDMT::get_nbeams)
        .def_property_readonly(
            "memory_usage", &DDMT::get_memory_usage,
            "DDMTMemoryUsage: bytes held now (see estimate_memory).")
        .def(
            "execute",
            [](DDMT& ddmt, const py::array& waterfall_obj) {
                const auto waterfall = require_contiguous_array<float>(
                    waterfall_obj, "DDMT.execute", "float32");
                const auto& plan = ddmt.get_plan();
                const auto ndim  = waterfall.ndim();
                if (ndim != 2 && ndim != 3) {
                    throw std::invalid_argument(
                        "DDMT.execute: waterfall must be 2D (nchans, "
                        "nsamps) or 3D (nbeams, nchans, nsamps)");
                }
                const auto* shape = waterfall.shape();
                const auto nbeams =
                    ndim == 3 ? static_cast<SizeType>(shape[0]) : 1;
                const auto nsamps_in = static_cast<SizeType>(shape[ndim - 1]);
                const auto nchans    = plan.get_nchans();
                if (nbeams != ddmt.get_nbeams()) {
                    throw std::invalid_argument(fmt::format(
                        "DDMT.execute: waterfall has {} beam(s), but "
                        "this instance was configured for {}",
                        nbeams, ddmt.get_nbeams()));
                }
                if (static_cast<SizeType>(shape[ndim - 2]) != nchans) {
                    throw std::invalid_argument(fmt::format(
                        "DDMT.execute: waterfall has {} channels, but "
                        "this instance was configured for {}",
                        shape[ndim - 2], nchans));
                }
                const auto checked_mul = [](SizeType a, SizeType b) {
                    if (b != 0 &&
                        a > std::numeric_limits<SizeType>::max() / b) {
                        throw std::invalid_argument(
                            "DDMT.execute: waterfall shape is too large");
                    }
                    return a * b;
                };
                const auto expect =
                    checked_mul(checked_mul(nbeams, nchans), nsamps_in);
                if (static_cast<SizeType>(waterfall.size()) != expect) {
                    throw std::invalid_argument(fmt::format(
                        "DDMT.execute: waterfall has {} elements, expected "
                        "nbeams * nchans * nsamps = {}",
                        waterfall.size(), expect));
                }
                const auto nsamps_out = ddmt.get_output_nsamps(nsamps_in);
                const auto dm_count   = plan.get_container().dm_arr.size();

                py::array_t<float, py::array::c_style> dmt(
                    ndim == 3
                        ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(
                                                       nbeams),
                                                   static_cast<py::ssize_t>(
                                                       dm_count),
                                                   static_cast<py::ssize_t>(
                                                       nsamps_out)}
                        : std::vector<py::ssize_t>{
                              static_cast<py::ssize_t>(dm_count),
                              static_cast<py::ssize_t>(nsamps_out)});
                const std::span<const float> in_span(waterfall.data(),
                                                     waterfall.size());
                const std::span<float> out_span =
                    nsamps_out > 0
                        ? std::span<float>(dmt.mutable_data(), dmt.size())
                        : std::span<float>{};
                {
                    const py::gil_scoped_release release;
                    ddmt.execute(in_span, out_span);
                }
                return dmt;
            },
            "waterfall"_a.noconvert(),
            R"doc(
            Dedisperse a float32 waterfall of shape ``(nchans, nsamps)``, or
            ``(nbeams, nchans, nsamps)`` if this instance's nbeams > 1.
            Produces ``(n_dm, output_nsamps)`` or ``(nbeams, n_dm, output_nsamps)``.
            )doc")
        .def(
            "execute",
            [](DDMT& ddmt, const py::array& waterfall_packed_obj,
               SizeType nsamps) {
                const auto waterfall_packed = require_contiguous_array<uint8_t>(
                    waterfall_packed_obj, "DDMT.execute", "uint8");
                const auto& plan      = ddmt.get_plan();
                const auto& plan_c    = plan.get_container();
                const auto nsamps_out = ddmt.get_output_nsamps(nsamps);
                const auto dm_count   = plan_c.dm_arr.size();
                const auto nbeams     = ddmt.get_nbeams();

                py::array_t<int32_t, py::array::c_style> dmt(
                    nbeams > 1
                        ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(
                                                       nbeams),
                                                   static_cast<py::ssize_t>(
                                                       dm_count),
                                                   static_cast<py::ssize_t>(
                                                       nsamps_out)}
                        : std::vector<py::ssize_t>{
                              static_cast<py::ssize_t>(dm_count),
                              static_cast<py::ssize_t>(nsamps_out)});
                const std::span<const uint8_t> in_span(waterfall_packed.data(),
                                                       waterfall_packed.size());
                const std::span<int32_t> out_span =
                    nsamps_out > 0
                        ? std::span<int32_t>(dmt.mutable_data(), dmt.size())
                        : std::span<int32_t>{};
                {
                    const py::gil_scoped_release release;
                    ddmt.execute(in_span, nsamps, out_span);
                }
                return dmt;
            },
            "waterfall_packed"_a.noconvert(), py::kw_only(), "nsamps"_a,
            R"doc(
            Dedisperse a channel-major packed integer waterfall of shape
            ``(nchans, row_bytes)``, or ``(nbeams, nchans, row_bytes)`` if
            this instance's nbeams > 1.
            Produces ``(n_dm, output_nsamps)`` or
            ``(nbeams, n_dm, output_nsamps)``.
            )doc")
        .def(
            "execute_time_major",
            [](DDMT& ddmt, const py::array& filterbank_packed_obj,
               SizeType nsamps) {
                const auto filterbank_packed =
                    require_contiguous_array<uint8_t>(filterbank_packed_obj,
                                                      "DDMT.execute_time_major",
                                                      "uint8");
                const auto& plan_c    = ddmt.get_plan().get_container();
                const auto nsamps_out = ddmt.get_output_nsamps(nsamps);
                const auto dm_count   = plan_c.dm_arr.size();
                const auto nbeams     = ddmt.get_nbeams();

                py::array_t<int32_t, py::array::c_style> dmt(
                    nbeams > 1
                        ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(
                                                       nbeams),
                                                   static_cast<py::ssize_t>(
                                                       dm_count),
                                                   static_cast<py::ssize_t>(
                                                       nsamps_out)}
                        : std::vector<py::ssize_t>{
                              static_cast<py::ssize_t>(dm_count),
                              static_cast<py::ssize_t>(nsamps_out)});
                // Always called, even with no output yet: the block still
                // feeds the stream history.
                const std::span<const uint8_t> in_span(
                    filterbank_packed.data(), filterbank_packed.size());
                const std::span<int32_t> out_span(dmt.mutable_data(),
                                                  dmt.size());
                {
                    const py::gil_scoped_release release;
                    ddmt.execute_time_major(in_span, nsamps, out_span);
                }
                return dmt;
            },
            "filterbank_packed"_a.noconvert(), py::kw_only(), "nsamps"_a,
            R"doc(
            Dedisperse a time-major packed filterbank of shape
            ``(nsamps, samp_bytes)``, or ``(nbeams, nsamps, samp_bytes)`` if
            this instance's nbeams > 1.
            Produces ``(n_dm, output_nsamps)`` or
            ``(nbeams, n_dm, output_nsamps)``, where ``output_nsamps =
            get_output_nsamps(nsamps)``. Shares the stream history with
            :meth:`execute`: consecutive calls continue one stream; call
            :meth:`reset_history` to start a new one.
            )doc")
        .def(
            "execute",
            [](DDMT& ddmt, const DeviceArray& waterfall, const py::object& out,
               const py::object& stream) {
                const DeviceCall call =
                    device_call("DDMT.execute", ddmt.backend(), ddmt.device(),
                                waterfall, stream);
                const auto input =
                    call.consume<float>(waterfall.obj, "waterfall", false);
                const auto nsamps = ddmt_waterfall_nsamps(ddmt, call, input);
                return ddmt_execute_device<float, float>(
                    ddmt, call, waterfall.obj, input, nsamps,
                    input.tensor().ndim == 3, out,
                    [&](DeviceSpan<const float> in, DeviceSpan<float> dmt,
                        Stream s) { ddmt.execute(in, dmt, s); });
            },
            "waterfall"_a, py::kw_only(), "out"_a = py::none(),
            "stream"_a = py::none(),
            R"doc(
            Dedisperse device memory (GPU backends), without copies.

            Parameters
            ----------
            waterfall : array
                C-contiguous float32 PyTorch, CuPy, Numba or other
                ``__dlpack__`` / ``__cuda_array_interface__`` array on this
                instance's device, shaped as for NumPy input.
                Another dtype or layout raises ``TypeError``.
            out : array, optional
                Writeable C-contiguous device array of at least
                ``nbeams * n_dm * output_nsamps`` elements (float32; int32
                for the packed overloads). Default: a new array from the
                input's library.
            stream : int or stream, optional
                ``cudaStream_t`` / ``hipStream_t`` handle as an int, or a
                ``torch.cuda.Stream`` / ``cupy.cuda.Stream`` / Numba stream.
                Default: the current stream of the input's library on its
                device, as for :meth:`FDMT.execute`.

            Returns
            -------
            array
                Device view ``(n_dm, output_nsamps)`` or ``(nbeams, n_dm,
                output_nsamps)`` of the output.

            Notes
            -----
            Asynchronous on ``stream``, with the stream history kept on the
            device. The packed :meth:`execute` and
            :meth:`execute_time_major` take ``(waterfall_packed, nsamps=...)``
            uint8 device arrays the same way.
            )doc")
        .def(
            "execute",
            [](DDMT& ddmt, const DeviceArray& waterfall_packed, SizeType nsamps,
               const py::object& out, const py::object& stream) {
                return ddmt_packed_device(
                    ddmt, "DDMT.execute", waterfall_packed, nsamps, out, stream,
                    [&](DeviceSpan<const uint8_t> in, SizeType n,
                        DeviceSpan<int32_t> dmt,
                        Stream s) { ddmt.execute(in, n, dmt, s); });
            },
            "waterfall_packed"_a, py::kw_only(), "nsamps"_a,
            "out"_a = py::none(), "stream"_a = py::none(),
            "Packed analogue of the device-memory execute() (int32 output).")
        .def(
            "execute_time_major",
            [](DDMT& ddmt, const DeviceArray& filterbank_packed,
               SizeType nsamps, const py::object& out,
               const py::object& stream) {
                return ddmt_packed_device(
                    ddmt, "DDMT.execute_time_major", filterbank_packed, nsamps,
                    out, stream,
                    [&](DeviceSpan<const uint8_t> in, SizeType n,
                        DeviceSpan<int32_t> dmt,
                        Stream s) { ddmt.execute_time_major(in, n, dmt, s); });
            },
            "filterbank_packed"_a, py::kw_only(), "nsamps"_a,
            "out"_a = py::none(), "stream"_a = py::none(),
            "Device-memory execute_time_major() (GPU backends, int32 output).")
        .def("set_gulp_size", &DDMT::set_gulp_size, "gulp_size"_a,
             R"doc(
            Set the chunk length (input samples) that host-memory calls on a
            GPU backend stream through the device. 0 restores the default.
            Results do not depend on it; the CPU backend ignores it.
            )doc")
        .def_property_readonly("gulp_size", &DDMT::get_gulp_size)
        .def("get_output_nsamps", &DDMT::get_output_nsamps, "input_nsamps"_a)
        .def("reset_history", &DDMT::reset_history)
        .def("history_state_size", &DDMT::history_state_size)
        .def("save_history",
             [](const DDMT& ddmt) -> py::object {
                 const auto sz   = ddmt.history_state_size();
                 const auto save = [&](auto tag) -> py::object {
                     using T = decltype(tag);
                     py::array_t<T, py::array::c_style> hist(
                         static_cast<py::ssize_t>(sz));
                     const std::span<T> out(hist.mutable_data(), hist.size());
                     {
                         const py::gil_scoped_release release;
                         ddmt.save_history(out);
                     }
                     return hist;
                 };
                 return ddmt.get_plan().get_nbits() == 32 ? save(float{})
                                                          : save(uint8_t{});
             })
        .def(
            "load_history",
            [](DDMT& ddmt, const py::array& hist_obj) {
                // Exact dtype and C order, as every host input: plan nbits
                // 32 keeps float32 history, packed plans uint8.
                const auto load = [&](auto tag, const char* dtype_name) {
                    using T         = decltype(tag);
                    const auto hist = require_contiguous_array<T>(
                        hist_obj, "DDMT.load_history", dtype_name);
                    const std::span<const T> in(hist.data(), hist.size());
                    const py::gil_scoped_release release;
                    ddmt.load_history(in);
                };
                if (ddmt.get_plan().get_nbits() == 32) {
                    load(float{}, "float32");
                } else {
                    load(uint8_t{}, "uint8");
                }
            },
            py::arg("history").noconvert())
        .def(
            "save_history",
            [](const DDMT& ddmt, const DeviceArray& out,
               const py::object& stream) {
                return ddmt.get_plan().get_nbits() == 32
                           ? save_history_device<float>(
                                 ddmt, "DDMT.save_history", out, stream)
                           : save_history_device<uint8_t>(
                                 ddmt, "DDMT.save_history", out, stream);
            },
            "out"_a, py::kw_only(), "stream"_a = py::none(),
            "Save the streaming history into ``out``, a device array of "
            "history_state_size() elements (float32 at nbits=32, else uint8; "
            "GPU backends, asynchronous on ``stream``). Returns ``out``.")
        .def(
            "load_history",
            [](DDMT& ddmt, const DeviceArray& history,
               const py::object& stream) {
                if (ddmt.get_plan().get_nbits() == 32) {
                    load_history_device<float>(ddmt, "DDMT.load_history",
                                               history, stream);
                } else {
                    load_history_device<uint8_t>(ddmt, "DDMT.load_history",
                                                 history, stream);
                }
            },
            "history"_a, py::kw_only(), "stream"_a = py::none(),
            "Restore the streaming history from a device array (GPU "
            "backends, asynchronous on ``stream``).");

    py::class_<SDMT, DDMT> sdmt_cls(mod, "SDMT",
                                    R"doc(
        Subband-shared Dispersion Measure Transform (SDMT).

        Computes exactly the DDMT sums, with the same delay table, DM grid
        and kill mask, but shares partial sums between DM trials: within
        16-channel subbands, trials whose integer delays agree (relative to
        a per-trial base) have identical partial sums, computed once.
        Nothing is approximated. Integer output is bit-identical to DDMT;
        float32 output differs only by rounding (order of additions).
        Dense DM grids save the most additions (~7x fewer additions at 4096
        channels and 2049 trials: ~3.5x faster than DDMT on a CPU, ~2.5x on
        an NVIDIA L40S); coarse or sparse grids fall back to direct sums
        per subband (where DDMT's uninterpreted loop has lower overhead).

        Same parameters, methods, backends and streaming behaviour as
        :class:`DDMT`. On a GPU, grids that share too little (sparse or
        coarse) run the DDMT kernel instead, with the same result.
        )doc");
    def_ddmt_inits<SDMT>(sdmt_cls);
}

} // namespace dmt
