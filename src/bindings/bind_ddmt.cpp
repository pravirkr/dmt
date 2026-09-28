#include "bindings/bind.hpp"

#include <algorithm>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dmt/algorithms/ddmt.hpp"
#include "dmt/algorithms/sdmt.hpp"
#include "dmt/common/plans.hpp"
#include "pybind_utils.hpp"

namespace dmt {
using algorithms::DDMT;
using algorithms::SDMT;
using plans::DDMTPlan;
using plans::LevinConfig;

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

/// Binds the constructor set shared by DDMT and SDMT.
template <typename T, typename Cls> void def_ddmt_inits(Cls& cls) {
    cls.def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                        float dm_max, float dm_step, float dm_min, int nthreads,
                        SizeType nbits,
                        std::optional<py::array_t<uint8_t>> kill_mask,
                        SizeType nbeams, std::string_view backend, int device) {
                std::vector<uint8_t> km_vec;
                if (kill_mask.has_value()) {
                    km_vec.assign(kill_mask->data(),
                                  kill_mask->data() + kill_mask->size());
                }
                return T(f_min, f_max, nchans, tsamp, dm_max, dm_step, dm_min,
                         make_exec(backend, nthreads, device), nbits, km_vec,
                         nbeams);
            }),
            "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "dm_max"_a,
            "dm_step"_a, "dm_min"_a = 0.0F, "nthreads"_a = 1, "nbits"_a = 32,
            "kill_mask"_a = py::none(), "nbeams"_a = 1, py::kw_only(),
            "backend"_a = "cpu", "device"_a = 0)
        .def(
            py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                        const py::array_t<float>& dm_arr, int nthreads,
                        SizeType nbits,
                        std::optional<py::array_t<uint8_t>> kill_mask,
                        SizeType nbeams, std::string_view backend, int device) {
                std::vector<uint8_t> km_vec;
                if (kill_mask.has_value()) {
                    km_vec.assign(kill_mask->data(),
                                  kill_mask->data() + kill_mask->size());
                }
                return T(f_min, f_max, nchans, tsamp,
                         std::span<const float>(dm_arr.data(), dm_arr.size()),
                         make_exec(backend, nthreads, device), nbits, km_vec,
                         nbeams);
            }),
            "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "dm_arr"_a,
            "nthreads"_a = 1, "nbits"_a = 32, "kill_mask"_a = py::none(),
            "nbeams"_a = 1, py::kw_only(), "backend"_a = "cpu", "device"_a = 0)
        .def(
            py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                        const LevinConfig& levin, int nthreads, SizeType nbits,
                        std::optional<py::array_t<uint8_t>> kill_mask,
                        SizeType nbeams, std::string_view backend, int device) {
                std::vector<uint8_t> km_vec;
                if (kill_mask.has_value()) {
                    km_vec.assign(kill_mask->data(),
                                  kill_mask->data() + kill_mask->size());
                }
                return T(f_min, f_max, nchans, tsamp, levin,
                         make_exec(backend, nthreads, device), nbits, km_vec,
                         nbeams);
            }),
            "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "levin"_a,
            "nthreads"_a = 1, "nbits"_a = 32, "kill_mask"_a = py::none(),
            "nbeams"_a = 1, py::kw_only(), "backend"_a = "cpu", "device"_a = 0)
        .def(py::init([](const DDMTPlan& plan, int nthreads, SizeType nbeams,
                         std::string_view backend, int device) {
                 return T(plan, make_exec(backend, nthreads, device), nbeams);
             }),
             "plan"_a, "nthreads"_a = 1, "nbeams"_a = 1, py::kw_only(),
             "backend"_a = "cpu", "device"_a = 0);
}

void bind_ddmt(py::module_& mod) {
    py::class_<DDMT> ddmt_cls(mod, "DDMT",
                              R"doc(
        Direct Dispersion Measure Transform (DDMT) incoherent dedispersion.

        Supports float32 waterfalls, packed low-bit integers (1, 2, 4, 8, 16 bits),
        channel kill masks, streaming history across consecutive chunks, and both
        channel-major and time-major filterbank layouts.

        Parameters
        ----------
        f_min, f_max : float
            Band edges in MHz.
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
            Number of OpenMP worker threads (default 1).
        nbits : int, optional
            Bit precision per sample (32 for float, or 1, 2, 4, 8, 16 for packed integer).
        kill_mask : numpy.ndarray, optional
            Per-channel mask (uint8 array, 1 = keep, 0 = mask out).
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

        Input and output are NumPy (host) arrays on every backend; a GPU
        backend pipelines the host transfers and blocks until the result is
        on the host.
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
        .def(
            "execute",
            [](DDMT& ddmt,
               const py::array_t<float, py::array::c_style>& waterfall) {
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
                if (nbeams != ddmt.get_nbeams()) {
                    throw std::invalid_argument(std::format(
                        "DDMT.execute: waterfall has {} beam(s), but "
                        "this instance was configured for {}",
                        nbeams, ddmt.get_nbeams()));
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
                if (nsamps_out > 0) {
                    ddmt.execute(
                        std::span<const float>(waterfall.data(),
                                               waterfall.size()),
                        std::span<float>(dmt.mutable_data(), dmt.size()));
                } else {
                    ddmt.execute(std::span<const float>(waterfall.data(),
                                                        waterfall.size()),
                                 std::span<float>{});
                }
                return dmt;
            },
            "waterfall"_a,
            R"doc(
            Dedisperse a float32 waterfall of shape ``(nchans, nsamps)``, or
            ``(nbeams, nchans, nsamps)`` if this instance's nbeams > 1.
            Produces ``(n_dm, output_nsamps)`` or ``(nbeams, n_dm, output_nsamps)``.
            )doc")
        .def(
            "execute",
            [](DDMT& ddmt,
               const py::array_t<uint8_t, py::array::c_style>& waterfall_packed,
               SizeType nsamps) {
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
                if (nsamps_out > 0) {
                    ddmt.execute(
                        std::span<const uint8_t>(waterfall_packed.data(),
                                                 waterfall_packed.size()),
                        nsamps,
                        std::span<int32_t>(dmt.mutable_data(), dmt.size()));
                } else {
                    ddmt.execute(
                        std::span<const uint8_t>(waterfall_packed.data(),
                                                 waterfall_packed.size()),
                        nsamps, std::span<int32_t>{});
                }
                return dmt;
            },
            "waterfall_packed"_a, "nsamps"_a,
            R"doc(
            Dedisperse a channel-major packed integer waterfall of shape
            ``(nchans, row_bytes)``, or ``(nbeams, nchans, row_bytes)`` if
            this instance's nbeams > 1.
            Produces ``(n_dm, output_nsamps)`` or
            ``(nbeams, n_dm, output_nsamps)``.
            )doc")
        .def(
            "execute_time_major",
            [](DDMT& ddmt,
               const py::array_t<uint8_t, py::array::c_style>&
                   filterbank_packed,
               SizeType nsamps) {
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
                ddmt.execute_time_major(
                    std::span<const uint8_t>(filterbank_packed.data(),
                                             filterbank_packed.size()),
                    nsamps, std::span<int32_t>(dmt.mutable_data(), dmt.size()));
                return dmt;
            },
            "filterbank_packed"_a, "nsamps"_a,
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
                 const auto& plan = ddmt.get_plan();
                 const auto sz    = ddmt.history_state_size();
                 if (plan.get_nbits() == 32) {
                     py::array_t<float, py::array::c_style> hist(sz);
                     ddmt.save_history(
                         std::span<float>(hist.mutable_data(), hist.size()));
                     return hist;
                 } else {
                     py::array_t<uint8_t, py::array::c_style> hist(sz);
                     ddmt.save_history(
                         std::span<uint8_t>(hist.mutable_data(), hist.size()));
                     return hist;
                 }
             })
        .def(
            "load_history",
            [](DDMT& ddmt, const py::array& hist_obj) {
                const auto& plan = ddmt.get_plan();
                const auto nbits = plan.get_nbits();
                // Validate the *actual* dtype before casting: py::array_t's
                // implicit conversion would otherwise silently truncate e.g. a
                // float32 array into uint8 instead of raising a clear error.
                if (nbits == 32) {
                    if (hist_obj.dtype().kind() != 'f' ||
                        hist_obj.itemsize() != 4) {
                        throw std::invalid_argument(std::format(
                            "DDMT.load_history: plan nbits=32 expects a "
                            "float32 array, got dtype '{}'",
                            std::string(py::str(hist_obj.dtype()))));
                    }
                    auto hist =
                        py::array_t<float, py::array::c_style>(hist_obj);
                    ddmt.load_history(
                        std::span<const float>(hist.data(), hist.size()));
                } else {
                    if (hist_obj.dtype().kind() != 'u' ||
                        hist_obj.itemsize() != 1) {
                        throw std::invalid_argument(std::format(
                            "DDMT.load_history: plan nbits={} expects a "
                            "uint8 array, got dtype '{}'",
                            nbits, std::string(py::str(hist_obj.dtype()))));
                    }
                    auto hist =
                        py::array_t<uint8_t, py::array::c_style>(hist_obj);
                    ddmt.load_history(
                        std::span<const uint8_t>(hist.data(), hist.size()));
                }
            },
            "history"_a);

    py::class_<SDMT, DDMT> sdmt_cls(mod, "SDMT",
                                    R"doc(
        Subband-shared Dispersion Measure Transform (SDMT).

        Computes exactly the DDMT sums, with the same delay table, DM grid
        and kill mask, but shares partial sums between DM trials: within
        16-channel subbands, trials whose integer delays agree (relative to
        a per-trial base) have identical partial sums, computed once.
        Nothing is approximated. Integer output is bit-identical to DDMT;
        float32 output differs only by rounding (order of additions).
        Dense DM grids save the most additions (~7x fewer additions, yielding
        ~3.5x wall-clock speedup at 4096 channels and 2049 trials); coarse or
        sparse grids fall back to direct sums per subband (where DDMT's
        uninterpreted loop has lower overhead).

        Same parameters, methods and streaming behaviour as :class:`DDMT`.
        CPU backend only (``backend='cpu'``).
        )doc");
    def_ddmt_inits<SDMT>(sdmt_cls);
}

} // namespace dmt
