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
#include "dmt/common/plans.hpp"
#include "pybind_utils.hpp"

namespace dmt {
using algorithms::DDMTCPU;
using plans::DDMTPlan;
using plans::LevinConfig;

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

void bind_ddmt(py::module_& mod) {
    py::class_<DDMTCPU>(mod, "DDMTCPU",
                        R"doc(
        Direct Dispersion Measure Transform (DDMT) incoherent dedispersion on the CPU.

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
        )doc")
        .def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                         float dm_max, float dm_step, float dm_min,
                         int nthreads, SizeType nbits,
                         std::optional<py::array_t<uint8_t>> kill_mask,
                         SizeType nbeams) {
                 std::vector<uint8_t> km_vec;
                 if (kill_mask.has_value()) {
                     km_vec.assign(kill_mask->data(), kill_mask->data() + kill_mask->size());
                 }
                 return DDMTCPU(f_min, f_max, nchans, tsamp, dm_max, dm_step, dm_min,
                                nthreads, nbits, km_vec, nbeams);
             }),
             "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "dm_max"_a, "dm_step"_a,
             "dm_min"_a = 0.0F, "nthreads"_a = 1, "nbits"_a = 32,
             "kill_mask"_a = py::none(), "nbeams"_a = 1)
        .def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                         const py::array_t<float>& dm_arr,
                         int nthreads, SizeType nbits,
                         std::optional<py::array_t<uint8_t>> kill_mask,
                         SizeType nbeams) {
                 std::vector<uint8_t> km_vec;
                 if (kill_mask.has_value()) {
                     km_vec.assign(kill_mask->data(), kill_mask->data() + kill_mask->size());
                 }
                 return DDMTCPU(f_min, f_max, nchans, tsamp,
                                std::span<const float>(dm_arr.data(), dm_arr.size()),
                                nthreads, nbits, km_vec, nbeams);
             }),
             "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "dm_arr"_a,
             "nthreads"_a = 1, "nbits"_a = 32, "kill_mask"_a = py::none(),
             "nbeams"_a = 1)
        .def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                         const LevinConfig& levin,
                         int nthreads, SizeType nbits,
                         std::optional<py::array_t<uint8_t>> kill_mask,
                         SizeType nbeams) {
                 std::vector<uint8_t> km_vec;
                 if (kill_mask.has_value()) {
                     km_vec.assign(kill_mask->data(), kill_mask->data() + kill_mask->size());
                 }
                 return DDMTCPU(f_min, f_max, nchans, tsamp, levin,
                                nthreads, nbits, km_vec, nbeams);
             }),
             "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "levin"_a,
             "nthreads"_a = 1, "nbits"_a = 32, "kill_mask"_a = py::none(),
             "nbeams"_a = 1)
        .def(py::init<const DDMTPlan&, int, SizeType>(), "plan"_a,
             "nthreads"_a = 1, "nbeams"_a = 1)
        .def_property_readonly("plan", &DDMTCPU::get_plan)
        .def_property_readonly("nbeams", &DDMTCPU::get_nbeams)
        .def(
            "execute",
            [](DDMTCPU& ddmt,
               const py::array_t<float, py::array::c_style>& waterfall) {
                const auto& plan = ddmt.get_plan();
                const auto ndim  = waterfall.ndim();
                if (ndim != 2 && ndim != 3) {
                    throw std::invalid_argument(
                        "DDMTCPU.execute: waterfall must be 2D (nchans, "
                        "nsamps) or 3D (nbeams, nchans, nsamps)");
                }
                const auto* shape    = waterfall.shape();
                const auto nbeams    = ndim == 3 ? static_cast<SizeType>(shape[0]) : 1;
                const auto nsamps_in = static_cast<SizeType>(shape[ndim - 1]);
                if (nbeams != ddmt.get_nbeams()) {
                    throw std::invalid_argument(std::format(
                        "DDMTCPU.execute: waterfall has {} beam(s), but "
                        "this instance was configured for {}",
                        nbeams, ddmt.get_nbeams()));
                }
                const auto nsamps_out = ddmt.get_output_nsamps(nsamps_in);
                const auto dm_count   = plan.get_container().dm_arr.size();

                py::array_t<float, py::array::c_style> dmt(
                    ndim == 3
                        ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(nbeams),
                                                   static_cast<py::ssize_t>(dm_count),
                                                   static_cast<py::ssize_t>(nsamps_out)}
                        : std::vector<py::ssize_t>{static_cast<py::ssize_t>(dm_count),
                                                   static_cast<py::ssize_t>(nsamps_out)});
                if (nsamps_out > 0) {
                    ddmt.execute(
                        std::span<const float>(waterfall.data(), waterfall.size()),
                        std::span<float>(dmt.mutable_data(), dmt.size()));
                } else {
                    ddmt.execute(
                        std::span<const float>(waterfall.data(), waterfall.size()),
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
            [](DDMTCPU& ddmt,
               const py::array_t<uint8_t, py::array::c_style>& waterfall_packed,
               SizeType nsamps) {
                const auto& plan      = ddmt.get_plan();
                const auto& plan_c    = plan.get_container();
                const auto nsamps_out = ddmt.get_output_nsamps(nsamps);
                const auto dm_count   = plan_c.dm_arr.size();
                const auto nbeams     = ddmt.get_nbeams();

                py::array_t<int32_t, py::array::c_style> dmt(
                    nbeams > 1
                        ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(nbeams),
                                                   static_cast<py::ssize_t>(dm_count),
                                                   static_cast<py::ssize_t>(nsamps_out)}
                        : std::vector<py::ssize_t>{static_cast<py::ssize_t>(dm_count),
                                                   static_cast<py::ssize_t>(nsamps_out)});
                if (nsamps_out > 0) {
                    ddmt.execute(
                        std::span<const uint8_t>(waterfall_packed.data(), waterfall_packed.size()),
                        nsamps,
                        std::span<int32_t>(dmt.mutable_data(), dmt.size()));
                } else {
                    ddmt.execute(
                        std::span<const uint8_t>(waterfall_packed.data(), waterfall_packed.size()),
                        nsamps,
                        std::span<int32_t>{});
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
            [](DDMTCPU& ddmt,
               const py::array_t<uint8_t, py::array::c_style>& filterbank_packed,
               SizeType nsamps) {
                const auto& plan      = ddmt.get_plan();
                const auto& plan_c    = plan.get_container();
                const auto max_delay  = *std::ranges::max_element(plan_c.delay_table);
                const auto nsamps_out = nsamps > max_delay ? nsamps - max_delay : 0;
                const auto dm_count   = plan_c.dm_arr.size();
                const auto nbeams     = ddmt.get_nbeams();

                py::array_t<int32_t, py::array::c_style> dmt(
                    nbeams > 1
                        ? std::vector<py::ssize_t>{static_cast<py::ssize_t>(nbeams),
                                                   static_cast<py::ssize_t>(dm_count),
                                                   static_cast<py::ssize_t>(nsamps_out)}
                        : std::vector<py::ssize_t>{static_cast<py::ssize_t>(dm_count),
                                                   static_cast<py::ssize_t>(nsamps_out)});
                if (nsamps_out > 0) {
                    ddmt.execute_time_major(
                        std::span<const uint8_t>(filterbank_packed.data(), filterbank_packed.size()),
                        nsamps,
                        std::span<int32_t>(dmt.mutable_data(), dmt.size()));
                }
                return dmt;
            },
            "filterbank_packed"_a, "nsamps"_a,
            R"doc(
            Dedisperse a time-major packed filterbank of shape
            ``(nsamps, samp_bytes)``, or ``(nbeams, nsamps, samp_bytes)`` if
            this instance's nbeams > 1.
            Produces ``(n_dm, nsamps - max_delay)`` or
            ``(nbeams, n_dm, nsamps - max_delay)``.
            )doc")
        .def("get_output_nsamps", &DDMTCPU::get_output_nsamps, "input_nsamps"_a)
        .def("reset_history", &DDMTCPU::reset_history)
        .def("history_state_size", &DDMTCPU::history_state_size)
        .def("save_history", [](const DDMTCPU& ddmt) -> py::object {
            const auto& plan = ddmt.get_plan();
            const auto sz    = ddmt.history_state_size();
            if (plan.get_nbits() == 32) {
                py::array_t<float, py::array::c_style> hist(sz);
                if (!ddmt.save_history(
                        std::span<float>(hist.mutable_data(), hist.size()))) {
                    throw std::runtime_error(
                        "DDMTCPU.save_history: stream is not fully warmed up "
                        "yet (call execute() with enough samples first)");
                }
                return hist;
            } else {
                py::array_t<uint8_t, py::array::c_style> hist(sz);
                if (!ddmt.save_history(
                        std::span<uint8_t>(hist.mutable_data(), hist.size()))) {
                    throw std::runtime_error(
                        "DDMTCPU.save_history: stream is not fully warmed up "
                        "yet (call execute() with enough samples first)");
                }
                return hist;
            }
        })
        .def("load_history", [](DDMTCPU& ddmt, const py::array& hist_obj) {
            const auto& plan  = ddmt.get_plan();
            const auto nbits  = plan.get_nbits();
            // Validate the *actual* dtype before casting: py::array_t's
            // implicit conversion would otherwise silently truncate e.g. a
            // float32 array into uint8 instead of raising a clear error.
            if (nbits == 32) {
                if (hist_obj.dtype().kind() != 'f' || hist_obj.itemsize() != 4) {
                    throw std::invalid_argument(std::format(
                        "DDMTCPU.load_history: plan nbits=32 expects a "
                        "float32 array, got dtype '{}'",
                        std::string(py::str(hist_obj.dtype()))));
                }
                auto hist = py::array_t<float, py::array::c_style>(hist_obj);
                if (!ddmt.load_history(
                        std::span<const float>(hist.data(), hist.size()))) {
                    throw std::runtime_error(std::format(
                        "DDMTCPU.load_history: expected {} float elements, got {}",
                        ddmt.history_state_size(), hist.size()));
                }
            } else {
                if (hist_obj.dtype().kind() != 'u' || hist_obj.itemsize() != 1) {
                    throw std::invalid_argument(std::format(
                        "DDMTCPU.load_history: plan nbits={} expects a "
                        "uint8 array, got dtype '{}'",
                        nbits, std::string(py::str(hist_obj.dtype()))));
                }
                auto hist = py::array_t<uint8_t, py::array::c_style>(hist_obj);
                if (!ddmt.load_history(
                        std::span<const uint8_t>(hist.data(), hist.size()))) {
                    throw std::runtime_error(std::format(
                        "DDMTCPU.load_history: expected {} uint8 bytes, got {}",
                        ddmt.history_state_size(), hist.size()));
                }
            }
        }, "history"_a);
}

} // namespace dmt
