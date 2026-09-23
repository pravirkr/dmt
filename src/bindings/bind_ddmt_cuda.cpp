#include "bindings/bind_cuda.hpp"

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
using algorithms::DDMTCUDA;
using plans::DDMTPlan;
using plans::LevinConfig;

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

void bind_ddmt_cuda(py::module_& mod) {
    py::class_<DDMTCUDA>(mod, "DDMTCUDA",
                         R"doc(
        Direct Dispersion Measure Transform (DDMT) incoherent dedispersion on CUDA.

        GPU counterpart to DDMTCPU, supporting float32 waterfalls, low-bit packed
        integers (1, 2, 4, 8, 16 bits), channel kill masks, constant memory fractional
        delay lookups, and double-buffered asynchronous execution.
        )doc")
        .def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                         float dm_max, float dm_step, float dm_min,
                         int device_id, SizeType nbits,
                         std::optional<py::array_t<uint8_t>> kill_mask,
                         SizeType nbeams) {
                 std::vector<uint8_t> km_vec;
                 if (kill_mask.has_value()) {
                     km_vec.assign(kill_mask->data(), kill_mask->data() + kill_mask->size());
                 }
                 return DDMTCUDA(f_min, f_max, nchans, tsamp, dm_max, dm_step, dm_min,
                                 device_id, nbits, km_vec, nbeams);
             }),
             "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "dm_max"_a, "dm_step"_a,
             "dm_min"_a = 0.0F, "device_id"_a = 0, "nbits"_a = 32,
             "kill_mask"_a = py::none(), "nbeams"_a = 1)
        .def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                         const py::array_t<float>& dm_arr,
                         int device_id, SizeType nbits,
                         std::optional<py::array_t<uint8_t>> kill_mask,
                         SizeType nbeams) {
                 std::vector<uint8_t> km_vec;
                 if (kill_mask.has_value()) {
                     km_vec.assign(kill_mask->data(), kill_mask->data() + kill_mask->size());
                 }
                 return DDMTCUDA(f_min, f_max, nchans, tsamp,
                                 std::vector<float>(dm_arr.data(), dm_arr.data() + dm_arr.size()),
                                 device_id, nbits, km_vec, nbeams);
             }),
             "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "dm_arr"_a,
             "device_id"_a = 0, "nbits"_a = 32, "kill_mask"_a = py::none(),
             "nbeams"_a = 1)
        .def(py::init([](float f_min, float f_max, SizeType nchans, float tsamp,
                         const LevinConfig& levin,
                         int device_id, SizeType nbits,
                         std::optional<py::array_t<uint8_t>> kill_mask,
                         SizeType nbeams) {
                 std::vector<uint8_t> km_vec;
                 if (kill_mask.has_value()) {
                     km_vec.assign(kill_mask->data(), kill_mask->data() + kill_mask->size());
                 }
                 return DDMTCUDA(f_min, f_max, nchans, tsamp, levin,
                                 device_id, nbits, km_vec, nbeams);
             }),
             "f_min"_a, "f_max"_a, "nchans"_a, "tsamp"_a, "levin"_a,
             "device_id"_a = 0, "nbits"_a = 32, "kill_mask"_a = py::none(),
             "nbeams"_a = 1)
        .def(py::init<const DDMTPlan&, int, SizeType>(), "plan"_a,
             "device_id"_a = 0, "nbeams"_a = 1)
        .def_property_readonly("plan", &DDMTCUDA::get_plan)
        .def_property_readonly("nbeams", &DDMTCUDA::get_nbeams)
        .def(
            "execute",
            [](DDMTCUDA& ddmt,
               const py::array_t<float, py::array::c_style>& waterfall) {
                const auto& plan = ddmt.get_plan();
                const auto ndim  = waterfall.ndim();
                if (ndim != 2 && ndim != 3) {
                    throw std::invalid_argument(
                        "DDMTCUDA.execute: waterfall must be 2D (nchans, "
                        "nsamps) or 3D (nbeams, nchans, nsamps)");
                }
                const auto* shape    = waterfall.shape();
                const auto nbeams    = ndim == 3 ? static_cast<SizeType>(shape[0]) : 1;
                const auto nsamps_in = static_cast<SizeType>(shape[ndim - 1]);
                if (nbeams != ddmt.get_nbeams()) {
                    throw std::invalid_argument(std::format(
                        "DDMTCUDA.execute: waterfall has {} beam(s), but "
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
            "waterfall"_a)
        .def(
            "execute",
            [](DDMTCUDA& ddmt,
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
            "waterfall_packed"_a, "nsamps"_a)
        .def(
            "execute_time_major",
            [](DDMTCUDA& ddmt,
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
            "filterbank_packed"_a, "nsamps"_a)
        .def("get_output_nsamps", &DDMTCUDA::get_output_nsamps, "input_nsamps"_a)
        .def("reset_history", &DDMTCUDA::reset_history)
        .def("history_state_size", &DDMTCUDA::history_state_size)
        .def("save_history", [](const DDMTCUDA& ddmt) -> py::object {
            const auto& plan = ddmt.get_plan();
            const auto sz    = ddmt.history_state_size();
            if (plan.get_nbits() == 32) {
                py::array_t<float, py::array::c_style> hist(sz);
                if (!ddmt.save_history(
                        std::span<float>(hist.mutable_data(), hist.size()))) {
                    throw std::runtime_error(
                        "DDMTCUDA.save_history: stream is not fully warmed up "
                        "yet (call execute() with enough samples first)");
                }
                return hist;
            } else {
                py::array_t<uint8_t, py::array::c_style> hist(sz);
                if (!ddmt.save_history(
                        std::span<uint8_t>(hist.mutable_data(), hist.size()))) {
                    throw std::runtime_error(
                        "DDMTCUDA.save_history: stream is not fully warmed up "
                        "yet (call execute() with enough samples first)");
                }
                return hist;
            }
        })
        .def("load_history", [](DDMTCUDA& ddmt, const py::array& hist_obj) {
            const auto& plan  = ddmt.get_plan();
            const auto nbits  = plan.get_nbits();
            // Validate the *actual* dtype before casting: py::array_t's
            // implicit conversion would otherwise silently truncate e.g. a
            // float32 array into uint8 instead of raising a clear error.
            if (nbits == 32) {
                if (hist_obj.dtype().kind() != 'f' || hist_obj.itemsize() != 4) {
                    throw std::invalid_argument(std::format(
                        "DDMTCUDA.load_history: plan nbits=32 expects a "
                        "float32 array, got dtype '{}'",
                        std::string(py::str(hist_obj.dtype()))));
                }
                auto hist = py::array_t<float, py::array::c_style>(hist_obj);
                if (!ddmt.load_history(
                        std::span<const float>(hist.data(), hist.size()))) {
                    throw std::runtime_error(std::format(
                        "DDMTCUDA.load_history: expected {} float elements, got {}",
                        ddmt.history_state_size(), hist.size()));
                }
            } else {
                if (hist_obj.dtype().kind() != 'u' || hist_obj.itemsize() != 1) {
                    throw std::invalid_argument(std::format(
                        "DDMTCUDA.load_history: plan nbits={} expects a "
                        "uint8 array, got dtype '{}'",
                        nbits, std::string(py::str(hist_obj.dtype()))));
                }
                auto hist = py::array_t<uint8_t, py::array::c_style>(hist_obj);
                if (!ddmt.load_history(
                        std::span<const uint8_t>(hist.data(), hist.size()))) {
                    throw std::runtime_error(std::format(
                        "DDMTCUDA.load_history: expected {} uint8 bytes, got {}",
                        ddmt.history_state_size(), hist.size()));
                }
            }
        }, "history"_a);
}

} // namespace dmt
