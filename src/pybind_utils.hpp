#pragma once

#include <format>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dmt/common/types.hpp"

namespace dmt {

namespace py = pybind11;

// helper function to avoid making a copy when returning a py::array_t
// source: https://github.com/pybind/pybind11/issues/1042#issuecomment-642215028
template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence&& seq) {
    auto size = seq.size();
    auto data = seq.data();
    std::unique_ptr<Sequence> seq_ptr =
        std::make_unique<Sequence>(std::forward<Sequence>(seq));
    auto capsule = py::capsule(seq_ptr.get(), [](void* p) {
        std::unique_ptr<Sequence>(reinterpret_cast<Sequence*>(p)); // NOLINT
    });
    seq_ptr.release();
    return py::array(size, data, capsule);
}

template <typename Sequence>
inline py::array_t<typename Sequence::value_type>
as_pyarray_ref(const Sequence& seq) {
    auto size        = seq.size();
    const auto* data = seq.data();
    return py::array_t<typename Sequence::value_type>(size, data);
}

template <typename T>
inline py::list
as_listof_pyarray(const std::vector<std::vector<T>>& vec_of_vecs) {
    py::list result;
    for (const auto& inner : vec_of_vecs) {
        result.append(
            py::array_t<T>(static_cast<ssize_t>(inner.size()), inner.data()));
    }
    return result;
}

inline std::vector<IndexType> extract_dt_grid(const py::object& obj) {
    std::vector<IndexType> result;
    if (py::isinstance<py::array>(obj)) {
        auto arr =
            py::array_t<IndexType,
                        py::array::c_style | py::array::forcecast>::ensure(obj);
        if (!arr) {
            throw py::value_error(
                "dt_grid could not be converted to 1D integer array");
        }
        if (arr.ndim() != 1) {
            throw py::value_error("dt_grid must be a 1D array");
        }
        result.assign(arr.data(), arr.data() + arr.size());
    } else if (py::isinstance<py::sequence>(obj)) {
        auto seq = py::reinterpret_borrow<py::sequence>(obj);
        result.reserve(seq.size());
        for (const auto& item : seq) {
            result.push_back(item.cast<IndexType>());
        }
    } else {
        throw py::value_error("dt_grid must be a list, tuple, or 1D array");
    }
    return result;
}

inline std::vector<float> extract_dm_grid(const py::object& obj) {
    std::vector<float> result;
    if (py::isinstance<py::array>(obj)) {
        auto arr = py::array_t<float, py::array::c_style |
                                          py::array::forcecast>::ensure(obj);
        if (!arr) {
            throw py::value_error(
                "dm_grid could not be converted to 1D float array");
        }
        if (arr.ndim() != 1) {
            throw py::value_error("dm_grid must be a 1D array");
        }
        result.assign(arr.data(), arr.data() + arr.size());
    } else if (py::isinstance<py::sequence>(obj)) {
        auto seq = py::reinterpret_borrow<py::sequence>(obj);
        result.reserve(seq.size());
        for (const auto& item : seq) {
            result.push_back(item.cast<float>());
        }
    } else {
        throw py::value_error("dm_grid must be a list, tuple, or 1D array");
    }
    return result;
}

enum class CustomGridType { kDt, kDm };

inline std::pair<CustomGridType, py::object>
resolve_custom_grid(const py::object& dt_grid,
                    const py::object& dt_arr,
                    const py::object& dm_grid,
                    const py::object& dm_arr) {
    if (!dt_grid.is_none() && !dt_arr.is_none()) {
        throw py::value_error("Cannot provide both dt_grid and dt_arr");
    }
    if (!dm_grid.is_none() && !dm_arr.is_none()) {
        throw py::value_error("Cannot provide both dm_grid and dm_arr");
    }
    const py::object dt_obj = dt_grid.is_none() ? dt_arr : dt_grid;
    const py::object dm_obj = dm_grid.is_none() ? dm_arr : dm_grid;

    if (!dt_obj.is_none() && !dm_obj.is_none()) {
        throw py::value_error("Cannot provide both dt_grid and dm_grid");
    }
    if (dt_obj.is_none() && dm_obj.is_none()) {
        throw py::value_error("Either dt_grid (or dt_arr) or dm_grid (or "
                              "dm_arr) must be provided");
    }
    if (!dt_obj.is_none()) {
        return {CustomGridType::kDt, dt_obj};
    }
    return {CustomGridType::kDm, dm_obj};
}

// The engine's output buffer: the caller's `out` (validated, reused as is)
// or a fresh array of nbeams * get_buffer_size() floats.
template <typename Engine>
py::array_t<float, py::array::c_style>
fdmt_output_buffer(const Engine& fdmt,
                   std::string_view name,
                   const std::optional<py::array>& out) {
    const auto needed = fdmt.get_nbeams() * fdmt.get_plan().get_buffer_size();
    if (!out.has_value()) {
        return py::array_t<float, py::array::c_style>(
            static_cast<py::ssize_t>(needed));
    }
    const auto& arr = *out;
    if (arr.dtype().kind() != 'f' || arr.itemsize() != 4 ||
        (arr.flags() & py::array::c_style) == 0 || !arr.writeable()) {
        throw py::type_error(std::format("{}.execute: out must be a writeable, "
                                         "C-contiguous float32 array",
                                         name));
    }
    if (static_cast<SizeType>(arr.size()) < needed) {
        throw std::invalid_argument(std::format(
            "{}.execute: out has {} elements, needs at least nbeams * "
            "plan.buffer_size = {}",
            name, arr.size(), needed));
    }
    return py::reinterpret_borrow<py::array_t<float, py::array::c_style>>(arr);
}

// FDMTCPU/FDMTCUDA execute(): runs `run(dmt_span)` into the output buffer
// and returns a zero-copy view of the transform: (ndms, nsamps) when
// `batched` is false (nbeams must be 1),
// else (nbeams, ndms, nsamps) with beams plan.buffer_size apart. The view's
// base is the whole buffer (each beam's scratch tail included). A fresh
// buffer is allocated per call unless `out` is given. `expected_size` is the
// full beam-major input size, used only for the error message.
template <typename Engine, typename Run>
py::object fdmt_execute_to_array(const Engine& fdmt,
                                 std::string_view name,
                                 bool batched,
                                 SizeType input_size,
                                 SizeType expected_size,
                                 const std::optional<py::array>& out,
                                 Run&& run) {
    const auto nbeams   = fdmt.get_nbeams();
    const auto& plan    = fdmt.get_plan();
    const auto ncoords  = static_cast<py::ssize_t>(plan.get_dmt_ndms());
    const auto nsamps   = static_cast<py::ssize_t>(plan.get_dmt_nsamps());
    const auto buf_size = static_cast<py::ssize_t>(plan.get_buffer_size());
    const auto fsize    = static_cast<py::ssize_t>(sizeof(float));
    if (!batched && nbeams != 1) {
        throw std::invalid_argument(
            std::format("{}: Invalid size of waterfall. Expected {}, got {}",
                        name, expected_size, input_size));
    }
    auto buf = fdmt_output_buffer(fdmt, name, out);
    run(std::span<float>(buf.mutable_data(), buf.size()));
    if (!batched) {
        return py::array_t<float>({ncoords, nsamps}, {nsamps * fsize, fsize},
                                  buf.data(), buf);
    }
    return py::array_t<float>(
        {static_cast<py::ssize_t>(nbeams), ncoords, nsamps},
        {buf_size * fsize, nsamps * fsize, fsize}, buf.data(), buf);
}

} // namespace dmt
