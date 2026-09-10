#pragma once

#include <memory>
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

inline std::vector<SizeType> extract_dt_grid(const py::object& obj) {
    std::vector<SizeType> result;
    if (py::isinstance<py::array>(obj)) {
        auto arr = py::array_t<SizeType, py::array::c_style |
                                             py::array::forcecast>::ensure(obj);
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
            result.push_back(item.cast<SizeType>());
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

} // namespace dmt