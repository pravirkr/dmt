#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <dlpack/dlpack.h>
#include <fmt/format.h>
#include <pybind11/gil_safe_call_once.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "dmt/common/backend.hpp"
#include "dmt/common/types.hpp"

namespace dmt {

namespace py = pybind11;

/// A Python array in GPU memory: an object whose `__dlpack_device__()` names
/// a CUDA, CUDA-managed or ROCm device (PyTorch, CuPy, JAX, ...). As a bound
/// argument it selects a method's device overload; NumPy and other host
/// arrays do not convert and fall through to the host overload.
struct DeviceArray {
    py::object obj;
    Device device;
};

/// The device of `obj` if it is a DLPack array in GPU memory.
inline std::optional<Device> gpu_device_of(py::handle obj) {
    if (py::isinstance<py::array>(obj) ||
        !py::hasattr(obj, "__dlpack_device__")) {
        return std::nullopt;
    }
    int type = 0;
    int id   = 0;
    try {
        const auto dev = py::tuple(obj.attr("__dlpack_device__")());
        type           = dev[0].cast<int>();
        id             = dev[1].cast<int>();
    } catch (const py::error_already_set&) {
        return std::nullopt;
    } catch (const py::cast_error&) {
        return std::nullopt;
    }
    switch (type) {
    case kDLCUDA:
    case kDLCUDAManaged:
        return Device{.backend = Backend::kCUDA, .id = id};
    case kDLROCM:
        return Device{.backend = Backend::kHIP, .id = id};
    default:
        return std::nullopt;
    }
}

/// A tensor consumed from `__dlpack__`. Owns the producer's managed tensor
/// and calls its deleter on destruction, so the memory stays valid while
/// this lives (keep_alive() extends that past the call).
class DLPackTensor {
public:
    DLPackTensor() = default;
    DLPackTensor(DLPackTensor&& other) noexcept
        : m_versioned(std::exchange(other.m_versioned, nullptr)),
          m_unversioned(std::exchange(other.m_unversioned, nullptr)) {}
    DLPackTensor& operator=(DLPackTensor&& other) noexcept {
        if (this != &other) {
            release();
            m_versioned   = std::exchange(other.m_versioned, nullptr);
            m_unversioned = std::exchange(other.m_unversioned, nullptr);
        }
        return *this;
    }
    DLPackTensor(const DLPackTensor&)            = delete;
    DLPackTensor& operator=(const DLPackTensor&) = delete;
    ~DLPackTensor() { release(); }

    /// Consumes `obj.__dlpack__(stream=stream)`: the producer orders its
    /// pending work on the array before `stream`, a DLPack stream value.
    static DLPackTensor consume(py::handle obj, std::intptr_t stream) {
        const auto dlpack = obj.attr("__dlpack__");
        py::object capsule;
        try {
            capsule = dlpack(py::arg("stream")      = stream,
                             py::arg("max_version") = py::make_tuple(
                                 DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION));
        } catch (py::error_already_set& e) {
            if (!e.matches(PyExc_TypeError)) {
                throw;
            }
            // Producers predating DLPack 1.0 take no max_version.
            capsule = dlpack(py::arg("stream") = stream);
        }
        PyObject* cap = capsule.ptr();
        DLPackTensor out;
        if (PyCapsule_IsValid(cap, "dltensor_versioned") != 0) {
            out.m_versioned = static_cast<DLManagedTensorVersioned*>(
                PyCapsule_GetPointer(cap, "dltensor_versioned"));
            mark_used(cap, "used_dltensor_versioned");
            const auto version = out.m_versioned->version;
            if (version.major != DLPACK_MAJOR_VERSION) {
                throw py::buffer_error(fmt::format(
                    "DLPack {}.{} tensors are not supported (this build "
                    "reads {}.x)",
                    version.major, version.minor, DLPACK_MAJOR_VERSION));
            }
        } else if (PyCapsule_IsValid(cap, "dltensor") != 0) {
            out.m_unversioned = static_cast<DLManagedTensor*>(
                PyCapsule_GetPointer(cap, "dltensor"));
            mark_used(cap, "used_dltensor");
        } else {
            throw py::type_error(
                "__dlpack__() did not return an unused DLPack capsule");
        }
        return out;
    }

    [[nodiscard]] const DLTensor& tensor() const noexcept {
        return m_versioned != nullptr ? m_versioned->dl_tensor
                                      : m_unversioned->dl_tensor;
    }
    [[nodiscard]] bool read_only() const noexcept {
        return m_versioned != nullptr &&
               (m_versioned->flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0;
    }
    /// The producer exported a copy: writes would not reach its array.
    [[nodiscard]] bool copied() const noexcept {
        return m_versioned != nullptr &&
               (m_versioned->flags & DLPACK_FLAG_BITMASK_IS_COPIED) != 0;
    }
    [[nodiscard]] SizeType size() const noexcept {
        const auto& t = tensor();
        SizeType n    = 1;
        for (int32_t i = 0; i < t.ndim; ++i) {
            n *= static_cast<SizeType>(t.shape[i]);
        }
        return n;
    }
    [[nodiscard]] bool is_c_contiguous() const noexcept {
        const auto& t = tensor();
        if (t.strides == nullptr) {
            return true;
        }
        int64_t expected = 1;
        for (int32_t i = t.ndim - 1; i >= 0; --i) {
            if (t.shape[i] != 1 && t.strides[i] != expected) {
                return false;
            }
            expected *= t.shape[i];
        }
        return true;
    }

private:
    static void mark_used(PyObject* capsule, const char* used_name) {
        if (PyCapsule_SetName(capsule, used_name) != 0) {
            throw py::error_already_set();
        }
    }
    void release() noexcept {
        if (m_versioned != nullptr && m_versioned->deleter != nullptr) {
            m_versioned->deleter(m_versioned);
        }
        if (m_unversioned != nullptr && m_unversioned->deleter != nullptr) {
            m_unversioned->deleter(m_unversioned);
        }
        m_versioned   = nullptr;
        m_unversioned = nullptr;
    }

    DLManagedTensorVersioned* m_versioned = nullptr;
    DLManagedTensor* m_unversioned        = nullptr;
};

/// A Python object owning `tensor`, for memory an engine keeps using after
/// the call returns (the stepper's reset()).
inline py::capsule keep_alive(DLPackTensor&& tensor) {
    auto owned = std::make_unique<DLPackTensor>(std::move(tensor));
    py::capsule capsule(owned.get(), [](void* p) {
        delete static_cast<DLPackTensor*>(p); // NOLINT
    });
    owned.release();
    return capsule;
}

template <typename T> constexpr DLDataType dl_dtype() noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return {.code = kDLFloat, .bits = 32, .lanes = 1};
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return {.code = kDLInt, .bits = 32, .lanes = 1};
    } else {
        static_assert(std::is_same_v<T, uint8_t>);
        return {.code = kDLUInt, .bits = 8, .lanes = 1};
    }
}

template <typename T> constexpr const char* dtype_name() noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return "float32";
    } else if constexpr (std::is_same_v<T, int32_t>) {
        return "int32";
    } else {
        static_assert(std::is_same_v<T, uint8_t>);
        return "uint8";
    }
}

inline bool same_dtype(DLDataType a, DLDataType b) noexcept {
    return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

inline std::string dl_dtype_name(DLDataType t) {
    const char* kind = "unknown";
    switch (t.code) {
    case kDLInt:
        kind = "int";
        break;
    case kDLUInt:
        kind = "uint";
        break;
    case kDLFloat:
        kind = "float";
        break;
    case kDLBfloat:
        kind = "bfloat";
        break;
    case kDLComplex:
        kind = "complex";
        break;
    case kDLBool:
        return "bool";
    default:
        break;
    }
    return t.lanes == 1 ? fmt::format("{}{}", kind, t.bits)
                        : fmt::format("{}{}x{}", kind, t.bits, t.lanes);
}

/// Cached attribute of the dmtlib._device helper module.
inline py::object device_helper(const char* name) {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object>
        storage;
    return storage
        .call_once_and_store_result(
            [] { return py::module_::import("dmtlib._device"); })
        .get_stored()
        .attr(name);
}

/// Native stream handle of `stream`: an int, a stream object, or None for
/// the current stream of `like`'s array library (see dmtlib._device).
inline std::uintptr_t resolve_stream(const py::object& stream,
                                     py::handle like) {
    if (py::isinstance<py::int_>(stream)) {
        return stream.cast<std::uintptr_t>();
    }
    return device_helper("stream_handle")(stream, like).cast<std::uintptr_t>();
}

/// A stepper call's stream: None keeps the one given to reset().
inline Stream stepper_stream(const py::object& stream) {
    if (stream.is_none()) {
        return {};
    }
    return {reinterpret_cast<void*>(resolve_stream(stream, py::none()))};
}

/// The device and stream of one device-memory call.
struct DeviceCall {
    std::string_view what; ///< e.g. "FDMT.execute", for messages
    Device device;
    std::uintptr_t stream = 0; ///< native handle; 0 = default stream

    [[nodiscard]] Stream native_stream() const noexcept {
        return {reinterpret_cast<void*>(stream)}; // NOLINT
    }

    /// `stream` as a `__dlpack__(stream=...)` value: CUDA spells the legacy
    /// default stream 1 (0 is ambiguous there), ROCm spells it 0.
    [[nodiscard]] std::intptr_t dlpack_stream() const noexcept {
        if (stream == 0 && device.backend == Backend::kCUDA) {
            return 1;
        }
        return static_cast<std::intptr_t>(stream);
    }

    /// Consumes `obj`, which must be a device array on this call's device.
    [[nodiscard]] DLPackTensor consume(py::handle obj,
                                       std::string_view arg) const {
        const auto dev = gpu_device_of(obj);
        if (!dev || dev->backend != device.backend || dev->id != device.id) {
            throw py::type_error(fmt::format(
                "{}: {} must be a device array on {}:{} (like the input)", what,
                arg, to_string(device.backend), device.id));
        }
        return DLPackTensor::consume(obj, dlpack_stream());
    }

    /// Checks that `t` is a C-contiguous T array (and writeable in place
    /// when `writable`).
    template <typename T>
    void check(const DLPackTensor& t,
               std::string_view arg,
               bool writable = false) const {
        if (!same_dtype(t.tensor().dtype, dl_dtype<T>())) {
            throw py::type_error(
                fmt::format("{}: {} must be a {} array, got {}", what, arg,
                            dtype_name<T>(), dl_dtype_name(t.tensor().dtype)));
        }
        if (!t.is_c_contiguous()) {
            throw py::type_error(
                fmt::format("{}: {} must be C-contiguous", what, arg));
        }
        if (writable && (t.read_only() || t.copied())) {
            throw py::type_error(fmt::format(
                "{}: {} must be writeable in place (the producer exported a "
                "read-only array or a copy)",
                what, arg));
        }
    }

    template <typename T>
    [[nodiscard]] DLPackTensor
    consume(py::handle obj, std::string_view arg, bool writable) const {
        auto t = consume(obj, arg);
        check<T>(t, arg, writable);
        return t;
    }

    /// `t` as the DeviceSpan an engine takes (no copy: the same allocation,
    /// offset and device).
    template <typename T>
    [[nodiscard]] DeviceSpan<T> span(const DLPackTensor& t) const noexcept {
        DeviceSpan<T> s;
        s.handle      = t.tensor().data;
        s.byte_offset = t.tensor().byte_offset;
        s.count       = t.size();
        s.device      = device;
        return s;
    }
};

/// Starts a device-memory call of an engine on `backend`:`device_id` with
/// input array `input`: checks that they share a GPU and resolves `stream`.
inline DeviceCall device_call(std::string_view what,
                              Backend backend,
                              int device_id,
                              const DeviceArray& input,
                              const py::object& stream) {
    const auto& dev = input.device;
    if (backend == Backend::kCPU) {
        throw std::invalid_argument(fmt::format(
            "{}: got a {} device array, but this instance runs on the cpu "
            "backend; construct it with backend='{}' or pass a NumPy array",
            what, to_string(dev.backend), to_string(dev.backend)));
    }
    if (dev.backend != backend || dev.id != device_id) {
        throw std::invalid_argument(fmt::format(
            "{}: the array is on {}:{} but this instance runs on {}:{}", what,
            to_string(dev.backend), dev.id, to_string(backend), device_id));
    }
    return {.what   = what,
            .device = dev,
            .stream = resolve_stream(stream, input.obj)};
}

/// The output of a device call: the array returned to Python and its memory.
struct DeviceOutput {
    py::object array;
    DLPackTensor tensor;
};

/// `out` (a writeable, C-contiguous T device array of at least prod(shape)
/// elements on the call's device, returned as a `shape` view of its start),
/// or a new `shape` array from `like`'s array library, allocated on the
/// call's stream.
template <typename T>
DeviceOutput device_output(const DeviceCall& call,
                           py::handle like,
                           const py::object& out,
                           const std::vector<SizeType>& shape) {
    SizeType count = 1;
    py::tuple shape_tuple(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i) {
        count *= shape[i];
        shape_tuple[i] = py::int_(shape[i]);
    }
    const bool fresh = out.is_none();
    py::object array =
        fresh ? device_helper("empty")(like, shape_tuple, dtype_name<T>(),
                                       call.stream)
              : out;
    auto tensor = call.consume<T>(array, "out", /*writable=*/true);
    if (tensor.size() < count) {
        throw std::invalid_argument(
            fmt::format("{}: out has {} elements, needs at least {}", call.what,
                        tensor.size(), count));
    }
    if (!fresh) {
        array = device_helper("view")(out, count, shape_tuple);
    }
    return {.array = std::move(array), .tensor = std::move(tensor)};
}

/// The shape of a DLPack tensor.
inline std::vector<SizeType> shape_of(const DLPackTensor& t) {
    const auto& dl = t.tensor();
    return std::vector<SizeType>(dl.shape, dl.shape + dl.ndim);
}

/// Device-memory save_history(): the engine's streaming history into `out`
/// (T elements), asynchronously on the stream. Returns `out`.
template <typename T, typename Engine>
py::object save_history_device(const Engine& eng,
                               std::string_view what,
                               const DeviceArray& out,
                               const py::object& stream) {
    const DeviceCall call =
        device_call(what, eng.backend(), eng.device(), out, stream);
    const auto t = call.consume<T>(out.obj, "out", /*writable=*/true);
    {
        const py::gil_scoped_release release;
        eng.save_history(call.span<T>(t), call.native_stream());
    }
    return out.obj;
}

/// Device-memory load_history(), asynchronously on the stream.
template <typename T, typename Engine>
void load_history_device(Engine& eng,
                         std::string_view what,
                         const DeviceArray& history,
                         const py::object& stream) {
    const DeviceCall call =
        device_call(what, eng.backend(), eng.device(), history, stream);
    const auto t = call.consume<T>(history.obj, "history", false);
    const py::gil_scoped_release release;
    eng.load_history(call.span<const T>(t), call.native_stream());
}

/// Checks a float32 DDMT-family waterfall (nchans, nsamps) or (nbeams,
/// nchans, nsamps) against the engine; returns nsamps.
template <typename Engine>
SizeType ddmt_waterfall_nsamps(const Engine& eng,
                               const DeviceCall& call,
                               const DLPackTensor& waterfall) {
    const auto shape = shape_of(waterfall);
    if (shape.size() != 2 && shape.size() != 3) {
        throw std::invalid_argument(
            fmt::format("{}: waterfall must be 2D (nchans, nsamps) or 3D "
                        "(nbeams, nchans, nsamps)",
                        call.what));
    }
    const auto nbeams = shape.size() == 3 ? shape[0] : 1;
    if (nbeams != eng.get_nbeams()) {
        throw std::invalid_argument(
            fmt::format("{}: waterfall has {} beam(s), but this instance was "
                        "configured for {}",
                        call.what, nbeams, eng.get_nbeams()));
    }
    const auto nchans = shape[shape.size() - 2];
    if (nchans != eng.get_plan().get_nchans()) {
        throw std::invalid_argument(
            fmt::format("{}: waterfall has {} channels, but this instance was "
                        "configured for {}",
                        call.what, nchans, eng.get_plan().get_nchans()));
    }
    return shape.back();
}

/// Device-memory execute() of the DDMT family (DDMT, SDMT, DDMTFFT): `In`
/// input, `Out` output of shape (n_dm, output_nsamps), or (nbeams, n_dm,
/// output_nsamps) when `beam_axis`; `like` is the input array. `run(in, out,
/// stream)` calls the engine with the GIL released.
template <typename In, typename Out, typename Engine, typename Run>
py::object ddmt_execute_device(Engine& eng,
                               const DeviceCall& call,
                               py::handle like,
                               const DLPackTensor& input,
                               SizeType nsamps,
                               bool beam_axis,
                               const py::object& out,
                               Run&& run) {
    std::vector<SizeType> shape{eng.get_plan().get_container().dm_arr.size(),
                                eng.get_output_nsamps(nsamps)};
    if (beam_axis) {
        shape.insert(shape.begin(), eng.get_nbeams());
    }
    auto result = device_output<Out>(call, like, out, shape);
    {
        const py::gil_scoped_release release;
        run(call.span<const In>(input), call.span<Out>(result.tensor),
            call.native_stream());
    }
    return std::move(result.array);
}

} // namespace dmt

namespace pybind11::detail {

template <> struct type_caster<dmt::DeviceArray> {
    PYBIND11_TYPE_CASTER(dmt::DeviceArray, const_name("DeviceArray"));

    bool load(handle src, bool /*convert*/) {
        const auto device = dmt::gpu_device_of(src);
        if (!device) {
            return false;
        }
        value = {.obj = reinterpret_borrow<object>(src), .device = *device};
        return true;
    }

    static handle cast(const dmt::DeviceArray& array,
                       return_value_policy /*policy*/,
                       handle /*parent*/) {
        return array.obj.inc_ref();
    }
};

} // namespace pybind11::detail
