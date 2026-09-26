#include "bindings/bind_cuda.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

#include <cuda/std/span>
#include <cuda_runtime.h>

#include "dmt/dmt.hpp"
#include "pybind_utils.hpp"

namespace dmt {
using algorithms::FDMTCUDA;
using algorithms::FDMTFFTCUDA;
using algorithms::kFDMTAutoFuse;

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

namespace {

void check_cuda(cudaError_t err, std::string_view what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::format("FDMTCUDA.{}: {}", what, cudaGetErrorString(err)));
    }
}

// Float input as a C-contiguous float32 array (other float dtypes are cast);
// uint8 must go through the packed overload.
py::array_t<float, py::array::c_style> float_waterfall(const py::array& obj,
                                                       std::string_view fn) {
    if (obj.dtype().kind() == 'u' && obj.itemsize() == 1) {
        throw py::type_error(std::format(
            "FDMTCUDA.{0}: got a uint8 waterfall; pass nbits for packed input "
            "({0}(waterfall_packed, nbits)) or convert to float32 explicitly",
            fn));
    }
    auto arr =
        py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(
            obj);
    if (!arr || (arr.ndim() != 2 && arr.ndim() != 3)) {
        throw std::runtime_error("Input waterfall must be a 2D (nchans, "
                                 "nsamps) or 3D (nbeams, nchans, nsamps) "
                                 "NumPy array.");
    }
    return arr;
}

py::array_t<uint8_t, py::array::c_style> packed_waterfall(const py::array& obj,
                                                          std::string_view fn) {
    if (obj.dtype().kind() != 'u' || obj.itemsize() != 1) {
        throw py::type_error(std::format(
            "FDMTCUDA.{}: nbits given, so waterfall must be a packed uint8 "
            "array",
            fn));
    }
    auto arr = py::array_t<uint8_t, py::array::c_style>::ensure(obj);
    if (!arr || (arr.ndim() != 2 && arr.ndim() != 3)) {
        throw std::runtime_error(
            "Packed waterfall must be a 2D (nchans, row_bytes) or 3D (nbeams, "
            "nchans, row_bytes) uint8 NumPy array.");
    }
    return arr;
}

// Host copy of the device floats at `d_ptr`, shaped `shape`.
py::array_t<float> device_to_host(const float* d_ptr,
                                  std::vector<py::ssize_t> shape) {
    py::array_t<float, py::array::c_style> host(shape);
    check_cuda(cudaMemcpy(host.mutable_data(), d_ptr,
                          static_cast<std::size_t>(host.size()) * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "view");
    return host;
}

// Device buffers of one Python-driven stepper block, owned by the FDMTCUDA
// Python object (a capsule attribute): the staged input and the
// nbeams * plan.buffer_size dmt / ping-pong buffer. Allocated on the first
// reset() and reused.
class FDMTCUDAStepperStage {
public:
    explicit FDMTCUDAStepperStage(int device_id) : m_device_id(device_id) {}
    ~FDMTCUDAStepperStage() {
        cudaSetDevice(m_device_id);
        cudaFree(m_waterfall);
        cudaFree(m_dmt);
    }
    FDMTCUDAStepperStage(const FDMTCUDAStepperStage&)            = delete;
    FDMTCUDAStepperStage& operator=(const FDMTCUDAStepperStage&) = delete;
    FDMTCUDAStepperStage(FDMTCUDAStepperStage&&)                 = delete;
    FDMTCUDAStepperStage& operator=(FDMTCUDAStepperStage&&)      = delete;

    // Copies `bytes` of host input into the staged device input.
    const void* upload(const void* host, std::size_t bytes) {
        check_cuda(cudaSetDevice(m_device_id), "reset");
        if (bytes > m_waterfall_bytes) {
            cudaFree(m_waterfall);
            m_waterfall       = nullptr;
            m_waterfall_bytes = 0;
            check_cuda(cudaMalloc(&m_waterfall, bytes), "reset");
            m_waterfall_bytes = bytes;
        }
        check_cuda(cudaMemcpy(m_waterfall, host, bytes, cudaMemcpyHostToDevice),
                   "reset");
        return m_waterfall;
    }

    cuda::std::span<float> dmt_span(const FDMTCUDA& fdmt) {
        const auto elems =
            fdmt.get_nbeams() * fdmt.get_plan().get_buffer_size();
        if (elems > m_dmt_elems) {
            cudaFree(m_dmt);
            m_dmt       = nullptr;
            m_dmt_elems = 0;
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&m_dmt),
                                  elems * sizeof(float)),
                       "reset");
            m_dmt_elems = elems;
        }
        return {m_dmt, elems};
    }

    // Each beam's leading get_dmt_size() values (the root transform).
    py::array_t<float> result_to_host(const FDMTCUDA& fdmt) const {
        const auto& plan  = fdmt.get_plan();
        const auto nbeams = fdmt.get_nbeams();
        const auto ndms   = static_cast<py::ssize_t>(plan.get_dmt_ndms());
        const auto nsamps = static_cast<py::ssize_t>(plan.get_dmt_nsamps());
        std::vector<py::ssize_t> shape{ndms, nsamps};
        if (nbeams > 1) {
            shape.insert(shape.begin(), static_cast<py::ssize_t>(nbeams));
        }
        py::array_t<float, py::array::c_style> host(shape);
        const auto row = plan.get_dmt_size() * sizeof(float);
        check_cuda(cudaMemcpy2D(host.mutable_data(), row, m_dmt,
                                plan.get_buffer_size() * sizeof(float), row,
                                nbeams, cudaMemcpyDeviceToHost),
                   "finalize");
        return host;
    }

private:
    int m_device_id;
    void* m_waterfall{nullptr};
    std::size_t m_waterfall_bytes{0};
    float* m_dmt{nullptr};
    SizeType m_dmt_elems{0};
};

FDMTCUDAStepperStage& stepper_stage(py::object& self, const FDMTCUDA& fdmt) {
    if (!py::hasattr(self, "_stepper_stage")) {
        auto* stage = new FDMTCUDAStepperStage(fdmt.get_device_id()); // NOLINT
        self.attr("_stepper_stage") = py::capsule(stage, [](void* p) {
            delete static_cast<FDMTCUDAStepperStage*>(p); // NOLINT
        });
    }
    return *static_cast<FDMTCUDAStepperStage*>(
        self.attr("_stepper_stage").cast<py::capsule>().get_pointer());
}

} // namespace

void bind_fdmt_cuda(py::module_& mod) {
    py::class_<FDMTCUDA>(mod, "FDMTCUDA", py::dynamic_attr(),
                         R"doc(
        Incoherent Fast Dispersion Measure Transform on CUDA.

        Same constructor arguments as :class:`~dmtlib.libdmt.FDMTCPU`, with
        ``device_id`` instead of ``nthreads``. ``fuse_levels=None`` (default)
        picks the fused-kernel depth from the plan (shared-memory tile of at
        least 256 samples in 48 KiB); an explicit depth is reduced until it
        fits the device's shared memory.

        See also
        --------
        dmtlib.libdmt.FDMTCPU
        )doc")
        .def(py::init([](float f_min, float f_max, SizeType nchans,
                         SizeType nsamps, float tsamp, IndexType dt_max,
                         IndexType dt_min, SizeType dt_step,
                         bool use_box_smearing, std::string_view mode,
                         int verbose, int device_id, SizeType nbeams,
                         std::optional<SizeType> fuse_levels, bool int_tree) {
                 return FDMTCUDA(f_min, f_max, nchans, nsamps, tsamp, dt_max,
                                 dt_min, dt_step, use_box_smearing, mode,
                                 verbose, device_id, nbeams,
                                 fuse_levels.value_or(kFDMTAutoFuse), int_tree);
             }),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("dt_max"),
             py::arg("dt_min") = 0, py::arg("dt_step") = 1,
             py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
             py::arg("verbose") = 0, py::arg("device_id") = 0,
             py::arg("nbeams") = 1, py::arg("fuse_levels") = py::none(),
             py::arg("int_tree") = true)
        .def(py::init([](float f_min, float f_max, SizeType nchans,
                         SizeType nsamps, float tsamp,
                         const py::object& dt_grid, const py::object& dt_arr,
                         const py::object& dm_grid, const py::object& dm_arr,
                         bool use_box_smearing, std::string_view mode,
                         int verbose, int device_id, SizeType nbeams,
                         std::optional<SizeType> fuse_levels, bool int_tree) {
                 const auto [type, obj] =
                     resolve_custom_grid(dt_grid, dt_arr, dm_grid, dm_arr);
                 const auto fuse = fuse_levels.value_or(kFDMTAutoFuse);
                 if (type == CustomGridType::kDt) {
                     return FDMTCUDA(f_min, f_max, nchans, nsamps, tsamp,
                                     extract_dt_grid(obj), use_box_smearing,
                                     mode, verbose, device_id, nbeams, fuse,
                                     int_tree);
                 }
                 return FDMTCUDA(f_min, f_max, nchans, nsamps, tsamp,
                                 extract_dm_grid(obj), use_box_smearing, mode,
                                 verbose, device_id, nbeams, fuse, int_tree);
             }),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::kw_only(),
             py::arg("dt_grid") = py::none(), py::arg("dt_arr") = py::none(),
             py::arg("dm_grid") = py::none(), py::arg("dm_arr") = py::none(),
             py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
             py::arg("verbose") = 0, py::arg("device_id") = 0,
             py::arg("nbeams") = 1, py::arg("fuse_levels") = py::none(),
             py::arg("int_tree") = true)
        .def_property_readonly(
            "plan", &FDMTCUDA::get_plan,
            "Get the FDMTPlan object containing transform details.")
        .def_property_readonly("nbeams", &FDMTCUDA::get_nbeams)
        .def_property_readonly("dt_grid_final",
                               [](FDMTCUDA& fdmt) {
                                   return as_pyarray(
                                       fdmt.get_plan().get_dt_grid_final());
                               })
        .def_property_readonly("dm_grid_final",
                               [](FDMTCUDA& fdmt) {
                                   return as_pyarray(
                                       fdmt.get_plan().get_dm_grid_final());
                               })
        .def("get_dt_grid_final",
             [](FDMTCUDA& fdmt) {
                 return as_pyarray(fdmt.get_plan().get_dt_grid_final());
             })
        .def("get_dm_grid_final",
             [](FDMTCUDA& fdmt) {
                 return as_pyarray(fdmt.get_plan().get_dm_grid_final());
             })
        .def(
            "execute",
            [](FDMTCUDA& fdmt, const py::array& waterfall_obj,
               const std::optional<py::array>& out) -> py::object {
                const auto waterfall =
                    float_waterfall(waterfall_obj, "execute");
                return fdmt_execute_to_array(
                    fdmt, "FDMTCUDA", waterfall.ndim() == 3,
                    static_cast<SizeType>(waterfall.size()),
                    fdmt.get_nbeams() * fdmt.get_plan().get_nchans() *
                        fdmt.get_plan().get_nsamps(),
                    out, [&](std::span<float> dmt) {
                        fdmt.execute(std::span<const float>(waterfall.data(),
                                                            waterfall.size()),
                                     dmt);
                    });
            },
            py::arg("waterfall"), py::kw_only(), py::arg("out") = py::none(),
            R"doc(
            Run the FDMT transform on the GPU from host arrays.

            Same arguments, layout and return value as
            :meth:`dmtlib.libdmt.FDMTCPU.execute` (including ``out``). The
            input is copied to a persistent device staging buffer and only the
            result is copied back; blocks until it is on the host.
            )doc")
        .def(
            "execute",
            [](FDMTCUDA& fdmt, const py::array& waterfall_obj, SizeType nbits,
               const std::optional<py::array>& out) -> py::object {
                const auto packed = packed_waterfall(waterfall_obj, "execute");
                return fdmt_execute_to_array(
                    fdmt, "FDMTCUDA", packed.ndim() == 3,
                    static_cast<SizeType>(packed.size()),
                    fdmt.get_nbeams() * fdmt.get_plan().get_nchans() *
                        (((fdmt.get_plan().get_nsamps() * nbits) + 7) / 8),
                    out, [&](std::span<float> dmt) {
                        fdmt.execute(std::span<const uint8_t>(packed.data(),
                                                              packed.size()),
                                     nbits, dmt);
                    });
            },
            py::arg("waterfall_packed"), py::arg("nbits"), py::kw_only(),
            py::arg("out") = py::none(),
            R"doc(
            Run the FDMT transform on packed low-bit input on the GPU. Only the
            packed bytes are copied to the device. Same layout and output as
            :meth:`dmtlib.libdmt.FDMTCPU.execute` with ``nbits``.
            )doc")
        .def_property_readonly(
            "fuse_levels", &FDMTCUDA::get_fuse_levels,
            "Fusion depth the device execute() uses (0 = unfused).")
        .def_property_readonly("int_tree", &FDMTCUDA::get_int_tree,
                               "Whether packed input uses the integer tree.")
        .def_property_readonly("device_id", &FDMTCUDA::get_device_id)
        .def_property_readonly(
            "memory_usage", &FDMTCUDA::get_memory_usage,
            "FDMTMemoryUsage: device bytes allocated by the engine.")
        .def("get_effective_variance", &FDMTCUDA::get_effective_variance,
             py::arg("dm_idx"), py::arg("boxcar_width") = 1,
             "Theoretical noise variance for a DM trial and boxcar width "
             "(same as FDMTCPU).")
        .def("get_effective_sigma", &FDMTCUDA::get_effective_sigma,
             py::arg("dm_idx"), py::arg("boxcar_width") = 1,
             "Theoretical noise standard deviation (same as FDMTCPU).")
        .def(
            "get_effective_variance_grid",
            [](const FDMTCUDA& fdmt, SizeType boxcar_width) {
                return as_pyarray(
                    fdmt.get_effective_variance_grid(boxcar_width));
            },
            py::arg("boxcar_width") = 1)
        .def(
            "get_effective_sigma_grid",
            [](const FDMTCUDA& fdmt, SizeType boxcar_width) {
                return as_pyarray(fdmt.get_effective_sigma_grid(boxcar_width));
            },
            py::arg("boxcar_width") = 1)
        // Stepper from host arrays: the block is staged in device buffers
        // owned by this Python object; views and finalize() return host
        // copies.
        .def(
            "reset",
            [](py::object self, const py::array& waterfall_obj,
               SizeType nbits) {
                auto& fdmt        = self.cast<FDMTCUDA&>();
                const auto packed = packed_waterfall(waterfall_obj, "reset");
                auto& stage       = stepper_stage(self, fdmt);
                const auto* d_wf = stage.upload(packed.data(), packed.nbytes());
                fdmt.reset(cuda::std::span<const uint8_t>(
                               static_cast<const uint8_t*>(d_wf),
                               static_cast<SizeType>(packed.size())),
                           nbits, stage.dmt_span(fdmt));
            },
            py::arg("waterfall_packed"), py::arg("nbits"),
            "Stage a packed low-bit block on the device and start the stepper "
            "(see execute(waterfall_packed, nbits)).")
        .def(
            "reset",
            [](py::object self, const py::array& waterfall_obj) {
                auto& fdmt           = self.cast<FDMTCUDA&>();
                const auto waterfall = float_waterfall(waterfall_obj, "reset");
                auto& stage          = stepper_stage(self, fdmt);
                const auto* d_wf =
                    stage.upload(waterfall.data(), waterfall.nbytes());
                fdmt.reset(cuda::std::span<const float>(
                               static_cast<const float*>(d_wf),
                               static_cast<SizeType>(waterfall.size())),
                           stage.dmt_span(fdmt));
            },
            py::arg("waterfall"),
            R"doc(
            Stage a host block on the device and start the stepper at level 0.

            As on the CPU: in ``mode='valid'`` a block must be finalized
            before the next ``reset``/``execute`` (else RuntimeError);
            :meth:`reset_history` abandons it. The stepper never fuses.
            )doc")
        .def(
            "advance",
            [](FDMTCUDA& fdmt, SizeType levels) { fdmt.advance(levels); },
            py::arg("levels") = 1, "Advance execution by `levels` levels.")
        .def(
            "advance_until_remaining",
            [](FDMTCUDA& fdmt, SizeType remaining) {
                fdmt.advance_until_remaining(remaining);
            },
            py::arg("remaining_levels"),
            "Advance until `remaining_levels` levels remain before the root.")
        .def(
            "view_level_data",
            [](const FDMTCUDA& fdmt) {
                const auto span = fdmt.view_level_data();
                return device_to_host(span.data(),
                                      {static_cast<py::ssize_t>(span.size())});
            },
            "Host copy of the current level's state (flat).")
        .def(
            "view_subband_data",
            [](const FDMTCUDA& fdmt, SizeType subband_idx) {
                const auto v = fdmt.view_subband(subband_idx);
                return device_to_host(v.data.data(),
                                      {static_cast<py::ssize_t>(v.ndt),
                                       static_cast<py::ssize_t>(v.nsamps)});
            },
            py::arg("subband_idx"),
            "Host copy (ndt, nsamps) of one sub-band at the current level.")
        .def(
            "view_subband",
            [](const FDMTCUDA& fdmt, SizeType subband_idx) {
                const auto v = fdmt.view_subband(subband_idx);
                py::dict out;
                out["data"] = device_to_host(
                    v.data.data(), {static_cast<py::ssize_t>(v.ndt),
                                    static_cast<py::ssize_t>(v.nsamps)});
                out["subband_idx"] = v.subband_idx;
                out["ndt"]         = v.ndt;
                out["nsamps"]      = v.nsamps;
                out["f_start"]     = v.f_start;
                out["f_end"]       = v.f_end;
                out["dt_grid"]     = py::array_t<IndexType>(
                    static_cast<py::ssize_t>(v.dt_grid.size()),
                    v.dt_grid.data());
                return out;
            },
            py::arg("subband_idx"),
            "Host copy of one sub-band plus its metadata, as a dict.")
        .def(
            "finalize",
            [](py::object self) {
                auto& fdmt = self.cast<FDMTCUDA&>();
                fdmt.finalize();
                return stepper_stage(self, fdmt).result_to_host(fdmt);
            },
            "Advance to the root and return a host copy of the transform, "
            "(n_delays, n_times) or (nbeams, n_delays, n_times).")
        .def_property_readonly("current_level", &FDMTCUDA::current_level)
        .def_property_readonly("total_levels", &FDMTCUDA::total_levels)
        .def_property_readonly("remaining_levels", &FDMTCUDA::remaining_levels)
        .def_property_readonly("num_subbands", &FDMTCUDA::num_subbands)
        .def_property_readonly("is_finished", &FDMTCUDA::is_finished)
        .def("reset_history", &FDMTCUDA::reset_history,
             "Reset the valid-mode streaming history (a cold start) and "
             "abandon any unfinished stepper block.");

    py::class_<FDMTFFTCUDA>(mod, "FDMTFFTCUDA", py::dynamic_attr(),
                            R"doc(
        FFT-domain FDMT on CUDA.

        Same constructor arguments as :class:`~dmtlib.libdmt.FDMTFFTCPU`, with
        ``device_id`` instead of ``nthreads``.
        )doc")
        .def(py::init<float, float, SizeType, SizeType, float, IndexType,
                      IndexType, SizeType, bool, std::string_view, bool, int,
                      SizeType>(),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::arg("dt_max"),
             py::arg("dt_min") = 0, py::arg("dt_step") = 1,
             py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
             py::arg("verbose") = 0, py::arg("device_id") = 0,
             py::arg("nbeams") = 1)
        .def(py::init([](float f_min, float f_max, SizeType nchans,
                         SizeType nsamps, float tsamp,
                         const py::object& dt_grid, const py::object& dt_arr,
                         const py::object& dm_grid, const py::object& dm_arr,
                         bool use_box_smearing, std::string_view mode,
                         int verbose, int device_id, SizeType nbeams) {
                 const auto [type, obj] =
                     resolve_custom_grid(dt_grid, dt_arr, dm_grid, dm_arr);
                 if (type == CustomGridType::kDt) {
                     return FDMTFFTCUDA(f_min, f_max, nchans, nsamps, tsamp,
                                        extract_dt_grid(obj), use_box_smearing,
                                        mode, verbose, device_id, nbeams);
                 }
                 return FDMTFFTCUDA(f_min, f_max, nchans, nsamps, tsamp,
                                    extract_dm_grid(obj), use_box_smearing,
                                    mode, verbose, device_id, nbeams);
             }),
             py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
             py::arg("nsamps"), py::arg("tsamp"), py::kw_only(),
             py::arg("dt_grid") = py::none(), py::arg("dt_arr") = py::none(),
             py::arg("dm_grid") = py::none(), py::arg("dm_arr") = py::none(),
             py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
             py::arg("verbose") = 0, py::arg("device_id") = 0,
             py::arg("nbeams") = 1)
        .def_property_readonly("plan", &FDMTFFTCUDA::get_plan)
        .def_property_readonly("nbeams", &FDMTFFTCUDA::get_nbeams)
        .def_property_readonly("dt_grid_final",
                               [](FDMTFFTCUDA& fdmt) {
                                   return as_pyarray(
                                       fdmt.get_plan().get_dt_grid_final());
                               })
        .def_property_readonly("dm_grid_final",
                               [](FDMTFFTCUDA& fdmt) {
                                   return as_pyarray(
                                       fdmt.get_plan().get_dm_grid_final());
                               })
        .def(
            "execute",
            [](FDMTFFTCUDA& fdmt,
               const py::array_t<float, py::array::c_style>& waterfall)
                -> py::object {
                const auto nbeams     = fdmt.get_nbeams();
                const auto& plan      = fdmt.get_plan();
                const auto ndms       = plan.get_dmt_ndms();
                const auto nsamps_out = plan.get_dmt_nsamps();
                if (waterfall.ndim() == 2) {
                    py::array_t<float, py::array::c_style> result(
                        {ndms, nsamps_out});
                    fdmt.execute(
                        std::span<const float>(waterfall.data(),
                                               waterfall.size()),
                        std::span<float>(result.mutable_data(), result.size()));
                    return result;
                }
                if (waterfall.ndim() == 3) {
                    py::array_t<float, py::array::c_style> result(
                        {nbeams, ndms, nsamps_out});
                    fdmt.execute(
                        std::span<const float>(waterfall.data(),
                                               waterfall.size()),
                        std::span<float>(result.mutable_data(), result.size()));
                    return result;
                }
                throw std::runtime_error("Input waterfall must be 2D or 3D.");
            },
            py::arg("waterfall"))
        .def(
            "reset",
            [](py::object self,
               const py::array_t<float, py::array::c_style>& waterfall,
               std::optional<py::array_t<float, py::array::c_style>> dmt_opt) {
                auto& fdmt = self.cast<FDMTFFTCUDA&>();
                py::array_t<float, py::array::c_style> dmt;
                if (dmt_opt.has_value()) {
                    dmt = *dmt_opt;
                } else {
                    dmt = py::array_t<float, py::array::c_style>(
                        fdmt.get_nbeams() * fdmt.get_plan().get_dmt_size());
                }
                self.attr("_waterfall_buffer") = waterfall;
                self.attr("_dmt_buffer")       = dmt;
                fdmt.reset(
                    std::span<const float>(waterfall.data(), waterfall.size()),
                    std::span<float>(dmt.mutable_data(), dmt.size()));
            },
            py::arg("waterfall"), py::arg("dmt") = py::none())
        .def(
            "advance",
            [](FDMTFFTCUDA& fdmt, SizeType levels) { fdmt.advance(levels); },
            py::arg("levels") = 1)
        .def(
            "advance_until_remaining",
            [](FDMTFFTCUDA& fdmt, SizeType remaining) {
                fdmt.advance_until_remaining(remaining);
            },
            py::arg("remaining_levels"))
        .def("view_level_data",
             [](py::object self) {
                 auto& fdmt = self.cast<FDMTFFTCUDA&>();
                 auto span  = fdmt.view_level_data();
                 py::array_t<float, py::array::c_style> host(
                     static_cast<py::ssize_t>(span.size()));
                 cudaMemcpy(host.mutable_data(), span.data(),
                            span.size() * sizeof(float),
                            cudaMemcpyDeviceToHost);
                 self.attr("_view_level_host") = host;
                 return host;
             })
        .def(
            "view_subband_data",
            [](py::object self, SizeType subband_idx) {
                auto& fdmt = self.cast<FDMTFFTCUDA&>();
                auto v     = fdmt.view_subband(subband_idx);
                py::array_t<float, py::array::c_style> host({v.ndt, v.nsamps});
                cudaMemcpy(host.mutable_data(), v.data.data(),
                           v.data.size() * sizeof(float),
                           cudaMemcpyDeviceToHost);
                return host;
            },
            py::arg("subband_idx"))
        .def(
            "view_subband",
            [](py::object self, SizeType subband_idx) {
                auto& fdmt = self.cast<FDMTFFTCUDA&>();
                auto v     = fdmt.view_subband(subband_idx);
                py::array_t<float, py::array::c_style> host({v.ndt, v.nsamps});
                cudaMemcpy(host.mutable_data(), v.data.data(),
                           v.data.size() * sizeof(float),
                           cudaMemcpyDeviceToHost);
                py::dict out;
                out["data"]        = host;
                out["subband_idx"] = v.subband_idx;
                out["ndt"]         = v.ndt;
                out["nsamps"]      = v.nsamps;
                out["f_start"]     = v.f_start;
                out["f_end"]       = v.f_end;
                return out;
            },
            py::arg("subband_idx"))
        .def("finalize",
             [](py::object self) {
                 auto& fdmt = self.cast<FDMTFFTCUDA&>();
                 fdmt.finalize();
                 py::array_t<float, py::array::c_style> dmt =
                     self.attr("_dmt_buffer")
                         .cast<py::array_t<float, py::array::c_style>>();
                 const auto& plan   = fdmt.get_plan();
                 const auto ncoords = plan.get_dmt_ndms();
                 const auto nsamps  = plan.get_dmt_nsamps();
                 return py::array_t<float>(
                     {ncoords, nsamps}, {nsamps * sizeof(float), sizeof(float)},
                     dmt.data(), self);
             })
        .def("reset_history", &FDMTFFTCUDA::reset_history)
        .def("get_effective_variance", &FDMTFFTCUDA::get_effective_variance,
             py::arg("dm_idx"), py::arg("boxcar_width") = 1)
        .def("get_effective_sigma", &FDMTFFTCUDA::get_effective_sigma,
             py::arg("dm_idx"), py::arg("boxcar_width") = 1)
        .def(
            "get_effective_variance_grid",
            [](const FDMTFFTCUDA& fdmt, SizeType boxcar_width) {
                return as_pyarray(
                    fdmt.get_effective_variance_grid(boxcar_width));
            },
            py::arg("boxcar_width") = 1)
        .def(
            "get_effective_sigma_grid",
            [](const FDMTFFTCUDA& fdmt, SizeType boxcar_width) {
                return as_pyarray(fdmt.get_effective_sigma_grid(boxcar_width));
            },
            py::arg("boxcar_width") = 1)
        .def_property_readonly("current_level", &FDMTFFTCUDA::current_level)
        .def_property_readonly("total_levels", &FDMTFFTCUDA::total_levels)
        .def_property_readonly("remaining_levels",
                               &FDMTFFTCUDA::remaining_levels)
        .def_property_readonly("num_subbands", &FDMTFFTCUDA::num_subbands)
        .def_property_readonly("is_finished", &FDMTFFTCUDA::is_finished);
}

} // namespace dmt
