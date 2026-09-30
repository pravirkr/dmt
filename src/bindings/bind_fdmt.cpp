#include "bindings/bind.hpp"

#include <algorithm>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dmt/dmt.hpp"
#include "pybind_utils.hpp"

namespace dmt {
using algorithms::FDMT;
using algorithms::FDMTFFT;
using algorithms::FDMTMemoryUsage;
using algorithms::kFDMTAutoFuse;
using plans::FDMTPlan;

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

namespace {

std::vector<uint8_t>
fft_kill_mask(const std::optional<py::array_t<uint8_t>>& m) {
    std::vector<uint8_t> v;
    if (m.has_value()) {
        v.assign(m->data(), m->data() + m->size());
    }
    return v;
}

// FDMTFFT output for an input of `ndim` dimensions.
py::array_t<float, py::array::c_style> fft_output(const FDMTFFT& fdmt,
                                                  py::ssize_t ndim) {
    const auto& plan = fdmt.get_plan();
    const auto ndms  = plan.get_dmt_ndms();
    const auto nout  = plan.get_dmt_nsamps();
    if (ndim == 2) {
        if (fdmt.get_nbeams() != 1) {
            throw std::invalid_argument(
                "FDMTFFT: a 2D waterfall needs nbeams == 1");
        }
        return py::array_t<float, py::array::c_style>({ndms, nout});
    }
    if (ndim == 3) {
        return py::array_t<float, py::array::c_style>(
            {fdmt.get_nbeams(), ndms, nout});
    }
    throw std::runtime_error("Packed waterfall must be a 2D or 3D (beams "
                             "first) uint8 NumPy array.");
}

} // namespace

void bind_fdmt(py::module_& mod) {
    py::class_<FDMTMemoryUsage>(mod, "FDMTMemoryUsage", R"doc(
        Memory an FDMT engine allocates at construction, in bytes (host
        memory on the CPU backend, device memory on a GPU backend). ``execute`` allocates
        nothing further; ``output`` is the caller-side output buffer per call
        (managed automatically in Python) and is not part of ``total``.
        )doc")
        .def_readonly("plan", &FDMTMemoryUsage::plan)
        .def_readonly("state", &FDMTMemoryUsage::state)
        .def_readonly("history", &FDMTMemoryUsage::history)
        .def_readonly("workspace", &FDMTMemoryUsage::workspace)
        .def_readonly("output", &FDMTMemoryUsage::output)
        .def_property_readonly("total", &FDMTMemoryUsage::total)
        .def("__repr__", [](const FDMTMemoryUsage& m) {
            return std::format(
                "FDMTMemoryUsage(plan={}, state={}, history={}, workspace={}, "
                "total={}, output={})",
                m.plan, m.state, m.history, m.workspace, m.total(), m.output);
        });

    py::class_<FDMT>(mod, "FDMT", py::dynamic_attr(),
                     R"doc(
        Incoherent Fast Dispersion Measure Transform.

        Transforms a frequency–time waterfall into a DM/delay–time plane.
        ``mode='valid'`` keeps a streaming history so consecutive blocks
        join without a wrap artefact.

        Parameters
        ----------
        f_min, f_max : float
            Band edges in MHz.
        nchans, nsamps : int
            Channels and samples per block.
        tsamp : float
            Sampling interval in seconds.
        dt_max : int
            Maximum delay trial in samples (regular grid constructor).
        dt_min, dt_step : int, optional
            Delay-grid range and stride.
        use_box_smearing : bool, optional
            Account for intra-channel smearing in the tree.
        mode : {'valid', 'full', 'roll'}, optional
            Output time alignment.
        nthreads : int, optional
            OpenMP threads on the CPU backend (default 1).
        nbeams : int, optional
            Independent beams packed as ``(nbeams, nchans, nsamps)``.
        fuse_levels : int or None, optional
            Performance parameter (most users keep the default): ``execute``
            fuses level-0 initialisation with the first ``fuse_levels`` tree
            merges (cache-resident channel groups on the CPU, shared-memory
            tiles on a GPU). 0 = original level-by-level path; ``None``
            (default) picks the depth from the plan (and ``nthreads`` on the
            CPU). Bit-identical output.
        int_tree : bool, optional
            Performance parameter: packed input stores tree levels as
            uint8/uint16 where the exact bound allows (default True). Set
            False to inspect every level of a packed stepper run.
        backend : {'cpu', 'cuda', 'hip'}, optional
            Keyword-only. Where to run (default ``'cpu'``). See
            :func:`available_backends` for the backends in this build; any
            other raises ``ValueError``.
        device : int, optional
            Keyword-only. Device ordinal on a GPU backend (default 0).
        dt_grid, dt_arr, dm_grid, dm_arr : array_like, optional
            Keyword-only custom trial grid. Provide exactly one of these.

        Input and output are NumPy (host) arrays on every backend; a GPU
        backend copies the input to the device and the result back, and
        blocks until it is on the host. Every backend gives bit-identical
        results.

        All working memory is allocated by the constructor (see
        :attr:`memory_usage`); ``execute`` allocates only its output array,
        and nothing when ``out=`` is given.

        See also
        --------
        FDMTPlan, FDMTFFT, compute_fdmt, available_backends
        )doc")
        .def(py::init([](float f_min, float f_max, SizeType nchans,
                         SizeType nsamps, float tsamp, IndexType dt_max,
                         IndexType dt_min, SizeType dt_step,
                         bool use_box_smearing, std::string_view mode,
                         int nthreads, SizeType nbeams,
                         std::optional<SizeType> fuse_levels, bool int_tree,
                         std::string_view backend, int device) {
                 return FDMT(f_min, f_max, nchans, nsamps, tsamp, dt_max,
                             dt_min, dt_step, use_box_smearing, mode,
                             make_exec(backend, nthreads, device), nbeams,
                             fuse_levels.value_or(kFDMTAutoFuse), int_tree);
             }),
             "f_min"_a, "f_max"_a, "nchans"_a, "nsamps"_a, "tsamp"_a,
             "dt_max"_a, "dt_min"_a = 0, "dt_step"_a = 1,
             "use_box_smearing"_a = true, "mode"_a = "valid", "nthreads"_a = 1,
             "nbeams"_a = 1, "fuse_levels"_a = py::none(), "int_tree"_a = true,
             py::kw_only(), "backend"_a = "cpu", "device"_a = 0)
        .def(
            py::init([](float f_min, float f_max, SizeType nchans,
                        SizeType nsamps, float tsamp, const py::object& dt_grid,
                        const py::object& dt_arr, const py::object& dm_grid,
                        const py::object& dm_arr, bool use_box_smearing,
                        std::string_view mode, int nthreads, SizeType nbeams,
                        std::optional<SizeType> fuse_levels, bool int_tree,
                        std::string_view backend, int device) {
                const auto [type, obj] =
                    resolve_custom_grid(dt_grid, dt_arr, dm_grid, dm_arr);
                const auto fuse = fuse_levels.value_or(kFDMTAutoFuse);
                const auto exec = make_exec(backend, nthreads, device);
                if (type == CustomGridType::kDt) {
                    return FDMT(f_min, f_max, nchans, nsamps, tsamp,
                                extract_dt_grid(obj), use_box_smearing, mode,
                                exec, nbeams, fuse, int_tree);
                }
                return FDMT(f_min, f_max, nchans, nsamps, tsamp,
                            extract_dm_grid(obj), use_box_smearing, mode, exec,
                            nbeams, fuse, int_tree);
            }),
            py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
            py::arg("nsamps"), py::arg("tsamp"), py::kw_only(),
            py::arg("dt_grid") = py::none(), py::arg("dt_arr") = py::none(),
            py::arg("dm_grid") = py::none(), py::arg("dm_arr") = py::none(),
            py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
            py::arg("nthreads") = 1, py::arg("nbeams") = 1,
            py::arg("fuse_levels") = py::none(), py::arg("int_tree") = true,
            py::arg("backend") = "cpu", py::arg("device") = 0)
        .def_property_readonly(
            "plan", &FDMT::get_plan,
            "Get the FDMTPlan object containing transform details.")
        .def_property_readonly(
            "backend",
            [](const FDMT& fdmt) {
                return std::string(to_string(fdmt.backend()));
            },
            "Backend this instance runs on ('cpu', 'cuda', ...).")
        .def_property_readonly("nthreads", &FDMT::nthreads,
                               "OpenMP threads (CPU backend); 1 on a GPU.")
        .def_property_readonly("device", &FDMT::device,
                               "Device ordinal (GPU backend); -1 on the CPU.")
        .def_property_readonly(
            "fuse_levels", &FDMT::get_fuse_levels,
            "Fusion depth execute() uses (0 = unfused); resolves the "
            "automatic default for this plan and backend.")
        .def_property_readonly("int_tree", &FDMT::get_int_tree,
                               "Whether packed input uses the integer tree.")
        .def_property_readonly(
            "memory_usage", &FDMT::get_memory_usage,
            "FDMTMemoryUsage: bytes allocated at construction (host memory on "
            "the CPU backend, device memory on a GPU).")
        .def("summary", &FDMT::summary,
             "Human-readable summary: plan, engine configuration and memory.")
        .def_property_readonly(
            "nbeams", &FDMT::get_nbeams,
            "Number of beams this instance processes together (see the "
            "nbeams constructor argument). Supports 2D arrays (nchans, nsamps) "
            "when nbeams=1, and 3D arrays (nbeams, nchans, nsamps) when "
            "nbeams>=1.")
        .def_property_readonly("dt_grid_final",
                               [](FDMT& fdmt) {
                                   return as_pyarray(
                                       fdmt.get_plan().get_dt_grid_final());
                               })
        .def_property_readonly("dm_grid_final",
                               [](FDMT& fdmt) {
                                   return as_pyarray(
                                       fdmt.get_plan().get_dm_grid_final());
                               })
        .def("get_dt_grid_final",
             [](FDMT& fdmt) {
                 return as_pyarray(fdmt.get_plan().get_dt_grid_final());
             })
        .def("get_dm_grid_final",
             [](FDMT& fdmt) {
                 return as_pyarray(fdmt.get_plan().get_dm_grid_final());
             })
        .def("get_effective_variance", &FDMT::get_effective_variance,
             py::arg("dm_idx"), py::arg("boxcar_width") = 1,
             R"doc(
             Compute theoretical noise variance for a DM trial and boxcar width.

             Parameters
             ----------
             dm_idx : int
                 Index into the final DM/dt trial grid.
             boxcar_width : int, optional
                 Width of subsequent boxcar filter in samples (default 1).

             Returns
             -------
             float
                 Predicted noise variance scaled from input noise.
             )doc")
        .def("get_effective_sigma", &FDMT::get_effective_sigma,
             py::arg("dm_idx"), py::arg("boxcar_width") = 1,
             R"doc(
             Compute theoretical noise standard deviation (sigma) for a DM trial and boxcar width.

             Parameters
             ----------
             dm_idx : int
                 Index into the final DM/dt trial grid.
             boxcar_width : int, optional
                 Width of subsequent boxcar filter in samples (default 1).

             Returns
             -------
             float
                 Predicted noise standard deviation (sqrt(variance)).
             )doc")
        .def(
            "get_effective_variance_grid",
            [](const FDMT& fdmt, SizeType boxcar_width) {
                return as_pyarray(
                    fdmt.get_effective_variance_grid(boxcar_width));
            },
            py::arg("boxcar_width") = 1,
            R"doc(
            Compute theoretical noise variance for all DM trials across the grid.

            Parameters
            ----------
            boxcar_width : int, optional
                Width of subsequent boxcar filter in samples (default 1).

            Returns
            -------
            numpy.ndarray
                1D float32 array of shape ``(n_dm,)`` of theoretical variances.
            )doc")
        .def(
            "get_effective_sigma_grid",
            [](const FDMT& fdmt, SizeType boxcar_width) {
                return as_pyarray(fdmt.get_effective_sigma_grid(boxcar_width));
            },
            py::arg("boxcar_width") = 1,
            R"doc(
            Compute theoretical noise standard deviation (sigma) for all DM trials.

            Parameters
            ----------
            boxcar_width : int, optional
                Width of subsequent boxcar filter in samples (default 1).

            Returns
            -------
            numpy.ndarray
                1D float32 array of shape ``(n_dm,)`` of theoretical standard deviations.
            )doc")
        // execute takes 2d or 3d array as input, and returns 2d or 3d array as
        // output
        .def(
            "execute",
            [](FDMT& fdmt, const py::array& waterfall_obj, SizeType nbits,
               const std::optional<py::array>& out) -> py::object {
                if (waterfall_obj.dtype().kind() != 'u' ||
                    waterfall_obj.itemsize() != 1) {
                    throw py::type_error(
                        "FDMT.execute: nbits given, so waterfall must be a "
                        "packed uint8 array");
                }
                const auto packed =
                    py::array_t<uint8_t, py::array::c_style>::ensure(
                        waterfall_obj);
                if (!packed || (packed.ndim() != 2 && packed.ndim() != 3)) {
                    throw std::runtime_error(
                        "Packed waterfall must be a 2D (nchans, row_bytes) or "
                        "3D (nbeams, nchans, row_bytes) uint8 NumPy array.");
                }
                return fdmt_execute_to_array(
                    fdmt, "FDMT", packed.ndim() == 3,
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
            // No numpydoc sections here: pybind11 joins the overload
            // docstrings, and the float overload below documents both.
            "Run the FDMT transform on packed low-bit integer input "
            "(``waterfall_packed``, ``nbits``); see below.")
        .def(
            "execute",
            [](FDMT& fdmt, const py::array& waterfall_obj,
               const std::optional<py::array>& out) -> py::object {
                if (waterfall_obj.dtype().kind() == 'u' &&
                    waterfall_obj.itemsize() == 1) {
                    throw py::type_error(
                        "FDMT.execute: got a uint8 waterfall; pass nbits "
                        "for packed input (execute(waterfall_packed, nbits)) "
                        "or convert to float32 explicitly");
                }
                const auto waterfall = py::array_t<
                    float, py::array::c_style |
                               py::array::forcecast>::ensure(waterfall_obj);
                if (!waterfall ||
                    (waterfall.ndim() != 2 && waterfall.ndim() != 3)) {
                    throw std::runtime_error("Input waterfall must be a 2D "
                                             "(nchans, nsamps) or 3D (nbeams, "
                                             "nchans, nsamps) NumPy array.");
                }
                return fdmt_execute_to_array(
                    fdmt, "FDMT", waterfall.ndim() == 3,
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
            Run the FDMT transform on float or packed low-bit input.

            Parameters
            ----------
            waterfall : numpy.ndarray
                ``(nchans, nsamps)`` for ``nbeams=1``, or ``(nbeams, nchans,
                nsamps)``. Other float dtypes are cast to float32 (a copy);
                uint8 input needs ``nbits`` (packed overload).
            waterfall_packed : numpy.ndarray, dtype uint8
                Packed overload, called as ``execute(waterfall_packed, nbits)``:
                C-contiguous ``(nchans, row_bytes)`` or ``(nbeams, nchans,
                row_bytes)``; each row holds ``nsamps`` unsigned samples of
                ``nbits`` bits, LSB-first for ``nbits < 8`` (same convention
                as :class:`DDMT`), ``row_bytes = ceil(nsamps*nbits/8)``.
                The result is identical to float input with the same values.
            nbits : int
                Packed sample width: 1, 2, 4, 8 or 16.
            out : numpy.ndarray, optional
                Writeable C-contiguous float32 buffer of at least
                ``nbeams * plan.buffer_size`` elements, reused instead of
                allocating a new one per call (e.g.
                ``np.empty(fdmt.nbeams * fdmt.plan.buffer_size, np.float32)``).

            Returns
            -------
            numpy.ndarray
                float32 view ``(n_delays, n_times)`` or ``(nbeams, n_delays,
                n_times)``; ``n_times`` is ``plan.dmt_nsamps``. It is a
                zero-copy view into the output buffer, which also holds each
                beam's scratch tail (``plan.buffer_size`` floats per beam in
                total), so a retained result keeps that whole buffer alive;
                call ``.copy()`` to keep only the transform. With ``out``, the
                next call overwrites the returned view.

            Notes
            -----
            In ``valid`` mode, successive calls keep inter-block history:
            pass contiguous, non-overlapping blocks. Call
            :meth:`reset_history` to start a new stream.
            )doc")
        .def(
            "reset",
            [](py::object self,
               const py::array_t<uint8_t, py::array::c_style>& waterfall_packed,
               SizeType nbits,
               std::optional<py::array_t<float, py::array::c_style>> dmt_opt) {
                auto& fdmt = self.cast<FDMT&>();
                if (waterfall_packed.ndim() != 2 &&
                    waterfall_packed.ndim() != 3) {
                    throw std::runtime_error(
                        "Packed waterfall must be a 2D (nchans, row_bytes) or "
                        "3D (nbeams, nchans, row_bytes) uint8 NumPy array.");
                }
                auto dmt = stepper_dmt_buffer(fdmt, dmt_opt);
                self.attr("_waterfall_buffer") = waterfall_packed;
                self.attr("_dmt_buffer")       = dmt;
                fdmt.reset(std::span<const uint8_t>(waterfall_packed.data(),
                                                    waterfall_packed.size()),
                           nbits,
                           std::span<float>(dmt.mutable_data(), dmt.size()));
            },
            py::arg("waterfall_packed").noconvert(), py::arg("nbits"),
            py::arg("dmt") = py::none(),
            "Reset and initialize the stepper with a packed low-bit waterfall "
            "(see execute(waterfall_packed, nbits)) and optional dmt buffer "
            "of at least nbeams * plan.buffer_size floats.")
        .def(
            "reset",
            [](py::object self, const py::array& waterfall_obj,
               std::optional<py::array_t<float, py::array::c_style>> dmt_opt) {
                auto& fdmt = self.cast<FDMT&>();
                if (waterfall_obj.dtype().kind() == 'u' &&
                    waterfall_obj.itemsize() == 1) {
                    throw py::type_error(
                        "FDMT.reset: got a uint8 waterfall; pass nbits for "
                        "packed input (reset(waterfall_packed, nbits)) or "
                        "convert to float32 explicitly");
                }
                const auto waterfall = py::array_t<
                    float, py::array::c_style |
                               py::array::forcecast>::ensure(waterfall_obj);
                if (!waterfall ||
                    (waterfall.ndim() != 2 && waterfall.ndim() != 3)) {
                    throw std::runtime_error("Input waterfall must be a 2D "
                                             "(nchans, nsamps) or 3D (nbeams, "
                                             "nchans, nsamps) NumPy array.");
                }
                auto dmt = stepper_dmt_buffer(fdmt, dmt_opt);
                self.attr("_waterfall_buffer") = waterfall;
                self.attr("_dmt_buffer")       = dmt;
                fdmt.reset(
                    std::span<const float>(waterfall.data(), waterfall.size()),
                    std::span<float>(dmt.mutable_data(), dmt.size()));
            },
            py::arg("waterfall"), py::arg("dmt") = py::none(),
            R"doc(
            Reset and initialize the stepper with waterfall and optional dmt
            buffer (at least ``nbeams * plan.buffer_size`` floats).

            In ``mode='valid'`` a block must be finalized before the next
            ``reset``/``execute`` (else RuntimeError); :meth:`reset_history`
            abandons it. The stepper never fuses. On a GPU backend the block
            is staged on the device, and the ``view_*`` methods return a host
            snapshot of the current level.
            )doc")
        .def(
            "advance",
            [](FDMT& fdmt, SizeType levels) { fdmt.advance(levels); },
            py::arg("levels") = 1,
            "Advance execution by a given number of levels.")
        .def(
            "advance_until_remaining",
            [](FDMT& fdmt, SizeType remaining) {
                fdmt.advance_until_remaining(remaining);
            },
            py::arg("remaining_levels"),
            "Advance execution until N levels remain before root.")
        .def("view_level_data",
             [](py::object self) {
                 auto& fdmt = self.cast<FDMT&>();
                 auto span  = fdmt.view_level_data();
                 return py::array_t<float>(span.size(), span.data(), self);
             })
        .def(
            "view_subband_data",
            [](py::object self, SizeType subband_idx) {
                auto& fdmt    = self.cast<FDMT&>();
                auto sub_view = fdmt.view_subband(subband_idx);
                return py::array_t<float>(
                    {sub_view.ndt, sub_view.nsamps},
                    {sub_view.nsamps * sizeof(float), sizeof(float)},
                    sub_view.data.data(), self);
            },
            py::arg("subband_idx"))
        .def(
            "view_subband",
            [](py::object self, SizeType subband_idx) {
                auto& fdmt = self.cast<FDMT&>();
                return fdmt.view_subband(subband_idx);
            },
            py::arg("subband_idx"))
        .def(
            "finalize",
            [](py::object self) {
                auto& fdmt = self.cast<FDMT&>();
                fdmt.finalize();
                py::array_t<float, py::array::c_style> dmt =
                    self.attr("_dmt_buffer")
                        .cast<py::array_t<float, py::array::c_style>>();
                const auto& plan  = fdmt.get_plan();
                const auto ndms   = plan.get_dmt_ndms();
                const auto nsamps = plan.get_dmt_nsamps();
                const auto fsize  = sizeof(float);
                if (fdmt.get_nbeams() == 1) {
                    return py::array_t<float>({ndms, nsamps},
                                              {nsamps * fsize, fsize},
                                              dmt.data(), self);
                }
                return py::array_t<float>(
                    {fdmt.get_nbeams(), ndms, nsamps},
                    {plan.get_buffer_size() * fsize, nsamps * fsize, fsize},
                    dmt.data(), self);
            },
            "Advance to the root and return the transform, (n_delays, "
            "n_times) or (nbeams, n_delays, n_times), as a view into the dmt "
            "buffer given to reset().")
        .def_property_readonly("current_level", &FDMT::current_level)
        .def_property_readonly("total_levels", &FDMT::total_levels)
        .def_property_readonly("remaining_levels", &FDMT::remaining_levels)
        .def_property_readonly("num_subbands", &FDMT::num_subbands)
        .def_property_readonly("is_finished", &FDMT::is_finished)
        .def("reset_history", &FDMT::reset_history,
             "Reset the internal history buffers for valid-mode streaming.")
        .def("history_state_size", &FDMT::history_state_size,
             R"doc(
             Size (in float32 elements) of this instance's streaming history state.
             Zero for "full" or "roll" modes.
             )doc")
        .def(
            "save_history",
            [](const FDMT& fdmt) -> py::object {
                const auto sz = fdmt.history_state_size();
                py::array_t<float, py::array::c_style> hist(sz);
                fdmt.save_history(
                    std::span<float>(hist.mutable_data(), hist.size()));
                return hist;
            },
            R"doc(
        Save the internal streaming history state into a 1D float32 NumPy array.
        Enables time-multiplexing multiple beams or sub-streams on a single FDMT instance.
        )doc")
        .def(
            "load_history",
            [](FDMT& fdmt, const py::array& hist_obj) {
                if (hist_obj.dtype().kind() != 'f' ||
                    hist_obj.itemsize() != 4) {
                    throw std::invalid_argument(
                        "FDMT.load_history: expected a float32 array");
                }
                auto hist =
                    hist_obj.cast<py::array_t<float, py::array::c_style>>();
                if (static_cast<SizeType>(hist.size()) !=
                    fdmt.history_state_size()) {
                    throw std::invalid_argument(
                        std::format("FDMT.load_history: expected buffer of "
                                    "size {}, got {}",
                                    fdmt.history_state_size(), hist.size()));
                }
                fdmt.load_history(
                    std::span<const float>(hist.data(), hist.size()));
            },
            py::arg("history"),
            R"doc(
        Restore previously saved streaming history state from a 1D float32 NumPy array.
        )doc");

    py::class_<FDMTFFT>(mod, "FDMTFFT", py::dynamic_attr(),
                        R"doc(
        FFT-domain FDMT.

        Same interface as :class:`FDMT`, but delay shifts are applied
        with FFTs. Numerically close to ``mode='roll'`` on :class:`FDMT`.

        Parameters
        ----------
        f_min, f_max : float
            Band edges in MHz.
        nchans, nsamps : int
            Channels and samples per block.
        tsamp : float
            Sampling interval in seconds.
        dt_max : int
            Maximum delay trial in samples (regular grid constructor).
        dt_min, dt_step : int, optional
            Delay-grid range and stride.
        use_box_smearing : bool, optional
            Account for intra-channel smearing.
        mode : {'valid', 'full', 'roll'}, optional
            Output time alignment.
        nthreads : int, optional
            OpenMP / FFTW threads.
        nbeams : int, optional
            Packed independent beams.
        backend : {'cpu', 'cuda', 'hip'}, optional
            Keyword-only. Where to run (default ``'cpu'``); see
            :func:`available_backends`.
        device : int, optional
            Keyword-only. Device ordinal on a GPU backend (default 0).
        dt_grid, dt_arr, dm_grid, dm_arr : array_like, optional
            Keyword-only custom trial grid.
        fractional_delays : bool, optional
            Keyword-only. True by default, and by the nature of the
            Fourier-domain tree: each merge shifts by a real-valued,
            least-squares delay (band-limited interpolation) instead of the
            rounded integer delay of the time-domain FDMT, which roughly
            halves the per-channel misalignment. In valid mode the output
            then lags the input by 64 samples (:attr:`output_latency`).
            ``False`` rounds the shifts like :class:`FDMT` and exists only for
            equivalence tests against FDMT.
        kill_mask : numpy.ndarray, optional
            Keyword-only. Per-channel uint8 mask (1 = keep, 0 = kill);
            killed channels read as zero.

        See also
        --------
        FDMT, compute_fdmt_fft
        )doc")
        .def(
            py::init([](float f_min, float f_max, SizeType nchans,
                        SizeType nsamps, float tsamp, IndexType dt_max,
                        IndexType dt_min, SizeType dt_step,
                        bool use_box_smearing, std::string_view mode,
                        int nthreads, SizeType nbeams, std::string_view backend,
                        int device, bool fractional_delays,
                        const std::optional<py::array_t<uint8_t>>& kill_mask) {
                const auto mask = fft_kill_mask(kill_mask);
                return FDMTFFT(f_min, f_max, nchans, nsamps, tsamp, dt_max,
                               dt_min, dt_step, use_box_smearing, mode,
                               make_exec(backend, nthreads, device), nbeams,
                               fractional_delays, mask);
            }),
            "f_min"_a, "f_max"_a, "nchans"_a, "nsamps"_a, "tsamp"_a, "dt_max"_a,
            "dt_min"_a = 0, "dt_step"_a = 1, "use_box_smearing"_a = true,
            "mode"_a = "valid", "nthreads"_a = 1, "nbeams"_a = 1, py::kw_only(),
            "backend"_a = "cpu", "device"_a = 0, "fractional_delays"_a = true,
            "kill_mask"_a = py::none())
        .def(
            py::init([](float f_min, float f_max, SizeType nchans,
                        SizeType nsamps, float tsamp, const py::object& dt_grid,
                        const py::object& dt_arr, const py::object& dm_grid,
                        const py::object& dm_arr, bool use_box_smearing,
                        std::string_view mode, int nthreads, SizeType nbeams,
                        std::string_view backend, int device,
                        bool fractional_delays,
                        const std::optional<py::array_t<uint8_t>>& kill_mask) {
                const auto [type, obj] =
                    resolve_custom_grid(dt_grid, dt_arr, dm_grid, dm_arr);
                const auto exec = make_exec(backend, nthreads, device);
                const auto mask = fft_kill_mask(kill_mask);
                if (type == CustomGridType::kDt) {
                    return std::make_unique<FDMTFFT>(
                        f_min, f_max, nchans, nsamps, tsamp,
                        extract_dt_grid(obj), use_box_smearing, mode, exec,
                        nbeams, fractional_delays, mask);
                }
                return std::make_unique<FDMTFFT>(
                    f_min, f_max, nchans, nsamps, tsamp, extract_dm_grid(obj),
                    use_box_smearing, mode, exec, nbeams, fractional_delays,
                    mask);
            }),
            py::arg("f_min"), py::arg("f_max"), py::arg("nchans"),
            py::arg("nsamps"), py::arg("tsamp"), py::kw_only(),
            py::arg("dt_grid") = py::none(), py::arg("dt_arr") = py::none(),
            py::arg("dm_grid") = py::none(), py::arg("dm_arr") = py::none(),
            py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
            py::arg("nthreads") = 1, py::arg("nbeams") = 1,
            py::arg("backend") = "cpu", py::arg("device") = 0,
            py::arg("fractional_delays") = true,
            py::arg("kill_mask") = py::none())
        .def_property_readonly(
            "backend",
            [](const FDMTFFT& fdmt) {
                return std::string(to_string(fdmt.backend()));
            },
            "Backend this instance runs on ('cpu', 'cuda', ...).")
        .def_property_readonly("nthreads", &FDMTFFT::nthreads)
        .def_property_readonly("device", &FDMTFFT::device)
        .def_property_readonly("fractional_delays", &FDMTFFT::fractional_delays,
                               "True if merges use real-valued (least-squares) "
                               "delays (the default; False is the "
                               "FDMT-equivalence test mode).")
        .def_property_readonly(
            "output_latency", &FDMTFFT::get_output_latency,
            "Samples the output lags the input (the fractional-delay "
            "look-ahead in valid mode, else 0).")
        .def_property_readonly(
            "suggested_nsamps", &FDMTFFT::get_suggested_nsamps,
            "Valid mode: block length keeping >= 80% of every transform as "
            "output (else the plan's nsamps).")
        .def_property_readonly("plan", &FDMTFFT::get_plan)
        .def_property_readonly("nbeams", &FDMTFFT::get_nbeams)
        .def_property_readonly("dt_grid_final",
                               [](FDMTFFT& fdmt) {
                                   return as_pyarray(
                                       fdmt.get_plan().get_dt_grid_final());
                               })
        .def_property_readonly("dm_grid_final",
                               [](FDMTFFT& fdmt) {
                                   return as_pyarray(
                                       fdmt.get_plan().get_dm_grid_final());
                               })
        .def("get_dt_grid_final",
             [](FDMTFFT& fdmt) {
                 return as_pyarray(fdmt.get_plan().get_dt_grid_final());
             })
        .def("get_dm_grid_final",
             [](FDMTFFT& fdmt) {
                 return as_pyarray(fdmt.get_plan().get_dm_grid_final());
             })
        .def("get_effective_variance", &FDMTFFT::get_effective_variance,
             py::arg("dm_idx"), py::arg("boxcar_width") = 1)
        .def("get_effective_sigma", &FDMTFFT::get_effective_sigma,
             py::arg("dm_idx"), py::arg("boxcar_width") = 1)
        .def(
            "get_effective_variance_grid",
            [](const FDMTFFT& fdmt, SizeType boxcar_width) {
                return as_pyarray(
                    fdmt.get_effective_variance_grid(boxcar_width));
            },
            py::arg("boxcar_width") = 1)
        .def(
            "get_effective_sigma_grid",
            [](const FDMTFFT& fdmt, SizeType boxcar_width) {
                return as_pyarray(fdmt.get_effective_sigma_grid(boxcar_width));
            },
            py::arg("boxcar_width") = 1)
        .def(
            "execute",
            [](FDMTFFT& fdmt,
               const py::array_t<uint8_t, py::array::c_style>& packed,
               SizeType nbits) {
                auto out = fft_output(fdmt, packed.ndim());
                fdmt.execute(
                    std::span<const uint8_t>(packed.data(), packed.size()),
                    nbits, std::span<float>(out.mutable_data(), out.size()));
                return out;
            },
            py::arg("waterfall_packed").noconvert(), py::arg("nbits"),
            "Run the transform on packed low-bit unsigned input: "
            "``(nchans, row_bytes)`` or ``(nbeams, nchans, row_bytes)`` uint8 "
            "rows at ``nbits`` (1, 2, 4, 8, 16) per sample, LSB first; float32 "
            "output.")
        .def(
            "execute_time_major",
            [](FDMTFFT& fdmt,
               const py::array_t<uint8_t, py::array::c_style>& packed,
               SizeType nbits) {
                auto out = fft_output(fdmt, packed.ndim());
                fdmt.execute_time_major(
                    std::span<const uint8_t>(packed.data(), packed.size()),
                    nbits, std::span<float>(out.mutable_data(), out.size()));
                return out;
            },
            py::arg("filterbank_packed").noconvert(), py::arg("nbits"),
            "Run the transform on a time-major packed filterbank: "
            "``(nsamps, sample_bytes)`` or ``(nbeams, nsamps, sample_bytes)`` "
            "uint8 at ``nbits``; shares the streaming history.")
        .def(
            "reset",
            [](py::object self,
               const py::array_t<uint8_t, py::array::c_style>& packed,
               SizeType nbits,
               std::optional<py::array_t<float, py::array::c_style>> dmt_opt) {
                auto& fdmt = self.cast<FDMTFFT&>();
                py::array_t<float, py::array::c_style> dmt =
                    dmt_opt.has_value()
                        ? *dmt_opt
                        : py::array_t<float, py::array::c_style>(
                              fdmt.get_nbeams() *
                              fdmt.get_plan().get_dmt_size());
                self.attr("_waterfall_buffer") = packed;
                self.attr("_dmt_buffer")       = dmt;
                fdmt.reset(
                    std::span<const uint8_t>(packed.data(), packed.size()),
                    nbits, std::span<float>(dmt.mutable_data(), dmt.size()));
            },
            py::arg("waterfall_packed").noconvert(), py::arg("nbits"),
            py::arg("dmt") = py::none(),
            "Initialize the stepper with a packed low-bit waterfall (see the "
            "packed execute()).")
        .def(
            "execute",
            [](FDMTFFT& fdmt,
               const py::array_t<float, py::array::c_style>& waterfall)
                -> py::object {
                const auto nbeams     = fdmt.get_nbeams();
                const auto& plan      = fdmt.get_plan();
                const auto ndms       = plan.get_dmt_ndms();
                const auto nsamps_out = plan.get_dmt_nsamps();

                if (waterfall.ndim() == 2) {
                    if (nbeams != 1) {
                        throw std::invalid_argument(std::format(
                            "FDMTFFT: Invalid size of waterfall. Expected "
                            "{}, "
                            "got {}",
                            nbeams * plan.get_nchans() * plan.get_nsamps(),
                            waterfall.size()));
                    }
                    py::array_t<float, py::array::c_style> result(
                        {ndms, nsamps_out});
                    fdmt.execute(
                        std::span<const float>(waterfall.data(),
                                               waterfall.size()),
                        std::span<float>(result.mutable_data(), result.size()));
                    return result;
                }
                if (waterfall.ndim() == 3) {
                    if (static_cast<SizeType>(waterfall.shape(0)) != nbeams) {
                        throw std::invalid_argument(std::format(
                            "FDMTFFT: Invalid size of waterfall. Expected "
                            "{}, "
                            "got {}",
                            nbeams * plan.get_nchans() * plan.get_nsamps(),
                            waterfall.size()));
                    }
                    py::array_t<float, py::array::c_style> result(
                        {nbeams, ndms, nsamps_out});
                    fdmt.execute(
                        std::span<const float>(waterfall.data(),
                                               waterfall.size()),
                        std::span<float>(result.mutable_data(), result.size()));
                    return result;
                }
                throw std::runtime_error("Input waterfall must be a 2D "
                                         "(nchans, nsamps) or 3D (nbeams, "
                                         "nchans, nsamps) NumPy array.");
            },
            py::arg("waterfall"),
            R"doc(
            Run the FFT-domain FDMT transform.

            Parameters
            ----------
            waterfall : numpy.ndarray, dtype float32
                C-contiguous ``(nchans, nsamps)`` or
                ``(nbeams, nchans, nsamps)``.

            Returns
            -------
            numpy.ndarray
                ``(n_delays, n_times)`` or ``(nbeams, n_delays, n_times)``.
            )doc")
        .def(
            "reset",
            [](py::object self,
               const py::array_t<float, py::array::c_style>& waterfall,
               std::optional<py::array_t<float, py::array::c_style>> dmt_opt) {
                auto& fdmt = self.cast<FDMTFFT&>();
                py::array_t<float, py::array::c_style> dmt;
                if (dmt_opt.has_value()) {
                    dmt = *dmt_opt;
                } else {
                    const auto nbeams = fdmt.get_nbeams();
                    dmt               = py::array_t<float, py::array::c_style>(
                        nbeams * fdmt.get_plan().get_dmt_size());
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
            [](FDMTFFT& fdmt, SizeType levels) { fdmt.advance(levels); },
            py::arg("levels") = 1)
        .def(
            "advance_until_remaining",
            [](FDMTFFT& fdmt, SizeType remaining) {
                fdmt.advance_until_remaining(remaining);
            },
            py::arg("remaining_levels"))
        .def("view_level_data",
             [](py::object self) {
                 auto& fdmt = self.cast<FDMTFFT&>();
                 auto span  = fdmt.view_level_data();
                 return py::array_t<float>(span.size(), span.data(), self);
             })
        .def(
            "view_subband_data",
            [](py::object self, SizeType subband_idx) {
                auto& fdmt    = self.cast<FDMTFFT&>();
                auto sub_view = fdmt.view_subband(subband_idx);
                return py::array_t<float>(
                    {sub_view.ndt, sub_view.nsamps},
                    {sub_view.nsamps * sizeof(float), sizeof(float)},
                    sub_view.data.data(), self);
            },
            py::arg("subband_idx"))
        .def(
            "view_subband",
            [](py::object self, SizeType subband_idx) {
                auto& fdmt = self.cast<FDMTFFT&>();
                return fdmt.view_subband(subband_idx);
            },
            py::arg("subband_idx"))
        .def("finalize",
             [](py::object self) {
                 auto& fdmt = self.cast<FDMTFFT&>();
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
        .def_property_readonly("current_level", &FDMTFFT::current_level)
        .def_property_readonly("total_levels", &FDMTFFT::total_levels)
        .def_property_readonly("remaining_levels", &FDMTFFT::remaining_levels)
        .def_property_readonly("num_subbands", &FDMTFFT::num_subbands)
        .def_property_readonly("is_finished", &FDMTFFT::is_finished)
        .def("reset_history", &FDMTFFT::reset_history)
        .def("history_state_size", &FDMTFFT::history_state_size,
             "Floats of the valid-mode streaming history (0 in full and roll "
             "mode); the layout (beam, channel, sample) is the same on every "
             "backend.")
        .def(
            "save_history",
            [](const FDMTFFT& fdmt) {
                py::array_t<float, py::array::c_style> hist(
                    fdmt.history_state_size());
                fdmt.save_history(
                    std::span<float>(hist.mutable_data(), hist.size()));
                return hist;
            },
            "Save the streaming history into a 1D float32 array.")
        .def(
            "load_history",
            [](FDMTFFT& fdmt,
               const py::array_t<float, py::array::c_style>& hist) {
                fdmt.load_history(
                    std::span<const float>(hist.data(), hist.size()));
            },
            py::arg("history").noconvert(),
            "Restore a streaming history saved by save_history().");

    mod.def(
        "compute_fdmt_fft",
        [](const py::array_t<float, py::array::c_style>& waterfall, float f_min,
           float f_max, SizeType nchans, SizeType nsamps, float tsamp,
           IndexType dt_max, IndexType dt_min, SizeType dt_step,
           bool use_box_smearing, std::string_view mode, int nthreads,
           SizeType nbeams, std::string_view backend, int device,
           bool fractional_delays) {
            const auto exec       = make_exec(backend, nthreads, device);
            auto [dmt, fdmt_plan] = algorithms::compute_fdmt_fft(
                std::span<const float>(waterfall.data(), waterfall.size()),
                f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, dt_step,
                use_box_smearing, mode, exec, nbeams, fractional_delays);
            return std::make_tuple(as_pyarray(std::move(dmt)), fdmt_plan);
        },
        py::arg("waterfall"), py::arg("f_min"), py::arg("f_max"),
        py::arg("nchans"), py::arg("nsamps"), py::arg("tsamp"),
        py::arg("dt_max"), py::arg("dt_min") = 0, py::arg("dt_step") = 1,
        py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
        py::arg("nthreads") = 1, py::arg("nbeams") = 1, py::kw_only(),
        py::arg("backend") = "cpu", py::arg("device") = 0,
        py::arg("fractional_delays") = true,
        R"doc(
        One-shot FFT-domain FDMT.

        Runs the FFT-domain FDMT on an input waterfall array of shape (nchans, nsamps),
        returning a tuple (dmt_matrix, plan). ``fractional_delays`` is True by
        default (see :class:`FDMTFFT`); False only for FDMT-equivalence tests.
        )doc");

    mod.def(
        "compute_fdmt_fft",
        [](const py::array_t<float, py::array::c_style>& waterfall, float f_min,
           float f_max, SizeType nchans, SizeType nsamps, float tsamp,
           const py::object& dt_grid, const py::object& dt_arr,
           const py::object& dm_grid, const py::object& dm_arr,
           bool use_box_smearing, std::string_view mode, int nthreads,
           SizeType nbeams, std::string_view backend, int device,
           bool fractional_delays) {
            const auto exec = make_exec(backend, nthreads, device);
            const auto [type, obj] =
                resolve_custom_grid(dt_grid, dt_arr, dm_grid, dm_arr);
            if (type == CustomGridType::kDt) {
                auto [dmt, fdmt_plan] = algorithms::compute_fdmt_fft(
                    std::span<const float>(waterfall.data(), waterfall.size()),
                    f_min, f_max, nchans, nsamps, tsamp, extract_dt_grid(obj),
                    use_box_smearing, mode, exec, nbeams, fractional_delays);
                return std::make_tuple(as_pyarray(std::move(dmt)), fdmt_plan);
            }
            auto [dmt, fdmt_plan] = algorithms::compute_fdmt_fft(
                std::span<const float>(waterfall.data(), waterfall.size()),
                f_min, f_max, nchans, nsamps, tsamp, extract_dm_grid(obj),
                use_box_smearing, mode, exec, nbeams, fractional_delays);
            return std::make_tuple(as_pyarray(std::move(dmt)), fdmt_plan);
        },
        py::arg("waterfall"), py::arg("f_min"), py::arg("f_max"),
        py::arg("nchans"), py::arg("nsamps"), py::arg("tsamp"), py::kw_only(),
        py::arg("dt_grid") = py::none(), py::arg("dt_arr") = py::none(),
        py::arg("dm_grid") = py::none(), py::arg("dm_arr") = py::none(),
        py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
        py::arg("nthreads") = 1, py::arg("nbeams") = 1,
        py::arg("backend") = "cpu", py::arg("device") = 0,
        py::arg("fractional_delays") = true);

    mod.def(

        "compute_fdmt",
        [](const py::array_t<float, py::array::c_style>& waterfall, float f_min,
           float f_max, SizeType nchans, SizeType nsamps, float tsamp,
           IndexType dt_max, IndexType dt_min, SizeType dt_step,
           bool use_box_smearing, std::string_view mode, int nthreads,
           SizeType nbeams, std::string_view backend, int device) {
            const auto exec       = make_exec(backend, nthreads, device);
            auto [dmt, fdmt_plan] = algorithms::compute_fdmt(
                std::span<const float>(waterfall.data(), waterfall.size()),
                f_min, f_max, nchans, nsamps, tsamp, dt_max, dt_min, dt_step,
                use_box_smearing, mode, exec, nbeams);
            return std::make_tuple(as_pyarray(std::move(dmt)), fdmt_plan);
        },
        py::arg("waterfall"), py::arg("f_min"), py::arg("f_max"),
        py::arg("nchans"), py::arg("nsamps"), py::arg("tsamp"),
        py::arg("dt_max"), py::arg("dt_min") = 0, py::arg("dt_step") = 1,
        py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
        py::arg("nthreads") = 1, py::arg("nbeams") = 1, py::kw_only(),
        py::arg("backend") = "cpu", py::arg("device") = 0,
        R"doc(
        One-shot incoherent FDMT.

        Runs the time-domain FDMT on an input waterfall array of shape (nchans, nsamps),
        returning a tuple (dmt_matrix, plan).
        )doc");

    mod.def(
        "compute_fdmt",
        [](const py::array_t<float, py::array::c_style>& waterfall, float f_min,
           float f_max, SizeType nchans, SizeType nsamps, float tsamp,
           const py::object& dt_grid, const py::object& dt_arr,
           const py::object& dm_grid, const py::object& dm_arr,
           bool use_box_smearing, std::string_view mode, int nthreads,
           SizeType nbeams, std::string_view backend, int device) {
            const auto exec = make_exec(backend, nthreads, device);
            const auto [type, obj] =
                resolve_custom_grid(dt_grid, dt_arr, dm_grid, dm_arr);
            if (type == CustomGridType::kDt) {
                auto [dmt, fdmt_plan] = algorithms::compute_fdmt(
                    std::span<const float>(waterfall.data(), waterfall.size()),
                    f_min, f_max, nchans, nsamps, tsamp, extract_dt_grid(obj),
                    use_box_smearing, mode, exec, nbeams);
                return std::make_tuple(as_pyarray(std::move(dmt)), fdmt_plan);
            }
            auto [dmt, fdmt_plan] = algorithms::compute_fdmt(
                std::span<const float>(waterfall.data(), waterfall.size()),
                f_min, f_max, nchans, nsamps, tsamp, extract_dm_grid(obj),
                use_box_smearing, mode, exec, nbeams);
            return std::make_tuple(as_pyarray(std::move(dmt)), fdmt_plan);
        },
        py::arg("waterfall"), py::arg("f_min"), py::arg("f_max"),
        py::arg("nchans"), py::arg("nsamps"), py::arg("tsamp"), py::kw_only(),
        py::arg("dt_grid") = py::none(), py::arg("dt_arr") = py::none(),
        py::arg("dm_grid") = py::none(), py::arg("dm_arr") = py::none(),
        py::arg("use_box_smearing") = true, py::arg("mode") = "valid",
        py::arg("nthreads") = 1, py::arg("nbeams") = 1,
        py::arg("backend") = "cpu", py::arg("device") = 0);

    mod.def(
        "add_frb_track",
        [](py::array_t<float, py::array::c_style>& waterfall,
           const FDMTPlan& plan, SizeType dm_idx, float amplitude,
           IndexType toffset, SizeType width) {
            algorithms::add_frb_track(
                std::span<float>(waterfall.mutable_data(), waterfall.size()),
                plan, dm_idx, amplitude, toffset, width);
        },
        py::arg("waterfall"), py::arg("plan"), py::arg("dm_idx"),
        py::arg("amplitude") = 1.0F, py::arg("toffset") = 0,
        py::arg("width") = 1,
        R"doc(
        Injects a synthetic dispersed pulse ("FRB track") into a waterfall
        array in place, so that it lands exactly on a given final DM/dt
        trial at a chosen time sample after running the FDMT transform.

        Parameters
        ----------
        waterfall : np.ndarray, shape (nchans, nsamps)
            Modified in place (amplitude is added, not overwritten).
        plan : FDMTPlan
            The plan whose trial grid and channel layout to target.
        dm_idx : int
            Index into the plan's final DM/dt trial grid.
        amplitude : float
            Amplitude added per channel, per injected sample.
        toffset : int
            Reference-channel time sample at which the pulse should peak.
        width : int
            Number of consecutive samples per channel to inject (>=1).
        )doc");
}

} // namespace dmt
