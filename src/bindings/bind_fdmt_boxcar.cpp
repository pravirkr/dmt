#include <algorithm>
#include <cstring>
#include <optional>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dmt/dmt.hpp"

#include "bindings/bind.hpp"
#include "pybind_utils.hpp"

namespace dmt {
namespace py = pybind11;
using namespace pybind11::literals; // NOLINT
using algorithms::FDMTBoxcar;
using algorithms::FDMTBoxcarMemoryUsage;
using algorithms::FDMTFFTBoxcar;

namespace {
struct BoxcarResult {
    py::array_t<RawCandidate> candidates;
    CandidateCounts counts;
    uint64_t input_sample_begin;
    SizeType input_nsamps;
};
BoxcarResult own_result(CandidateResultView view) {
    py::array_t<RawCandidate> array(
        static_cast<py::ssize_t>(view.candidates.size()));
    if (!view.candidates.empty()) {
        std::memcpy(array.mutable_data(), view.candidates.data(),
                    view.candidates.size_bytes());
    }
    return {std::move(array), view, view.input_sample_begin, view.input_nsamps};
}
} // namespace

void bind_fdmt_boxcar(py::module_& mod) {
    PYBIND11_NUMPY_DTYPE(RawCandidate, peak_sample, begin_sample, end_sample,
                         snr, dm, dm_idx, filter_idx, width, beam_idx,
                         subband_level, subband_idx, chan_begin, chan_end,
                         f_low_mhz, f_high_mhz, staggered);
    py::enum_<PeakFindingMode>(mod, "PeakFindingMode")
        .value("CONTIGUOUS_PEAK_PER_WIDTH",
               PeakFindingMode::kContiguousPeakPerWidth)
        .value("CONTIGUOUS_PEAK_ACROSS_WIDTHS",
               PeakFindingMode::kContiguousPeakAcrossWidths)
        .value("ALL_ABOVE_THRESHOLD", PeakFindingMode::kAllAboveThreshold);
    py::class_<BoxcarConfig>(mod, "BoxcarConfig")
        .def(py::init<SizeType, float, float, float, PeakFindingMode, SizeType,
                      SizeType, bool, SizeType, float>(),
             "max_boxcar"_a = 4096, py::kw_only(), "wtsp"_a = 2.0F,
             "detect_thresh"_a = 6.0F, "noise_sigma"_a = 1.0F,
             "peak_mode"_a      = PeakFindingMode::kContiguousPeakPerWidth,
             "max_candidates"_a = 131072, "max_cands_per_dm"_a = 4096,
             "skip_cold_prefix"_a = true, "input_nbits"_a = 32,
             "input_mean"_a = 0.0F)
        .def_readwrite("max_boxcar", &BoxcarConfig::max_boxcar)
        .def_readwrite("wtsp", &BoxcarConfig::wtsp)
        .def_readwrite("detect_thresh", &BoxcarConfig::detect_thresh)
        .def_readwrite("noise_sigma", &BoxcarConfig::noise_sigma)
        .def_readwrite("input_nbits", &BoxcarConfig::input_nbits)
        .def_readwrite("input_mean", &BoxcarConfig::input_mean)
        .def_readwrite("peak_mode", &BoxcarConfig::peak_mode)
        .def_readwrite("max_candidates", &BoxcarConfig::max_candidates)
        .def_readwrite("max_cands_per_dm", &BoxcarConfig::max_cands_per_dm)
        .def_readwrite("skip_cold_prefix", &BoxcarConfig::skip_cold_prefix);
    py::class_<BoxcarResult>(mod, "BoxcarResult")
        .def_readonly("candidates", &BoxcarResult::candidates)
        .def_readonly("input_sample_begin", &BoxcarResult::input_sample_begin)
        .def_readonly("input_nsamps", &BoxcarResult::input_nsamps)
        .def_property_readonly(
            "emitted_count",
            [](const BoxcarResult& r) { return r.counts.emitted_count; })
        .def_property_readonly(
            "total_detected",
            [](const BoxcarResult& r) { return r.counts.total_detected; })
        .def_property_readonly(
            "dropped_global",
            [](const BoxcarResult& r) { return r.counts.dropped_global; })
        .def_property_readonly(
            "dropped_per_dm",
            [](const BoxcarResult& r) { return r.counts.dropped_per_dm; })
        .def_property_readonly("overflowed", [](const BoxcarResult& r) {
            return r.counts.overflowed();
        });
    py::class_<plans::BoxcarPlanePlan>(mod, "BoxcarPlanePlan")
        .def_readonly("level", &plans::BoxcarPlanePlan::level)
        .def_readonly("subband_level", &plans::BoxcarPlanePlan::subband_level)
        .def_readonly("subband_idx", &plans::BoxcarPlanePlan::subband_idx)
        .def_readonly("staggered", &plans::BoxcarPlanePlan::staggered)
        .def_readonly("chan_begin", &plans::BoxcarPlanePlan::chan_begin)
        .def_readonly("chan_end", &plans::BoxcarPlanePlan::chan_end)
        .def_readonly("f_low_mhz", &plans::BoxcarPlanePlan::f_low_mhz)
        .def_readonly("f_high_mhz", &plans::BoxcarPlanePlan::f_high_mhz)
        .def_readonly("widths", &plans::BoxcarPlanePlan::widths)
        .def_readonly("dm_grid", &plans::BoxcarPlanePlan::dm_grid)
        .def_readonly("warmup_samples", &plans::BoxcarPlanePlan::warmup_samples)
        .def_readonly("inv_sigma_lut", &plans::BoxcarPlanePlan::inv_sigma_lut)
        .def_readonly("integer_mean_lut",
                      &plans::BoxcarPlanePlan::integer_mean_lut)
        .def_readonly("integer_thresh_lut",
                      &plans::BoxcarPlanePlan::integer_thresh_lut)
        .def_readonly("raw_thresh_lut",
                      &plans::BoxcarPlanePlan::raw_thresh_lut);
    py::class_<FDMTBoxcarMemoryUsage>(mod, "FDMTBoxcarMemoryUsage")
        .def_readonly("plan", &FDMTBoxcarMemoryUsage::plan)
        .def_readonly("state", &FDMTBoxcarMemoryUsage::state)
        .def_readonly("history", &FDMTBoxcarMemoryUsage::history)
        .def_readonly("peaks", &FDMTBoxcarMemoryUsage::peaks)
        .def_readonly("workspace", &FDMTBoxcarMemoryUsage::workspace)
        .def_readonly("candidates", &FDMTBoxcarMemoryUsage::candidates)
        .def_readonly("staging", &FDMTBoxcarMemoryUsage::staging)
        .def_property_readonly("total", &FDMTBoxcarMemoryUsage::total);
    mod.def("generate_box_width_trials", &plans::generate_box_width_trials,
            "max_width"_a, "wtsp"_a = 2.0F);
    py::class_<FDMTBoxcar>(mod, "FDMTBoxcar", R"doc(
        Streaming valid-mode FDMT and analytical boxcar detection.

        Input is contiguous float32, (channels, samples) for one beam or
        (beams, channels, samples), with zero mean and independent noise of
        known common sigma. Set config.input_nbits to 1/2/4/8/16 for packed
        uint8 bytes, with input_mean and noise_sigma in code units. Each
        channel row has ceil(samples * nbits / 8) bytes, LSB-first; 16-bit
        samples are little-endian. Root DM-time output is never materialized.
        execute() returns owned structured candidate arrays. Contiguous
        peaks cross blocks; call flush() at end of stream. Sample indices
        are absolute trailing-window endpoints, not filter centers.
        Host calls on GPU backends block and return candidates only.

        Subband search follows the plan: with ``FDMTPlan(...,
        subband_depth=d)`` every depth ``k = 0..d`` is searched, i.e. its
        ``2**k`` equal subbands and the ``2**k - 1`` staggered subbands of
        the same width centred on their boundaries. Candidates carry
        ``subband_level`` (the depth), ``subband_idx``, the channel extent
        and ``staggered``.
        )doc")
        .def(py::init([](plans::FDMTPlan plan, BoxcarConfig config, bool smear,
                         int nthreads, SizeType nbeams,
                         std::optional<SizeType> fuse,
                         std::optional<SizeType> chunk,
                         std::string_view backend, int device) {
                 return FDMTBoxcar(std::move(plan), config, smear,
                                   make_exec(backend, nthreads, device), nbeams,
                                   fuse.value_or(algorithms::kFDMTAutoFuse),
                                   chunk.value_or(0));
             }),
             "plan"_a, "config"_a = BoxcarConfig{}, "use_box_smearing"_a = true,
             "nthreads"_a = 1, "nbeams"_a = 1, "fuse_levels"_a = py::none(),
             "chunk_nsamps"_a = py::none(), py::kw_only(), "backend"_a = "cpu",
             "device"_a = 0)
        .def(
            "execute",
            [](FDMTBoxcar& search, const py::array& input) {
                const auto& plan  = search.get_plan();
                const bool packed = search.get_config().input_nbits != 32;
                const auto row_size =
                    packed ? ((plan.get_nsamps() *
                                   search.get_config().input_nbits +
                               7) /
                              8)
                           : plan.get_nsamps();
                if (!input.dtype().is(packed ? py::dtype::of<uint8_t>()
                                             : py::dtype::of<float>()) ||
                    (input.flags() & py::array::c_style) == 0 ||
                    !((input.ndim() == 2 && search.get_nbeams() == 1 &&
                       input.shape(0) ==
                           static_cast<py::ssize_t>(plan.get_nchans()) &&
                       input.shape(1) == static_cast<py::ssize_t>(row_size)) ||
                      (input.ndim() == 3 &&
                       input.shape(0) ==
                           static_cast<py::ssize_t>(search.get_nbeams()) &&
                       input.shape(1) ==
                           static_cast<py::ssize_t>(plan.get_nchans()) &&
                       input.shape(2) == static_cast<py::ssize_t>(row_size)))) {
                    throw std::invalid_argument(
                        "FDMTBoxcar: expected contiguous float32 samples or "
                        "packed uint8 bytes, with (channels, row_size) or "
                        "(beams, channels, row_size) matching plan/config");
                }
                CandidateResultView view;
                {
                    py::gil_scoped_release release;
                    if (packed) {
                        view = search.execute(
                            std::span(static_cast<const uint8_t*>(input.data()),
                                      static_cast<SizeType>(input.size())));
                    } else {
                        view = search.execute(
                            std::span(static_cast<const float*>(input.data()),
                                      static_cast<SizeType>(input.size())));
                    }
                }
                return own_result(view);
            },
            "waterfall"_a)
        .def("flush",
             [](FDMTBoxcar& search) {
                 CandidateResultView view;
                 {
                     py::gil_scoped_release release;
                     view = search.flush();
                 }
                 return own_result(view);
             })
        .def("reset_history",
             [](FDMTBoxcar& search) {
                 py::gil_scoped_release release;
                 search.reset_history();
             })
        .def_property_readonly("plan", &FDMTBoxcar::get_plan,
                               py::return_value_policy::reference_internal)
        .def_property_readonly(
            "config",
            [](const FDMTBoxcar& search) { return search.get_config(); })
        .def_property_readonly("search_plans",
                               [](const FDMTBoxcar& search) {
                                   const auto planes =
                                       search.get_search_plans();
                                   return std::vector<plans::BoxcarPlanePlan>(
                                       planes.begin(), planes.end());
                               })
        .def_property_readonly("nbeams", &FDMTBoxcar::get_nbeams)
        .def_property_readonly("chunk_nsamps", &FDMTBoxcar::get_chunk_nsamps)
        .def_property_readonly("fuse_levels", &FDMTBoxcar::get_fuse_levels)
        .def_property_readonly("nthreads", &FDMTBoxcar::nthreads)
        .def_property_readonly("device", &FDMTBoxcar::device)
        .def_property_readonly("backend",
                               [](const FDMTBoxcar& search) {
                                   return std::string(
                                       to_string(search.backend()));
                               })
        .def_property_readonly("memory_usage", &FDMTBoxcar::get_memory_usage)
        .def("summary", &FDMTBoxcar::summary)
        .def("__repr__", &FDMTBoxcar::summary);
    py::class_<FDMTFFTBoxcar>(mod, "FDMTFFTBoxcar", R"doc(
        Streaming valid-mode FDMT-FFT and analytical boxcar detection.

        The Fourier-domain counterpart of FDMTBoxcar, with the same config,
        result and candidate fields, for contiguous float32 input of shape
        (channels, samples) or (beams, channels, samples). Each overlap-save
        segment's DM rows are searched straight from the inverse transform;
        no DM-time plane is allocated. Full-band search only (plan
        subband_depth 0). With fractional delays (the default) the search lags the input
        by output_latency samples; candidate sample indices are input-stream
        indices, comparable with FDMTBoxcar's. Channels with kill_mask 0 read
        as zero and are excluded from the noise model.
        )doc")
        .def(py::init([](plans::FDMTPlan plan, BoxcarConfig config, bool smear,
                         int nthreads, SizeType nbeams, bool frac,
                         const py::object& kill, std::string_view backend,
                         int device) {
                 const auto mask = extract_kill_mask(kill);
                 return FDMTFFTBoxcar(std::move(plan), config, smear,
                                      make_exec(backend, nthreads, device),
                                      nbeams, frac, mask);
             }),
             "plan"_a, "config"_a = BoxcarConfig{}, "use_box_smearing"_a = true,
             "nthreads"_a = 1, "nbeams"_a = 1, "fractional_delays"_a = true,
             "kill_mask"_a = py::none(), py::kw_only(), "backend"_a = "cpu",
             "device"_a = 0)
        .def(
            "execute",
            [](FDMTFFTBoxcar& search, const py::array& input) {
                const auto& plan  = search.get_plan();
                const auto nchans = static_cast<py::ssize_t>(plan.get_nchans());
                const auto nsamps = static_cast<py::ssize_t>(plan.get_nsamps());
                if (!input.dtype().is(py::dtype::of<float>()) ||
                    (input.flags() & py::array::c_style) == 0 ||
                    !((input.ndim() == 2 && search.get_nbeams() == 1 &&
                       input.shape(0) == nchans && input.shape(1) == nsamps) ||
                      (input.ndim() == 3 &&
                       input.shape(0) ==
                           static_cast<py::ssize_t>(search.get_nbeams()) &&
                       input.shape(1) == nchans && input.shape(2) == nsamps))) {
                    throw std::invalid_argument(
                        "FDMTFFTBoxcar: expected contiguous float32 of shape "
                        "(channels, samples) or (beams, channels, samples)");
                }
                CandidateResultView view;
                {
                    py::gil_scoped_release release;
                    view = search.execute(
                        std::span(static_cast<const float*>(input.data()),
                                  static_cast<SizeType>(input.size())));
                }
                return own_result(view);
            },
            "waterfall"_a)
        .def("flush",
             [](FDMTFFTBoxcar& search) {
                 CandidateResultView view;
                 {
                     py::gil_scoped_release release;
                     view = search.flush();
                 }
                 return own_result(view);
             })
        .def("reset_history",
             [](FDMTFFTBoxcar& search) {
                 py::gil_scoped_release release;
                 search.reset_history();
             })
        .def_property_readonly("plan", &FDMTFFTBoxcar::get_plan,
                               py::return_value_policy::reference_internal)
        .def_property_readonly(
            "config",
            [](const FDMTFFTBoxcar& search) { return search.get_config(); })
        .def_property_readonly("search_plans",
                               [](const FDMTFFTBoxcar& search) {
                                   const auto planes =
                                       search.get_search_plans();
                                   return std::vector<plans::BoxcarPlanePlan>(
                                       planes.begin(), planes.end());
                               })
        .def_property_readonly("nbeams", &FDMTFFTBoxcar::get_nbeams)
        .def_property_readonly("chunk_nsamps", &FDMTFFTBoxcar::get_chunk_nsamps)
        .def_property_readonly("output_latency",
                               &FDMTFFTBoxcar::get_output_latency)
        .def_property_readonly("nthreads", &FDMTFFTBoxcar::nthreads)
        .def_property_readonly("device", &FDMTFFTBoxcar::device)
        .def_property_readonly("backend",
                               [](const FDMTFFTBoxcar& search) {
                                   return std::string(
                                       to_string(search.backend()));
                               })
        .def_property_readonly("memory_usage", &FDMTFFTBoxcar::get_memory_usage)
        .def("summary", &FDMTFFTBoxcar::summary)
        .def("__repr__", &FDMTFFTBoxcar::summary);
}
} // namespace dmt
