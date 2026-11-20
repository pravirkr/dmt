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
using algorithms::FDMTBoxcarMemoryEstimate;
using algorithms::FDMTBoxcarMemoryUsage;
using algorithms::FDMTFFTBoxcar;

namespace {
struct BoxcarResult {
    py::array_t<RawCandidate> candidates;
    CandidateCounts counts;
    uint64_t input_sample_begin;
    SizeType input_nsamps;
    py::array_t<CandidateCounts> plane_counts;
};
BoxcarResult own_result(CandidateResultView view) {
    py::array_t<RawCandidate> array(
        static_cast<py::ssize_t>(view.candidates.size()));
    if (!view.candidates.empty()) {
        std::memcpy(array.mutable_data(), view.candidates.data(),
                    view.candidates.size_bytes());
    }
    py::array_t<CandidateCounts> planes(
        static_cast<py::ssize_t>(view.plane_counts.size()));
    if (!view.plane_counts.empty()) {
        std::memcpy(planes.mutable_data(), view.plane_counts.data(),
                    view.plane_counts.size_bytes());
    }
    return {std::move(array), view, view.input_sample_begin, view.input_nsamps,
            std::move(planes)};
}
} // namespace

void bind_fdmt_boxcar(py::module_& mod) {
    PYBIND11_NUMPY_DTYPE(RawCandidate, peak_sample, begin_sample, end_sample,
                         snr, dm, dm_idx, filter_idx, width, beam_idx,
                         subband_level, subband_idx, chan_begin, chan_end,
                         f_low_mhz, f_high_mhz, staggered);
    PYBIND11_NUMPY_DTYPE(CandidateCounts, emitted_count, total_detected,
                         dropped_global, dropped_per_dm);
    py::enum_<PeakFindingMode>(mod, "PeakFindingMode")
        .value("CONTIGUOUS_PEAK_PER_WIDTH",
               PeakFindingMode::kContiguousPeakPerWidth)
        .value("CONTIGUOUS_PEAK_ACROSS_WIDTHS",
               PeakFindingMode::kContiguousPeakAcrossWidths)
        .value("ALL_ABOVE_THRESHOLD", PeakFindingMode::kAllAboveThreshold);
    py::enum_<OverflowPolicy>(mod, "OverflowPolicy", R"doc(
        What a call keeps when more candidates cross the threshold than fit.

        ``DROP_NEW`` (default): first come; which candidates survive an
        overflow is unspecified. ``KEEP_BRIGHTEST``: exactly the brightest:
        each DM row keeps its ``max_cands_per_dm`` highest-S/N candidates of
        the call, and the call the ``max_candidates`` highest of those (ties
        to the earlier peak sample). Needs ``max_cands_per_dm`` in 1..1024,
        e.g. 16 (a per-row store counted in the memory budget).
        )doc")
        .value("DROP_NEW", OverflowPolicy::kDropNew)
        .value("KEEP_BRIGHTEST", OverflowPolicy::kKeepBrightest);
    py::enum_<BoxcarScreenMode>(
        mod, "BoxcarScreenMode",
        "Stage-1 tile screening mode for analytical boxcar search.")
        .value("PREFIX", BoxcarScreenMode::kPrefix,
               "Prefix-sum / range-minimum tile screen (exact).")
        .value("KADANE", BoxcarScreenMode::kKadane,
               "Lossless multi-bucket Kadane tile gate + exact Stage-2 "
               "prefix-sum scoring.")
        .value("NONE", BoxcarScreenMode::kNone,
               "Disable Stage-1 screening and score every tile directly "
               "(exact).")
        .value("AUTO", BoxcarScreenMode::kAuto,
               "Default: the fastest exact mode of the backend (KADANE on "
               "the CPU, PREFIX on a GPU).");
    py::class_<BoxcarConfig>(mod, "BoxcarConfig", R"doc(
        Analytical boxcar and trapezoid matched-filter search settings.

        ``max_boxcar`` (1..2^20, at most 4096 widths) bounds the geometrically
        spaced width bank generated with ``wtsp``
        (:func:`generate_box_width_trials`). Set ``use_trapezoid=True`` to
        replace widths >= 3 with ``{1, 2, 1}`` half-weight edge-tapered
        trapezoids. ``input_nbits`` selects float32 (32) or packed unsigned
        codes (1/2/4/8/16) calibrated by ``input_mean`` and ``noise_sigma``.
        ``overflow_policy`` (:class:`OverflowPolicy`, default ``DROP_NEW``)
        chooses between first-come overflow and exact ``KEEP_BRIGHTEST``
        top-S/N selection. ``screen_mode`` selects the Stage-1 tile screening
        algorithm (:class:`BoxcarScreenMode`, default
        ``BoxcarScreenMode.AUTO``): ``AUTO`` (``KADANE`` on the CPU,
        ``PREFIX`` on a GPU), ``KADANE``, ``PREFIX`` and ``NONE`` are lossless
        and produce identical exact candidates. ``per_call_masks=True``
        declares that :class:`FDMTBoxcar` will get per-call kill masks: their
        requirements and memory are checked and reserved at construction (and
        in ``estimate_memory``) instead of at the first masked call.
        ``max_candidates_per_plane`` (0: off) gives each searched plane its own
        share of a call's candidates (its first ones under ``DROP_NEW``, its
        brightest under ``KEEP_BRIGHTEST``), so a flooded plane cannot crowd
        out the others; planes x share must not exceed ``max_candidates``.
        )doc")
        .def(py::init([](SizeType max_boxcar, float wtsp, float detect_thresh,
                         float noise_sigma, PeakFindingMode peak_mode,
                         SizeType max_candidates, SizeType max_cands_per_dm,
                         bool skip_cold_prefix, SizeType input_nbits,
                         float input_mean, bool use_trapezoid,
                         OverflowPolicy overflow_policy,
                         BoxcarScreenMode screen_mode, bool per_call_masks,
                         SizeType max_candidates_per_plane) {
                 BoxcarConfig c(max_boxcar, wtsp, detect_thresh, noise_sigma,
                                peak_mode, max_candidates, max_cands_per_dm,
                                skip_cold_prefix, input_nbits, input_mean,
                                use_trapezoid, overflow_policy, screen_mode);
                 c.per_call_masks           = per_call_masks;
                 c.max_candidates_per_plane = max_candidates_per_plane;
                 return c;
             }),
             "max_boxcar"_a = 4096, py::kw_only(), "wtsp"_a = 2.0F,
             "detect_thresh"_a = 6.0F, "noise_sigma"_a = 1.0F,
             "peak_mode"_a      = PeakFindingMode::kContiguousPeakPerWidth,
             "max_candidates"_a = 131072, "max_cands_per_dm"_a = 4096,
             "skip_cold_prefix"_a = true, "input_nbits"_a = 32,
             "input_mean"_a = 0.0F, "use_trapezoid"_a = false,
             "overflow_policy"_a = OverflowPolicy::kDropNew,
             "screen_mode"_a     = BoxcarScreenMode::kAuto,
             "per_call_masks"_a = false, "max_candidates_per_plane"_a = 0)
        .def_readwrite("max_boxcar", &BoxcarConfig::max_boxcar)
        .def_readwrite("wtsp", &BoxcarConfig::wtsp)
        .def_readwrite("detect_thresh", &BoxcarConfig::detect_thresh)
        .def_readwrite("noise_sigma", &BoxcarConfig::noise_sigma)
        .def_readwrite("input_nbits", &BoxcarConfig::input_nbits)
        .def_readwrite("input_mean", &BoxcarConfig::input_mean)
        .def_readwrite("use_trapezoid", &BoxcarConfig::use_trapezoid)
        .def_readwrite("peak_mode", &BoxcarConfig::peak_mode)
        .def_readwrite("max_candidates", &BoxcarConfig::max_candidates)
        .def_readwrite("max_cands_per_dm", &BoxcarConfig::max_cands_per_dm)
        .def_readwrite("skip_cold_prefix", &BoxcarConfig::skip_cold_prefix)
        .def_readwrite("overflow_policy", &BoxcarConfig::overflow_policy)
        .def_readwrite("screen_mode", &BoxcarConfig::screen_mode)
        .def_readwrite("per_call_masks", &BoxcarConfig::per_call_masks)
        .def_readwrite("max_candidates_per_plane",
                       &BoxcarConfig::max_candidates_per_plane);
    py::class_<BoxcarResult>(mod, "BoxcarResult")
        .def_readonly("candidates", &BoxcarResult::candidates)
        .def_readonly("input_sample_begin", &BoxcarResult::input_sample_begin)
        .def_readonly("input_nsamps", &BoxcarResult::input_nsamps)
        .def_readonly("plane_counts", &BoxcarResult::plane_counts,
                      R"doc(
            Structured array of the counters of each searched plane
            (``emitted_count``, ``total_detected``, ``dropped_global``,
            ``dropped_per_dm``), in ``search_plans`` order; they sum to the
            call's counters.
            )doc")
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
        .def_readonly("coord_indices", &plans::BoxcarPlanePlan::coord_indices)
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
        .def_readonly("host", &FDMTBoxcarMemoryUsage::host,
                      "Of total, bytes in host memory on a GPU backend "
                      "(reported, not budgeted); 0 on the CPU.")
        .def_property_readonly("total", &FDMTBoxcarMemoryUsage::total)
        .def_property_readonly(
            "budgeted", &FDMTBoxcarMemoryUsage::budgeted,
            "The part of total that max_memory_bytes bounds: device memory "
            "on a GPU, everything on the CPU.");
    py::class_<FDMTBoxcarMemoryEstimate>(mod, "FDMTBoxcarMemoryEstimate", R"doc(
        What an instance built with the same arguments would allocate, from
        :meth:`FDMTBoxcar.estimate_memory` (computed without allocating it).

        ``usage`` equals that instance's ``memory_usage``; ``chunk_nsamps``
        its ``chunk_nsamps`` (on a GPU the staging of host (NumPy) input is
        part of ``usage``);
        ``search_path`` the transform's path (the ``Transform:`` line of
        ``summary()``; empty for the tree path) and ``search_latency`` its
        ``search_latency``; ``max_memory_bytes`` the budget it was made with
        (from :meth:`FDMTBoxcar.estimate_minimum_memory`: the smallest that
        builds, at least ``usage.budgeted``).
        )doc")
        .def_readonly("usage", &FDMTBoxcarMemoryEstimate::usage)
        .def_readonly("chunk_nsamps", &FDMTBoxcarMemoryEstimate::chunk_nsamps)
        .def_readonly("search_path", &FDMTBoxcarMemoryEstimate::search_path)
        .def_readonly("search_latency",
                      &FDMTBoxcarMemoryEstimate::search_latency)
        .def_readonly("max_memory_bytes",
                      &FDMTBoxcarMemoryEstimate::max_memory_bytes)
        .def("__repr__", [](const FDMTBoxcarMemoryEstimate& e) {
            return fmt::format(
                "FDMTBoxcarMemoryEstimate(total={}, chunk_nsamps={}, "
                "search_latency={}, search_path='{}', max_memory_bytes={})",
                e.usage.total(), e.chunk_nsamps, e.search_latency,
                e.search_path, e.max_memory_bytes);
        });
    mod.def("generate_box_width_trials", &plans::generate_box_width_trials,
            "max_width"_a, "wtsp"_a = 2.0F, R"doc(
        Boxcar widths searched for max_boxcar=max_width and this wtsp.

        From 1, each next width is max(w + 1, ceil(w * wtsp)) while it does
        not exceed max_width; max_width ends the bank when the steps stop
        below it (9 at wtsp 2: 1, 2, 4, 8, 9).
        )doc");
    py::class_<FDMTBoxcar>(mod, "FDMTBoxcar", R"doc(
        Streaming valid-mode FDMT and analytical boxcar detection.

        Input is contiguous float32, (channels, samples) for one beam or
        (beams, channels, samples), with zero mean and independent noise of
        known common sigma. Set config.input_nbits to 1/2/4/8/16 for packed
        uint8 bytes, with input_mean and noise_sigma in code units. Each
        channel row has ceil(samples * nbits / 8) bytes, LSB-first; 16-bit
        samples are little-endian. Root DM-time output is never materialized.
        execute() returns owned structured candidate arrays. Contiguous
        peaks cross blocks; call drain() at end of stream (flush() does not
        search the samples a skewed-path subband row lags). Sample indices
        are absolute trailing-window endpoints, not filter centers.
        Host calls on GPU backends block and return candidates only.

        Subband search follows the plan: with ``FDMTPlan(...,
        subband_depth=d)`` every depth ``k = 0..d`` is searched, i.e. its
        ``2**k`` equal subbands and the ``2**k - 1`` staggered subbands of
        the same width centred on their boundaries. Candidates carry
        ``subband_level`` (the depth), ``subband_idx``, the channel extent
        and ``staggered``.

        ``max_memory_bytes`` (keyword-only, default 10 GiB) bounds
        ``memory_usage.budgeted``: everything the instance allocates on the
        CPU, device memory on a GPU (host copies and result storage are
        reported in ``memory_usage.host``, not budgeted). Plans whose FDMT tree
        history does not fit (UWL-like bands at high time resolution) run the
        skewed-window path, as does packed input on the CPU with blocks over
        2048 samples (faster there); their subband rows are then searched with a fixed
        per-row latency of up to the dispersion delay to the bottom of the
        band, so the final such samples of a stream are searched by a later
        call or by :meth:`drain`. Raises ``ValueError`` when nothing fits. On a GPU host
        (NumPy) input is copied one chunk at a time into a buffer reserved
        inside that budget, so :meth:`execute` allocates nothing.

        ``kill_mask`` (keyword-only): ``None`` or one bool/integer per
        channel, low frequency first. Channels with 0 read as zero (code 0
        for packed input) and are left out of every row's noise model, so
        S/N stays calibrated on the remaining channels. On the CPU each
        block is copied once to zero them; on a GPU the transform reads them
        as zero, without a copy. :meth:`execute` also takes a mask per block
        (float or packed input), ANDed with this one.

        ``config.screen_mode`` (:class:`BoxcarScreenMode`, default ``AUTO``)
        chooses the Stage-1 tile screening mode on both CPU and GPU:
        ``PREFIX``, ``KADANE`` (lossless multi-bucket Kadane gate), ``NONE``
        and ``AUTO`` (``KADANE`` on the CPU, ``PREFIX`` on a GPU) all emit
        identical exact candidates.
        )doc")
        .def(py::init([](plans::FDMTPlan plan, BoxcarConfig config, bool smear,
                         int nthreads, SizeType nbeams,
                         std::optional<SizeType> fuse,
                         std::optional<SizeType> chunk,
                         std::string_view backend, int device,
                         SizeType max_memory_bytes, const py::object& kill) {
                 const auto mask = extract_kill_mask(kill);
                 return FDMTBoxcar(std::move(plan), config, smear,
                                   make_exec(backend, nthreads, device), nbeams,
                                   fuse.value_or(algorithms::kFDMTAutoFuse),
                                   chunk.value_or(0), max_memory_bytes, mask);
             }),
             "plan"_a, "config"_a = BoxcarConfig{}, "use_box_smearing"_a = true,
             "nthreads"_a = 1, "nbeams"_a = 1, "fuse_levels"_a = py::none(),
             "chunk_nsamps"_a = py::none(), py::kw_only(), "backend"_a = "cpu",
             "device"_a           = 0,
             "max_memory_bytes"_a = algorithms::kFDMTDefaultMaxMemory,
             "kill_mask"_a        = py::none())
        .def_static(
            "estimate_memory",
            [](const plans::FDMTPlan& plan, BoxcarConfig config, bool smear,
               int nthreads, SizeType nbeams, std::optional<SizeType> fuse,
               std::optional<SizeType> chunk, std::string_view backend,
               int device, SizeType max_memory_bytes, const py::object& kill) {
                const auto mask = extract_kill_mask(kill);
                return FDMTBoxcar::estimate_memory(
                    plan, config, smear, make_exec(backend, nthreads, device),
                    nbeams, fuse.value_or(algorithms::kFDMTAutoFuse),
                    chunk.value_or(0), max_memory_bytes, mask);
            },
            "plan"_a, "config"_a = BoxcarConfig{}, "use_box_smearing"_a = true,
            "nthreads"_a = 1, "nbeams"_a = 1, "fuse_levels"_a = py::none(),
            "chunk_nsamps"_a = py::none(), py::kw_only(), "backend"_a = "cpu",
            "device"_a           = 0,
            "max_memory_bytes"_a = algorithms::kFDMTDefaultMaxMemory,
            "kill_mask"_a        = py::none(),
            R"doc(
            Memory an ``FDMTBoxcar`` built with the same arguments would
            allocate, and its chunk, without allocating it
            (:class:`FDMTBoxcarMemoryEstimate`).

            Exact: the constructor's decisions and tables are computed and
            every working buffer is sized, not allocated. Raises as the
            constructor would when nothing fits ``max_memory_bytes``. GPU
            backends read the device's limits (no device memory is allocated;
            the CUDA context is not counted). Per-call kill masks are
            included when declared (``BoxcarConfig.per_call_masks``);
            otherwise they add their noise tables on first use.
            )doc")
        .def_static(
            "estimate_minimum_memory",
            [](const plans::FDMTPlan& plan, BoxcarConfig config, bool smear,
               int nthreads, SizeType nbeams, std::optional<SizeType> fuse,
               std::optional<SizeType> chunk, std::string_view backend,
               int device, const py::object& kill) {
                const auto mask = extract_kill_mask(kill);
                return FDMTBoxcar::estimate_minimum_memory(
                    plan, config, smear, make_exec(backend, nthreads, device),
                    nbeams, fuse.value_or(algorithms::kFDMTAutoFuse),
                    chunk.value_or(0), mask);
            },
            "plan"_a, "config"_a = BoxcarConfig{}, "use_box_smearing"_a = true,
            "nthreads"_a = 1, "nbeams"_a = 1, "fuse_levels"_a = py::none(),
            "chunk_nsamps"_a = py::none(), py::kw_only(), "backend"_a = "cpu",
            "device"_a = 0, "kill_mask"_a = py::none(),
            R"doc(
            The smallest configuration of these arguments: what
            :meth:`estimate_memory` returns at the smallest
            ``max_memory_bytes`` an ``FDMTBoxcar`` can be built with (the
            result's ``max_memory_bytes``; one byte less raises). A few
            estimates long. Raises as :meth:`estimate_memory` for anything
            but the budget.
            )doc")
        .def(
            "execute",
            [](FDMTBoxcar& search, const py::array& input,
               const py::object& kill) {
                const auto mask   = extract_kill_mask(kill);
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
                                      static_cast<SizeType>(input.size())),
                            std::span<const uint8_t>(mask));
                    } else {
                        view = search.execute(
                            std::span(static_cast<const float*>(input.data()),
                                      static_cast<SizeType>(input.size())),
                            std::span<const uint8_t>(mask));
                    }
                }
                return own_result(view);
            },
            "waterfall"_a, "kill_mask"_a = py::none(),
            R"doc(
            Search one block and return its candidates.

            ``kill_mask``: ``None`` or one bool/integer per channel, low
            frequency first (0 = killed), for this block only. It is ANDed
            with the constructor's mask; ``None`` means the constructor's
            mask alone. On packed input, channels it kills (and the
            constructor's mask keeps) read the code ``input_mean``, which
            must then be a whole number (else ``ValueError``): the expected
            sums stay the constructor's, and only the thresholds and S/N
            scales follow the mask. When the mask changes, the
            endpoints whose input reaches back before the block are scored
            with the variance of every channel live in their reach (no
            extra false alarms, S/N there slightly low); any block length
            works. The first mask needs its noise tables within
            ``max_memory_bytes``; otherwise ``ValueError``, with the stream
            unchanged.
            )doc")
        .def("flush",
             [](FDMTBoxcar& search) {
                 CandidateResultView view;
                 {
                     py::gil_scoped_release release;
                     view = search.flush();
                 }
                 return own_result(view);
             })
        .def(
            "drain",
            [](FDMTBoxcar& search) {
                CandidateResultView view;
                {
                    py::gil_scoped_release release;
                    view = search.drain();
                }
                return own_result(view);
            },
            R"doc(
            End of stream: search the stream up to its last sample and emit
            the open excursions (as :meth:`flush`). Without it the last
            :attr:`search_latency` samples are never searched. The stream ends:
            ``execute`` raises until :meth:`reset_history`.
            )doc")
        .def("get_candidate_frontier", &FDMTBoxcar::get_candidate_frontier,
             R"doc(
            The oldest sample a later candidate can still have: every
            candidate emitted by a later :meth:`execute`, :meth:`flush` or
            :meth:`drain` has ``begin_sample`` (hence ``peak_sample`` and
            ``end_sample``) at or after it. It is the smallest
            ``begin_sample`` of the excursions still open, or
            :attr:`searched_nsamps` when none is open — for a ``chan_dm``
            plan minus :attr:`FDMTPlan.level0_causal_delay` clamped at 0,
            matching candidate times. It never decreases until
            :meth:`reset_history` (then 0). Use it as the watermark
            for releasing candidate groups, whatever the length of an
            excursion.
            )doc")
        .def_property_readonly(
            "searched_nsamps", &FDMTBoxcar::get_searched_nsamps,
            "FDMT output samples [0, n) searched by every row so far (after "
            ":meth:`drain`, the whole stream; for ``chan_dm != 0`` this is the "
            "raw count without subtracting "
            ":attr:`FDMTPlan.level0_causal_delay`).")
        .def_property_readonly(
            "search_latency", &FDMTBoxcar::get_search_latency,
            "Input samples a stream sample waits for before every searched "
            "row has searched it (the largest subband row lag on the skewed "
            "paths; 0 otherwise).")
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

        ``max_memory_bytes`` (keyword-only, default 10 GiB) bounds
        ``memory_usage.budgeted``, as on FDMTBoxcar (device memory on a
        GPU):
        over it the overlap-save segments are shortened (:attr:`chunk_nsamps`)
        until it fits. Raises ``ValueError`` when even the shortest segment
        does not fit. On a GPU the device copy of a host (NumPy) block is
        reserved inside that budget at construction. ``config.screen_mode``
        (:class:`BoxcarScreenMode`) applies to every segment's DM-row search
        as on :class:`FDMTBoxcar`.
        )doc")
        .def(py::init([](plans::FDMTPlan plan, BoxcarConfig config, bool smear,
                         int nthreads, SizeType nbeams, bool frac,
                         const py::object& kill, std::string_view backend,
                         int device, SizeType max_memory_bytes) {
                 const auto mask = extract_kill_mask(kill);
                 return FDMTFFTBoxcar(std::move(plan), config, smear,
                                      make_exec(backend, nthreads, device),
                                      nbeams, frac, mask, max_memory_bytes);
             }),
             "plan"_a, "config"_a = BoxcarConfig{}, "use_box_smearing"_a = true,
             "nthreads"_a = 1, "nbeams"_a = 1, "fractional_delays"_a = true,
             "kill_mask"_a = py::none(), py::kw_only(), "backend"_a = "cpu",
             "device"_a           = 0,
             "max_memory_bytes"_a = algorithms::kFDMTDefaultMaxMemory)
        .def_static(
            "estimate_memory",
            [](const plans::FDMTPlan& plan, BoxcarConfig config, bool smear,
               int nthreads, SizeType nbeams, bool frac, const py::object& kill,
               std::string_view backend, int device,
               SizeType max_memory_bytes) {
                const auto mask = extract_kill_mask(kill);
                return FDMTFFTBoxcar::estimate_memory(
                    plan, config, smear, make_exec(backend, nthreads, device),
                    nbeams, frac, mask, max_memory_bytes);
            },
            "plan"_a, "config"_a = BoxcarConfig{}, "use_box_smearing"_a = true,
            "nthreads"_a = 1, "nbeams"_a = 1, "fractional_delays"_a = true,
            "kill_mask"_a = py::none(), py::kw_only(), "backend"_a = "cpu",
            "device"_a           = 0,
            "max_memory_bytes"_a = algorithms::kFDMTDefaultMaxMemory,
            R"doc(
            Memory an ``FDMTFFTBoxcar`` built with the same arguments would
            allocate, and its chunk (the overlap-save segment), without
            allocating it (:class:`FDMTBoxcarMemoryEstimate`). Raises as the
            constructor would when nothing fits ``max_memory_bytes``.
            )doc")
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
        .def(
            "drain",
            [](FDMTFFTBoxcar& search) {
                CandidateResultView view;
                {
                    py::gil_scoped_release release;
                    view = search.drain();
                }
                return own_result(view);
            },
            R"doc(
            End of stream: search the stream up to its last sample and emit
            the open excursions (as :meth:`flush`). Without it the last
            :attr:`output_latency` samples are never searched. The stream ends:
            ``execute`` raises until :meth:`reset_history`.
            )doc")
        .def("get_candidate_frontier", &FDMTFFTBoxcar::get_candidate_frontier,
             R"doc(
            The oldest sample a later candidate can still have: every
            candidate emitted by a later :meth:`execute`, :meth:`flush` or
            :meth:`drain` has ``begin_sample`` (hence ``peak_sample`` and
            ``end_sample``) at or after it. It is the smallest
            ``begin_sample`` of the excursions still open, or
            :attr:`searched_nsamps` when none is open — for a ``chan_dm``
            plan minus :attr:`FDMTPlan.level0_causal_delay` clamped at 0,
            matching candidate times. It never decreases until
            :meth:`reset_history` (then 0). Use it as the watermark
            for releasing candidate groups, whatever the length of an
            excursion.
            )doc")
        .def_property_readonly(
            "searched_nsamps", &FDMTFFTBoxcar::get_searched_nsamps,
            "FDMT output samples [0, n) searched by every row so far (after "
            ":meth:`drain`, the whole stream; for ``chan_dm != 0`` this is the "
            "raw count without subtracting "
            ":attr:`FDMTPlan.level0_causal_delay`).")
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
