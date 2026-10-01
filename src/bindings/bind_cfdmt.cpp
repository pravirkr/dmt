#include "bindings/bind.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dmt/dmt.hpp"
#include "dmt/utils/simulate.hpp"
#include "pybind_utils.hpp"

namespace dmt {
using algorithms::CohFDMT;

namespace py = pybind11;
using namespace pybind11::literals; // NOLINT

namespace {

using ByteArray = py::array_t<uint8_t, py::array::c_style>;

// A C-contiguous 1-byte array (uint8 or int8, any shape) viewed as bytes.
std::span<const uint8_t> byte_view(const py::array& arr) {
    if (arr.itemsize() != 1 ||
        (arr.dtype().kind() != 'u' && arr.dtype().kind() != 'i')) {
        throw py::type_error(
            "CohFDMT.execute: baseband blocks must be uint8 or int8 arrays "
            "(the config's BasebandFormat decides the decoding)");
    }
    if ((arr.flags() & py::array::c_style) == 0) {
        throw py::type_error(
            "CohFDMT.execute: baseband blocks must be C-contiguous");
    }
    return {static_cast<const uint8_t*>(arr.data()),
            static_cast<SizeType>(arr.size())};
}

py::array_t<float, py::array::c_style>
output_buffer(const CohFDMT& coh, const std::optional<py::array>& out) {
    const auto needed = coh.get_dmt_size();
    if (!out.has_value()) {
        return py::array_t<float, py::array::c_style>(
            static_cast<py::ssize_t>(needed));
    }
    const auto& arr = *out;
    if (arr.dtype().kind() != 'f' || arr.itemsize() != 4 ||
        (arr.flags() & py::array::c_style) == 0 || !arr.writeable()) {
        throw py::type_error("CohFDMT.execute: out must be a writeable, "
                             "C-contiguous float32 array");
    }
    if (static_cast<SizeType>(arr.size()) < needed) {
        throw std::invalid_argument(
            std::format("CohFDMT.execute: out has {} elements, needs at least "
                        "dmt_size = {}",
                        arr.size(), needed));
    }
    return py::reinterpret_borrow<py::array_t<float, py::array::c_style>>(arr);
}

py::object coh_execute(const CohFDMT& coh,
                       const py::object& data,
                       const std::optional<py::array>& out) {
    // Keep every group alive (and contiguous) for the call.
    std::vector<py::array> arrays;
    if (py::isinstance<py::list>(data) || py::isinstance<py::tuple>(data)) {
        for (const auto& item : data) {
            arrays.push_back(py::array::ensure(item));
        }
    } else {
        arrays.push_back(py::array::ensure(data));
    }
    std::vector<std::span<const uint8_t>> groups;
    groups.reserve(arrays.size());
    for (const auto& a : arrays) {
        if (!a) {
            throw py::type_error("CohFDMT.execute: expected a NumPy array or "
                                 "a list of arrays (one per subband group)");
        }
        groups.push_back(byte_view(a));
    }
    auto buf = output_buffer(coh, out);
    {
        const py::gil_scoped_release release;
        coh.execute<uint8_t>(std::span<const std::span<const uint8_t>>(groups),
                             std::span<float>(buf.mutable_data(), buf.size()));
    }
    const auto& plan = coh.get_plan();
    const auto ndm   = static_cast<py::ssize_t>(plan.get_ndm());
    const auto nout  = static_cast<py::ssize_t>(plan.get_output_nsamps());
    const auto fsize = static_cast<py::ssize_t>(sizeof(float));
    return py::array_t<float>({ndm, nout}, {nout * fsize, fsize}, buf.data(),
                              buf);
}

} // namespace

void bind_cfdmt(py::module_& mod) {
    mod.def(
        "generate_pure_frb",
        [](SizeType nchans, SizeType nsamps, float f_min, float f_max,
           SizeType dt, float pulse_toa, float amplitude = 1.0F) {
            auto [arr, nsamps_dispersed] = utils::generate_pure_frb(
                nchans, nsamps, f_min, f_max, dt, pulse_toa, amplitude);
            return std::make_tuple(as_pyarray(std::move(arr)),
                                   nsamps_dispersed);
        },
        "nchans"_a, "nsamps"_a, "f_min"_a, "f_max"_a, "dt"_a, "pulse_toa"_a,
        "amplitude"_a = 1.0F,
        R"doc(
        Inject a noise-free dispersed pulse into a zero waterfall.

        Parameters
        ----------
        nchans, nsamps : int
            Waterfall shape.
        f_min, f_max : float
            Band edges in MHz.
        dt : int
            Total delay across the band, in samples.
        pulse_toa : float
            Pulse time of arrival at the lowest channel, in samples.
        amplitude : float, optional
            Peak amplitude (default 1).

        Returns
        -------
        waterfall : numpy.ndarray
            Flattened ``float32`` array of length ``nchans * nsamps``.
        n_dispersed : int
            Number of samples that received energy.
        )doc");

    py::class_<BasebandFormat>(mod, "BasebandFormat",
                               R"doc(
        Layout and encoding of a block of complex dual-polarisation baseband.

        Parameters
        ----------
        order : str, optional
            Axis order, outermost first: any permutation of the tokens
            ``P`` (polarisation), ``RI`` (real/imaginary), ``T`` (time) and
            ``F`` (subband). ``"FTPRI"`` (default) is GUPPI raw; ``"PRITF"``
            is LOFAR; ``"TFPRI"`` is time-major PSRDADA.
        nbits : {8, 4, 2}, optional
            Bits per real element (default 8).
        is_signed : bool, optional
            Two's complement (default) or offset binary (``uint8 - 128``,
            4-bit ``code - 8``). Ignored for 2-bit.
        msb_first : bool, optional
            Sub-byte packing: first element in the high bits (default; GUPPI
            4-bit, real in the high nibble) or the low bits (VDIF 2-bit).
        levels_2bit : tuple of 4 float, optional
            Value of each 2-bit code (default VDIF: -3.3359, -1, 1, 3.3359).
        )doc")
        .def(py::init([](std::string order, SizeType nbits, bool is_signed,
                         bool msb_first, std::array<float, 4> levels_2bit) {
                 return BasebandFormat{.order       = std::move(order),
                                       .nbits       = nbits,
                                       .is_signed   = is_signed,
                                       .msb_first   = msb_first,
                                       .levels_2bit = levels_2bit};
             }),
             "order"_a = "FTPRI", "nbits"_a = 8, "is_signed"_a = true,
             "msb_first"_a = true,
             "levels_2bit"_a =
                 std::array<float, 4>{-3.3359F, -1.0F, 1.0F, 3.3359F})
        .def_readwrite("order", &BasebandFormat::order)
        .def_readwrite("nbits", &BasebandFormat::nbits)
        .def_readwrite("is_signed", &BasebandFormat::is_signed)
        .def_readwrite("msb_first", &BasebandFormat::msb_first)
        .def_readwrite("levels_2bit", &BasebandFormat::levels_2bit)
        .def("__repr__", [](const BasebandFormat& f) {
            return std::format(
                "BasebandFormat(order='{}', nbits={}, is_signed={}, "
                "msb_first={})",
                f.order, f.nbits, f.is_signed ? "True" : "False",
                f.msb_first ? "True" : "False");
        });

    py::class_<CohFDMTConfig>(mod, "CohFDMTConfig",
                              R"doc(
        Parameters of a CohFDMT search (see :class:`CohFDMTPlan`).

        Parameters
        ----------
        f_center : float
            Centre of the whole band (all subbands) in MHz.
        bw_sub : float
            Subband bandwidth in MHz; the sampling interval is 1 / bw_sub.
        nsub : int
            Number of subbands (all groups).
        t_p : float
            Target detected time resolution in seconds (rounded to n_p
            subband samples, n_p 2,3,5,7-smooth).
        dm_min, dm_max : float
            DM search range in pc cm^-3.
        block_nsamps : int, optional
            Raw samples per subband per block (0 = automatic).
        nbin : int, optional
            Forward FFT length, a multiple of n_p (0 = automatic).
        smear_tol : float, optional
            Largest residual intra-channel smearing, in output samples
            (default 1, the Zackay & Ofek criterion).
        filter_leakage : float, optional
            Share of the channel filter's impulse-response energy allowed
            past each FFT block's overlap (default 1e-4). Smaller values
            lengthen the overlap.
        dt_step : int, optional
            Stride between fine delay trials (default 1).
        normalize : bool, optional
            Normalise every channel to zero mean and unit variance
            (default True).
        format : BasebandFormat, optional
            Input layout and encoding (default GUPPI FTPRI int8).
        subband_groups : list of int, optional
            Subbands per input group (e.g. per GUPPI node file); must sum
            to ``nsub``. Empty (default) = one group.
        )doc")
        .def(py::init([](float f_center, float bw_sub, SizeType nsub, float t_p,
                         float dm_min, float dm_max, SizeType block_nsamps,
                         SizeType nbin, float smear_tol, float filter_leakage,
                         SizeType dt_step, bool normalize,
                         const BasebandFormat& format,
                         std::vector<SizeType> subband_groups) {
                 return CohFDMTConfig{.f_center       = f_center,
                                      .bw_sub         = bw_sub,
                                      .nsub           = nsub,
                                      .t_p            = t_p,
                                      .dm_min         = dm_min,
                                      .dm_max         = dm_max,
                                      .block_nsamps   = block_nsamps,
                                      .nbin           = nbin,
                                      .smear_tol      = smear_tol,
                                      .filter_leakage = filter_leakage,
                                      .dt_step        = dt_step,
                                      .normalize      = normalize,
                                      .format         = format,
                                      .subband_groups =
                                          std::move(subband_groups)};
             }),
             "f_center"_a, "bw_sub"_a, "nsub"_a, "t_p"_a, "dm_min"_a,
             "dm_max"_a, py::kw_only(), "block_nsamps"_a = 0, "nbin"_a = 0,
             "smear_tol"_a = 1.0F, "filter_leakage"_a = 1.0E-4F,
             "dt_step"_a = 1, "normalize"_a = true,
             "format"_a         = BasebandFormat{},
             "subband_groups"_a = std::vector<SizeType>{})
        .def_readwrite("f_center", &CohFDMTConfig::f_center)
        .def_readwrite("bw_sub", &CohFDMTConfig::bw_sub)
        .def_readwrite("nsub", &CohFDMTConfig::nsub)
        .def_readwrite("t_p", &CohFDMTConfig::t_p)
        .def_readwrite("dm_min", &CohFDMTConfig::dm_min)
        .def_readwrite("dm_max", &CohFDMTConfig::dm_max)
        .def_readwrite("block_nsamps", &CohFDMTConfig::block_nsamps)
        .def_readwrite("nbin", &CohFDMTConfig::nbin)
        .def_readwrite("smear_tol", &CohFDMTConfig::smear_tol)
        .def_readwrite("filter_leakage", &CohFDMTConfig::filter_leakage)
        .def_readwrite("dt_step", &CohFDMTConfig::dt_step)
        .def_readwrite("normalize", &CohFDMTConfig::normalize)
        .def_readwrite("format", &CohFDMTConfig::format)
        .def_readwrite("subband_groups", &CohFDMTConfig::subband_groups);

    py::class_<CohFDMT>(mod, "CohFDMT",
                        R"doc(
        Hybrid coherent + FDMT dedispersion search of baseband voltages.

        Stateless: every :meth:`execute` searches one self-contained block of
        ``plan.block_nsamps`` raw samples per subband. Advance the read
        position by ``plan.stride_nsamps`` between blocks (blocks overlap by
        the dispersion sweep); the ``plan.output_nsamps`` valid samples per
        row of consecutive blocks then tile the time axis. Output sample j
        of a block starting at raw sample s0 is the arrival time at
        ``plan.f_ref`` of ``s0 * plan.tbin + plan.output_time_offset + j *
        plan.tsamp``.

        Parameters
        ----------
        config : CohFDMTConfig
            The search.
        nthreads : int, optional
            OpenMP threads on the CPU backend.
        backend : {'cpu', 'cuda', 'hip'}, optional
            Keyword-only. Where to run (default ``'cpu'``).
        device : int, optional
            Keyword-only. Device ordinal on a GPU backend (default 0).

        See also
        --------
        CohFDMTPlan, BasebandFormat, simulate_baseband
        )doc")
        .def(py::init([](const CohFDMTConfig& config, int nthreads,
                         std::string_view backend, int device) {
                 return CohFDMT(config, make_exec(backend, nthreads, device));
             }),
             "config"_a, "nthreads"_a = 1, py::kw_only(), "backend"_a = "cpu",
             "device"_a = 0)
        .def_property_readonly(
            "backend",
            [](const CohFDMT& coh) {
                return std::string(to_string(coh.backend()));
            },
            "Backend this instance runs on ('cpu', 'cuda', ...).")
        .def_property_readonly("nthreads", &CohFDMT::nthreads)
        .def_property_readonly("device", &CohFDMT::device)
        .def_property_readonly("plan", &CohFDMT::get_plan,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("block_nsamps", &CohFDMT::get_block_nsamps)
        .def_property_readonly("stride_nsamps", &CohFDMT::get_stride_nsamps)
        .def_property_readonly("output_nsamps", &CohFDMT::get_output_nsamps)
        .def_property_readonly("dmt_size", &CohFDMT::get_dmt_size)
        .def("input_size", &CohFDMT::get_input_size, "igroup"_a = 0,
             "Bytes of subband group ``igroup`` per block.")
        .def(
            "memory_usage",
            [](const CohFDMT& coh) {
                const auto m = coh.get_memory_usage();
                return py::dict("spectrum"_a  = m.spectrum,
                                "waterfall"_a = m.waterfall, "fdmt"_a = m.fdmt,
                                "workspace"_a = m.workspace,
                                "output"_a = m.output, "total"_a = m.total());
            },
            "Bytes allocated by the engine, by kind (``output`` is the "
            "caller's result buffer and not part of ``total``).")
        .def("execute", &coh_execute, "data"_a, py::kw_only(),
             "out"_a = py::none(),
             R"doc(
             Search one baseband block.

             Parameters
             ----------
             data : numpy.ndarray or list of numpy.ndarray
                 C-contiguous uint8/int8 block of ``input_size()`` bytes (any
                 shape), or one such array per subband group.
             out : numpy.ndarray, optional
                 Keyword-only float32 buffer of at least ``dmt_size``
                 elements to reuse (else one is allocated).

             Returns
             -------
             numpy.ndarray
                 ``(plan.ndm, plan.output_nsamps)`` float32 view; row i is
                 DM ``plan.dm_grid_final[i]``.
             )doc");

    mod.def(
        "simulate_baseband",
        [](float f_center, float bw_sub, SizeType nsub, SizeType nsamps,
           const std::vector<std::array<double, 4>>& pulses, float noise_sigma,
           uint64_t seed, int nthreads) {
            std::vector<utils::BasebandPulse> ps;
            for (const auto& p : pulses) {
                ps.push_back({.dm        = p[0],
                              .t_arrival = p[1],
                              .fluence   = p[2],
                              .width     = p[3]});
            }
            std::vector<ComplexType> v;
            {
                const py::gil_scoped_release release;
                v = utils::simulate_baseband(f_center, bw_sub, nsub, nsamps, ps,
                                             noise_sigma, seed, nthreads);
            }
            auto arr = as_pyarray(std::move(v));
            return arr.reshape({py::ssize_t{2}, static_cast<py::ssize_t>(nsub),
                                static_cast<py::ssize_t>(nsamps)});
        },
        "f_center"_a, "bw_sub"_a, "nsub"_a, "nsamps"_a,
        "pulses"_a      = std::vector<std::array<double, 4>>{},
        "noise_sigma"_a = 0.0F, "seed"_a = 42, "nthreads"_a = 1,
        R"doc(
        Simulate complex dual-polarisation baseband with dispersed pulses.

        Each pulse is built with the exact cold-plasma phase at every RF
        frequency (one transform per subband, circular in time), so it
        arrives at frequency f at ``t_arrival + K * dm / f**2``.

        Parameters
        ----------
        f_center, bw_sub : float
            Band centre and subband bandwidth in MHz.
        nsub, nsamps : int
            Subbands and samples per subband.
        pulses : list of (dm, t_arrival, fluence, width), optional
            DM in pc cm^-3, arrival time at infinite frequency in s after
            sample 0, energy per polarisation and subband, Gaussian width in
            s (0 = impulse).
        noise_sigma : float, optional
            Noise standard deviation per real component (default 0).
        seed : int, optional
            Noise seed.
        nthreads : int, optional
            Threads for the transforms.

        Returns
        -------
        numpy.ndarray
            complex64 array of shape ``(2, nsub, nsamps)``.
        )doc");

    mod.def(
        "pack_baseband",
        [](const py::array_t<std::complex<float>,
                             py::array::c_style | py::array::forcecast>&
               voltages,
           const BasebandFormat& format, float scale, SizeType sub_begin,
           SizeType sub_count, SizeType t_begin, SizeType t_count) {
            if (voltages.ndim() != 3 || voltages.shape(0) != 2) {
                throw std::invalid_argument(
                    "pack_baseband: voltages must have shape (2, nsub, "
                    "nsamps)");
            }
            const auto nsub   = static_cast<SizeType>(voltages.shape(1));
            const auto nsamps = static_cast<SizeType>(voltages.shape(2));
            auto bytes        = utils::pack_baseband(
                std::span<const ComplexType>(voltages.data(), voltages.size()),
                nsub, nsamps, format, scale, sub_begin, sub_count, t_begin,
                t_count);
            return as_pyarray(std::move(bytes));
        },
        "voltages"_a, "format"_a = BasebandFormat{}, "scale"_a = 1.0F,
        "sub_begin"_a = 0, "sub_count"_a = 0, "t_begin"_a = 0, "t_count"_a = 0,
        R"doc(
        Quantise ``(2, nsub, nsamps)`` voltages into a baseband block.

        Each component becomes ``round(v * scale)`` clipped to the format's
        range (2-bit: the nearest level). ``sub_begin, sub_count`` and
        ``t_begin, t_count`` select subbands and samples (0 = to the end).

        Returns
        -------
        numpy.ndarray
            uint8 bytes in ``format``.
        )doc");
}

} // namespace dmt
