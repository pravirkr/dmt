#include "dmt/fdmt_history.hpp"

#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>

#include <fmt/format.h>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"

namespace dmt::algorithms::detail {

namespace {

/// splitmix64 finaliser: every input bit reaches every output bit.
constexpr uint64_t mix64(uint64_t x) noexcept {
    x ^= x >> 30U;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27U;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31U;
    return x;
}

/// Order-sensitive hash of a sequence of integers.
struct LayoutHasher {
    uint64_t h{0x9E3779B97F4A7C15ULL};
    void add(uint64_t v) noexcept {
        h = mix64(h ^ (v + 0x9E3779B97F4A7C15ULL));
    }
};

/// The 16-bit piece at header[at .. at + words), or throws: every header
/// word is an exact integer in [0, 65535].
uint64_t get16(std::span<const float> header, SizeType at, SizeType words) {
    uint64_t v = 0;
    for (SizeType i = 0; i < words; ++i) {
        const float x = header[at + i];
        // Negated so that NaN fails too.
        if (!(x >= 0.0F && x <= 65535.0F) || // NOLINT(*-simplify-boolean-expr)
            x != std::floor(x)) {
            throw std::invalid_argument(
                "FDMT::load_history: not an FDMT history (no valid header; "
                "buffers saved before dmt 0.8 have none and cannot be "
                "loaded)");
        }
        v |= static_cast<uint64_t>(x) << (16 * i);
    }
    return v;
}

} // namespace

uint64_t fdmt_tree_history_hash(const plans::FDMTPlan& plan,
                                SizeType nbeams,
                                bool use_box_smearing) {
    LayoutHasher hash;
    hash.add(nbeams);
    hash.add(use_box_smearing ? 1 : 0);
    hash.add(plan.get_nchans());
    hash.add(plan.get_history_size());
    hash.add(plan.get_tree_history_size());
    const auto& pc        = plan.get_container();
    const SizeType niters = plan.get_niters();
    hash.add(niters);
    for (SizeType l = 1; l <= niters; ++l) {
        const auto& coords = pc.coordinates_sum[l];
        hash.add(coords.size());
        for (const auto& c : coords) {
            if (c.delay == 0) {
                continue; // no history slot
            }
            hash.add(c.i_sub);
            hash.add(c.i_dt);
            hash.add(c.i_coord_head);
            hash.add(c.delay);
            hash.add(c.hist_offset);
        }
    }
    return hash.h;
}

void write_fdmt_tree_history_header(std::span<float> header,
                                    uint64_t layout_hash,
                                    uint64_t stream_pos) noexcept {
    const auto put16 = [&](SizeType at, uint64_t v, SizeType words) {
        for (SizeType i = 0; i < words; ++i) {
            header[at + i] = static_cast<float>((v >> (16 * i)) & 0xFFFFU);
        }
    };
    for (SizeType i = 0; i < kFDMTHistHeader; ++i) {
        header[i] = 0.0F;
    }
    put16(0, kFDMTTreeHistMagic, 1);
    put16(1, kFDMTTreeHistVersion, 1);
    put16(2, layout_hash, 4);
    put16(6, stream_pos, 4);
}

uint64_t check_fdmt_tree_history_header(std::span<const float> header,
                                        uint64_t layout_hash) {
    const uint64_t magic = get16(header, 0, 1);
    if (magic == 0) {
        throw std::invalid_argument(
            "FDMT::load_history: the buffer has no history header (a zeroed "
            "buffer is not a saved history; start a stream cold with "
            "reset_history())");
    }
    if (magic == kFDMTSkewHistMagic) {
        throw std::invalid_argument(
            "FDMT::load_history: the history was saved on the skewed-window "
            "path, but this engine runs the plain tree (a different plan or "
            "max_memory_bytes)");
    }
    if (magic != kFDMTTreeHistMagic) {
        throw std::invalid_argument(
            "FDMT::load_history: not an FDMT history (no valid header; "
            "buffers saved before dmt 0.8 have none and cannot be loaded)");
    }
    const uint64_t version = get16(header, 1, 1);
    if (version != kFDMTTreeHistVersion) {
        throw std::invalid_argument(fmt::format(
            "FDMT::load_history: history version {} is not supported (this "
            "build reads version {})",
            version, kFDMTTreeHistVersion));
    }
    if (get16(header, 2, 4) != layout_hash) {
        throw std::invalid_argument(
            "FDMT::load_history: the history was saved by an FDMT with a "
            "different configuration (plan: DM grid or tree; nbeams; "
            "use_box_smearing)");
    }
    return get16(header, 6, 4);
}

void throw_fdmt_history_size(SizeType expected, SizeType got) {
    throw std::invalid_argument(fmt::format(
        "FDMT::load_history: Invalid input size. Expected {}, got {}{}",
        expected, got,
        got + kFDMTHistHeader == expected
            ? " (the size of a history saved before dmt 0.8, without the "
              "16-word header; it cannot be loaded)"
            : ""));
}

} // namespace dmt::algorithms::detail
