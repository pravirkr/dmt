#pragma once

// Backend-neutral host descriptions of the FDMTFFT tree for engines that run
// it as data-parallel programs over frequency bins (the GPU engine), and the
// overlap-save segment rule shared by every engine.
//
// Every frequency bin k runs the same tree: node = tail + head * W^(s * k)
// (W = exp(-2 pi i / N)), so a program is a list of such merges applied to a
// tile of bins at once. A *stage* computes the nodes of level `level_out`
// from those of `level_in` (level 0: the per-channel delay nodes, formed on
// the fly from the channel spectra and the level-0 windows). Its programs
// each own a run of consecutive output nodes together with everything they
// need at the levels in between (their "cone"), numbered locally so that the
// intermediate levels fit in on-chip memory: only the stage's input and
// output levels are stored in device memory. Cones of neighbouring programs
// can overlap at the lower levels; plan_tree_stages() picks the stage
// boundaries that minimise the device-memory traffic plus the (redundant)
// merge work. A single-level stage always fits, so every plan has a plan.
//
// Nvcc includes this header: declarations only, no FFT library headers.

#include <cstdint>
#include <optional>
#include <vector>

#include "dmt/common/plans.hpp"
#include "dmt/common/types.hpp"
#include "dmt/modes.hpp"

namespace dmt::algorithms::fdmt_fft {

struct Geometry;

/// `head` of a copy node (out = tail).
inline constexpr std::uint32_t kTreeCopy = 0xFFFFFFFFU;

/// Producer of one node: out = in[tail] + in[head] * W^(shift[sid] * k).
struct TreeOp {
    std::uint32_t tail;
    std::uint32_t head; // kTreeCopy: out = in[tail]
    std::uint32_t sid;  // index into TreeDag::shift
};

/// The plan's tree as producers per node.
struct TreeDag {
    std::vector<SizeType> ncoords;         // nodes per level 0..L
    std::vector<std::vector<TreeOp>> prod; // [l][node] for l >= 1 (global
                                           // indices of level l - 1)
    std::vector<std::uint32_t> l0_chan;    // level-0 node -> channel
    std::vector<std::uint32_t> l0_shift;   // level-0 node -> |dt| (samples)
    std::vector<double> shift;             // merge shift (samples) per sid
    SizeType dt0_max{0};
    SizeType nchans{0};

    [[nodiscard]] SizeType levels() const noexcept {
        return ncoords.empty() ? 0 : ncoords.size() - 1;
    }
    [[nodiscard]] SizeType nops() const noexcept;
};

/// @brief The plan's tree. With fractional delays every merge has its own
/// shift (sid = merge op id, levels in order); otherwise sid is the plan's
/// integer delay.
[[nodiscard]] TreeDag
make_tree_dag(const plans::FDMTPlan& plan, bool fractional, bool box);

/// Programs of one stage (see the file comment), flattened. Program p writes
/// output nodes [out_begin[p], out_begin[p] + n_out(p)) of level_out; its
/// ops for level level_in + 1 + i are ops[lvl_off[p * nlev + i],
/// lvl_off[p * nlev + i + 1]), op j producing local node j. The first level's
/// ops index level_in nodes globally (level-0 nodes when level_in == 0), the
/// others the previous level's local nodes; local nodes of the last level
/// are output nodes out_begin[p] + j.
struct TreeStage {
    SizeType level_in{};
    SizeType level_out{};
    SizeType max_nodes{}; // largest local intermediate level (0: none)
    std::vector<std::uint32_t> out_begin;
    std::vector<std::uint32_t> lvl_off; // nprograms * nlevels + 1 entries
    std::vector<TreeOp> ops;
    SizeType in_rows{}; // input rows read, summed over programs

    [[nodiscard]] SizeType nprograms() const noexcept {
        return out_begin.size();
    }
    [[nodiscard]] SizeType nlevels() const noexcept {
        return level_out - level_in;
    }
};

/// @brief Stage level_in -> level_out whose intermediate levels hold at most
/// `budget` nodes per program, or nullopt if one output node's cone alone
/// exceeds it. Programs own at most `max_out` output nodes.
[[nodiscard]] std::optional<TreeStage> make_tree_stage(const TreeDag& dag,
                                                       SizeType level_in,
                                                       SizeType level_out,
                                                       SizeType budget,
                                                       SizeType max_out);

/// @brief Stages from level 0 to the root minimising (per bin) device-memory
/// traffic plus merge work, each within `budget` intermediate nodes.
[[nodiscard]] std::vector<TreeStage>
plan_tree_stages(const TreeDag& dag, SizeType budget, SizeType max_out);

/// @brief One stage per level (level l - 1 -> l, l = 1..L): the stepper.
[[nodiscard]] std::vector<TreeStage> level_tree_stages(const TreeDag& dag,
                                                       SizeType max_out);

/// Overlap-save segments of execute(). The virtual input of a channel is
/// v = [zeros(zeros) | overlap (valid) | block | zeros...]; segment j
/// transforms v[j * hop, j * hop + n_fft) and yields output samples
/// [j * hop, (j + 1) * hop) from transform sample skip on. One segment is
/// the single-transform case (skip = the geometry's skip, zeros = 0).
struct Segments {
    SizeType n_fft{};
    SizeType nseg{1};
    SizeType hop{};
    SizeType skip{};
    SizeType zeros{};
};

/// @brief execute()'s transform length: the single transform, or overlap-save
/// segments of a length that transforms faster per output sample (FFTW's
/// deterministic cost estimate, so every engine and run picks the same
/// segments and hence the same result bits). `max_len` > 0 caps the length
/// (memory; tests) where the mode allows segments (not roll).
[[nodiscard]] Segments choose_segments(const Geometry& geom,
                                       FDMTMode mode,
                                       SizeType nsamps_out,
                                       SizeType tree_nodes,
                                       SizeType nchans,
                                       SizeType ndms,
                                       SizeType max_len = 0);

} // namespace dmt::algorithms::fdmt_fft
