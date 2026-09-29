#pragma once

/**
 * @file sdmt_gpu_plan.hpp
 * @brief Host-side planner of the SDMT GPU kernel (sdmt_kernel.cuh).
 *
 * The GPU SDMT computes, per block of TDM consecutive DM trials and per
 * subband of 16 consecutive active channels, the exact DDMT subband sums
 * through a fixed two-level hierarchy:
 *
 *  - L1 nodes: 4-channel groups. Write a trial's group delays as
 *    m1(d) + r(d, j) with m1 the smallest. Trials with the same r share
 *    one node N1[tau] = sum_j x[c_j][tau + r_j], read at tau = t + m1(d).
 *  - L2 nodes: the subband. Write the group minima as m2(d) + o(d, g).
 *    Trials with the same (L1 node, o) for every group share one node
 *    N2[tau] = sum_g N1_g[tau + o_g], read at tau = t + m2(d).
 *  - Combine: every trial adds its subband's N2 at t + m2(d).
 *
 * Nothing is approximated: a node is shared only by trials whose integer
 * delays over its channels agree up to a common shift. The planner lays out
 * every (tile, subband) as a small program over one shared-memory arena of
 * elements: staged channel windows, L1 nodes, L2 nodes, and one arena
 * address per trial. A node with a single source is an alias (an address
 * into its source) and costs nothing.
 */

#include <algorithm>
#include <cstdint>
#include <map>
#include <span>
#include <vector>

#include "dmt/common/types.hpp"

namespace dmt::algorithms::sdmt_gpu {

/// Active channels per subband and per L1 group.
inline constexpr int kSubband = 16;
inline constexpr int kGroup   = 4;
/// Most nodes (L1 + L2) of one program; larger programs are split.
inline constexpr int kMaxNodes = 128;

/// One (tile, subband) program: index ranges into the tables below.
struct SubProg {
    int win0;   // first window
    int nwin;   // windows to stage
    int n1_0;   // first L1 node
    int n1;     // L1 nodes to compute
    int n2_0;   // first L2 node
    int n2;     // L2 nodes to compute
    int trial0; // first of TDM trial addresses
    int wlen;   // window length (all windows of the subband); window j
                // starts at arena element j * wlen
    int bmin;   // smallest window base
    int bmax;   // largest window base
};

/// Channel window j of a subband: arena[j * wlen + i] =
/// x[chan][t0 + base + i], i < wlen.
struct Window {
    int chan; // channel (row within the beam)
    int base;
};

/// Node: arena[dst + i] = sum_k arena[src[k] + i], i < len, k < nsrc.
struct Node {
    int dst;
    int len;
    int nsrc;
    int pad;
    int src[4];
};

struct Plan {
    int tdm{0};
    int bt{0};
    int nsub{0};      // most subband programs of a tile
    int arena{0};     // elements the largest program uses
    int max_nodes{0}; // largest n1 + n2 of a program
    double ops{0};    // shared-memory element loads + stores, all programs
    double ddmt{0};   // brute-force additions, all tiles (for comparison)
    std::vector<SubProg> subs; // programs, tile by tile
    std::vector<int> tile_sub; // [tile + 1]: first program of each tile
    std::vector<Window> wins;
    std::vector<Node> nodes;
    std::vector<int> trials; // [tile][subband][TDM] arena addresses
};

/**
 * @brief Builds the programs for DM tiles of @p tdm trials and time tiles of
 * @p bt output samples.
 * @details
 * Each tile's active channels are covered by consecutive subbands of up to
 * kSubband channels. A subband whose program needs more than @p max_arena
 * elements or @p max_nodes nodes is split in halves (down to single
 * channels, whose "program" is a plain delayed read), so coarse channels or
 * wide delay spreads still fit.
 * @param delays (ndm, nchans) integer delay table.
 * @param active Active (unmasked) channel indices.
 * @return The plan; arena > max_arena if even one channel does not fit.
 */
inline Plan build_plan(std::span<const int> delays,
                       std::span<const int> active,
                       SizeType nchans,
                       SizeType ndm,
                       int tdm,
                       int bt,
                       int max_arena,
                       int max_nodes) {
    Plan p;
    p.tdm         = tdm;
    p.bt          = bt;
    const auto na = static_cast<int>(active.size());
    const int ntile =
        static_cast<int>((ndm + static_cast<SizeType>(tdm) - 1) / tdm);
    p.ddmt = static_cast<double>(ntile) * tdm * na * bt;

    std::vector<int> m1(static_cast<SizeType>(tdm) * 4);  // [d][g]
    std::vector<int> id1(static_cast<SizeType>(tdm) * 4); // [d][g]
    std::vector<int> m2(static_cast<SizeType>(tdm));
    std::vector<int> id2(static_cast<SizeType>(tdm));
    std::vector<int> key;

    struct Tmp {
        int lo;
        int hi;
        int nsrc;
        int src[4]; // L1: window index; L2: L1 node index
        int off[4]; // L1: start in the window at tau = lo; L2: o_g
        int addr;
    };
    std::vector<Tmp> t1;
    std::vector<Tmp> t2;
    std::map<std::vector<int>, int> map1;
    std::map<std::vector<int>, int> map2;

    p.tile_sub.push_back(0);
    for (int t = 0; t < ntile; ++t) {
        auto delay = [&](int d, int k) {
            const auto dm = std::min<SizeType>(
                static_cast<SizeType>((t * tdm) + d), ndm - 1);
            return delays[(dm * nchans) + static_cast<SizeType>(active[k])];
        };
        // Appends the program of active channels [k0, k0 + nc); returns its
        // arena size and adds its operations to ops.
        auto build_sub = [&](int k0, int nc, double& ops) {
            int arena = 0;
            SubProg sp{};
            sp.win0 = static_cast<int>(p.wins.size());
            sp.nwin = nc;
            // Windows start at base_c, the channel's smallest delay in the
            // tile, and all have the subband's largest bt + spread_c
            // samples (one length keeps the kernel's staging simple).
            std::vector<int> wbase(static_cast<SizeType>(nc));
            std::vector<int> waddr(static_cast<SizeType>(nc));
            int wlen = bt;
            for (int j = 0; j < nc; ++j) {
                int mn = delay(0, k0 + j);
                int mx = mn;
                for (int d = 1; d < tdm; ++d) {
                    const int v = delay(d, k0 + j);
                    mn          = std::min(mn, v);
                    mx          = std::max(mx, v);
                }
                wbase[j] = mn;
                wlen     = std::max(wlen, bt + (mx - mn));
            }
            for (int j = 0; j < nc; ++j) {
                waddr[j] = j * wlen;
                p.wins.push_back({active[k0 + j], wbase[j]});
            }
            sp.wlen           = wlen;
            sp.bmin           = *std::ranges::min_element(wbase);
            sp.bmax           = *std::ranges::max_element(wbase);
            const int arena_a = nc * wlen;
            ops += arena_a;
            const int ng = (nc + kGroup - 1) / kGroup;
            // L1 nodes, per group.
            t1.clear();
            for (int g = 0; g < ng; ++g) {
                const int j0 = g * kGroup;
                const int j1 = std::min(j0 + kGroup, nc);
                map1.clear();
                for (int d = 0; d < tdm; ++d) {
                    int mn = delay(d, k0 + j0);
                    for (int j = j0 + 1; j < j1; ++j) {
                        mn = std::min(mn, delay(d, k0 + j));
                    }
                    key.clear();
                    for (int j = j0; j < j1; ++j) {
                        key.push_back(delay(d, k0 + j) - mn);
                    }
                    auto [it, fresh] =
                        map1.try_emplace(key, static_cast<int>(t1.size()));
                    if (fresh) {
                        Tmp n{};
                        n.lo   = mn;
                        n.hi   = mn;
                        n.nsrc = j1 - j0;
                        for (int j = j0; j < j1; ++j) {
                            n.src[j - j0] = j;
                            n.off[j - j0] = key[j - j0];
                        }
                        t1.push_back(n);
                    }
                    auto& n          = t1[it->second];
                    n.lo             = std::min(n.lo, mn);
                    n.hi             = std::max(n.hi, mn);
                    m1[(d * 4) + g]  = mn;
                    id1[(d * 4) + g] = it->second;
                }
            }
            // L2 nodes.
            t2.clear();
            map2.clear();
            for (int d = 0; d < tdm; ++d) {
                int mn = m1[d * 4];
                for (int g = 1; g < ng; ++g) {
                    mn = std::min(mn, m1[(d * 4) + g]);
                }
                key.clear();
                for (int g = 0; g < ng; ++g) {
                    key.push_back(id1[(d * 4) + g]);
                    key.push_back(m1[(d * 4) + g] - mn);
                }
                auto [it, fresh] =
                    map2.try_emplace(key, static_cast<int>(t2.size()));
                if (fresh) {
                    Tmp n{};
                    n.lo   = mn;
                    n.hi   = mn;
                    n.nsrc = ng;
                    for (int g = 0; g < ng; ++g) {
                        n.src[g] = key[2 * g];
                        n.off[g] = key[(2 * g) + 1];
                    }
                    t2.push_back(n);
                }
                auto& n = t2[it->second];
                n.lo    = std::min(n.lo, mn);
                n.hi    = std::max(n.hi, mn);
                m2[d]   = mn;
                id2[d]  = it->second;
            }
            // Layout. L1 source k of node n, element i (tau = lo + i), is
            // window src[k] at lo + i + off[k] - wbase: address waddr +
            // lo + off - wbase. Aliases (one source) take no space. L2 nodes
            // overlay the windows unless an alias still points into them.
            bool alias_win = false;
            int pos        = arena_a; // L1 region after the windows
            sp.n1_0        = static_cast<int>(p.nodes.size());
            for (auto& n : t1) {
                const int len = bt + (n.hi - n.lo);
                int s[4];
                for (int k = 0; k < n.nsrc; ++k) {
                    s[k] = waddr[n.src[k]] + n.lo + n.off[k] - wbase[n.src[k]];
                }
                if (n.nsrc == 1) {
                    n.addr    = s[0];
                    alias_win = true;
                    continue;
                }
                n.addr = pos;
                Node nd{};
                nd.dst  = pos;
                nd.len  = len;
                nd.nsrc = n.nsrc;
                for (int k = 0; k < 4; ++k) {
                    nd.src[k] = k < n.nsrc ? s[k] : s[0];
                }
                p.nodes.push_back(nd);
                ops += static_cast<double>(n.nsrc + 1) * len;
                pos += len;
            }
            sp.n1            = static_cast<int>(p.nodes.size()) - sp.n1_0;
            const int l1_end = pos;
            // L2 nodes first fill the window region (free once the L1 nodes
            // are computed) unless an alias still points into it, then
            // continue after the L1 region.
            int lo_pos       = alias_win ? l1_end : 0;
            const int lo_end = alias_win ? l1_end : arena_a;
            int hi_pos       = l1_end;
            sp.n2_0          = static_cast<int>(p.nodes.size());
            for (auto& n : t2) {
                const int len = bt + (n.hi - n.lo);
                int s[4];
                for (int g = 0; g < n.nsrc; ++g) {
                    const auto& c = t1[n.src[g]];
                    // N1 element at tau = lo2 + i + o_g.
                    s[g] = c.addr + (n.lo + n.off[g] - c.lo);
                }
                if (n.nsrc == 1) {
                    n.addr = s[0];
                    continue;
                }
                int& pos_ref = lo_pos + len <= lo_end ? lo_pos : hi_pos;
                n.addr       = pos_ref;
                pos_ref += len;
                Node nd{};
                nd.dst  = n.addr;
                nd.len  = len;
                nd.nsrc = n.nsrc;
                for (int k = 0; k < 4; ++k) {
                    nd.src[k] = k < n.nsrc ? s[k] : s[0];
                }
                p.nodes.push_back(nd);
                ops += static_cast<double>(n.nsrc + 1) * len;
            }
            sp.n2            = static_cast<int>(p.nodes.size()) - sp.n2_0;
            const int l2_end = hi_pos;
            arena            = std::max(l1_end, l2_end);
            // Trials: N2 element at tau = t + m2(d).
            sp.trial0 = static_cast<int>(p.trials.size());
            for (int d = 0; d < tdm; ++d) {
                const auto& n = t2[id2[d]];
                p.trials.push_back(n.addr + (m2[d] - n.lo));
            }
            ops += static_cast<double>(tdm) * bt;
            p.subs.push_back(sp);
            return arena;
        };
        for (int k0 = 0; k0 < na;) {
            int nc = std::min(kSubband, na - k0);
            while (true) {
                const auto nsubs  = p.subs.size();
                const auto nwins  = p.wins.size();
                const auto nnodes = p.nodes.size();
                const auto ntr    = p.trials.size();
                double ops        = 0;
                const int arena   = build_sub(k0, nc, ops);
                const int nn      = static_cast<int>(p.nodes.size() - nnodes);
                if ((arena <= max_arena && nn <= max_nodes) || nc == 1) {
                    p.arena     = std::max(p.arena, arena);
                    p.max_nodes = std::max(p.max_nodes, nn);
                    p.ops += ops;
                    break;
                }
                p.subs.resize(nsubs);
                p.wins.resize(nwins);
                p.nodes.resize(nnodes);
                p.trials.resize(ntr);
                nc /= 2;
            }
            if (p.arena > max_arena) {
                return p;
            }
            k0 += nc;
        }
        p.tile_sub.push_back(static_cast<int>(p.subs.size()));
        p.nsub = std::max(p.nsub, p.tile_sub[t + 1] - p.tile_sub[t]);
    }
    return p;
}

} // namespace dmt::algorithms::sdmt_gpu
