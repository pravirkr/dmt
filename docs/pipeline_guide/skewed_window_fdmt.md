# Streaming FDMT on skewed windows

A valid-mode FDMT that keeps no per-merge tree history above level 4:
`lib/dmt/fdmt_skew.hpp` (`FDMTSkewCpu`, `compute_skew_windows`,
`estimate_skew_memory`), tested by `tests/cpp/fdmt_skew_cpu_t.cpp` and
measured by `bench/fdmt_skew_cpu_b.cpp`, `bench/fdmt_skew_mem_b.cpp` and
the `parkes_uwl` boxcar suite. Both `FDMT` and `FDMTBoxcar` automatically
select this path on the CPU (and its device counterpart in
`lib/dmt/fdmt_direct.hpp` / `lib/cuda/fdmt_direct.cuh` on CUDA) when the
tree path does not fit the memory budget (`max_memory_bytes`, 10 GiB by
default; `parkes_uwl`), and it supports the subband search
(`subband_depth > 0`) as well as the full band.

## The problem

The streaming tree ([streaming and history](streaming_and_history.md))
keeps, at every merge, the last `delay` samples of its head input, so the
tree history is the sum of all merge delays (`plan.tree_history_size`). On a
dense delay grid that grows as $0.755\,\Delta t_{\max}^2$ floats, and on the
Levin grids as trials $\times \Delta t_{\max}$: 10.1 GB for the
`parkes_uwl` preset ($\Delta t_{\max} = 253\,891$, 4749 trials), doubled on
a GPU. The chunk state (two levels of `rows × chunk` floats) is 2.1 GB at
chunk 1024 and 8.5 GB at 4096. The standard streaming tree therefore does
not fit a 10 GB budget on `parkes_uwl` at any chunk length.

## The idea

Walk the plan top-down. A root row is read at delay 0. A merge at level
$l$ with head delay $d$ passes its delay range $[s_{lo}, s_{hi}]$ to its
tail child and $[s_{lo} + d, s_{hi} + d]$ to its head child; a copy passes
its own. Row $r$ then has $s_{lo}(r)$ and $s_{hi}(r)$, the smallest and
largest accumulated head delay over every path to a trial that reads it.
For an output chunk $[T, T + n)$ the row is needed exactly on

$$[T - s_{hi}(r),\; T + n - s_{lo}(r)),$$

its *skewed window*. A parent computed on its own window finds both
children inside theirs, so a chunk needs nothing of the past except the
window overlap $s_{hi} - s_{lo}$, the row's *spread*, which the row keeps as
a short ring: every chunk computes $n$ new samples per row and carries the
last `spread` over. The per-row rings replace the per-merge histories.

A row's spread is set by how many trials read it. A level-0 row (one
channel at one box length) is read by every trial, so its spread is about
$\Delta t_{\max}$; a row at level 4 and above is read by a narrow DM range
and its spread is tiny (tables below). The two cannot be mixed: a level-4
row evaluated in the past needs its children in the past. So levels 0–3 are
never materialised. A level-4 row is the sum of its 16 leaves (channel, box
length, accumulated delay; the tables of `fdmt_direct.hpp`, which the CUDA
direct path also uses), each read as a difference of the channel's running
sum.

The *leaf level* $K$ at which rows are read from the raw ring need not be
4. Below the fusion level $F$ every row is computed on a *local* window
(placed inside its level-$F$ ancestor's), so a lower $K$ is possible: its
rows have fewer leaves (2 at $K = 1$) and the levels between $K$ and $F$
become the tree's cheap binary merges inside the group scratch, at the cost
of recomputing a larger local overlap and a little more group memory. On
the CPU the leaf gathers dominate float input (double running sums, 16
leaves per row at $K = 4$), so the engine picks the leaf level with the
chunk (below).

Three further observations make the chunk state and raw input ring compact
too:

1. **Per-channel causal raw input rings and narrow running-sum stripes.**
   Channel $c$ is only ever read up to its own largest leaf lookback
   $\text{hist}(c) = \max_{r \ni c}(s_{hi}(r) + d_{\text{acc}} + L_0)$,
   which increases from the top of the band ($f_{\max}$) to the bottom
   ($f_{\min}$). Allocation per channel is $R(c) = \text{hist}(c) + n$,
   saving $1.68\times$ of raw-ring memory on `parkes_uwl`. Likewise, within
   each level-4 subband $s$, all rows have delays inside $[s_{lo,\min}(s),
   s_{hi,\max}(s)]$, so its running-sum stripe only covers a window of
   length $n + (s_{hi,\max}(s) - s_{lo,\min}(s)) + e_{\max}(s)$ ($8.5\times$
   narrower on `parkes_uwl`).
2. **Group fusion across levels $K \dots F$.** Each level-$F$ subband group
   ($F = 8$ on `parkes_uwl`, 256 channels) is independent. Each thread
   evaluates its group's level-$K$ subbands and merges levels $K+1 \dots F$
   inside per-thread group scratch (`grp_a_`, `grp_b_`), placing `out_f` in
   the tail of the group scratch after level $F-1$'s head/tail inputs, and
   only levels $F \dots L-1$ use the full-band ping-pong buffers (`state_a_`,
   `state_b_`). On `parkes_uwl` the widest full-band level drops from
   240 636 rows (level 4) to 28 284 rows (level 9), an $8.5\times$ reduction
   in full-band state memory.
3. **Subband batching and target-ring `DirectTopD` on CUDA.** On the GPU,
   `fdmt_direct.hpp` evaluates level $K$ on skewed windows directly from the
   sliding running-sum window, merges levels $K+1 \dots k_{\text{top}}$ one
   coordinate-weighted subband batch at a time (up to 24 batches) in a shared
   ping-pong workspace using compact per-row causal rings, and reads level
   $k_{\text{top}}$ in `DirectTopD` at relative offset $d - s_{lo}(r) \le
   \text{spread}(r)$.

**Exactness.** Levels 0–$K$ are summed in integer arithmetic (uint32
running sums, exact box sums, the plan's box-mean recipes for packed input)
and converted once to float at level $K$, the tree's own int → float
conversion.
The merges above are the tree's additions with the tree's operand order.
So for packed input and for float input holding integers (a filterbank),
while the level-4 row values stay below $2^{24}$ on the plan's scale (true
for 8-bit data on all four presets) and every level-0 box sum below
$2^{31}$, the output is bit-identical to the streaming tree engine for any
chunk length and any switch level; the unit test checks this on the
`cryopaf_lband` and `meerkat_lband` geometry over three consecutive blocks,
float and packed, with and without box smearing, at levels 3, 4 and 5,
at leaf levels 1 and 2 below fusion from level 4, and with uneven chunks, and the hidden `[.skew_preset]` case on the full
presets (DM 0–2000, 32 768-sample blocks). Non-integer float input is summed
in double running sums on the CPU (float on the GPU) and agrees within float
rounding (measured: within $1.5 \times 10^{-7}$ of the output scale on the
`cryopaf_lband` and `meerkat_lband` geometry).

## Window spreads

`dmt_tests "[.skew_report]"` prints these for the four benchmark presets
(DM 0–2000 on the piecewise Levin grid). Largest and mean spread per level,
in samples:

| level | `cryopaf_lband` (Δt 8877) | `meerkat_lband` (Δt 13 876) | `parkes_uwl` (Δt 253 891) |
| ---: | ---: | ---: | ---: |
| 1 | 2325 / 557 | 6134 / 1188 | 127 501 / 3397 |
| 2 | 1333 / 292 | 3114 / 634 | 49 906 / 1695 |
| 3 | 743 / 149 | 1610 / 328 | 30 103 / 837 |
| 4 | 332 / 74 | 870 / 166 | 11 302 / 425 |
| 5 | 168 / 36 | 514 / 83 | 6451 / 230 |
| 6 | 86 / 17 | 278 / 41 | 3641 / 132 |
| 8 | 23 / 2.5 | 66 / 8.9 | 909 / 43 |
| 10 | 2 / 0.0 | 10 / 1.4 | 166 / 8.6 |
| 12 | – | 0 | 0 |

Summed over levels 4 to the root, the overlap a chunk would recompute (or,
here, copies in and out of the rings) relative to the rows' chunk work:

| chunk | `cryopaf_lband` | `meerkat_lband` | `parkes_uwl` |
| ---: | ---: | ---: | ---: |
| 1024 | 2.2% | 4.4% | 19.8% |
| 4096 | 0.6% | 1.1% | 5.0% |

(at level 3: 1.0% / 1.9% / 8.4% at chunk 4096; at level 5: 0.3% / 0.6% /
3.0%).

## Subband search

A subband search reads rows below the root: at depth $d$ every row of
levels $\text{root} - d \dots \text{root} - 1$, tree and staggered, also
the delays the full-band grid never reads. `compute_skew_windows(plan,
level, d)` (`lib/dmt/fdmt_skew_windows.hpp`) gives each such row a window
and a fixed **lag**: at the chunk whose root samples are $[T, T + n)$, the
row is emitted for stream samples $[T - \text{lag}, T - \text{lag} + n)$,
a slice of its own window $[T - s_{hi}, T + n - s_{lo})$.

- A row the root already reads takes $\text{lag} = s_{lo}$: its window is
  unchanged.
- Any other row takes the lag that grows its children's windows least.
  Its tail must be read at $[\text{lag}, \text{lag}]$ and its head at
  $[\text{lag} + d, \text{lag} + d]$; where both fall inside the children's
  existing windows, nothing grows. Ties go to the previous row of the same
  subband, since adjacent delays have nearly equal lags. A staggered row's
  two children are consistent in this sense whenever a DM exists that uses
  both, so its lag fits them.

The subband rows therefore ride on the full band's windows: on
`parkes_uwl` the mean window spread at every level is within 1% of the
full-band search's (level 3: 844 against 837 samples), and the state grows
only by the extra rows. The lag of a row is about the dispersion delay from
its subband down to the bottom of the band: zero for the lowest subbands,
up to the whole 16 s sweep for the top ones. A subband candidate thus
arrives at about the time the full-band search would report the same burst,
with the tree path's sample indices, S/N and widths; only the call that
returns it moves. `flush()` adds no samples, so at the end of a stream the
last `lag` samples of each subband row are searched only if it continues.

- **CPU:** `FDMTSkewCpu(..., search_depth)` keeps the fused levels below
  the searched ones (`plan.max_fuse_levels`) and hands each searched level
  to the sink right after its merge (`FDMTSearchInput::row_offset` /
  `row_lag`).
- **GPU (windowed top):** the skewed direct levels and the subband-batched
  mid levels compute only level $k_{top}$'s $n$ newest samples (local
  windows); each level-$k_{top}$ row carries the rest of its window,
  $\text{spread}(r)$ samples, over from the previous chunk (88 MB on
  `parkes_uwl`). Levels $k_{top} + 1 \dots \text{root}$ are merged over
  their full windows in the state halves, with no history and no target
  rings (which would need 13 GB at depth 4). The local windows of levels
  3–6 are still wide (mean spread 753 samples at level 3 against a chunk of
  1024), so the windowed top runs best at long chunks: 2.8 s per block at
  chunk 1024, 1.7 s at 2048 and 1.25 s at 4096 on `parkes_uwl` uint8.
  The skewed level-3 kernel stages each slice's channels once for up to 16
  rows; a slice groups rows whose shifts lie within 1024 samples (256 left
  about one row per slice on these local windows: 1.27 → 1.17 s per block
  for uint8, 1.82 → 1.62 s for float32).

Both are tested against the tree path (`tests/cpp/fdmt_skew_cpu_t.cpp`,
`[skew][boxcar]`): identical candidates and S/N at depths 1, 3 and 5,
float and packed, CPU and GPU.

## Memory budget and chunk length

`FDMT` and `FDMTBoxcar` take the tree path when it fits `max_memory_bytes`
(10 GiB by default) and the skewed path otherwise, at the longest chunk
that fits:

- **CPU:** from `estimate_skew_memory()` (0.4 s on `parkes_uwl`), with a
  tenth of headroom for the merge tables; the allocation is then checked.
  It takes the first (chunk, leaf level) that fits from a list in measured
  speed order (`FDMTCpuEngine::skew_candidates`; `parkes_uwl`, 8 threads,
  s per 2.1 s block):

  | ring | chunk 4096 | chunk 2048 |
  | :--- | :--- | :--- |
  | float | $K=1$ 4.96, $K=2$ 6.6, $K=3$ 8.8, $K=4$ 13.3 | $K=1$ 7.4, $K=2$ 9.4, $K=3$ 11.1, $K=4$ 15.3 |
  | 8-bit | $K=2$ 4.08, $K=3$ 4.38, $K=4$ 4.69, $K=1$ 4.77 | $K=4$ 5.99, $K=2$ 6.28 |

  A float ring prefers $K = 1$ at 4096, then $K = 2$, then $K = 1$ at 2048
  ahead of $K \ge 3$ at 4096; an 8-bit ring prefers $K = 2$ (its integer
  running sums make the leaves cheap). Chunks of 1024 and below follow.
  An explicit `chunk_nsamps` takes the fastest leaf level that fits it.
  The ring width is known at construction only for an `FDMTBoxcar` with a
  packed configuration (`BoxcarConfig` input bits). A bare `FDMT` is built
  for float input and keeps the float choice ($K = 1$) when it is later fed
  packed blocks: 4.76 s per block on 8-bit `parkes_uwl` instead of 4.08 s.
- **GPU:** by building the engine (about 1 s) and measuring it. The
  windowed top tries 4096 and 2048 first, but only builds a longer chunk
  that is predicted to fit: its memory is affine in the chunk (rows of
  chunk + spread samples, rings of reach + chunk), so two measured engines
  (the automatic chunk and half of it) extrapolate exactly. Construction
  thus never holds more than the budget (it used to build the 12 GiB
  float32 engine at chunk 4096 to find it over). Any other path keeps its
  automatic chunk and halves it while over budget.
- **FDMTBoxcar** gives the FDMT the budget minus its own carry, peak,
  candidate and threshold buffers, and halves an automatic chunk again if
  the measured total is still over.

## Memory

`parkes_uwl` (32768-sample blocks, DM 0–2000, 4749 trials), FDMTBoxcar with
boxcars to 100 ms at wtsp 1.5, default budget; working memory is
`get_memory_usage().total()` plus the input block (GiB = $2^{30}$ bytes):

| search | backend | chunk | working memory | tree path |
| :--- | :--- | ---: | ---: | ---: |
| full band, uint8 | CPU, 8 threads | 4096 | 5.2 GiB | 12–16 GB |
| full band, float32 | CPU, 8 threads | 4096 | 9.9 GiB | 12–16 GB |
| depth 4, uint8 | CPU, 8 threads | 4096 | 6.2 GiB | 17.4 GB |
| depth 4, float32 | CPU, 8 threads | 2048 | 9.2 GiB | 17.4 GB |
| full band, uint8 | GPU | 1024 | 2.1 GiB | 9–15 GB |
| full band, float32 | GPU | 1024 | 2.7 GiB | 9–15 GB |
| depth 4, uint8 | GPU | 4096 | 9.4 GiB | 15.0 GB |
| depth 4, float32 | GPU | 2048 | 10.8 GiB (9.96 engine) | 14.0 GB |
| depth 5, uint8 | GPU | 2048 | 8.1 GB engine | 16.3 GB |

The largest parts are the raw input ring (each channel's dispersion reach
plus a chunk: about 1 GB at 1 byte per sample, 4 GB for float input) and,
for a subband search, the boxcar's carried samples (`max_boxcar` per row:
218k rows at depth 4, 1.4 GB in float or uint32).

## Throughput

`parkes_uwl`, FDMTBoxcar, steady-state streaming after warm-up (AMD EPYC
9654 with 8 threads one per CCD; NVIDIA L40S, device-resident), from the
boxcar suite (`bench/results/parkes_uwl/plots/summary.md`):

| search | GPU | CPU, 8 threads |
| :--- | ---: | ---: |
| full band, uint8 | 0.36 s per block (5.8× real time) | 4.17 s (0.50×) |
| full band, float32 | 0.36 s (5.8×) | 5.25 s (0.40×) |
| depth 4, uint8 | 1.17 s (1.79×) | 14.3 s (0.15×) |
| depth 4, float32 | 1.62 s (1.30×) | 20.1 s (0.10×) |

(The CPU full-band rows are not in the suite; they come from a standalone
FDMTBoxcar loop over 5 blocks on the same machine and threads. The float32 row
was 13.5 s before the leaf level was chosen by speed, see the budget
section above.)

The GPU searches depth 4 at the native 64 µs faster than real time within
10 GiB; the CPU does not, but keeps up from 256 µs (the real-time sweep on
the [Benchmarks](../benchmarks.md#parkes-uwl-search) page).

## Scope and current limits

- **Automatic selection:** valid mode, one beam, automatic fusion (CPU:
  `fuse_levels != 0`), and for a subband search a depth that leaves the
  leaf and fused levels below the searched ones (CPU: `niters >= depth +
  5`; GPU: `niters >= depth + 5` with the direct levels at 3). Otherwise
  the tree path is taken, and construction throws if it does not fit the
  budget (for example `nbeams > 1` on `parkes_uwl`).
- **Stepper:** `reset()` / `advance()` need whole-block chunks, so they are
  not available on the skewed path.
- **History serialization:** `save_history()` / `load_history()` work on both
  backends. The saved state is the skewed engine's own: the raw input ring,
  the level-F (CPU), target or carried level-$k_{top}$ (GPU) rings and the
  stream position. That is far smaller than the tree history it replaces,
  and a shared engine resumes on the fast path at once. On the CPU the raw
  ring is saved at its own width: while only packed 8- or 16-bit blocks
  have arrived, three samples go into one (8-bit) or two (16-bit) exact
  24-bit integer float words, so an 8-bit `parkes_uwl` checkpoint is
  1.97 GB instead of 5.88 GB (float layout). `history_state_size()`
  therefore follows the input (compact, then the float size after a float
  block or before any input); `load_history()` accepts either layout and
  rejects altered words. A buffer is accepted
  only by an engine with the same skewed layout: the same plan and
  `chunk_nsamps`, and on the CPU the same thread count, which can change the
  fusion level. Anything else raises `invalid_argument`, as does a
  tree-format buffer.
- **Switching input type mid-stream** (float and packed blocks on one engine)
  keeps the history. Rows that straddle the switch are recomputed from the
  raw samples with the new block's box-mean convention. The tree instead keeps
  the previous block's convention for rows it had already stored. So for up to
  `dt_max` samples after a switch the two differ by at most the packed-input
  rounding (half an LSB per channel), and every later sample is identical.
- **Packed input on the GPU skewed path** uses the tree's rounded box means.
  The direct top sums its leaves in a different order than the tree, though,
  so once a level's sums exceed $2^{24}$ units of $2^{-F}$ (as on
  `parkes_uwl`), packed output agrees with the CPU within float rounding
  rather than bit for bit. Float input carries the same caveat as every GPU
  direct path.
