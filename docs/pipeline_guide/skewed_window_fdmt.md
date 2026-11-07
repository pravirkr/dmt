# Streaming FDMT on skewed windows

A valid-mode FDMT that keeps no per-merge tree history above level 4:
`lib/dmt/fdmt_skew.hpp` (`FDMTSkewCpu`, `compute_skew_windows`,
`estimate_skew_memory`), tested by `tests/cpp/fdmt_skew_cpu_t.cpp` and
measured by `bench/fdmt_skew_cpu_b.cpp` and `bench/fdmt_skew_mem_b.cpp`.
Both `FDMT` and `FDMTBoxcar` automatically select this path on the CPU (and
its device counterpart in `lib/dmt/fdmt_direct.hpp` /
`lib/cuda/fdmt_direct.cuh` on CUDA) when a plan's tree history exceeds 2 GiB
(`parkes_uwl`).

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

**Exactness.** Levels 0–4 are summed in integer arithmetic (uint32 running
sums, exact box sums, the plan's box-mean recipes for packed input) and
converted once to float at level 4, the tree's own int → float conversion.
The merges above are the tree's additions with the tree's operand order.
So for packed input and for float input holding integers (a filterbank),
while the level-4 row values stay below $2^{24}$ on the plan's scale (true
for 8-bit data on all four presets) and every level-0 box sum below
$2^{31}$, the output is bit-identical to the streaming tree engine for any
chunk length and any switch level; the unit test checks this on the
`cryopaf_lband` and `meerkat_lband` geometry over three consecutive blocks,
float and packed, with and without box smearing, at levels 3, 4 and 5, and
with uneven chunks, and the hidden `[.skew_preset]` case on the full
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

## Memory

From `estimate_skew_memory` (`FDMTSkewCpu`, automatic `fuse_level = 8` on
`parkes_uwl`), 8 threads, GiB = $2^{30}$ bytes (GB = $10^9$ bytes in
parentheses). "raw ring" is the per-channel causal input ring (1 byte per
sample for packed 8-bit input, 4 for float), "state + group scratch" the
full-band ping-pong state above `fuse_level` plus the 8 threads' group
scratch (`grp_a_`, `grp_b_`), "rings" the per-row causal window overlaps,
and "plan" the compact leaf, row and merge tables.

| preset, chunk, level | raw ring | state + group scratch | rings + stripes + plan | total (8-bit) | total (float) | tree state + history |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cryopaf_lband`, 4096, 4 | 0.01 GB | 0.08 GB | 0.02 GB | 0.11 GB | 0.14 GB | 0.33 + 0.07 = 0.39 GB |
| `meerkat_lband`, 4096, 4 | 0.04 GB | 0.14 GB | 0.04 GB | 0.22 GB | 0.36 GB | 0.59 + 0.26 = 0.85 GB |
| `parkes_uwl`, 1024, 4 (F=8) | 1.01 GB | 0.11 GB | 0.88 GB | **1.87 GiB (2.00 GB)** | **5.91 GiB (6.35 GB)** | 2.13 + 10.14 = 12.3 GB |
| `parkes_uwl`, 2048, 4 (F=8) | 1.02 GB | 0.19 GB | 0.88 GB | **2.36 GiB (2.53 GB)** | **6.41 GiB (6.88 GB)** | 4.23 + 10.14 = 14.4 GB |
| `parkes_uwl`, 4096, 4 (F=8) | 1.03 GB | 0.34 GB | 0.89 GB | **3.34 GiB (3.58 GB)** | **7.41 GiB (7.95 GB)** | 8.54 + 10.14 = 18.7 GB |

At level 4 with group fusion to level 8, the 10.1 GB UWL tree history
becomes 0.73 GB of per-row rings and the 2.1–8.5 GB full-band chunk state
becomes 0.11–0.34 GB (plus 8 threads' group scratch). **`parkes_uwl` fits
well under 10 GB on both 8-bit and float32 input on CPU and GPU**:

- **CPU (`dmt_bench_mem`, AMD EPYC 9654, 8 threads):**
  - `parkes_uwl`, `chunk = 1024`, 8-bit: **1910.6 MiB (1.87 GiB)** engine, **3242 MiB** process peak RSS (`FDMTBoxcar` working memory: **2.72 GB**).
  - `parkes_uwl`, `chunk = 4096`, 8-bit: **3418.8 MiB (3.34 GiB)** engine, **4750 MiB** process peak RSS.
  - `parkes_uwl`, `chunk = 1024`, float32: **6051.9 MiB (5.91 GiB)** engine (`FDMTBoxcar` working memory: **7.70 GB**).
- **GPU (`BM_fdmt_boxcar_cuda/parkes_uwl`, NVIDIA L40S):**
  - FDMT engine allocation: **1883.9 MiB (1.84 GiB)** (down from 9.05 GB).
  - Total working memory (including input/output/scorer): **2.41 GB** (`fused_uint8`), **3.08 GB** (`fused` float32), **3.51 GB** (`transform`).

## Throughput

Measured on `parkes_uwl` (32 768-sample blocks = 2.10 s of data, DM 0–2000,
4749 trials, steady-state streaming after warm-up):

- **NVIDIA L40S (`BM_fdmt_boxcar_cuda/parkes_uwl`):**
  - `transform` (float32): **352.9 ms** (**5.94×** real time, 3.51 GB)
  - `fused` (`FDMTBoxcar`, float32, wtsp 1.5): **355.3 ms** (**5.90×** real time, 3.08 GB)
  - `fused_uint8` (`FDMTBoxcar`, 8-bit, wtsp 1.5): **342.4 ms** (**6.12×** real time, 2.41 GB)
- **AMD EPYC 9654, 8 threads (`OMP_PLACES='{0}:8:8'`):**
  - `bm_skew/packed8_level4`, `chunk = 4096`: **5.86 s** (engine 3.34 GiB, peak RSS 4.64 GiB)
  - `bm_skew/packed8_level4`, `chunk = 1024`: **10.33 s** (engine 1.87 GiB, peak RSS 3.17 GiB)
  - `BM_fdmt_boxcar_stream/parkes_uwl/uint8/wtsp:1.5/cpu8`, `chunk = 1024`: **9.22 s** (working memory 2.72 GB)

## Scope and current limits

- **Automatic selection (CPU):** `FDMT` and `FDMTBoxcar` use the
  skewed-window engine when the level-0 plus tree history would exceed
  2 GiB, in `"valid"` mode, with one beam, no subband search
  (`subband_depth == 0`), fusion not disabled (`fuse_levels != 0`) and a
  chunk shorter than the block (such plans default to `chunk_nsamps = 1024`).
  Otherwise, for example with `nbeams > 1`, the engine silently keeps the
  tree and allocates its full history.
- **Automatic selection (GPU):** the skewed direct levels and target-ring
  direct top replace the unskewed direct top when the latter's rings exceed
  2 GiB at every candidate level (`"valid"` mode, automatic fusion).
- **Stepper:** `reset()` / `advance()` need whole-block chunks, so they are
  not available on the skewed path.
- **History serialization:** `save_history()` / `load_history()` work on both
  backends. The saved state is the skewed engine's own: the raw input ring,
  the level-F (CPU) or target (GPU) rings and the stream position. That is
  far smaller than the tree history it replaces, and a shared engine resumes
  on the fast path at once. A buffer is accepted only by an engine with the
  same skewed layout: the same plan and `chunk_nsamps`, and on the CPU the
  same thread count, which can change the fusion level. Anything else raises
  `invalid_argument`, as does a tree-format buffer.
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
