# Streaming and history

A real-time pipeline gets data in blocks of, say, 1024 or 32 768 samples. A
dispersed pulse near the end of one block has its low-frequency part in the
next. If you dedisperse each block on its own, that pulse is cut in two.
`dmt` engines avoid this by keeping a short history between calls, so a
stream can be cut into blocks anywhere.

## FDMT in `mode="valid"`

The history lives inside the tree, not on the raw input. Every merge node with
a delay keeps the last `delay` samples of its input from the previous block.
This is overlap-save applied node by node: streaming reproduces one long
transform bit for bit, for every DM trial, and no overlap samples are
recomputed.

What that means in practice:

- **Construct once, call once per block.** Each `execute()` (or stepper
  `reset()`) consumes one block and returns exactly `nsamps` outputs
  (`plan.dmt_nsamps`).
- **Pass contiguous, non-overlapping blocks.** The outputs are then
  contiguous too, and concatenating them equals a full-mode transform of the
  whole stream. Don't overlap blocks yourself; the history already supplies
  the past.
- **The first block starts cold.** After construction or `reset_history()`
  the history is zero. The first `startup_samples` outputs
  (`fdmt.startup_samples`, C++ `plan.get_startup_samples()`) are partial sums
  over the channels whose data has arrived. Usually that is
  $\max(|\Delta t_{\min}|, |\Delta t_{\max}|)$, a little more for FDMT-FFT
  because of its interpolation guard. These samples are written, not
  withheld, so mask or drop them. With real, nonzero-mean data they form a
  ramp that grows with DM and puts a lot of power at low Fourier frequencies
  (see [periodicity searches](periodicity.md)).
- **Any block size works.** `nsamps` can be smaller than $\Delta t_{\max}$; the
  history then spans several blocks. Choose it for latency and throughput.
- **Blocks and chunks are different things.** A block (`nsamps`) is what one
  `execute()` consumes and returns. Inside a block the engine runs the tree in
  chunks of `chunk_nsamps` samples, so its working state is two levels of
  rows × chunk rather than rows × `nsamps`, and the same overlap-save history
  joins the chunks. Some paths give bit-identical output for any chunk:
  packed input, integer-valued input, explicit fusion depths, and the CPU
  tree. Automatic GPU direct levels agree within float rounding. By default
  the chunk is the longest power of two ≥ 2048 whose state fits in 2 GiB,
  capped at `nsamps`, and shorter on a GPU when that keeps the state in L2.
  For example, for the `cryopaf_low` preset (512 channels at 512 µs, DM 0–2000)
  with 32 768-sample blocks, a whole-block tree state is 5.5 GiB; the
  automatic 8192-sample chunk needs 1.4 GiB and gives the same output.
- **Finish a stepped block before starting the next.** Stopping early raises
  rather than silently corrupting the history (see
  {ref}`stepper rules <stepper-rules>`). The stepper runs whole blocks, so
  construct a stepping engine with `chunk_nsamps=nsamps`.

```{warning}
Call `reset_history()` when the stream breaks (a new observation, dropped
data), and **never between consecutive blocks**.
```

:::{dropdown} What the history holds
:icon: info

The history has two parts, both kept per beam:

- **Level-0 input history** (`plan.history_size`, C++ `get_history_size()`):
  the last `dt0_max` samples of every channel, where `dt0_max` is the largest
  level-0 delay. That is `nchans × dt0_max` floats, only what level 0 reads
  from before the block.
- **Tree history** (`plan.tree_history_size`): the last `delay` samples of each
  merge node's head input. On the CPU it is a ring buffer, so a block moves
  $O(\min(\text{delay}, N_\text{samps}))$ samples per node instead of
  $O(\text{delay})$. On a GPU it is double-buffered.
:::

```{note}
`mode="full"` and `mode="roll"` keep no history; each block stands alone.
```

`FDMTBoxcar` and `FDMTFFTBoxcar` also carry open excursions across blocks:
a candidate is returned when its excursion ends, possibly many blocks after
it began. `get_candidate_frontier()` is the oldest `begin_sample` a later
call can still return, so every candidate that begins before it has
already been returned. Use it, not a fixed lag, to decide when candidates
near a sample can be grouped and released (see the
[Python API](../api/python_api.md)).

## DDMT

`DDMT` (and `SDMT`, `DDMT-FFT`) always streams. Its history is the raw input
tail: the last $\Delta t_{\max}$ samples of every beam and channel, prepended
to the next block. Unlike FDMT it withholds what it cannot compute yet:
`get_output_nsamps(n)` is `n - max_delay` on a cold start and `n` once the
stream is warm. Concatenating the outputs of consecutive blocks equals one
call on the whole stream, bit for bit. Output sample `k` of the stream is the
sum for a pulse arriving at `plan.f_ref` (default `f_min`) at input sample
`k + plan.time_offset`.

- **One history for every overload.** Every `execute()` overload shares this one history: float and packed input,
  channel-major and time-major (`execute_time_major`), host and device memory.
  You can mix them within a stream. Call `reset_history()` to start a new one.
- **On a GPU** the history stays on the device. Device-memory calls are then one
  asynchronous launch sequence on the given stream, with no host
  synchronisation. Host-memory calls stream the block through the device in
  chunks of `gulp_size` samples (`set_gulp_size()`), overlapping copies and
  kernels.

## Several streams through one engine: `save_history` / `load_history`

If one thread serves several beams in turn (block *k* of beam 0, block *k* of
beam 1, block *k+1* of beam 0, ...), save and restore each beam's history
around its call:

::::{tab-set}
:::{tab-item} Python
:sync: python

```python
from dmtlib import FDMT

fdmt = FDMT(f_min=1200.0, f_max=1600.0, nchans=256, nsamps=1024,
            tsamp=1e-3, dt_max=100)
histories = {}

def process(beam_id, block):
    if beam_id in histories:
        fdmt.load_history(histories[beam_id])
    else:
        fdmt.reset_history()
    out = fdmt.execute(block)
    histories[beam_id] = fdmt.save_history()
    return out
```
:::
:::{tab-item} C++
:sync: cpp

```cpp
std::vector<float> beam0(fdmt.history_state_size());
fdmt.save_history(std::span<float>(beam0));
fdmt.load_history(std::span<const float>(beam0));
```
:::
::::

The saved state is only the small history (`history_state_size()` floats),
not the engine's working buffers, so one engine can serve many streams with
the same plan. It starts with a 16-word header (a magic number, the format
version, a hash of the layout and the stream position), followed by the
level-0 input history and then the tree history, each node's slot oldest
sample first. The layout is the same on the CPU and on a GPU, so a stream
saved on one continues on the other. Chunk length, fusion depth and thread
count do not change it. `load_history()` checks the header and raises
(`std::invalid_argument`, `ValueError` in Python) on a buffer saved by an
engine with a different DM grid or tree, beam count or box smearing, by the
skewed-window path, or without a header: a zeroed buffer, or one saved
before 0.8. Start a new stream with `reset_history()`, not with a zeroed
buffer. On a GPU the history lives in device memory. The C++ `DeviceSpan`
overloads are asynchronous on the given stream, except that
`load_history()` first copies the header to the host to check it; the host
overloads copy synchronously. Plans large enough to run on the
skewed-window path save that engine's own, much
smaller state, specific to its backend, which only an engine with the same
skewed layout accepts (same plan and `chunk_nsamps`, on the CPU also the
same thread count); a mismatched buffer raises. On the CPU that state keeps 8- and
16-bit input at its own width (three 8-bit samples per float word) while
only packed blocks have arrived, so its size follows the input: ask
`history_state_size()` when saving (the Python `save_history()` does), not
once at setup. `CohFDMT` needs none of this: it is
stateless across blocks.

:::{dropdown} GPU caveat: direct levels after `load_history()`
:icon: info

One GPU caveat applies. With many delays per channel, a GPU engine computes
its lowest and highest levels directly by default.
That keeps extra state the saved format does not hold: a running-sum window
per channel and a ring per level-4 row. After `load_history()` the engine
runs the plain tree, with the same output, until that state has refilled.
That takes up to the longest delay any trial applies to a level-4 row, about
21 000 samples for `cryopaf_low`. An engine shared by many streams therefore
spends the start of each block on the slower path. Where that matters, give
each stream its own engine. For beams processed together, `nbeams` is
simpler (see [multi-beam batching](multibeam.md)).
:::

```{seealso}
- [Periodicity searches](periodicity.md): what the startup ramp does to a power spectrum.
- [Multi-beam batching](multibeam.md): several beams through one engine.
```
