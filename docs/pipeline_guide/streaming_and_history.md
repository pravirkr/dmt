# Streaming and history

Engines keep a short history between calls, so a dispersed pulse that crosses
a block boundary is not cut in two. You can cut a stream into blocks anywhere.

## FDMT in `mode="valid"`

**Rules:**

- **One engine, one call per block.** Each `execute()` (or stepper `reset()`)
  consumes one block and returns `nsamps` outputs (`plan.dmt_nsamps`).
- **Contiguous, non-overlapping blocks.** Don't overlap blocks yourself; the
  history supplies the past. Concatenated outputs equal one full-mode
  transform of the whole stream, bit for bit.
- **Any block size.** `nsamps` may be smaller than $\Delta t_{\max}$; the
  history then spans several blocks. Pick it for latency and throughput.
- **The first block starts cold.** After construction or `reset_history()`
  the history is zero, and the first `fdmt.startup_samples` outputs (C++
  `plan.get_startup_samples()`) are partial sums. They are written, not
  withheld: mask or drop them. This is usually
  $\max(|\Delta t_{\min}|, |\Delta t_{\max}|)$, slightly more for FDMT-FFT.
- **Finish a stepped block before the next.** Otherwise the engine raises (see
  {ref}`stepper rules <stepper-rules>`).

```{warning}
Call `reset_history()` when the stream breaks (a new observation, dropped
data), **never between consecutive blocks**.
```

`mode="full"` and `mode="roll"` keep no history; each block stands alone.

### Blocks and chunks

A block (`nsamps`) is what one `execute()` consumes. Inside it the engine runs
the tree in chunks of `chunk_nsamps`, joined by the same history, so the
working state is sized by the chunk, not the block.

- **Default chunk.** The longest power of two ≥ 2048 whose state fits in
  2 GiB, capped at `nsamps`, and shorter on a GPU when that keeps the state in
  L2. For `cryopaf_low` with 32 768-sample blocks: 5.5 GiB for the whole
  block, 1.4 GiB with the automatic 8192-sample chunk.
- **Output.** Bit-identical for any chunk with packed input, integer-valued
  input, explicit fusion depths and the CPU tree. Automatic GPU direct levels
  agree to float rounding.
- **Stepper.** It runs whole blocks: construct it with `chunk_nsamps=nsamps`.

:::{dropdown} What the history holds
:icon: info

Both parts are kept per beam.

- **Level-0 input history** (`plan.history_size`, C++ `get_history_size()`):
  the last `dt0_max` samples of every channel, `nchans × dt0_max` floats.
- **Tree history** (`plan.tree_history_size`): the last `delay` samples of
  each merge node's head input (overlap-save, node by node). A ring buffer on
  the CPU, double-buffered on a GPU.
:::

### Candidate frontier

`FDMTBoxcar` and `FDMTFFTBoxcar` return a candidate when its excursion ends,
possibly many blocks after it began. `get_candidate_frontier()` is the oldest
`begin_sample` a later call can still return. Release or group candidates
older than it, instead of waiting a fixed lag (see the
[Python API](../api/python_api.md)).

## DDMT

`DDMT`, `SDMT` and `DDMTFFT` always stream. The history is the raw input tail,
the last $\Delta t_{\max}$ samples of every beam and channel.

- **Withheld samples.** Incomplete samples are withheld, not written:
  `get_output_nsamps(n)` is `n - max_delay` on a cold start, `n` once warm.
- **Exact.** Concatenated outputs equal one call on the whole stream, bit for
  bit.
- **Time reference.** Output sample `k` is the pulse arriving at `plan.f_ref`
  at input sample `k + plan.time_offset`.
- **One history for every overload.** Float and packed, channel- and
  time-major, host and device input share it; you can mix them in one stream.
- **GPU.** The history stays on the device. Device-memory calls are one
  asynchronous launch sequence with no host sync. Host-memory calls stream
  through the device in `gulp_size` chunks (`set_gulp_size()`).

## Several streams through one engine

To serve beams in turn from one thread, save and restore each beam's history
around its call:

::::{tab-set}
:::{tab-item} Python
:sync: python

```python
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

- **Size.** Only the history is saved (`history_state_size()` floats), not
  the working buffers.
- **Portable.** CPU and GPU use the same format, so a stream saved on one
  continues on the other. Chunk length, fusion depth and thread count don't
  change it.
- **Checked.** `load_history()` raises (`std::invalid_argument`, Python
  `ValueError`) on a buffer from a different DM grid or tree, beam count, box
  smearing or skewed layout, and on one without a header (a zeroed buffer, or
  one saved before 0.8). Start a new stream with `reset_history()`.
- **`CohFDMT`** is stateless across blocks and needs none of this.
- **For beams processed together**, `nbeams` is simpler (see
  [multi-beam batching](multibeam.md)).

:::{dropdown} Format details and GPU caveats
:icon: info

- **Layout.** A 16-word header (magic number, format version, layout hash,
  stream position), then the level-0 history, then the tree history, each
  node's slot oldest sample first.
- **Device memory.** The C++ `DeviceSpan` overloads are asynchronous on the
  given stream, except that `load_history()` first copies the header to the
  host to check it. The host overloads copy synchronously.
- **Skewed-window path.** Large plans that run on it save that engine's own,
  much smaller, backend-specific state. Only an engine with the same plan and
  `chunk_nsamps` (on the CPU also the same thread count) accepts it. On the
  CPU, 8- and 16-bit input is kept at its own width while only packed blocks
  have arrived, so the size changes: ask `history_state_size()` at each save
  (Python `save_history()` does).
- **GPU direct levels.** With many delays per channel, a GPU engine computes
  its lowest and highest levels directly, which keeps state the saved format
  does not hold. After `load_history()` it runs the plain tree (same output)
  until that state refills: up to the longest level-4 delay, about 21 000
  samples for `cryopaf_low`. If that cost matters, give each stream its own
  engine.
:::

```{seealso}
[Periodicity searches](periodicity.md) for what the startup ramp does to a
power spectrum.
```
