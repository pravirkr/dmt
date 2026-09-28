# Streaming & Overlap-Save History

Telescopes such as CHIME, FAST, MeerKAT, and ASKAP stream continuous gigabytes of data per second. Real-time pipelines must process incoming data in discrete time blocks (e.g. $N_{\text{samps}} = 1024$ or $2048$ samples per block) without losing dispersed signals that span across block boundaries.

---

## 1. The Block Boundary Challenge

A dispersed transient arriving near the end of Block $k$ has its high-frequency component in Block $k$, while its dispersed low-frequency tail extends into Block $k+1$.

If data blocks are dedispersed independently:
- **Boundary Loss**: Up to $50\% - 100\%$ of the pulse energy is truncated at block boundaries.
- **Edge Artifacts**: Spurious peaks appear at the edges of the output matrix.
- **Missed Detections**: High-DM candidates that span multiple blocks are systematically lost.

---

## 2. DMT's Overlap-Save Mechanism

In `mode="valid"`, `FDMT` and `DDMT` (on every backend) maintain **internal state history buffers**:

```
Block 0:  [─────── Waterfall Data Block 0 ───────]
                                           \ Tail / ──> [Save to History Buffer]
                                                       │
Block 1:  [Saved History] + [── Waterfall Data Block 1 ──]
                                           \ Tail / ──> [Update History Buffer]
```

### How It Works

The history is kept inside the tree, not on the raw input: every merge node
with a delay keeps the last `delay` samples of its input from the previous
block. That is overlap-save applied per node, so streaming reproduces a single
monolithic transform **bit-exactly for every DM trial**, with no redundant
recomputation of overlap samples.

1. **Construct once, call per block.** Every `execute()` (or stepper
   `reset()`) consumes one block and returns exactly $N_{\text{samps}}$
   output samples (`plan.dmt_nsamps`).
2. **Blocks must be contiguous and non-overlapping in time.** Then the output
   blocks are contiguous too: concatenating them equals one full-mode transform
   of the whole stream. Do not overlap blocks yourself; the history already
   supplies the past samples.
3. **Cold start.** After construction or `reset_history()`, the history is
   zero, so the first $\Delta t_{\max}$ samples of the first block are partial
   sums. Call `reset_history()` whenever the stream breaks (a new observation,
   dropped data).
4. **Block size is free.** `nsamps` may even be smaller than $\Delta t_{\max}$;
   the per-node history then spans several blocks. Choose it for latency and
   throughput (see [Performance](performance.md)).
5. **Stepper.** A stepped block must be finalized before the next one; stopping
   early raises instead of silently corrupting the history (see
   {ref}`stepper rules <stepper-rules>`).

`mode="full"` and `mode="roll"` keep no history; each block is independent.

### DDMT

`DDMT` always streams. Its history is the raw input tail: the last
$\Delta t_{\max}$ samples of every (beam, channel) row, prepended to the next
block. `get_output_nsamps(n)` is `n - max_delay` on a cold start and `n` once
the stream is warm, and concatenating the outputs of consecutive blocks equals
one call on the whole stream, bit for bit.

- Every `execute()` overload shares this one history: float and packed input,
  channel-major and time-major (`execute_time_major`), host and device memory.
  They may be mixed within a stream; call `reset_history()` to start a new one.
- On a GPU backend the history stays on the device. Device-memory calls are
  then a single asynchronous launch sequence on the given stream, with no
  host synchronisation; host-memory calls stream the block through the device
  in chunks of `gulp_size` samples (`set_gulp_size()`), overlapping copies and
  kernels.

---

## 3. Time-Division Multiplexing (`save_history` / `load_history`)

In multi-beam systems, a single compute thread may process multiple beams in a round-robin schedule:
- Cycle 1: Process Block $k$ of Beam 0.
- Cycle 2: Process Block $k$ of Beam 1.
- Cycle 3: Process Block $k+1$ of Beam 0.

To prevent Beam 1 from overwriting the streaming state of Beam 0, use the History State API:

```python
from dmtlib import FDMT

fdmt = FDMT(
    f_min=1200.0, f_max=1600.0, nchans=256, nsamps=1024,
    tsamp=1e-3, dt_max=100, mode="valid"
)

# Allocate storage for per-beam history states
beam_histories = {
    "beam_0": None,
    "beam_1": None,
}

def process_beam_block(beam_id, waterfall_block):
    # 1. Restore this beam's history state (if already warmed up)
    if beam_histories[beam_id] is not None:
        fdmt.load_history(beam_histories[beam_id])
    else:
        fdmt.reset_history()

    # 2. Execute transform on this block
    dmt_plane = fdmt.execute(waterfall_block)

    # 3. Save updated history state for this beam
    beam_histories[beam_id] = fdmt.save_history()

    return dmt_plane
```

### C++ Equivalent:
```cpp
// Query required state buffer size in floats
const size_t state_size = fdmt.history_state_size();
std::vector<float> beam0_state(state_size);

// Save state
fdmt.save_history(std::span<float>(beam0_state));

// Restore state
fdmt.load_history(std::span<const float>(beam0_state));
```

The saved state is only the small history (`history_state_size()` floats),
not the engine's working buffers, so one engine can serve many streams of the
same plan geometry. `CohFDMT` uses exactly this to share one fine FDMT across
all coarse-DM trials. On a GPU backend the history lives in device memory;
the `DeviceSpan` overloads (C++ only) are asynchronous on the given stream,
and the host overloads copy synchronously. A saved history belongs to the
backend that wrote it (the layout differs between backends).
For beams that are processed together, `nbeams` is usually simpler (see
[Multi-Beam Batching](multibeam.md)).
