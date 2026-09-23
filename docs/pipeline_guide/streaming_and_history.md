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

In `mode="valid"`, `FDMTCPU` and `DDMTCPU` maintain **internal state history buffers**:

```
Block 0:  [─────── Waterfall Data Block 0 ───────]
                                           \ Tail / ──> [Save to History Buffer]
                                                       │
Block 1:  [Saved History] + [── Waterfall Data Block 1 ──]
                                           \ Tail / ──> [Update History Buffer]
```

### How It Works:
1. **Cold Start**: On the first call to `execute()`, the history buffer is empty. The output length is $N_{\text{samps}} - \Delta t_{\max}$ valid samples.
2. **Streaming Steady State**: Intermediate delayed subband states up to $\Delta t_{\max}$ are retained in internal memory. On all subsequent calls, previous tail states are prepended before merging, producing a continuous stream of exactly $N_{\text{samps}}$ valid samples per block!
3. **Zero Energy Loss**: Signals crossing block boundaries are fully integrated with $100\%$ peak SNR recovery.

---

## 3. Time-Division Multiplexing (`save_history` / `load_history`)

In multi-beam systems, a single compute thread may process multiple beams in a round-robin schedule:
- Cycle 1: Process Block $k$ of Beam 0.
- Cycle 2: Process Block $k$ of Beam 1.
- Cycle 3: Process Block $k+1$ of Beam 0.

To prevent Beam 1 from overwriting the streaming state of Beam 0, use the History State API:

```python
from dmtlib import FDMTCPU

fdmt = FDMTCPU(
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
