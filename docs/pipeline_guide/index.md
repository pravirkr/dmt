# Pipeline Integration Guide

This guide covers real-world telescope pipeline engineering: continuous data streams, overlap-save history, multi-beam batching, exact noise calibration, and known quirks.

```{toctree}
:maxdepth: 1

streaming_and_history
multibeam
noise_calibration
quirks_and_gotchas
```

---

## Guide Topics

- [Streaming & Overlap-Save History](streaming_and_history.md): How `dmt` preserves 100% of dispersed signals across block boundaries without edge artifacts.
- [Multi-Beam Batching](multibeam.md): SIMD and GPU-accelerated batch processing across tied-array telescope beams.
- [Noise Calibration & SNR Scaling](noise_calibration.md): Converting raw integrated flux into true statistical Signal-to-Noise Ratio ($\text{SNR}$).
- [Quirks & Gotchas](quirks_and_gotchas.md): Common edge cases, buffer alignments, memory footprint guidelines, and debugging tips.
