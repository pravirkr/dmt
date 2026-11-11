# C++ quickstart

Include `<dmt/dmt.hpp>`. Engines are in `dmt::algorithms`, plans in
`dmt::plans`, and the backend choice (`dmt::Exec`) in `dmt`. To link `dmt`
into your project, see {ref}`installation <find-package>`.

## Minimal example

```cpp
#include <iostream>
#include <span>
#include <vector>

#include <dmt/dmt.hpp>

int main() {
    const float f_min    = 1200.0F;  // MHz, bottom edge of the band
    const float f_max    = 1600.0F;  // MHz, top edge of the band
    const size_t nchans  = 256;
    const size_t nsamps  = 1024;     // samples per block
    const float tsamp    = 1e-3F;    // s
    const int32_t dt_max = 128;      // largest delay trial, in samples

    // Construct once: plans the tree and allocates all working memory.
    dmt::algorithms::FDMT fdmt(f_min, f_max, nchans, nsamps, tsamp, dt_max,
                               /*dt_min=*/0, /*dt_step=*/1,
                               /*use_box_smearing=*/true, /*mode=*/"valid",
                               dmt::Exec::cpu(/*nthreads=*/4));

    const auto& plan = fdmt.get_plan();
    std::vector<float> waterfall(nchans * nsamps);     // (nchans, nsamps)
    std::vector<float> dmt_buf(plan.get_dmt_size());   // (ndms, dmt_nsamps)

    // Once per block, contiguous in time.
    fdmt.execute(waterfall, dmt_buf);
    // dmt_buf[i_dm * plan.get_dmt_nsamps() + t]; DMs: plan.get_dm_grid_final()
    std::cout << plan.get_dmt_ndms() << " DM trials\n";
    return 0;
}
```

- **Streaming.** In `"valid"` mode the engine keeps the history, so
  consecutive blocks give one continuous output. `reset_history()` starts a
  new stream. See [streaming](../pipeline_guide/streaming_and_history.md).
- **No allocation per call.** `execute()` only writes the output buffer;
  `get_memory_usage()` reports what the constructor allocated.
- **S/N.** The output is in box-mean units by default, with a different noise
  sigma per DM trial; see
  [noise calibration](../pipeline_guide/noise_calibration.md).

## Inputs and outputs

| input | overload |
| :--- | :--- |
| float32, `(nbeams, nchans, nsamps)` row-major | `execute(std::span<const float>, std::span<float>)` |
| packed unsigned 1/2/4/8/16-bit | `execute(std::span<const uint8_t>, nbits, std::span<float>)` |

- **Output.** float32, `(nbeams, ndms, dmt_nsamps)`, `nbeams *
  plan.get_dmt_size()` floats; see {ref}`output buffers <output-buffers>`.
- **Packed input.** Each channel row is `ceil(nsamps * nbits / 8)` bytes,
  samples LSB-first within a byte. Pass ≤ 8-bit data as is: it is faster
  than converting to float.
- **Channel order.** Rows ascend in frequency and `f_min`/`f_max` are band
  edges; see {ref}`channel order <channel-order>`. For SIGPROC headers:
  `f_min = min(fch1, fch1 + (nchans - 1) * foff) - |foff| / 2`,
  `f_max = f_min + nchans * |foff|`.
- **Custom grids.** Pass a `std::vector<int32_t>` of delays or a
  `std::vector<float>` of DMs instead of `dt_max, dt_min, dt_step`.
- **Errors.** Invalid sizes, `nbits` or configurations throw
  `std::invalid_argument`; stepper misuse throws `std::logic_error`.

```cpp
const auto row_bytes = (nsamps * 2 + 7) / 8;   // 2-bit, 4 samples per byte
std::vector<uint8_t> packed(nchans * row_bytes);
fdmt.execute(std::span<const uint8_t>(packed), /*nbits=*/2, dmt_buf);
```

## GPUs

The same class runs on a GPU with `dmt::Exec::cuda(device)` (or
`dmt::Exec::hip(device)`); `dmt::available_backends()` lists what the build
has. Host spans work on every backend. Device memory goes through
`dmt::DeviceSpan` and an optional `dmt::Stream` (a `cudaStream_t` converts),
asynchronously:

```cpp
dmt::algorithms::FDMT gpu(f_min, f_max, nchans, nsamps, tsamp, dt_max, 0, 1,
                          true, "valid", dmt::Exec::cuda(/*device=*/0));
gpu.execute(dmt::DeviceSpan<const float>(d_wf, nchans * nsamps),
            dmt::DeviceSpan<float>(d_dmt, gpu.get_plan().get_dmt_size()),
            stream);
cudaStreamSynchronize(stream);
```

The public headers include no CUDA header, so CPU-only and GPU builds expose
the same declarations.

## Stepper

To inspect intermediate subbands, for example stopping two levels before the
root:

```cpp
fdmt.reset(waterfall, dmt_buf);          // consume one block (level 0)
fdmt.advance_until_remaining(2);         // 4 subbands left
for (size_t s = 0; s < fdmt.num_subbands(); ++s) {
    const auto sub = fdmt.view_subband(s);   // zero-copy view
}
fdmt.finalize();                         // finish to the root, into dmt_buf
```

Construct a stepping engine with `chunk_nsamps = nsamps`, and from an
`FDMTPlan` with `subband_depth` to also get the staggered subbands. See
{ref}`stepper rules <stepper-rules>`.
