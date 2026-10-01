| Platform | Time per block | Real-time factor | Coarse trials | DM rows | Useful fraction | Engine memory |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| NVIDIA L40S | 2.9 ms | 47.78× | 1 | 37 | 79% | 956 MiB |
| NVIDIA L40S (host arrays) | 10.1 ms | 13.54× | 1 | 37 | 79% | 956 MiB |
| Intel(R) Xeon(R) Gold 6348H CPU @ 2.30GHz, 8 threads | 76.3 ms | 1.80× | 1 | 37 | 79% | 956 MiB |
| Apple M1 Pro, 8 threads | 96.9 ms | 1.56× | 1 | 37 | 80% | 1038 MiB |
| Apple M1 Pro, 1 thread | 616.2 ms | 0.25× | 1 | 37 | 80% | 1038 MiB |
| Intel(R) Xeon(R) Gold 6348H CPU @ 2.30GHz, 1 thread | 585.2 ms | 0.23× | 1 | 37 | 79% | 956 MiB |

dmt version(s): 0.7.0; FFTW planner: measure.
