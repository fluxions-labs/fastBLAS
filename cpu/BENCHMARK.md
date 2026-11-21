# CPU GEMM Benchmark Results (M1, best-of-5)

`OMP_NUM_THREADS=8 ./build/cpu/bench_gemm`


| Kernel                     | 256³ GFLOPS | 512³ GFLOPS | 1024³ GFLOPS |
|----------------------------|-------------|-------------|--------------|
| L1  naive                  |     2.126   |     1.820   |     1.664    |
| L2  reorder (i-k-j)        |     2.170   |     1.823   |     1.630    |
| L3  register accumulation  |     5.589   |     5.916   |     5.985    |
| L4  1×4 micro-kernel       |     8.424   |     7.332   |     6.604    |
| L5  cache blocking         |    13.797   |    13.508   |    10.932    |
| L6  packing                |    14.031   |    13.962   |    13.417    |
| L7  NEON 4×4 SIMD          |    36.471   |    36.014   |    33.593    |
| L8  OpenMP (8 threads)     |   129.311   |   165.484   |   183.188    |
| L9  8×8 wide micro-kernel  |    74.525   |    74.623   |    68.298    |
| L10 pipelining + prefetch  |    79.776   |    78.722   |    71.694    |
| L11 BLIS 5-loop            |    49.287   |    65.043   |    63.767    |
| L12 autotuned              |    59.223   |    65.863   |    65.929    |
| L14 Strassen (1-level)     |    25.608   |    59.791   |    68.439    |
| Apple Accelerate (AMX)     |  1126.003   |  1321.942   |  1210.151    |

**Best result:** L8 OpenMP, 183.2 GFLOPS at 1024³ — ~110× over the naive baseline,
~5× over the best single-threaded kernel, ~15% of Accelerate's AMX-backed ceiling
(the remaining gap is the AMX coprocessor, not reachable from user-space NEON).
