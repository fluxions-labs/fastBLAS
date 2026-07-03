

# fastBLAS

Learning GEMM optimization from first principles.

The repository starts with a scalar CPU implementation and builds toward cache-blocked, vectorized kernels. A parallel GPU track implements the same ideas in WGSL using WebGPU.

Every optimization is implemented as a separate kernel so performance and correctness can be measured independently.

## CPU

`cpu/JOURNEY.md`

Progression from a naive triple loop to a blocked, vectorized GEMM.

Implemented

* L1 — naive GEMM
* L2 — loop reordering
* L3 — register accumulation

Remaining

* L4 — micro-kernel
* L5 — cache blocking
* L6 — packing
* L7 — NEON vectorization
* L8 — OpenMP
* L9–L14 — wider kernels, prefetching, BLIS 5-loop structure, autotuning, Strassen, roofline analysis

## GPU

`gpu/JOURNEY_GPU.md`

The GPU track mirrors the CPU progression using WGSL.

Implemented

* benchmark harness
* naive reference kernel

Remaining

* G3 — coalesced access
* G4 — shared-memory tiling
* G5 — register tiling
* G6 — double buffering
* G7–G8 — parameter tuning

## Build

```bash
cmake -B build
cmake --build build --target test_gemm bench_gemm gpu_gemm
```

CPU

```bash
./build/cpu/test_gemm
./build/cpu/bench_gemm
```

GPU

```bash
./build/gpu_gemm
./build/gpu_gemm src/kernels/gemm/gemm_tiled_16x16.wgsl
```

## Repository layout

```text
cpu/
├── JOURNEY.md
├── include/blas.h
├── src/
├── benchmark/
└── test/

gpu/
├── JOURNEY_GPU.md
└── src/kernels/gemm/

benchmark/
└── gpu_gemm.cpp
```

## Workflow

1. Read the next level.
2. Implement the kernel.
3. Verify correctness.
4. Benchmark.
5. Compare against the reference implementation.

## Topics

* cache locality
* register blocking
* SIMD
* memory bandwidth
* instruction-level parallelism
* shared memory
* arithmetic intensity
* roofline analysis

## Target platform

**CPU:** Apple M1 (NEON)

**GPU:** Metal through WebGPU (`wgpu-native`)

Performance is compared against Apple Accelerate and MLX, which use AMX hardware unavailable to NEON or WGSL kernels.

## Current status

* CPU: L1–L3 complete
* GPU: benchmark harness and naive kernel complete
* Remaining levels are provided as implementation stubs.

