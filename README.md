# fastBLAS

Learning how to write a fast matrix multiply from scratch. Started on CPU with Apple M1/NEON, now extending to GPU with WebGPU.

The whole point: you write each kernel yourself. Read the problem, implement it, measure it, *then* read why it works. Don't skip ahead.

## Two tracks

**CPU track** — `cpu/JOURNEY.md` — goes from a naive 2 GFLOPS triple loop to 100+ GFLOPS via cache blocking, vectorization, register optimization, and all the micro-optimization tricks.

- **Done:** Levels 1–3 (naive, reorder, register accumulate)
- **Stubbed:** Levels 4–8 (micro-kernel, blocking, packing, NEON, OpenMP)
- **Stubbed:** Levels 9–14 (wide kernels, prefetch, full BLIS 5-loop, autotuning, Strassen, roofline)

**GPU track** — `gpu/JOURNEY_GPU.md` — same concepts but in WGSL. Shared memory = CPU cache. Register tiles = GPU register reuse. Double-buffering = prefetch.

- **Working:** harness + provided naive reference (~85 GFLOPS on M1)
- **Stubbed:** G3–G8 (coalescing, shared tiling, register tiling, double buffering, tuning)

## Build & run

```bash
cmake build
cmake --build build --target test_gemm bench_gemm gpu_gemm
```

CPU tests:
```bash
./build/cpu/test_gemm              # should print ALL TESTS PASSED
./build/cpu/bench_gemm             # shows L1-L3 + Apple Accelerate ceiling
```

GPU shaders:
```bash
./build/gpu_gemm                   # runs the provided reference
./build/gpu_gemm src/kernels/gemm/gemm_tiled_16x16.wgsl  # run your stub
```

GPU stubs will print `FAIL` until you implement them (they write zeros on purpose — the harness checks that).

## What's where

```
cpu/JOURNEY.md               — the problems and explanations
cpu/include/blas.h           — function signatures
cpu/src/level3/              — your kernels go here
cpu/benchmark/bench_gemm.cpp — timing harness (auto-finds Accelerate, MLX)
cpu/test/test_gemm.cpp       — correctness check

gpu/JOURNEY_GPU.md           — GPU curriculum
gpu/src/kernels/gemm/
  ├── gemm_naive.wgsl        — reference (provided)
  ├── gemm_tiled_*.wgsl      — your stubs
benchmark/gpu_gemm.cpp       — runs shaders, checks vs CPU
```

## How to use this

1. Open `cpu/JOURNEY.md` or `gpu/JOURNEY_GPU.md`
2. Read the problem. Don't look at the explanation yet.
3. Write the kernel (`.cpp` or `.wgsl`)
4. Record your actual measured numbers
5. Read the explanation to see why it worked (or didn't)

That's it. Repeat for each level.

## Why this works

Optimizing GEMM teaches you:
- memory hierarchy (L1/L2 cache, bandwidth, latency)
- instruction-level parallelism (FMA chains, register pressure)
- data movement (coalescing, cache lines, shared memory)
- when to stop optimizing (roofline, arithmetic intensity)

All of these show up in every performance-critical code you'll ever write.

## Target: M1 + Metal

CPU: Apple M1 NEON (32 registers, 2 FMA/cycle, ~2.6 peak GFLOPS)
GPU: Metal via wgpu-native (you'll hit ~85 GFLOPS naive, ~200+ with tiling)

Accelerate and MLX are the ceiling — they use AMX (the matrix coprocessor you can't access from NEON/WGSL). See how close you can get.

## Status

- L1–L3 done and measured
- L4–L14 ready for you to implement
- GPU harness working, naive reference ~85 GFLOPS, stubs ready

Pick L4 on CPU or G3 on GPU and go.

---

Built on [how-to-optimize-gemm](https://github.com/flame/how-to-optimize-gemm).
