# wgpu-blas: The GPU GEMM Journey

You built a fast GEMM on the CPU (`cpu/JOURNEY.md`). Now bring it to the GPU with
WebGPU compute shaders (WGSL), running on Metal through `wgpu-native`. Every CPU
optimization has a direct GPU analogue — you will recognise each step.

The harness is already written for you (`benchmark/gpu_gemm.cpp`), exactly like the
CPU bench/test were. It uploads A and B, runs a WGSL shader, reads C back, and
**checks it against a CPU reference**. You write the shaders.

**Rules** (same as the CPU journey)
- Write the WGSL yourself before reading the explanation.
- Record your real numbers in each "Your Numbers" table.
- A level is done when the harness prints `ok` for your shader and it is faster
  than the previous level.

**How to build and run**
```bash
cmake --build build --target gpu_gemm
# default: runs the provided naive reference
./build/gpu_gemm
# run specific shaders (naive + your tiled kernel) and compare:
./build/gpu_gemm src/kernels/gemm/gemm_naive.wgsl src/kernels/gemm/gemm_tiled_16x16.wgsl
```

The shaders live in `src/kernels/gemm/`. The provided `gemm_naive.wgsl` works and
matches the reference; the others are stubs that write zeros (so they FAIL) until
you implement them.

**A note on the numbers.** The reported seconds include submit + readback, not
just kernel time — fine for comparing shaders and watching the trend, but not an
absolute kernel benchmark. Isolating kernel time with **GPU timestamp queries** is
itself an exercise (G7). And a failing (zero-writing) stub prints `—` for GFLOPS,
because timing a kernel that computes nothing is meaningless.

---

## The CPU → GPU mapping (your roadmap)

| CPU (what you built)         | GPU / WGSL equivalent              | Level |
|------------------------------|------------------------------------|-------|
| naive triple loop (L1)       | one invocation per C element       | G2    |
| loop reorder / coalescing    | thread→element mapping, coalesced  | G3    |
| cache blocking (L5)          | workgroup shared-memory tiling     | G4    |
| register micro-kernel (L9)   | per-invocation register tile       | G5    |
| prefetch / pipelining (L10)  | double-buffered shared memory      | G6    |
| autotuning (L12)             | tune workgroup/tile sizes          | G7    |
| Accelerate / MLX ceiling     | MLX (Metal) matmul ceiling         | G8    |

---
---

# Level G1 — The Compute Harness (provided)

## What you get for free

`benchmark/gpu_gemm.cpp` is a complete WebGPU compute pipeline. Read it once — it
is the GPU analogue of everything you'd otherwise have to build by hand:

- **Device bring-up**: instance → adapter → device → queue.
- **Buffers**: storage buffers for A, B, C; a uniform buffer for `{M,N,K}`; a
  `MapRead` readback buffer.
- **Bind group layout** (binding 0..3): `A` read-only storage, `B` read-only
  storage, `C` storage, `dims` uniform — the contract every shader must match.
- **Pipeline**: shader module ← WGSL source, explicit pipeline layout, compute
  pipeline with entry point `main`.
- **Dispatch**: a 16×16 workgroup grid covering M×N (extra invocations are
  bounds-checked out in the shader).
- **Readback + check**: copies C to the mappable buffer, maps it, compares to a
  CPU reference within tolerance.

## Think Before You Move On

1. WebGPU has no "just call the GPU" path — every kernel needs buffers, a bind
   group, a pipeline, an encoder, a submit, and an async readback. Which of these
   costs are *per-call* and which are *one-time*? (This is why real GPU BLAS
   caches pipelines and reuses buffers.)
2. The readback is **asynchronous** (`wgpuBufferMapAsync` + a callback you poll
   with `wgpuInstanceProcessEvents`). Why can't the CPU just read GPU memory
   directly?
3. The dims buffer is a **uniform**, A/B/C are **storage**. What's the difference,
   and why does C need `read_write` while A/B are `read`?

## The binding contract (every shader must declare exactly this)

```wgsl
struct Dims { M: u32, N: u32, K: u32, _pad: u32, };
@group(0) @binding(0) var<storage, read>       A: array<f32>;
@group(0) @binding(1) var<storage, read>       B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;
@compute @workgroup_size(16, 16) fn main(/* builtins */) { ... }
```

---
---

# Level G2 — Naive WGSL GEMM

## The Problem

Already provided as `gemm_naive.wgsl` and used as the reference — read it, run it,
and **own the model**. One invocation `(row, col)` walks the full K dimension and
writes one element of C. This is the GPU L1.

Then reproduce it yourself in a fresh file to prove you understand the mapping:
`gid.y → row`, `gid.x → col`, bounds-check against `dims.M`/`dims.N`, accumulate
`A[row*K + k] * B[k*N + col]`.

## Think Before You Look

1. `gid.x` is the fastest-varying index across a workgroup. Should it map to the
   **row** or the **column** of C so that neighbouring invocations read
   neighbouring addresses of B? (Memory coalescing — the GPU version of the CPU
   loop-reorder win.)
2. How many global-memory loads does this kernel do per output element? (Same
   arithmetic-intensity problem as CPU L1 — that's why it's slow.)

## Your Numbers

| Size      | naive GFLOPS |
|-----------|--------------|
| 256×256   | ___          |
| 512×512   | ___          |
| 1024×1024 | ___          |

---
&nbsp;

&nbsp;

**STOP. Fill in your numbers before reading.**

&nbsp;

---

## Why It Works (read AFTER you have numbers)

Each invocation streams an entire row of A and column of B from global memory with
zero reuse — arithmetic intensity ≈ 0.25 FLOP/byte, just like the CPU naive
kernel. The GPU has thousands of threads to hide latency, so it is far faster than
the CPU naive loop in absolute terms, but it is still **memory-bound**: most
invocations are stalled on global loads. Every level from here raises reuse.

---
---

# Level G3 — Thread Mapping & Coalescing

## The Problem

Create `gemm_naive.wgsl`'s coalesced cousin. Keep one element per invocation, but
make sure consecutive invocations in a workgroup read **consecutive** global
addresses of B (and write consecutive addresses of C). Experiment with
`@workgroup_size(16,16)` vs `(32,8)` vs `(8,32)` and which builtin index maps to
the column.

## Think Before You Look

1. On a GPU, 32 (or 16) adjacent invocations issue memory requests together. If
   they read addresses that are contiguous, the hardware fuses them into **one**
   wide transaction (coalesced). If they're strided, it issues many. Which
   thread→element mapping gives contiguous B reads?
2. This is the *same* insight as CPU Level 2 (loop reorder for sequential access).
   What's the GPU word for "sequential access across the inner loop"?

## Your Numbers

| Size      | naive | coalesced | workgroup size that won |
|-----------|-------|-----------|-------------------------|
| 1024×1024 | ___   | ___       | ___                     |

---
&nbsp;

&nbsp;

**STOP. Fill in your numbers before reading.**

&nbsp;

---

## Why It Works (read AFTER you have numbers)

Coalescing turns N scattered memory transactions into N/32 wide ones — an up-to-32×
cut in memory traffic *requests*. It costs nothing but choosing the right index
mapping, which is why it's the first thing to get right. The GPU memory system
rewards the same locality the CPU cache does; the mechanism (warp/wavefront
coalescing vs cache lines) differs, the principle is identical.

---
---

# Level G4 — Workgroup Shared-Memory Tiling

## The Problem

Implement `gemm_tiled_16x16.wgsl` (stub provided). Each 16×16 workgroup
cooperatively computes a 16×16 tile of C. Stage a 16×16 tile of A and of B into
`var<workgroup>` shared memory, `workgroupBarrier()`, then each invocation
accumulates over the tile from on-chip memory. Loop over K in 16-wide steps.

This is the GPU analogue of CPU **cache blocking** (Level 5): shared memory is the
GPU's software-managed L1.

## Think Before You Look

1. Without tiling, each C element re-reads its row of A and column of B from
   global memory. With a 16×16 tile, how many times is each loaded A/B value
   **reused** from shared memory before eviction? (That reuse factor is the
   speedup.)
2. Why are the two `workgroupBarrier()` calls (after load, after compute)
   mandatory? What race happens if you drop either?
3. Shared memory is tiny (tens of KB per workgroup) and limits how many workgroups
   run concurrently (occupancy). How does tile size trade off against occupancy?

## Your Numbers

| Size      | coalesced | 16×16 tiled | Speedup |
|-----------|-----------|-------------|---------|
| 512×512   | ___       | ___         | ___x    |
| 1024×1024 | ___       | ___         | ___x    |
| 2048×2048 | ___       | ___         | ___x    |

---
&nbsp;

&nbsp;

**STOP. Fill in your numbers before reading.**

&nbsp;

---

## Why It Works (read AFTER you have numbers)

A 16×16 tile loads 16×16 values of A and B once into shared memory, then reuses
each 16 times across the tile's outputs — arithmetic intensity jumps ~16×, moving
the kernel off the global-memory bandwidth roof. Shared memory has roughly an
order of magnitude more bandwidth and lower latency than global memory, so the
inner accumulation runs at compute speed instead of memory speed. This is the
single biggest GPU GEMM win, exactly as cache blocking was on the CPU.

---
---

# Level G5 — Register Tiling (Thread Coarsening)

## The Problem

Implement `gemm_tiled_32x8.wgsl` (stub provided). Each invocation computes a small
**register tile** of C — e.g. 1×4, 4×4, or 8×8 — instead of one element, keeping
the partial sums in registers (local `var`s). Combine with the G4 shared staging
so a workgroup covers a larger C tile.

This is the GPU analogue of the CPU **register-blocked micro-kernel** (Levels
9–11). More output per thread amortises the shared-memory loads.

## Think Before You Look

1. With a 4×4 register tile, one thread loads 4 A values and 4 B values from shared
   memory and does 16 FMAs. Compare that load:FLOP ratio to the 1×1 case. Why does
   coarsening raise throughput?
2. Registers are finite per thread; a big tile reduces how many threads fit
   (occupancy). This is the *exact same* register-pressure trade-off as the CPU
   8×8 vs 12×12 micro-kernel. What's the GPU sweet spot on your hardware?
3. Fewer, fatter threads vs many, thin threads — which hides memory latency
   better, and why might the answer be "measure it"?

## Your Numbers

| Per-thread tile | 1024×1024 GFLOPS | 2048×2048 GFLOPS |
|-----------------|------------------|------------------|
| 1×1 (G4)        | ___              | ___              |
| 1×4             | ___              | ___              |
| 4×4             | ___              | ___              |
| 8×8             | ___              | ___              |

---
&nbsp;

&nbsp;

**STOP. Fill in your numbers before reading.**

&nbsp;

---

## Why It Works (read AFTER you have numbers)

Shared memory is fast but not free — every FMA that reads two shared values still
pays a shared-load. A register tile reads each shared value **once** and reuses it
across the whole micro-tile from registers (which are free), so the shared-memory
traffic per FLOP drops by the tile area. This is the GPU's version of "compute a
small tile of C while reusing loaded operands" — the same core idea as the CPU
micro-kernel, and the step that pushes a GPU GEMM from memory-bound to
compute-bound.

---
---

# Level G6 — Double Buffering (Latency Hiding)

## The Problem

Implement `gemm_shared_doublebuf.wgsl` (stub provided). Use **two** shared-memory
tiles per operand. While the math units consume tile `t`, prefetch tile `t+1` into
the other buffer from global memory, then swap. The barrier no longer forces the
loads and the compute to be serial.

This is the GPU analogue of CPU **software pipelining / prefetch** (Level 10).

## Think Before You Look

1. In G4, each K-step is `load → barrier → compute → barrier`: the loads and the
   compute never overlap. With double buffering, which two phases now run at the
   same time?
2. Double buffering doubles shared-memory use, lowering occupancy. When is that
   trade worth it — and when does the single-buffered kernel already hide latency
   via having enough resident workgroups?

## Your Numbers

| Size      | G5 single-buf | G6 double-buf | Speedup |
|-----------|---------------|---------------|---------|
| 1024×1024 | ___           | ___           | ___x    |
| 2048×2048 | ___           | ___           | ___x    |

---
&nbsp;

&nbsp;

**STOP. Fill in your numbers before reading.**

&nbsp;

---

## Why It Works (read AFTER you have numbers)

The single-buffered kernel stalls every K-step waiting for the next tile's global
loads to land. Double buffering issues those loads *ahead of time* into a spare
buffer, so global-memory latency is hidden behind the current tile's arithmetic —
the same overlap-loads-with-compute idea as CPU prefetch, expressed through
explicit shared-memory ping-pong. The win is largest when the kernel is latency-
bound and occupancy alone wasn't enough to hide it.

---
---

# Level G7 — Autotuning & Timestamp Queries

## The Problem

Two parts:

1. **Honest timing.** Add **GPU timestamp queries** around the compute pass so you
   measure *kernel* time, not submit+readback. (Create a `QuerySet`, write
   timestamps at pass begin/end, resolve to a buffer, read it back.)
2. **Autotune.** Sweep tile sizes, workgroup shapes, and per-thread register tiles
   for your G5/G6 kernel; time each; keep the winner. Persist it like the CPU
   Level 12 tuner.

This mirrors CPU **autotuning** (Level 12) — the optimum is empirical and
hardware-specific.

## Think Before You Look

1. Why is wall-clock around submit+map a poor kernel benchmark? What does a
   timestamp query measure that it doesn't?
2. The GPU search space (workgroup size × tile × buffering) is large. How do you
   bound the search — coarse grid, a shortlist of known-good configs, coordinate
   descent?
3. Why does the best config differ between the M1's GPU and, say, a discrete GPU —
   even for the same matrix size?

## Your Numbers

| Config (wg × tile × buf) | Kernel-time GFLOPS (1024³) |
|--------------------------|----------------------------|
| best from G4             | ___                        |
| best from G5             | ___                        |
| best from G6             | ___                        |
| autotuned winner         | ___                        |

---
&nbsp;

&nbsp;

**STOP. Fill in your numbers before reading.**

&nbsp;

---

## Why It Works (read AFTER you have numbers)

Just like on the CPU, there is no closed-form best tile size: it depends on shared-
memory capacity, register file size, scheduler behaviour, and memory-system
geometry. Timestamp queries give you the clean signal to tune against; the sweep
finds the empirical optimum; caching makes it reusable. Every production GPU GEMM
(cuBLAS, MPS, MLX) ships a tuned kernel database for exactly this reason.

---
---

# Level G8 — The MLX / Metal Ceiling

## The Problem

No new shader — find your ceiling. Benchmark Apple's **MLX** `matmul` (Metal
backend, and on Apple silicon it taps the AMX/Matrix units) on the same sizes, and
compare your best kernel against it. MLX is the GPU-side analogue of Accelerate on
the CPU: the "what the machine can really do" number.

If MLX is installed (`pip install mlx` for Python, or build the C++ lib), time a
warm `mx.matmul(A, B)` with `mx.eval`, in GFLOPS, for 1024/2048/4096. Otherwise
record Apple's published Metal GEMM throughput for your chip as the reference.

## Think Before You Look

1. MLX/MPS reach a large fraction of the GPU's (and AMX's) peak. Realistically,
   what fraction can a hand-written WGSL kernel hit — 30%? 60%? Why is the gap
   wider on GPU than the ~50% you reached vs Accelerate on CPU?
2. WGSL runs through `wgpu-native` → Metal, a portability layer. What does that
   abstraction cost versus a native Metal kernel that MLX can emit?
3. You've now built block → shared-tile → register-tile → double-buffer by hand —
   the same skeleton MLX's kernel uses. What's left that you *can't* express in
   portable WGSL? (simdgroup matrix ops, AMX, vendor intrinsics.)

## Your Numbers

| Size      | your best WGSL GFLOPS | MLX/Metal GFLOPS | % of ceiling |
|-----------|-----------------------|------------------|--------------|
| 1024×1024 | ___                   | ___              | ___%         |
| 2048×2048 | ___                   | ___              | ___%         |
| 4096×4096 | ___                   | ___              | ___%         |

---
&nbsp;

&nbsp;

**STOP. Fill in your numbers before reading.**

&nbsp;

---

## Why It Works (read AFTER you have numbers)

MLX and MPS use Metal `simdgroup_matrix` instructions (and on Apple silicon, the
AMX coprocessor) that a portable WGSL shader simply cannot emit — the same way
Accelerate's AMX put a ceiling on your NEON kernel. Reaching a healthy fraction of
that ceiling with portable, vendor-neutral WGSL is the real achievement: you now
understand every layer of a GPU GEMM, why the ceiling sits where it does, and
exactly which hardware features close the last gap.

You have now built, by hand, the full GEMM stack on **both** sides:

```
CPU:  OpenMP → cache blocking → packing → NEON micro-kernel → autotune
GPU:  dispatch grid → shared tiling → register tiling → double buffer → autotune
```

Same algorithm, two machines, one mental model.
