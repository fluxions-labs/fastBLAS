# cpu-blas: The GEMM Journey

You are building a BLAS-style single-precision GEMM from scratch, level by level.
Each level is a problem. Read the problem. Close this file. Write the code. Run the numbers.
Only open the "Why it works" section after your numbers are on paper.

The goal is to go from ~2 GFLOPS (what a student writes) to ~100+ GFLOPS (what a serious
library achieves), understanding every step in between.

**Rules**
- Write the code yourself before reading the explanation.
- Record your actual measured numbers in the "Your Numbers" table.
- Do not move to the next level until the test passes and you have real numbers.

**How to build and run**
```bash
cd cmake-build-debug
ninja bench_gemm test_gemm
./cpu/test_gemm
./cpu/bench_gemm
```

---

# Level 1 — Naive Triple Loop

## The Problem

Implement `sgemm_naive` in `src/level3/gemm_naive.cpp`.

```
C = alpha * A * B + beta * C
A : M×K row-major
B : K×N row-major
C : M×N row-major
```

Signature (already declared in `include/blas.h`):
```cpp
void sgemm_naive(int M, int N, int K,
                 float alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float beta,
                 float* C, int ldc);
```

Use three nested loops: `i` over M, `j` over N, `k` over K.
Accumulate `A[i,k] * B[k,j]` into a local sum, then write `alpha*sum + beta*C[i,j]` to C.

**Files to touch**
- `src/level3/gemm_naive.cpp` — write the implementation
- Already wired into `CMakeLists.txt`, `bench_gemm.cpp`, `test_gemm.cpp`

**Done when**
- `test_gemm` prints ALL TESTS PASSED
- You have GFLOPS numbers for 256, 512, 1024

## Think Before You Look

Before reading the explanation below, answer these in your head:

1. For a 1024×1024×1024 matrix, how many multiply-add operations does your kernel do?
2. In the innermost loop (`k` loop), what is the memory access pattern for A? For B? Are they sequential or strided?
3. Your M1 Air can do roughly 2,600 GFLOPS of single-precision math. You are getting ~2 GFLOPS. What is happening with the other 99.9%?

## Your Numbers

| Size      | GFLOPS | Time (s) |
|-----------|--------|----------|
| 256×256   | 2.099  | 0.016    |
| 512×512   | 1.805  | 0.149    |
| 1024×1024 | 1.612  | 1.334    |

---
&nbsp;

&nbsp;

&nbsp;

**STOP. Fill in your numbers above before reading.**

&nbsp;

&nbsp;

&nbsp;

---

## Why It Works (read AFTER you have numbers)

The formula is `2*M*N*K` FLOPs — one multiply and one add per `(i,j,k)` triple.

The innermost loop over `k` accesses:
- `A[i * lda + k]` — `k` increments, so this is **sequential** (good)
- `B[k * ldb + j]` — `k` increments by `ldb` (= N) floats between elements, so this is **strided** (bad)

Every access to B in the innermost loop jumps N×4 bytes forward — a cache miss every time for
large matrices. This is why GFLOPS drop as the matrix grows: you hit the memory wall.

**Expected baseline:** 1–3 GFLOPS on M1 Air (depending on matrix size).

---
---

# Level 2 — Loop Reorder

## The Problem

Implement `sgemm_reorder` in `src/level3/gemm_reorder.cpp`.

Same computation as Level 1. The only difference is the loop order.

**Files to touch**
- Create `src/level3/gemm_reorder.cpp`
- Add declaration to `include/blas.h`
- Add source file to `cpu/CMakeLists.txt` under `add_library(cpu_blas ...)`
- Uncomment the Level 2 row in `benchmark/bench_gemm.cpp`
- Uncomment the Level 2 block in `test/test_gemm.cpp`

**Done when**
- `test_gemm` still passes
- You have GFLOPS numbers and they are faster than Level 1

## Think Before You Look

Before writing the code, think:

1. There are 6 possible orderings of 3 loops: i-j-k, i-k-j, j-i-k, j-k-i, k-i-j, k-j-i.
   Without running anything, which one do you think is fastest for row-major matrices, and why?
2. For a row-major matrix, element `[row][col]` is at address `base + row*lda + col`.
   Which direction (incrementing row vs incrementing col) is sequential in memory?
3. In the Level 1 inner loop `for k: C[i,j] += A[i,k] * B[k,j]` — which of A, B, C is accessed
   with a stride > 1? What if you made `k` the middle loop and `j` the inner loop instead?

## Your Numbers

| Size      | Naive GFLOPS | Reorder GFLOPS | Speedup |
|-----------|--------------|----------------|---------|
| 256×256   | ___          | ___            | ___x    |
| 512×512   | ___          | ___            | ___x    |
| 1024×1024 | ___          | ___            | ___x    |

---
&nbsp;

&nbsp;

&nbsp;

**STOP. Fill in your numbers above before reading.**

&nbsp;

&nbsp;

&nbsp;

---

## Why It Works (read AFTER you have numbers)

The winning order is **i → k → j**.

With `k` in the middle and `j` innermost, the inner loop body becomes:
```cpp
C[i*ldc + j] += A[i*lda + k] * B[k*ldb + j];
```
Now:
- `A[i*lda + k]` — `k` changes (middle loop), `i` fixed. A is loaded once per middle iteration, reused N times across the inner j loop.
- `B[k*ldb + j]` — `j` changes (innermost). Sequential. Cache-friendly.
- `C[i*ldc + j]` — `j` changes. Sequential. Cache-friendly.

Three sequential streams in the inner loop instead of one sequential + one strided.
The hardware prefetcher can now predict and prefetch all three.

**Expected speedup:** 3–6× over naive for large matrices.

---
---

# Level 3 — Register Accumulation

## The Problem

Implement `sgemm_reg` in `src/level3/gemm_reg.cpp`.

Same loop order as Level 2 (i → k → j). Add one change: instead of writing back to
`C[i*ldc + j]` on every inner iteration, accumulate into a local `float` variable and
write C once per (i,j) pair.

**Files to touch**
- Create `src/level3/gemm_reg.cpp`
- Add declaration to `include/blas.h`
- Add to `CMakeLists.txt`
- Uncomment Level 3 rows in bench and test

**Done when**
- Tests pass
- GFLOPS >= Level 2

## Think Before You Look

1. In the i→k→j loop, how many times is `C[i*ldc + j]` read and written for a single
   (i,j) output element across the full k loop?
2. `C` lives in memory (DRAM or cache). A local `float sum` lives in a CPU register.
   How much faster is a register read vs a cache read? (Think: 0 cycles vs 4 cycles)
3. Why can the compiler sometimes do this optimisation automatically, and sometimes not?
   (Hint: think about aliasing — what if A, B, C pointers overlap?)

## Your Numbers

| Size      | Reorder GFLOPS | Register GFLOPS | Speedup |
|-----------|----------------|-----------------|---------|
| 256×256   | ___            | ___             | ___x    |
| 512×512   | ___            | ___             | ___x    |
| 1024×1024 | ___            | ___             | ___x    |

---
&nbsp;

&nbsp;

&nbsp;

**STOP. Fill in your numbers above before reading.**

&nbsp;

&nbsp;

&nbsp;

---

## Why It Works (read AFTER you have numbers)

Without the local variable, the compiler cannot assume that `C[i*ldc+j]` doesn't alias
with A or B (they are all raw pointers). So it must reload C from memory on every iteration
of the k loop — even though the address never changes within that loop.

By accumulating into `float sum`, you tell the compiler: this value lives in a register.
No aliasing possible. The compiler can now keep `sum` in a register for the entire k loop,
doing K multiply-adds before a single memory write.

This also enables the compiler to pipeline the multiply-add operations because the dependency
chain (sum += ...) is now explicit and register-local.

**Expected speedup:** 1.2–1.5× over Level 2. Modest, but it sets up the foundation for
the micro-kernel in the next levels.

---
---

# Level 4 — 1×4 Micro-kernel

## The Problem

Implement `sgemm_1x4` in `src/level3/gemm_1x4.cpp`.

Instead of computing one output element per inner loop, compute **4 output elements at once**
in a single pass over k. Specifically, for each row `i` of C, compute C[i,j], C[i,j+1],
C[i,j+2], C[i,j+3] simultaneously in the same k loop.

Use 4 local float accumulators: `c0, c1, c2, c3`.

Handle the tail case (when N is not divisible by 4) with a fallback.

**Files to touch**
- Create `src/level3/gemm_1x4.cpp`
- Add declaration to `include/blas.h`
- Add to `CMakeLists.txt`
- Uncomment Level 4 rows in bench and test

**Done when**
- Tests pass for non-square sizes too (e.g. 128×256×64)
- GFLOPS > Level 3

## Think Before You Look

1. In a 1×4 micro-kernel, what single value of A is shared across all 4 output columns?
   How many times would you load it from memory in a naive approach vs hoisting it?
2. You now have 4 accumulators (`c0..c3`) and 4 B-column pointers. How many registers is
   the compiler using for just the accumulators? An M1 has 32 float registers. Is this tight?
3. Sketch the memory access pattern: in one k-loop iteration, how many loads from A?
   How many from B? Are the B loads sequential?

## Your Numbers

| Size      | L3 GFLOPS | 1×4 GFLOPS | Speedup |
|-----------|-----------|------------|---------|
| 256×256   | ___       | ___        | ___x    |
| 512×512   | ___       | ___        | ___x    |
| 1024×1024 | ___       | ___        | ___x    |

---
&nbsp;

&nbsp;

&nbsp;

**STOP. Fill in your numbers above before reading.**

&nbsp;

&nbsp;

&nbsp;

---

## Why It Works (read AFTER you have numbers)

The key insight: `A[i*lda + k]` is the **same value** for all 4 output columns. In Level 3
you loaded it once per (i,j,k) — meaning 4 loads for 4 j-values. Here you load it once and
reuse it across all 4 accumulators.

```
a_ik = A[i*lda + k]          // 1 load
c0  += a_ik * B[k*ldb + j+0] // 1 load + 1 fmadd
c1  += a_ik * B[k*ldb + j+1] // 1 load + 1 fmadd
c2  += a_ik * B[k*ldb + j+2] // 1 load + 1 fmadd
c3  += a_ik * B[k*ldb + j+3] // 1 load + 1 fmadd
```

4 FLOPs, 5 loads (instead of 8 loads in a naive approach). Better compute-to-memory ratio.

The 4 B loads are also sequential, so the prefetcher handles them well.

This pattern — compute a small tile of C while reusing a loaded value of A — is the core
idea of every high-performance GEMM kernel. Levels 5 and 6 scale this idea up.

**Expected speedup:** 1.5–2.5× over Level 3.

---
---

# Level 5 — Cache Blocking (Tiling)

## The Problem

Implement `sgemm_block` in `src/level3/gemm_block.cpp`.

Add three outer blocking loops around your Level 4 micro-kernel. The idea: instead of
iterating over the full M, N, K ranges at once, process the computation in tiles of size
`MC × KC × NC` so that the active data fits in L1/L2 cache.

Define block sizes at the top of the file:
```cpp
constexpr int MC = 64;
constexpr int KC = 64;
constexpr int NC = 64;
```

Start with these. You will tune them later.

The loop structure (outermost to innermost):
```
for jc in 0..N step NC       // NC-wide column panel of C and B
  for kc in 0..K step KC     // KC-wide panel: a block of A columns, B rows
    for ic in 0..M step MC   // MC-wide row panel of C and A
      // inner micro-kernel: compute MC×NC tile of C
      // using Level 4 (or Level 3) kernel on the MC×KC × KC×NC sub-problem
```

**Files to touch**
- Create `src/level3/gemm_block.cpp`
- Add declaration to `include/blas.h`
- Add to `CMakeLists.txt`
- Uncomment Level 5 rows in bench and test

**Done when**
- Tests pass (including non-square sizes)
- Measurably faster than Level 4 for 1024×1024

## Think Before You Look

This is the hardest level so far. Spend real time on it.

1. Your M1 L1 data cache is 64 KB. If `MC=KC=NC=64` and you are working with floats (4 bytes),
   how many bytes are in an MC×KC tile of A? A KC×NC tile of B? An MC×NC tile of C?
   Do they fit in 64 KB together?
2. Without blocking: when computing C[i,j] for large K, you scan an entire row of A (K floats)
   and an entire column of B (K floats). If K=1024, that is 8 KB each. Where does that
   data live after the first pass? Does it still live there on the second pass (different j)?
3. With blocking: within the innermost `ic` loop, you are working on a fixed MC×KC tile of A
   and KC×NC tile of B repeatedly for all MC×NC outputs. Why does this help?
4. What happens if your block sizes are too small? Too large?

## Your Numbers

| Size      | L4 GFLOPS | Blocked GFLOPS | Speedup |
|-----------|-----------|----------------|---------|
| 256×256   | ___       | ___            | ___x    |
| 512×512   | ___       | ___            | ___x    |
| 1024×1024 | ___       | ___            | ___x    |

After you have numbers, try changing `MC`, `KC`, `NC` to different values (32, 128, 256).
Record what happens:

| MC  | KC  | NC  | 1024×1024 GFLOPS |
|-----|-----|-----|------------------|
| 64  | 64  | 64  | ___              |
| 128 | 64  | 128 | ___              |
| 32  | 32  | 32  | ___              |
| 256 | 64  | 256 | ___              |

---
&nbsp;

&nbsp;

&nbsp;

**STOP. Fill in your numbers above before reading.**

&nbsp;

&nbsp;

&nbsp;

---

## Why It Works (read AFTER you have numbers)

Without blocking, the working set for a 1024×1024 GEMM is 3 × 1024² × 4 bytes = 12 MB.
The M1 L1 cache is 64 KB. L2 is 4 MB. The data does not fit — you are constantly evicting
and reloading.

With blocking (MC=KC=NC=64):
- Active tile of A: 64×64×4 = 16 KB
- Active tile of B: 64×64×4 = 16 KB
- Active tile of C: 64×64×4 = 16 KB
- Total: 48 KB — fits in L1.

Within the innermost `ic` loop, A's MC×KC tile and B's KC×NC tile stay resident in L1.
You do MC×NC×KC = 64³ = 262,144 FLOPs on 48 KB of data.
Arithmetic intensity = 262144 FLOPs / 48000 bytes ≈ 5.5 FLOP/byte.
M1's L1 bandwidth is ~500 GB/s. 5.5 FLOP/byte × 500 GB/s ≈ 2.75 TFLOPS theoretical.
You will not hit that (micro-kernel is still scalar) but this removes the memory wall.

Choosing block sizes:
- Too small: too many cache misses at the boundary, low FLOP density per tile.
- Too large: tiles spill out of the target cache level, evicting each other.
- Sweet spot: tiles together fill but do not exceed the target cache (L1 or L2).

This is the Goto algorithm. Everything from here is making the inner kernel faster.

**Expected speedup:** 3–8× over Level 4 for large matrices.

---
---

# Level 6 — Packing

## The Problem

Implement `sgemm_packed` in `src/level3/gemm_packed.cpp`.

Before calling the micro-kernel, **copy the active tile of A and B into contiguous
temporary buffers** (`packed_A` and `packed_B`). Allocate these buffers once at the
start of the function using `std::vector<float>`.

The pack step for A (MC×KC tile starting at row `ic`, column `kc`):
```
for each row ir in 0..mc:
    for each col kr in 0..kc_size:
        packed_A[ir * kc_size + kr] = A[(ic+ir)*lda + (kc+kr)]
```

Then pass `packed_A` and `packed_B` to the micro-kernel with `lda = KC`, `ldb = NC`.

**Files to touch**
- Create `src/level3/gemm_packed.cpp`
- Add declaration to `include/blas.h`
- Add to `CMakeLists.txt`
- Uncomment Level 6 rows in bench and test

**Done when**
- Tests pass
- Numbers are better than or equal to Level 5

## Think Before You Look

1. Even after blocking, the A tile in memory looks like this (MC=4, KC=4, lda=1024):
   ```
   [a00][a01][a02][a03][...992 other floats...][a10][a11]...
   ```
   The 992 floats between rows are gaps. What happens to cache lines that contain those gaps?
2. After packing, `packed_A` looks like `[a00][a01][a02][a03][a10][a11][a12][a13]...`
   Fully contiguous. How does this affect cache line utilisation?
3. Packing costs time (you are copying data). When does that cost pay off?
4. OpenBLAS packs B in column-major order and A in row-major order.
   Why might the layout of the packed buffer matter for the micro-kernel?

## Your Numbers

| Size      | Blocked GFLOPS | Packed GFLOPS | Speedup |
|-----------|----------------|---------------|---------|
| 256×256   | ___            | ___           | ___x    |
| 512×512   | ___            | ___           | ___x    |
| 1024×1024 | ___            | ___           | ___x    |

---
&nbsp;

&nbsp;

&nbsp;

**STOP. Fill in your numbers above before reading.**

&nbsp;

&nbsp;

&nbsp;

---

## Why It Works (read AFTER you have numbers)

A cache line is 64 bytes = 16 floats. When the A tile has `lda=1024`, each row of the tile
is followed by 1024-KC floats that belong to a different tile. Those floats load into cache
but are never used — wasted bandwidth and wasted cache space.

Packing copies the tile into a contiguous buffer where every loaded cache line is 100% useful.

For B, packing also enables SIMD: the micro-kernel can load 4 or 8 consecutive floats from
`packed_B` knowing they are the next KC values of a single column — no stride arithmetic.

The packing cost (O(MC×KC + KC×NC) copies) is amortised over O(MC×KC×NC) FLOPs inside
the micro-kernel. For large enough tiles the ratio is favourable.

After Level 6 your kernel has the same macro-structure as OpenBLAS and BLIS:
block → pack → micro-kernel. The remaining gains come from vectorising the micro-kernel.

**Expected speedup:** 1.2–2× over Level 5, more pronounced at large sizes.

---
---

# Level 7 — NEON SIMD Micro-kernel

## The Problem

Implement `sgemm_neon` in `src/level3/gemm_neon.cpp`.

Replace the scalar inner loop with ARM NEON intrinsics. The micro-kernel computes a
**4×4 tile of C** per call, using NEON 128-bit registers that hold 4 floats each.

Include header:
```cpp
#include <arm_neon.h>
```

Key intrinsics you will need:
```cpp
float32x4_t vld1q_f32(const float*);   // load 4 floats
void         vst1q_f32(float*, float32x4_t); // store 4 floats
float32x4_t vdupq_n_f32(float);        // broadcast 1 float to all 4 lanes
float32x4_t vfmaq_f32(acc, a, b);      // acc = acc + a * b  (fused multiply-add)
float32x4_t vaddq_f32(a, b);           // a + b elementwise
```

The 4×4 micro-kernel structure (pseudo-code):
```
declare 4 float32x4_t accumulators: c_col0, c_col1, c_col2, c_col3 (all zeroed)
for p in 0..K:
    load a_row = 4 consecutive floats from packed_A (one row of the 4×KC tile)
    broadcast b0 = packed_B[p*4 + 0]  // scalar B element → all 4 lanes
    broadcast b1 = packed_B[p*4 + 1]
    broadcast b2 = packed_B[p*4 + 2]
    broadcast b3 = packed_B[p*4 + 3]
    c_col0 = vfmaq_f32(c_col0, a_row, b0)
    c_col1 = vfmaq_f32(c_col1, a_row, b1)
    c_col2 = vfmaq_f32(c_col2, a_row, b2)
    c_col3 = vfmaq_f32(c_col3, a_row, b3)
store c_col0..c_col3 back to C (with alpha/beta scaling)
```

Use this inside the blocking + packing structure from Level 6.

**Files to touch**
- Create `src/level3/gemm_neon.cpp`
- Add to `blas.h`, `CMakeLists.txt`, bench, test
- In `CMakeLists.txt`, add a separate `target_compile_options` for this file:
  `-O3 -march=armv8.2-a+fp16+dotprod` (enables NEON with dot-product extension on M1)

**Done when**
- Tests pass
- Numbers are significantly higher than Level 6

## Think Before You Look

1. A NEON `float32x4_t` register holds 4 floats and does 4 multiply-adds in one instruction.
   Theoretically, how many times faster should this be than the scalar Level 6 kernel?
2. In the micro-kernel above, how many NEON registers are in use at peak?
   (4 accumulators + 1 a_row + 4 broadcasts = 9 minimum.) M1 has 32 NEON registers.
   What would happen if you expanded to an 8×4 or 4×8 micro-kernel?
3. `vfmaq_f32` is a fused multiply-add: it computes `acc + a*b` in one instruction with
   no intermediate rounding. Why does this matter for both performance and accuracy?
4. If you did NOT pack A and B, could you still use `vld1q_f32` efficiently? Why or why not?

## Your Numbers

| Size      | Packed GFLOPS | NEON GFLOPS | Speedup |
|-----------|---------------|-------------|---------|
| 256×256   | ___           | ___         | ___x    |
| 512×512   | ___           | ___         | ___x    |
| 1024×1024 | ___           | ___         | ___x    |

**Bonus:** Run Apple's Accelerate BLAS for comparison:
```cpp
#include <Accelerate/Accelerate.h>
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
            1.0f, A, K, B, N, 0.0f, C, N);
```
Add it as a kernel row in bench_gemm. How close are you?

---
&nbsp;

&nbsp;

&nbsp;

**STOP. Fill in your numbers above before reading.**

&nbsp;

&nbsp;

&nbsp;

---

## Why It Works (read AFTER you have numbers)

Single M1 efficiency core: 2 NEON FP32 FMA units, each 128-bit (4 floats wide).
Peak per-core single-precision: 2 units × 4 floats × 2 (mul+add) × 3.2 GHz ≈ 51 GFLOPS.

Your 4×4 micro-kernel issues 4 `vfmaq_f32` instructions per k-iteration, each doing
4 FMAs = 16 FLOPs. That is 4 FP instructions per cycle (if the pipeline is full).

Why you will not hit 51 GFLOPS yet:
- Memory latency from packing buffers (even if they are in L1, loads have latency)
- The 4×4 tile is too small to hide load latency — accumulators are written before
  the next iteration's loads are ready
- No software pipelining / prefetch

To get closer to peak, real libraries use larger tiles (8×12 on modern ARM) and
arrange the micro-kernel so multiple in-flight FMA chains overlap. That is beyond
this project's scope — but you now understand why it matters.

**Expected:** 20–60 GFLOPS on M1 Air (depending on tile size and compiler quality).
Apple Accelerate hits ~120–180 GFLOPS on the same hardware using AMX (Apple Matrix
coprocessor, not accessible from user code). You are doing this with public NEON
intrinsics — 50% of Accelerate is a respectable result.

---
---

# Level 8 — OpenMP Multi-threading

## The Problem

Implement `sgemm_omp` in `src/level3/gemm_omp.cpp`.

Add `#pragma omp parallel for` to the outermost blocking loop (the `jc` or `ic` loop)
in your Level 6 or Level 7 implementation.

In `CMakeLists.txt`, find the `cpu_blas` library target and add:
```cmake
find_package(OpenMP REQUIRED)
target_link_libraries(cpu_blas PUBLIC OpenMP::OpenMP_CXX)
```

Run with: `OMP_NUM_THREADS=4 ./cpu/bench_gemm`

**Done when**
- Tests still pass (run single-threaded)
- GFLOPS scale with thread count for large matrices

## Think Before You Look

1. The M1 Air has 4 efficiency cores + 4 performance cores (8 total). If you parallelise
   perfectly, what is the theoretical max speedup over your single-threaded Level 7?
2. For small matrices (256×256), will multi-threading help or hurt? Why?
3. Multiple threads will each call the inner micro-kernel on different tiles of C.
   Do threads ever write to the same element of C? (Think carefully about which loop
   you parallelise and which tile of C each thread owns.)
4. Each thread needs its own `packed_A` and `packed_B` buffers. What happens if they
   share one buffer?

## Your Numbers

| Size      | L7 GFLOPS (1 thread) | OMP GFLOPS (4t) | OMP GFLOPS (8t) |
|-----------|----------------------|-----------------|-----------------|
| 256×256   | ___                  | ___             | ___             |
| 512×512   | ___                  | ___             | ___             |
| 1024×1024 | ___                  | ___             | ___             |

---
&nbsp;

&nbsp;

&nbsp;

**STOP. Fill in your numbers above before reading.**

&nbsp;

&nbsp;

&nbsp;

---

## Why It Works (read AFTER you have numbers)

GEMM is embarrassingly parallel at the tile level: each output tile `C[ic:ic+MC, jc:jc+NC]`
is independent. Parallelising the outer loop gives each thread a disjoint set of output
tiles with no write conflicts.

Each thread needs its own packing buffers — if they shared one, thread A's pack step
would overwrite thread B's buffer mid-kernel. Use `#pragma omp parallel` with a local
`std::vector<float>` declared inside the parallel region, not outside it.

Why small matrices don't scale:
- Thread launch/sync overhead is ~microseconds.
- A 256×256 GEMM at 30 GFLOPS takes ~0.5 ms.
- Spawning 8 threads adds ~0.1 ms overhead = 20% penalty.
- At 2048×2048 the GEMM takes 200 ms. 0.1 ms overhead = 0.05%. Scales fine.

After Level 8 you have built, by hand, all the layers of a production BLAS:
```
OpenMP (thread parallelism)
  └─ Cache blocking (MC/KC/NC tiles)
       └─ Packing (contiguous A and B panels)
            └─ NEON micro-kernel (SIMD 4×4 tile)
```
This is the exact structure of OpenBLAS, BLIS, and Eigen's GEMM.

---
---

# Advanced Track (Levels 9–14)

Levels 1–8 took you from a naive loop to a blocked, packed, vectorised, threaded
kernel — the full skeleton of a production BLAS. The advanced track is about
**closing the gap to the machine's peak** and **understanding why a real library
is shaped the way it is**. Same rules: write the kernel yourself, record real
numbers, read the explanation last.

A note on measuring from here on: the wins get smaller and noisier. Pin the
problem size large (≥1024), take the *best of many* runs, and watch GFLOPS — not
seconds. A 3% regression is real now; don't chase it past the noise floor.

---
---

# Level 9 — Wide Register-Blocked Micro-kernel

## The Problem

Implement `sgemm_micro` in `src/level3/gemm_micro.cpp`.

The Level 7 NEON kernel computed a **4×4** tile of C. That tile is too small: with
only 4 accumulator registers, the FMA units stall waiting for the next iteration's
loads. Widen the micro-kernel to an **8×8** tile (or **8×12**, BLIS's choice for
ARM) so that many independent FMA chains are in flight at once.

For an 8×8 tile you hold **16** `float32x4_t` accumulators (8 rows × 2 vectors of 4
columns). Each k-iteration:
- load 8 scalars from the packed A panel (one column of the 8×KC tile),
- load 2 vectors (8 floats) from the packed B panel (one row of the KC×8 tile),
- issue 16 `vfmaq_lane_f32` / `vfmaq_n_f32` FMAs.

Build this on top of the Level 5/6 blocking + packing structure. Use `MR=8`,
`NR=8` (or `NR=12`) as the micro-tile, and handle edge tiles (M, N not multiples
of MR/NR) with a scalar fallback.

**Files to touch**
- Create `src/level3/gemm_micro.cpp`
- Uncomment the Level 9 rows in bench and test, and the matching `CMakeLists.txt` line
- Compile this file with `-march=armv8.2-a+fp16+dotprod` (see CMake notes)

**Done when**
- Tests pass for non-square and edge sizes (e.g. 130×255×97)
- GFLOPS > Level 7 for 1024×1024

## Think Before You Look

1. The M1 firestorm core has **32** NEON registers. An 8×8 tile uses 16 for
   accumulators. How many are left for A/B operands? Is 8×12 (24 accumulators)
   still safe? What about 12×12?
2. Why does a *bigger* tile hide load latency? (Hint: how many independent FMA
   chains must be in flight to cover an ~4-cycle FMA latency on 2 FMA ports?)
3. The arithmetic intensity of an MR×NR micro-kernel is roughly
   `(2·MR·NR) / (4·(MR+NR))` FLOP/byte at the register level. Compute it for 4×4
   vs 8×8 vs 8×12. Why does bigger win?
4. `vfmaq_lane_f32` lets you multiply a vector by one *lane* of another vector
   without a separate broadcast. How does that cut the instruction count vs the
   Level 7 broadcast approach?

## Your Numbers

| Size      | L7 GFLOPS | 8×8 GFLOPS | 8×12 GFLOPS | Speedup |
|-----------|-----------|------------|-------------|---------|
| 512×512   | ___       | ___        | ___         | ___x    |
| 1024×1024 | ___       | ___        | ___         | ___x    |
| 2048×2048 | ___       | ___        | ___         | ___x    |

---
&nbsp;

&nbsp;

**STOP. Fill in your numbers above before reading.**

&nbsp;

&nbsp;

---

## Why It Works (read AFTER you have numbers)

A single FMA has latency (~4 cycles on Apple cores) but the core can *issue* 2
per cycle. To keep both FMA ports busy you need ≥ `2 ports × 4 cycles = 8`
independent FMA chains in flight. A 4×4 tile gives you 4 accumulators — only 4
chains — so the ports idle half the time. An 8×8 tile gives 16 chains: enough to
fully cover the latency and saturate both ports.

The limit is registers. 16 accumulators + a few operand registers fits inside 32.
Push to 12×12 (36 accumulators) and the compiler *spills* to the stack — every
spill is a load/store that defeats the purpose. The sweet spot (8×8, 8×12) is
exactly where real ARM micro-kernels live, and it is dictated by the register file
size, not by the algorithm.

`vfmaq_lane_f32` folds the broadcast into the FMA, so each k-iteration is pure
load + FMA with no shuffle overhead. This is the single most important kernel in
the whole library — everything else exists to feed it.

**Expected:** 1.3–2× over Level 7; 40–80+ GFLOPS single-threaded on M1.

---
---

# Level 10 — Software Pipelining & Prefetch

## The Problem

Implement `sgemm_pipe` in `src/level3/gemm_pipe.cpp`.

Start from your Level 9 micro-kernel. The FMA units are now fast enough that
**memory latency** to the packed panels (even in L1) is exposed. Hide it:

1. **Prefetch** the next panel's cache lines while computing the current tile:
   ```cpp
   __builtin_prefetch(packed_A + next_off, 0 /*read*/, 3 /*high locality*/);
   ```
2. **Unroll** the k-loop by 4 (or 8) so the scheduler sees more independent work
   and the loop overhead amortises.
3. Optionally **software-pipeline**: load iteration `k+1`'s operands *before*
   issuing iteration `k`'s FMAs, so loads and FMAs overlap.

**Files to touch**
- Create `src/level3/gemm_pipe.cpp`
- Uncomment Level 10 rows in bench, test, CMake

**Done when**
- Tests pass
- GFLOPS ≥ Level 9 (gains here are small — 5–15% — and size-dependent)

## Think Before You Look

1. Prefetch is a *hint*. What happens if you prefetch too early (data evicted
   before use) or too late (FMA already stalled)? How would you find the right
   prefetch distance experimentally?
2. The M1 hardware prefetcher already follows sequential streams well. For which
   access pattern does a *software* prefetch still help — the packed panels, or
   the strided original matrices?
3. Unrolling by 4 means 4× the in-flight loads. Does that risk running out of
   registers again? How does it interact with Level 9's 16 accumulators?

## Your Numbers

| Size      | L9 GFLOPS | +unroll | +prefetch | +pipeline |
|-----------|-----------|---------|-----------|-----------|
| 1024×1024 | ___       | ___     | ___       | ___       |
| 2048×2048 | ___       | ___     | ___       | ___       |

---
&nbsp;

&nbsp;

**STOP. Fill in your numbers above before reading.**

&nbsp;

&nbsp;

---

## Why It Works (read AFTER you have numbers)

The micro-kernel is a producer/consumer pipeline: loads *produce* operands, FMAs
*consume* them. If the consumer is faster than the producer, it stalls. Three
levers fix this:

- **Unrolling** exposes more independent iterations to the out-of-order engine, so
  it finds loads to issue ahead of the FMAs that need them.
- **Prefetch** pulls the *next* tile's lines toward L1 before the kernel touches
  them, converting a future miss-latency into background work.
- **Software pipelining** makes the load-ahead explicit in source, not relying on
  the OoO window being large enough.

These are the gains that separate a textbook kernel from a tuned one. They are
small (5–15%) and brittle — the right prefetch distance depends on the exact
chip — which is precisely why the *next* level stops guessing and measures.

**Expected:** 1.05–1.15× over Level 9.

---
---

# Level 11 — Multi-level Cache Blocking (the Goto / BLIS 5-loop)

## The Problem

Implement `sgemm_blis` in `src/level3/gemm_blis.cpp`.

Level 5 blocked for **one** cache level. Real libraries block for the whole
hierarchy with five nested loops around the micro-kernel, each tile sized for a
specific level of cache:

```
for jc in 0..N step NC        // NC: B panel column-block      (targets L3 / SLC)
  for kc in 0..K step KC       // KC: shared K dimension         (targets L1, with packed B)
    pack B(kc:kc+KC, jc:jc+NC) -> Bp           // KC×NC packed, stays in L2/L3
    for ic in 0..M step MC      // MC: A panel row-block          (targets L2)
      pack A(ic:ic+MC, kc:kc+KC) -> Ap         // MC×KC packed, stays in L2
      for jr in 0..NC step NR    // NR: micro-tile columns
        for ir in 0..MC step MR  // MR: micro-tile rows
          micro_kernel(Ap, Bp, C, MR, NR, KC)  // L9 8×8 / 8×12 kernel, A panel in L1
```

This is the **Goto algorithm**, the layout of BLIS and OpenBLAS. The packed `Bp`
panel lives in L2/L3 across the whole `ic` loop; the packed `Ap` panel and the
active strip of `Bp` live in L1 across the micro-kernel.

**Files to touch**
- Create `src/level3/gemm_blis.cpp`
- Uncomment Level 11 rows in bench, test, CMake

**Done when**
- Tests pass (including large + non-square)
- For 2048×2048 and 4096×4096, GFLOPS stays *flat* (no cliff) as size grows —
  that flatness is the whole point

## Think Before You Look

1. Why pack **B once per `kc`** (outside the `ic` loop) but **A once per
   `(ic,kc)`**? Which panel is reused more, and by how much?
2. M1 sizes: L1d ≈ 64–128 KB, L2 ≈ 4 MB (shared per cluster), SLC ≈ 8–16 MB.
   Pick `KC`, `MC`, `NC` so that: `KC×NR` floats fit in L1, `MC×KC` floats fit in
   L2, `KC×NC` floats fit in L3. Write down your numbers.
3. In Level 5, GFLOPS fell as the matrix outgrew L2. Why does the 5-loop keep it
   flat to arbitrary size?
4. Which loop should you eventually parallelise with OpenMP — `jc`, `ic`, or
   `jr`? (Foreshadows combining this with Level 8.)

## Your Numbers

| Size      | L5 GFLOPS | L9 GFLOPS | BLIS 5-loop GFLOPS |
|-----------|-----------|-----------|--------------------|
| 1024×1024 | ___       | ___       | ___                |
| 2048×2048 | ___       | ___       | ___                |
| 4096×4096 | ___       | ___       | ___                |

Sweep the block sizes (record the best 1024³ for each):

| MC  | KC  | NC  | GFLOPS |
|-----|-----|-----|--------|
| 128 | 256 | 4096| ___    |
| 256 | 256 | 4096| ___    |
| 256 | 512 | 8192| ___    |

---
&nbsp;

&nbsp;

**STOP. Fill in your numbers above before reading.**

&nbsp;

&nbsp;

---

## Why It Works (read AFTER you have numbers)

Each cache level has a size and a bandwidth. The 5-loop assigns each operand the
cache level whose capacity it fits and whose bandwidth it needs:

- **Packed B (`KC×NC`)** is streamed by every micro-kernel call across the entire
  `ic` loop — huge reuse — so it is packed once and parked in L2/L3.
- **Packed A (`MC×KC`)** is reused across the `jr` loop (every column micro-tile),
  so it is packed once per `(ic,kc)` and parked in L2.
- The **active A strip (`MR×KC`)** and **B strip (`KC×NR`)** are what the
  micro-kernel actually touches each call — sized for L1.

Because every operand is resident at the right level *before* the micro-kernel
needs it, GFLOPS no longer depends on matrix size: a 4096³ GEMM runs at the same
rate as 1024³. That size-independence is the defining property of a real BLAS, and
it is why `cblas_sgemm` doesn't fall off a cliff the way your Level 1 did.

**Expected:** matches or slightly beats Level 9 at 1024³, but — crucially — holds
that rate at 2048³ and 4096³ where the single-level blocked kernel decays.

---
---

# Level 12 — Autotuning

## The Problem

Implement `sgemm_tuned` in `src/level3/gemm_tuned.cpp` plus a small tuner.

You have a kernel with knobs: `MC, KC, NC, MR, NR`, prefetch distance, unroll
factor. The optimum is **empirical** — it depends on the exact silicon, the
compiler, even the OS scheduler. Stop guessing; search.

1. Define a parameter space (a list of `{MC,KC,NC}` and micro-kernel shapes).
2. For a representative problem size, time each configuration (best-of-N).
3. Cache the winner to a file (e.g. `gemm_tune.cache`, simple `key=value`), keyed
   by problem shape.
4. `sgemm_tuned` reads the cache (tuning on first run if cold) and dispatches the
   Level 11 kernel with the winning parameters.

**Files to touch**
- Create `src/level3/gemm_tuned.cpp` (and optionally `src/level3/tuner.{h,cpp}`)
- Uncomment Level 12 rows in bench, test, CMake

**Done when**
- Tuner runs, writes a cache, and `sgemm_tuned` reproduces the best config without
  re-searching on the second run
- Tuned GFLOPS ≥ your best hand-picked Level 11 number

## Think Before You Look

1. The search space is combinatorial. How do you keep tuning time bounded —
   coordinate descent (tune one knob at a time), a coarse grid, or a fixed
   shortlist of known-good configs?
2. Why key the cache by problem *shape*? Would the best block size for 256³ also
   be best for 4096³?
3. Measurement noise: if config A reads 51.2 and config B reads 51.0 GFLOPS, are
   they actually different? How many reps before you trust a winner?
4. This is exactly what OpenBLAS does at *build* time (per-CPU kernels) and what a
   JIT BLAS (libxsmm) does at *run* time. What are the trade-offs of each?

## Your Numbers

| Approach                    | 1024³ GFLOPS | 2048³ GFLOPS |
|-----------------------------|--------------|--------------|
| L11 default block sizes     | ___          | ___          |
| L11 hand-tuned (your best)  | ___          | ___          |
| L12 autotuned               | ___          | ___          |

---
&nbsp;

&nbsp;

**STOP. Fill in your numbers above before reading.**

&nbsp;

&nbsp;

---

## Why It Works (read AFTER you have numbers)

There is no closed-form best block size. It is a function of cache sizes,
associativity, line size, TLB reach, prefetcher behaviour, and the compiler's
register allocation — too many interacting variables to model precisely. So the
practical answer is *measurement*: enumerate plausible configs, time them on the
real machine, remember the winner.

Caching matters because tuning is expensive (seconds) but the result is reusable
(millions of calls). This is the same insight as the GPU autotuner you'll build on
the other track — and the reason BLAS libraries ship with either per-architecture
hand-tuned kernels (OpenBLAS) or a runtime search (libxsmm, and most GPU GEMM).

**Expected:** ties or modestly beats your best manual config — the real win is that
you stop hand-tuning and the kernel adapts to whatever machine it runs on.

---
---

# Level 13 — Roofline & Performance Modeling

## The Problem

No new kernel. Instead, **instrument and model** what you've built, and compare it
against the practical ceiling.

1. Add baseline rows to `bench_gemm` that call **Apple Accelerate**
   (`cblas_sgemm`) and, if available, **MLX** — these are the "what the machine can
   actually do" numbers. (CMake auto-detects both; see the build notes.)
2. For each of your kernels, compute **arithmetic intensity** (FLOP per byte of
   DRAM traffic) and plot achieved GFLOPS against it — the **roofline**.
3. Locate each kernel on the roofline: is it **memory-bound** (left of the ridge,
   under the slanted bandwidth roof) or **compute-bound** (right of the ridge,
   under the flat peak-FLOP roof)?

Estimate the two roofs for your M1:
- **Peak FP32**: `cores × FMA_ports × 4 lanes × 2 (mul+add) × clock`.
- **Peak DRAM bandwidth**: ~`70–100 GB/s` (LPDDR on M1 Air); measure with a STREAM
  loop if you want the real number.

**Files to touch**
- `benchmark/bench_gemm.cpp` — Accelerate/MLX rows + intensity/roofline output
- Optionally `benchmark/roofline.{h,cpp}` for the arithmetic

**Done when**
- The bench prints, per kernel: GFLOPS, % of peak FP32, and arithmetic intensity
- You can state, for each level, *why* it sits where it does on the roofline

## Think Before You Look

1. Naive GEMM streams O(N³) bytes from DRAM; blocked GEMM streams O(N³/√cache).
   How does that shift each kernel's arithmetic intensity — and thus which roof it
   hits?
2. Accelerate hits ~120–180 GFLOPS using the **AMX** coprocessor, which user NEON
   code cannot touch. Is that a fair comparison? What *is* your fair ceiling — peak
   NEON FP32?
3. If a kernel is at 90% of the NEON peak-FLOP roof, what is the *only* way left to
   go faster? (Hint: it's not the kernel.)

## Your Numbers

| Kernel        | GFLOPS | % NEON peak | Intensity (F/B) | Bound by   |
|---------------|--------|-------------|-----------------|------------|
| naive (L1)    | ___    | ___         | ___             | memory     |
| blocked (L5)  | ___    | ___         | ___             | ___        |
| micro (L9)    | ___    | ___         | ___             | ___        |
| blis+omp      | ___    | ___         | ___             | ___        |
| Accelerate    | ___    | ___ (uses AMX) | —            | —          |
| MLX           | ___    | —           | —               | —          |

---
&nbsp;

&nbsp;

**STOP. Fill in your numbers above before reading.**

&nbsp;

&nbsp;

---

## Why It Works (read AFTER you have numbers)

The roofline model says a kernel's max performance is
`min(peak_FLOPs, intensity × peak_bandwidth)`. Low-intensity kernels (naive GEMM,
≈0.25 F/B) are clamped by the bandwidth roof no matter how good the math units are
— that is the memory wall, drawn as a line. Every optimization in Levels 5–12
*raised the intensity* (blocking and packing cut DRAM traffic) until the kernel
slid past the ridge point and became compute-bound, where it bumps against the
peak-FLOP roof.

Once you're under the flat roof at ~80–90%, the kernel is essentially done — the
remaining gap to Accelerate is **AMX**, a dedicated matrix coprocessor you can't
reach from NEON. Knowing *which roof you're under* tells you whether to optimize
memory (packing, blocking) or compute (wider kernel, better ILP) — or to stop.
That diagnosis is the most valuable skill in the whole journey.

---
---

# Level 14 — One-level Strassen (optional, algorithmic)

## The Problem

Implement `sgemm_strassen` in `src/level3/gemm_strassen.cpp`.

Every level so far kept the `2·M·N·K` FLOP count and made each FLOP cheaper.
Strassen changes the **algorithm**: it multiplies two 2×2 block matrices with
**7** sub-multiplications instead of 8, trading one multiply for several adds.
Recurse once (split each matrix into 2×2 blocks of size N/2), compute the 7
products with your Level 11/12 kernel, and recombine.

**Files to touch**
- Create `src/level3/gemm_strassen.cpp`
- Uncomment Level 14 rows in bench, test, CMake

**Done when**
- Tests pass with a **looser tolerance** (Strassen is less numerically stable)
- For large N (≥2048), one Strassen level beats calling the L11 kernel directly

## Think Before You Look

1. One level cuts multiplies from 8 to 7 — a `7/8 = 0.875×` FLOP factor, ~14% in
   theory. Why do you rarely see the full 14% in practice?
2. Strassen needs temporary buffers for the 7 products and the sums. How much
   extra memory traffic does that add, and at what N does the FLOP saving finally
   outweigh it?
3. Why is the recursion stopped after one level (or a few) rather than going all
   the way down to scalars?
4. Why the looser tolerance? Which step loses accuracy?

## Your Numbers

| Size      | L11 GFLOPS | Strassen-1 GFLOPS | Effective speedup |
|-----------|------------|-------------------|-------------------|
| 2048×2048 | ___        | ___               | ___x              |
| 4096×4096 | ___        | ___               | ___x              |

---
&nbsp;

&nbsp;

**STOP. Fill in your numbers above before reading.**

&nbsp;

&nbsp;

---

## Why It Works (read AFTER you have numbers)

Strassen replaces 8 multiplies of size N/2 with 7 multiplies plus 18 adds of size
N/2. Multiplies are O((N/2)³) each; adds are only O((N/2)²). For large N the cubic
term dominates, so trading a multiply for quadratic adds is a net win — asymptotic
complexity drops from O(N³) to O(N^2.807).

In practice the crossover is high (often N ≥ 1000s) because the extra adds and
temporary buffers add memory traffic that hurts on a kernel that is already
bandwidth-sensitive, and the recursion breaks the clean blocking your lower levels
depend on. That's why production libraries use at most one or two Strassen levels,
and only for very large matrices. It's the one place in this journey where a better
*algorithm*, not better *hardware mapping*, buys the speedup — a fitting capstone.

---
---

# What's Next

You are now ready to bring this to the GPU.

The mapping is direct:

| CPU (what you built)         | GPU / WGSL equivalent              |
|------------------------------|------------------------------------|
| OpenMP threads               | Workgroup dispatch grid            |
| Cache blocking (MC/KC/NC)    | Tile loaded into workgroup memory  |
| Packing into contiguous buf  | Coalesced global memory loads      |
| NEON 4×4 micro-kernel        | Per-invocation register tile       |
| `vfmaq_f32` FMA              | `fma()` in WGSL                    |
| L1 cache (64 KB)             | Workgroup shared memory (32 KB)    |

Go back to `wgpu-blas/`. Start with a naive WGSL GEMM. Then tile it with workgroup
shared memory. You will recognise every step.
