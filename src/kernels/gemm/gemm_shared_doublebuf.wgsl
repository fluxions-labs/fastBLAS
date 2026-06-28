// ── GPU Level 6 — Double-buffered shared memory (latency hiding) ──────────────
// YOUR KERNEL. Writes zeros until you implement it (the harness will report FAIL).
//
// Goal: hide the global-memory latency of loading the next K-tile by prefetching
// it into a SECOND shared-memory buffer while the math units consume the current
// one — then swap. Two `var<workgroup>` tiles per operand, ping-ponged each K
// step. This is the GPU analogue of the CPU software-pipelining/prefetch level
// (Level 10): overlap loads with compute so the FMA units never stall.
//
// Build this on top of your working G4 (shared tiling) and G5 (register tiling).
// Keep the binding layout identical to gemm_naive.wgsl.

struct Dims { M: u32, N: u32, K: u32, _pad: u32, };

@group(0) @binding(0) var<storage, read>       A: array<f32>;
@group(0) @binding(1) var<storage, read>       B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

const TILE: u32 = 16u;
var<workgroup> Asub: array<f32, 512>;  // 2 × 16×16 (double buffer)
var<workgroup> Bsub: array<f32, 512>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= dims.M || col >= dims.N) {
        return;
    }

    // TODO: prefetch tile t+1 into the inactive buffer while computing tile t.
    C[row * dims.N + col] = 0.0;
}
