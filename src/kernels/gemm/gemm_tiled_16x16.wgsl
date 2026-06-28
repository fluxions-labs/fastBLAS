// ── GPU Level 4 — Workgroup shared-memory tiling (16×16) ──────────────────────
// YOUR KERNEL. The harness runs this against the CPU reference and reports the
// error; right now it writes zeros, so it FAILS until you implement it.
//
// Goal: each 16×16 workgroup cooperatively computes a 16×16 tile of C. Stage a
// 16×16 tile of A and of B into `var<workgroup>` shared memory, barrier, then
// each invocation accumulates over the tile from fast on-chip memory. Loop over
// K in steps of 16. This is the GPU analogue of CPU cache blocking (Level 5).
//
// Keep the binding layout and entry point identical to gemm_naive.wgsl.

struct Dims { M: u32, N: u32, K: u32, _pad: u32, };

@group(0) @binding(0) var<storage, read>       A: array<f32>;
@group(0) @binding(1) var<storage, read>       B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

const TILE: u32 = 16u;
var<workgroup> Asub: array<f32, 256>;  // 16*16
var<workgroup> Bsub: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id)  lid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= dims.M || col >= dims.N) {
        return;
    }

    // TODO: replace this stub with the shared-memory tiled accumulation.
    //   1. for t in 0..ceil(K/TILE):
    //        Asub[lid.y][lid.x] = A[row, t*TILE + lid.x]   (guard bounds)
    //        Bsub[lid.y][lid.x] = B[t*TILE + lid.y, col]   (guard bounds)
    //        workgroupBarrier();
    //        for k in 0..TILE: sum += Asub[lid.y][k] * Bsub[k][lid.x];
    //        workgroupBarrier();
    //   2. C[row, col] = sum;
    C[row * dims.N + col] = 0.0;
}
