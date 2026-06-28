// ── GPU Level 1 (provided reference): naive GEMM ──────────────────────────────
// C = A * B   (alpha=1, beta=0).  One invocation computes one element of C.
// A: M×K row-major, B: K×N row-major, C: M×N row-major — all flattened f32 arrays.
//
// This is the GPU analogue of the CPU Level 1 naive triple loop, and it is the
// correctness/performance reference the harness checks every other shader against.
// Your job in G2+ is to make this faster (tiling, shared memory, register tiles)
// while keeping the same result.

struct Dims {
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       A: array<f32>;
@group(0) @binding(1) var<storage, read>       B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;   // 0 .. M
    let col = gid.x;   // 0 .. N
    if (row >= dims.M || col >= dims.N) {
        return;
    }

    var sum = 0.0;
    for (var k: u32 = 0u; k < dims.K; k = k + 1u) {
        sum = sum + A[row * dims.K + k] * B[k * dims.N + col];
    }
    C[row * dims.N + col] = sum;
}
