// ── GPU Level 5 — Register tiling / thread coarsening ─────────────────────────
// YOUR KERNEL. Writes zeros until you implement it (the harness will report FAIL).
//
// Goal: each invocation computes a *micro-tile* of C (e.g. 1 row × 4 cols, or
// 4×4) held in registers, instead of a single element. Combine with the shared
// 32×8 staging so each workgroup covers a 32×8 (or larger) tile of C. This is the
// GPU analogue of the CPU register-blocked micro-kernel (Levels 9–11): more
// output per invocation amortises global/shared loads and raises arithmetic
// intensity. Tune the workgroup_size and the per-thread tile (TM×TN) together.
//
// Keep the binding layout identical to gemm_naive.wgsl.

struct Dims { M: u32, N: u32, K: u32, _pad: u32, };

@group(0) @binding(0) var<storage, read>       A: array<f32>;
@group(0) @binding(1) var<storage, read>       B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform>             dims: Dims;

@compute @workgroup_size(32, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= dims.M || col >= dims.N) {
        return;
    }

    // TODO: accumulate a TM×TN register tile across K, then write it out.
    C[row * dims.N + col] = 0.0;
}
