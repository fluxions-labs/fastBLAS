#include <algorithm>
#include <vector>
#include <arm_neon.h>
#include "blas.h"

// Goto/BLIS 5-loop: block sizes chosen so each packed panel targets a
// specific cache level. On M1: L1d ~64KB, L2 ~4MB (shared per cluster),
// SLC ~8-16MB. KC*NR floats should fit L1, MC*KC should fit L2,
// KC*NC should fit L3/SLC.
namespace {
constexpr int MC = 256;
constexpr int KC = 512;
constexpr int NC = 4096;
constexpr int MR = 8;
constexpr int NR = 8;

void micro_8x8(int mc, int nc, int kc,
                float alpha,
                const float* A, int lda,
                const float* B, int ldb,
                float beta,
                float* C, int ldc)
{
    int i8 = mc - (mc % MR);
    int j8 = nc - (nc % NR);

    for (int i = 0; i < i8; i += MR) {
        for (int j = 0; j < j8; j += NR) {
            float32x4_t c[8][2];
            for (int r = 0; r < 8; r++) {
                c[r][0] = vdupq_n_f32(0.0f);
                c[r][1] = vdupq_n_f32(0.0f);
            }

            for (int k = 0; k < kc; k++) {
                __builtin_prefetch(&B[(k + 1) * ldb + j], 0, 3);
                float32x4_t b0 = vld1q_f32(&B[k * ldb + j]);
                float32x4_t b1 = vld1q_f32(&B[k * ldb + j + 4]);
                for (int r = 0; r < 8; r++) {
                    float a_val = A[(i + r) * lda + k];
                    c[r][0] = vfmaq_n_f32(c[r][0], b0, a_val);
                    c[r][1] = vfmaq_n_f32(c[r][1], b1, a_val);
                }
            }

            float32x4_t valpha = vdupq_n_f32(alpha);
            float32x4_t vbeta  = vdupq_n_f32(beta);
            for (int r = 0; r < 8; r++) {
                float* cp0 = &C[(i + r) * ldc + j];
                float* cp1 = &C[(i + r) * ldc + j + 4];
                vst1q_f32(cp0, vfmaq_f32(vmulq_f32(valpha, c[r][0]), vbeta, vld1q_f32(cp0)));
                vst1q_f32(cp1, vfmaq_f32(vmulq_f32(valpha, c[r][1]), vbeta, vld1q_f32(cp1)));
            }
        }
    }

    for (int i = 0; i < mc; i++) {
        int jstart = (i < i8) ? j8 : 0;
        for (int j = jstart; j < nc; j++) {
            float sum = 0.0f;
            for (int k = 0; k < kc; k++)
                sum += A[i * lda + k] * B[k * ldb + j];
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

void pack_A(const float* A, int lda, int mc, int kc, float* packed) {
    for (int ir = 0; ir < mc; ir++)
        for (int kr = 0; kr < kc; kr++)
            packed[ir * kc + kr] = A[ir * lda + kr];
}

void pack_B(const float* B, int ldb, int kc, int nc, float* packed) {
    for (int kr = 0; kr < kc; kr++)
        for (int jr = 0; jr < nc; jr++)
            packed[kr * nc + jr] = B[kr * ldb + jr];
}
} // namespace

void sgemm_blis(int M, int N, int K,
                 float alpha,
                 const float* A, int lda,
                 const float* B, int ldb,
                 float beta,
                 float* C, int ldc)
{
    std::vector<float> packed_A(MC * KC);
    std::vector<float> packed_B(KC * NC);

    for (int jc = 0; jc < N; jc += NC) {
        int nc = std::min(NC, N - jc);
        for (int kc = 0; kc < K; kc += KC) {
            int kcs = std::min(KC, K - kc);
            float this_beta = (kc == 0) ? beta : 1.0f;
            // Pack B once per (jc, kc): reused across the whole ic loop below.
            pack_B(&B[kc * ldb + jc], ldb, kcs, nc, packed_B.data());
            for (int ic = 0; ic < M; ic += MC) {
                int mc = std::min(MC, M - ic);
                // Pack A once per (ic, kc): reused across all jr micro-tiles.
                pack_A(&A[ic * lda + kc], lda, mc, kcs, packed_A.data());
                micro_8x8(mc, nc, kcs,
                          alpha,
                          packed_A.data(), kcs,
                          packed_B.data(), nc,
                          this_beta,
                          &C[ic * ldc + jc], ldc);
            }
        }
    }
}
