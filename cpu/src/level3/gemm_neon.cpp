#include <algorithm>
#include <vector>
#include <arm_neon.h>
#include "blas.h"

namespace {
constexpr int MC = 64;
constexpr int KC = 64;
constexpr int NC = 64;

// 4x4 NEON micro-kernel: computes a 4-row x 4-col tile of C.
// packed_A: mc x kc, row-major, lda == kc
// packed_B: kc x nc, row-major, ldb == nc
void micro_4x4(int mc, int nc, int kc,
                float alpha,
                const float* A, int lda,
                const float* B, int ldb,
                float beta,
                float* C, int ldc)
{
    int i4 = mc - (mc % 4);
    int j4 = nc - (nc % 4);

    for (int i = 0; i < i4; i += 4) {
        for (int j = 0; j < j4; j += 4) {
            float32x4_t c0 = vdupq_n_f32(0.0f);
            float32x4_t c1 = vdupq_n_f32(0.0f);
            float32x4_t c2 = vdupq_n_f32(0.0f);
            float32x4_t c3 = vdupq_n_f32(0.0f);

            for (int k = 0; k < kc; k++) {
                float32x4_t b_row = vld1q_f32(&B[k * ldb + j]);
                c0 = vfmaq_n_f32(c0, b_row, A[(i + 0) * lda + k]);
                c1 = vfmaq_n_f32(c1, b_row, A[(i + 1) * lda + k]);
                c2 = vfmaq_n_f32(c2, b_row, A[(i + 2) * lda + k]);
                c3 = vfmaq_n_f32(c3, b_row, A[(i + 3) * lda + k]);
            }

            float32x4_t valpha = vdupq_n_f32(alpha);
            float32x4_t vbeta  = vdupq_n_f32(beta);

            float* c0p = &C[(i + 0) * ldc + j];
            float* c1p = &C[(i + 1) * ldc + j];
            float* c2p = &C[(i + 2) * ldc + j];
            float* c3p = &C[(i + 3) * ldc + j];

            vst1q_f32(c0p, vfmaq_f32(vmulq_f32(valpha, c0), vbeta, vld1q_f32(c0p)));
            vst1q_f32(c1p, vfmaq_f32(vmulq_f32(valpha, c1), vbeta, vld1q_f32(c1p)));
            vst1q_f32(c2p, vfmaq_f32(vmulq_f32(valpha, c2), vbeta, vld1q_f32(c2p)));
            vst1q_f32(c3p, vfmaq_f32(vmulq_f32(valpha, c3), vbeta, vld1q_f32(c3p)));
        }
    }

    // Scalar fallback for edge rows/cols not covered by the 4x4 tile.
    for (int i = 0; i < mc; i++) {
        int jstart = (i < i4) ? j4 : 0;
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

void sgemm_neon(int M, int N, int K,
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
            pack_B(&B[kc * ldb + jc], ldb, kcs, nc, packed_B.data());
            for (int ic = 0; ic < M; ic += MC) {
                int mc = std::min(MC, M - ic);
                pack_A(&A[ic * lda + kc], lda, mc, kcs, packed_A.data());
                micro_4x4(mc, nc, kcs,
                          alpha,
                          packed_A.data(), kcs,
                          packed_B.data(), nc,
                          this_beta,
                          &C[ic * ldc + jc], ldc);
            }
        }
    }
}
