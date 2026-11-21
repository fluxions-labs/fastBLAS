#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <arm_neon.h>
#include "blas.h"

// Autotuned dispatch: search a small shortlist of {MC,KC,NC} configs for the
// given problem shape, cache the winner to a file keyed by shape, and reuse it
// on subsequent calls without re-searching.
namespace {
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

struct Config { int mc, kc, nc; };

void run_with_config(const Config& cfg, int M, int N, int K,
                      float alpha, const float* A, int lda,
                      const float* B, int ldb,
                      float beta, float* C, int ldc)
{
    std::vector<float> packed_A(static_cast<size_t>(cfg.mc) * cfg.kc);
    std::vector<float> packed_B(static_cast<size_t>(cfg.kc) * cfg.nc);

    for (int jc = 0; jc < N; jc += cfg.nc) {
        int nc = std::min(cfg.nc, N - jc);
        for (int kc = 0; kc < K; kc += cfg.kc) {
            int kcs = std::min(cfg.kc, K - kc);
            float this_beta = (kc == 0) ? beta : 1.0f;
            pack_B(&B[kc * ldb + jc], ldb, kcs, nc, packed_B.data());
            for (int ic = 0; ic < M; ic += cfg.mc) {
                int mc = std::min(cfg.mc, M - ic);
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

const std::vector<Config>& shortlist() {
    static const std::vector<Config> cfgs = {
        {128, 256, 1024},
        {256, 256, 4096},
        {256, 512, 4096},
        {256, 512, 8192},
        {64, 64, 64},
    };
    return cfgs;
}

std::string cache_path() { return "gemm_tune.cache"; }

std::string shape_key(int M, int N, int K) {
    return std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
}

bool read_cached_config(const std::string& key, Config& out) {
    FILE* f = fopen(cache_path().c_str(), "r");
    if (!f) return false;
    char line[256];
    bool found = false;
    while (fgets(line, sizeof(line), f)) {
        char k[128];
        int mc, kc, nc;
        if (sscanf(line, "%127[^=]=%d,%d,%d", k, &mc, &kc, &nc) == 4 && key == k) {
            out = {mc, kc, nc};
            found = true;
            break;
        }
    }
    fclose(f);
    return found;
}

void write_cached_config(const std::string& key, const Config& cfg) {
    FILE* f = fopen(cache_path().c_str(), "a");
    if (!f) return;
    fprintf(f, "%s=%d,%d,%d\n", key.c_str(), cfg.mc, cfg.kc, cfg.nc);
    fclose(f);
}

Config tune(int M, int N, int K) {
    std::string key = shape_key(M, N, K);
    Config cached;
    if (read_cached_config(key, cached)) return cached;

    std::vector<float> A(static_cast<size_t>(M) * K), B(static_cast<size_t>(K) * N),
        C(static_cast<size_t>(M) * N);
    for (auto& v : A) v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : B) v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : C) v = static_cast<float>(rand()) / RAND_MAX;

    double best_time = 1e18;
    Config best = shortlist().front();

    for (const auto& cfg : shortlist()) {
        std::vector<float> Ctmp = C;
        run_with_config(cfg, M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, Ctmp.data(), N);
        auto t0 = std::chrono::high_resolution_clock::now();
        run_with_config(cfg, M, N, K, 1.0f, A.data(), K, B.data(), N, 0.0f, Ctmp.data(), N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        if (secs < best_time) {
            best_time = secs;
            best = cfg;
        }
    }

    write_cached_config(key, best);
    return best;
}
} // namespace

void sgemm_tuned(int M, int N, int K,
                  float alpha,
                  const float* A, int lda,
                  const float* B, int ldb,
                  float beta,
                  float* C, int ldc)
{
    Config cfg = tune(M, N, K);
    run_with_config(cfg, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
