// ─────────────────────────────────────────────────────────────────────────────
// GPU GEMM harness (provided infrastructure — see gpu/JOURNEY_GPU.md, Level G1).
//
// End-to-end WebGPU compute pipeline: upload A/B, dispatch a WGSL GEMM shader,
// read C back, and check it against a CPU reference. Runs each shader file you
// pass on the command line; defaults to the provided naive reference shader.
//
//   ./gpu_gemm [shader1.wgsl shader2.wgsl ...]
//
// You write the shaders (G2+). This file you do not need to touch — it is the
// GPU analogue of cpu/benchmark/bench_gemm.cpp + cpu/test/test_gemm.cpp.
//
// NOTE on timing: the per-shader seconds include submit + readback, not just
// kernel time. Isolating kernel time needs GPU timestamp queries — that is a
// later curriculum exercise (see JOURNEY_GPU.md). The numbers are still valid for
// comparing shaders against each other and watching the trend toward the MLX/
// Metal ceiling.
// ─────────────────────────────────────────────────────────────────────────────
#include <webgpu/webgpu.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

static WGPUStringView sv(const char* s) {
    return WGPUStringView{ s, s ? strlen(s) : 0 };
}

// ── device bring-up (same pattern as main.cpp) ────────────────────────────────
struct Gpu {
    WGPUInstance instance = nullptr;
    WGPUAdapter  adapter  = nullptr;
    WGPUDevice   device   = nullptr;
    WGPUQueue    queue    = nullptr;
};

static void onAdapter(WGPURequestAdapterStatus status, WGPUAdapter a,
                      WGPUStringView, void* ud, void*) {
    if (status == WGPURequestAdapterStatus_Success) *(WGPUAdapter*)ud = a;
}
static void onDevice(WGPURequestDeviceStatus status, WGPUDevice d,
                     WGPUStringView, void* ud, void*) {
    if (status == WGPURequestDeviceStatus_Success) *(WGPUDevice*)ud = d;
}

static bool init_gpu(Gpu& g) {
    WGPUInstanceDescriptor idesc = {};
    g.instance = wgpuCreateInstance(&idesc);
    if (!g.instance) { fprintf(stderr, "no instance\n"); return false; }

    WGPURequestAdapterOptions opts = {};
    WGPURequestAdapterCallbackInfo aci = {};
    aci.callback = onAdapter; aci.userdata1 = &g.adapter;
    wgpuInstanceRequestAdapter(g.instance, &opts, aci);
    while (!g.adapter) wgpuInstanceProcessEvents(g.instance);

    WGPUDeviceDescriptor ddesc = {};
    WGPURequestDeviceCallbackInfo dci = {};
    dci.callback = onDevice; dci.userdata1 = &g.device;
    wgpuAdapterRequestDevice(g.adapter, &ddesc, dci);
    while (!g.device) wgpuInstanceProcessEvents(g.instance);

    g.queue = wgpuDeviceGetQueue(g.device);
    return g.queue != nullptr;
}

// ── helpers ───────────────────────────────────────────────────────────────────
static std::string read_file(const char* path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open shader %s\n", path); return {}; }
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

static WGPUBuffer make_buffer(WGPUDevice dev, uint64_t size, WGPUBufferUsage usage) {
    WGPUBufferDescriptor d = {};
    d.size = size;
    d.usage = usage;
    return wgpuDeviceCreateBuffer(dev, &d);
}

// Block until all submitted GPU work completes (used for timing).
static void wait_idle(Gpu& g) {
    bool done = false;
    WGPUQueueWorkDoneCallbackInfo ci = {};
    ci.mode = WGPUCallbackMode_AllowProcessEvents;
    ci.callback = [](WGPUQueueWorkDoneStatus, void* ud, void*) { *(bool*)ud = true; };
    ci.userdata1 = &done;
    wgpuQueueOnSubmittedWorkDone(g.queue, ci);
    while (!done) wgpuInstanceProcessEvents(g.instance);
}

// ── CPU reference (naive) ─────────────────────────────────────────────────────
static void cpu_gemm(int M, int N, int K, const float* A, const float* B, float* C) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += A[i*K + k] * B[k*N + j];
            C[i*N + j] = s;
        }
}

struct Dims { uint32_t M, N, K, pad; };

// Compile + run one shader. Returns best wall time over `reps`, fills `out` with C.
static double run_shader(Gpu& g, const std::string& wgsl, int M, int N, int K,
                         const std::vector<float>& A, const std::vector<float>& B,
                         std::vector<float>& out, int reps = 5) {
    // Shader module
    WGPUShaderSourceWGSL src = {};
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    src.code = sv(wgsl.c_str());
    WGPUShaderModuleDescriptor smd = {};
    smd.nextInChain = &src.chain;
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(g.device, &smd);

    // Buffers
    const uint64_t aSize = (uint64_t)M*K*sizeof(float);
    const uint64_t bSize = (uint64_t)K*N*sizeof(float);
    const uint64_t cSize = (uint64_t)M*N*sizeof(float);
    WGPUBuffer bufA = make_buffer(g.device, aSize, (WGPUBufferUsage)(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst));
    WGPUBuffer bufB = make_buffer(g.device, bSize, (WGPUBufferUsage)(WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst));
    WGPUBuffer bufC = make_buffer(g.device, cSize, (WGPUBufferUsage)(WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc));
    WGPUBuffer bufD = make_buffer(g.device, sizeof(Dims), (WGPUBufferUsage)(WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst));
    WGPUBuffer readback = make_buffer(g.device, cSize, (WGPUBufferUsage)(WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst));

    Dims dims{ (uint32_t)M, (uint32_t)N, (uint32_t)K, 0 };
    wgpuQueueWriteBuffer(g.queue, bufA, 0, A.data(), aSize);
    wgpuQueueWriteBuffer(g.queue, bufB, 0, B.data(), bSize);
    wgpuQueueWriteBuffer(g.queue, bufD, 0, &dims, sizeof(dims));

    // Bind group layout: 0=A ro-storage, 1=B ro-storage, 2=C storage, 3=dims uniform
    WGPUBindGroupLayoutEntry e[4] = {};
    for (int i = 0; i < 4; i++) { e[i].binding = (uint32_t)i; e[i].visibility = WGPUShaderStage_Compute; }
    e[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    e[1].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
    e[2].buffer.type = WGPUBufferBindingType_Storage;
    e[3].buffer.type = WGPUBufferBindingType_Uniform;
    WGPUBindGroupLayoutDescriptor bgld = {};
    bgld.entryCount = 4; bgld.entries = e;
    WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(g.device, &bgld);

    WGPUPipelineLayoutDescriptor pld = {};
    pld.bindGroupLayoutCount = 1; pld.bindGroupLayouts = &bgl;
    WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(g.device, &pld);

    WGPUComputePipelineDescriptor cpd = {};
    cpd.layout = pl;
    cpd.compute.module = mod;
    cpd.compute.entryPoint = sv("main");
    WGPUComputePipeline pipe = wgpuDeviceCreateComputePipeline(g.device, &cpd);

    WGPUBindGroupEntry be[4] = {};
    be[0].binding = 0; be[0].buffer = bufA; be[0].size = aSize;
    be[1].binding = 1; be[1].buffer = bufB; be[1].size = bSize;
    be[2].binding = 2; be[2].buffer = bufC; be[2].size = cSize;
    be[3].binding = 3; be[3].buffer = bufD; be[3].size = sizeof(Dims);
    WGPUBindGroupDescriptor bgd = {};
    bgd.layout = bgl; bgd.entryCount = 4; bgd.entries = be;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g.device, &bgd);

    // workgroup_size in the shaders is 16×16 (or 32×8); dispatch covers M×N with a
    // conservative 16×16 grid (extra invocations are bounds-checked out).
    const uint32_t gx = ((uint32_t)N + 15) / 16;
    const uint32_t gy = ((uint32_t)M + 15) / 16;

    auto dispatch_once = [&]() {
        WGPUCommandEncoderDescriptor ced = {};
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g.device, &ced);
        WGPUComputePassDescriptor cpdesc = {};
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &cpdesc);
        wgpuComputePassEncoderSetPipeline(pass, pipe);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, 1);
        wgpuComputePassEncoderEnd(pass);
        WGPUCommandBufferDescriptor cbd = {};
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cbd);
        wgpuQueueSubmit(g.queue, 1, &cmd);
    };

    // Warm-up + timed best-of-N (kernel + submit; see header note on timing).
    dispatch_once(); wait_idle(g);
    double best = 1e18;
    for (int r = 0; r < reps; r++) {
        auto t0 = Clock::now();
        dispatch_once();
        wait_idle(g);
        auto t1 = Clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        if (s < best) best = s;
    }

    // Read C back once for the correctness check.
    {
        WGPUCommandEncoderDescriptor ced = {};
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g.device, &ced);
        wgpuCommandEncoderCopyBufferToBuffer(enc, bufC, 0, readback, 0, cSize);
        WGPUCommandBufferDescriptor cbd = {};
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, &cbd);
        wgpuQueueSubmit(g.queue, 1, &cmd);

        bool mapped = false;
        WGPUBufferMapCallbackInfo mci = {};
        mci.mode = WGPUCallbackMode_AllowProcessEvents;
        mci.callback = [](WGPUMapAsyncStatus, WGPUStringView, void* ud, void*) { *(bool*)ud = true; };
        mci.userdata1 = &mapped;
        wgpuBufferMapAsync(readback, WGPUMapMode_Read, 0, cSize, mci);
        while (!mapped) wgpuInstanceProcessEvents(g.instance);

        const float* p = (const float*)wgpuBufferGetMappedRange(readback, 0, cSize);
        out.assign(p, p + (size_t)M*N);
        wgpuBufferUnmap(readback);
    }

    wgpuBindGroupRelease(bg);
    wgpuComputePipelineRelease(pipe);
    wgpuPipelineLayoutRelease(pl);
    wgpuBindGroupLayoutRelease(bgl);
    wgpuShaderModuleRelease(mod);
    wgpuBufferRelease(readback);
    wgpuBufferRelease(bufD);
    wgpuBufferRelease(bufC);
    wgpuBufferRelease(bufB);
    wgpuBufferRelease(bufA);
    return best;
}

int main(int argc, char** argv) {
    // Default shader: the provided naive reference. Pass more on the command line.
#ifndef WGPUBLAS_SHADER_DIR
#define WGPUBLAS_SHADER_DIR "src/kernels/gemm"
#endif
    std::string default_shader = std::string(WGPUBLAS_SHADER_DIR) + "/gemm_naive.wgsl";
    std::vector<const char*> shaders;
    if (argc > 1) for (int i = 1; i < argc; i++) shaders.push_back(argv[i]);
    else shaders.push_back(default_shader.c_str());

    Gpu g;
    if (!init_gpu(g)) { fprintf(stderr, "GPU init failed\n"); return 1; }
    printf("GPU ready.\n\n");

    const int sizes[] = { 256, 512, 1024 };
    printf("%-34s %14s %10s %8s %10s\n", "shader", "size", "GFLOPS", "time", "max_err");
    printf("%s\n", std::string(80, '-').c_str());

    bool all_ok = true;
    for (int sz : sizes) {
        const int M = sz, N = sz, K = sz;
        std::vector<float> A(M*K), B(K*N), Cref(M*N), Cgpu;
        for (auto& v : A) v = (float)rand() / RAND_MAX;
        for (auto& v : B) v = (float)rand() / RAND_MAX;
        cpu_gemm(M, N, K, A.data(), B.data(), Cref.data());

        for (const char* path : shaders) {
            std::string wgsl = read_file(path);
            if (wgsl.empty()) { all_ok = false; continue; }

            double secs = run_shader(g, wgsl, M, N, K, A, B, Cgpu);
            double gf = (2.0 * M * N * K) / 1e9 / secs;

            float max_err = 0.0f;
            for (int i = 0; i < M*N; i++)
                max_err = std::max(max_err, std::fabs(Cref[i] - Cgpu[i]));
            float tol = 1e-3f * (float)K;
            bool ok = max_err <= tol;
            all_ok &= ok;

            // strip directory for a compact label
            const char* base = strrchr(path, '/'); base = base ? base + 1 : path;
            // GFLOPS is only meaningful for a correct result; a stub that writes
            // zeros looks "fast" but computes nothing.
            char gfbuf[16];
            if (ok) snprintf(gfbuf, sizeof gfbuf, "%10.2f", gf);
            else    snprintf(gfbuf, sizeof gfbuf, "%10s", "—");
            printf("%-34s %5d×%4d×%4d %s %7.3fs %10.2e %s\n",
                   base, M, N, K, gfbuf, secs, max_err, ok ? "ok" : "FAIL");
        }
        printf("\n");
    }

    printf("%s\n", all_ok ? "ALL SHADERS MATCH REFERENCE"
                          : "SOME SHADERS FAIL (stubs write zeros until you implement them)");
    return all_ok ? 0 : 1;
}
