wgpu-blas/
│
├── CMakeLists.txt
├── README.md
├── LICENSE
│
├── external/                # third-party deps (wgpu-native)
│   └── wgpu-native/
│       ├── include/
│       └── lib/
│
├── include/                 # public headers
│   └── wgpublas/
│       ├── context.h
│       ├── buffer.h
│       ├── gemm.h
│       └── tuner.h
│
├── src/                     # implementation
│   │
│   ├── core/                # GPU infrastructure layer
│   │   ├── context.cpp
│   │   ├── buffer.cpp
│   │   ├── pipeline.cpp
│   │   └── dispatch.cpp
│   │
│   ├── kernels/             # WGSL compute shaders
│   │   └── gemm/
│   │       ├── gemm_naive.wgsl
│   │       ├── gemm_tiled_16x16.wgsl
│   │       ├── gemm_tiled_32x8.wgsl
│   │       └── gemm_shared_doublebuf.wgsl
│   │
│   ├── blas/                # BLAS-level operations
│   │   ├── gemm.cpp
│   │   └── gemm_utils.cpp
│   │
│   ├── autotune/            # performance tuning layer
│   │   ├── gemm_tuner.cpp
│   │   └── config_cache.cpp
│   │
│   └── utils/
│       ├── timer.cpp
│       └── logger.cpp
│
├── benchmark/
│   ├── bench_gemm.cpp
│   ├── cpu_reference.cpp
│   └── metrics.cpp
│
├── examples/
│   ├── simple_gemm.cpp
│   └── transformer_demo.cpp
│
└── build/