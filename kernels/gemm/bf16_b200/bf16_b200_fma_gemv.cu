#include "kittens.cuh"
#include "../common.cuh"

using namespace kittens;

/*
 * Optimized FMA-based GEMV: y = A * x
 *
 * Split-K strategy: the K dimension is partitioned across K_SPLITS block groups.
 * Each block computes a partial dot product for its K range and atomicAdds
 * the result into a float32 buffer. A tiny follow-up kernel converts to bf16.
 *
 * Grid: dim3(M_blocks, K_SPLITS)
 * blockIdx.x = which rows, blockIdx.y = which K split
 */

template <int _ROWS_PER_WARP, int _K_CHUNK, int _K_SPLITS>
struct fma_gemv_config {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int ROWS_PER_WARP = _ROWS_PER_WARP;
    static constexpr int K_CHUNK = _K_CHUNK;
    static constexpr int K_SPLITS = _K_SPLITS;
    static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;  // 8
    static constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;
};

// Split-K FMA GEMV kernel: partial dot products accumulated via atomicAdd to float32
template <typename Config>
__global__ void fma_gemv_splitk_kernel(
    float* __restrict__ y_partial,  // float32 accumulation buffer [M]
    const bf16* __restrict__ A,
    const bf16* __restrict__ x,
    int M, int K
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int base_row = blockIdx.x * Config::ROWS_PER_BLOCK + warp_id * Config::ROWS_PER_WARP;

    if (base_row >= M) return;

    // Compute K range for this split
    const int k_split = blockIdx.y;
    const int k_per_split = (K + Config::K_SPLITS - 1) / Config::K_SPLITS;
    // Align to 8 elements for vectorized loads (except last split)
    const int k_start = k_split * k_per_split;
    const int k_end = min(k_start + k_per_split, K);

    if (k_start >= K) return;

    __shared__ bf16 x_smem[Config::K_CHUNK];

    float acc[Config::ROWS_PER_WARP] = {0.0f};

    // Precompute row pointers (offset to k_start)
    const bf16* A_row_ptrs[Config::ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < Config::ROWS_PER_WARP; r++) {
        A_row_ptrs[r] = A + (base_row + r) * K;
    }

    constexpr int elems_per_thread = 8;
    constexpr int elems_per_pass = Config::BLOCK_SIZE * elems_per_thread;

    for (int k_base = k_start; k_base < k_end; k_base += Config::K_CHUNK) {
        const int k_remaining = min(Config::K_CHUNK, k_end - k_base);

        // Collaborative x load into shared memory using vectorized float4 loads
        for (int offset = threadIdx.x * elems_per_thread; offset < k_remaining; offset += elems_per_pass) {
            if (offset + elems_per_thread <= k_remaining) {
                float4 tmp = *reinterpret_cast<const float4*>(&x[k_base + offset]);
                *reinterpret_cast<float4*>(&x_smem[offset]) = tmp;
            } else {
                for (int i = 0; i < elems_per_thread && (offset + i) < k_remaining; i++) {
                    x_smem[offset + i] = x[k_base + offset + i];
                }
            }
        }
        __syncthreads();

        // Compute: each lane processes 8 elements at a time with stride 256
        const int lane_stride = 32 * elems_per_thread;  // 256

        for (int k = lane_id * elems_per_thread; k < k_remaining; k += lane_stride) {
            // Load x values from shared memory
            float4 x_vec = *reinterpret_cast<const float4*>(&x_smem[k]);
            const bf16* x_ptr = reinterpret_cast<const bf16*>(&x_vec);

            float xv[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                xv[i] = __bfloat162float(x_ptr[i]);
            }

            #pragma unroll
            for (int r = 0; r < Config::ROWS_PER_WARP; r++) {
                if (base_row + r < M) {
                    float4 a_vec = *reinterpret_cast<const float4*>(&A_row_ptrs[r][k_base + k]);
                    const bf16* a_ptr = reinterpret_cast<const bf16*>(&a_vec);

                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        acc[r] = fmaf(__bfloat162float(a_ptr[i]), xv[i], acc[r]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int r = 0; r < Config::ROWS_PER_WARP; r++) {
        #pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            acc[r] += __shfl_down_sync(0xffffffff, acc[r], offset);
        }
    }

    // Lane 0 writes partial results via atomicAdd to float32 buffer
    if (lane_id == 0) {
        #pragma unroll
        for (int r = 0; r < Config::ROWS_PER_WARP; r++) {
            if (base_row + r < M) {
                if (Config::K_SPLITS == 1) {
                    // No atomics needed for single split
                    y_partial[base_row + r] = acc[r];
                } else {
                    atomicAdd(&y_partial[base_row + r], acc[r]);
                }
            }
        }
    }
}

// Convert float32 accumulation buffer to bf16 output
__global__ void convert_f32_to_bf16(
    bf16* __restrict__ y,
    const float* __restrict__ y_f32,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        y[idx] = __float2bfloat16(y_f32[idx]);
    }
}

// Reference GEMV for correctness check
template <typename T>
__global__ void reference_gemv_kernel(T* y, const T* A, const T* x, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            float a_val = kittens::base_types::convertor<float, T>::convert(A[row * K + k]);
            float x_val = kittens::base_types::convertor<float, T>::convert(x[k]);
            acc += a_val * x_val;
        }
        y[row] = kittens::base_types::convertor<T, float>::convert(acc);
    }
}

template <typename T>
void reference_gemv(T* y, const T* A, const T* x, int M, int K) {
    dim3 block(256);
    dim3 grid((M + 255) / 256);
    reference_gemv_kernel<T><<<grid, block>>>(y, A, x, M, K);
}

template <typename Config>
__host__ double run_fma_gemv_benchmark(size_t M, size_t K, bool ncu = false) {
    std::cout << "--------------------  M=" << M << " K=" << K << "  --------------------\n";
    std::cout << "Config: ROWS_PER_WARP=" << Config::ROWS_PER_WARP
              << " K_CHUNK=" << Config::K_CHUNK
              << " K_SPLITS=" << Config::K_SPLITS
              << " BLOCK_SIZE=" << Config::BLOCK_SIZE << "\n";

    // Cooldown between configurations
    sleep_ms(500);

    // L2 cache eviction - multiple buffer groups
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = 2 * (size_t(M) * K + K + M);
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    // Allocate device memory
    std::vector<bf16*> d_A(arg_group_count);
    std::vector<bf16*> d_x(arg_group_count);
    std::vector<bf16*> d_y(arg_group_count);
    std::vector<float*> d_y_partial(arg_group_count);  // float32 accumulation buffer
    bf16* d_y_ref;

    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaMalloc(&d_A[i], M * K * sizeof(bf16)));
        CUDACHECK(cudaMalloc(&d_x[i], K * sizeof(bf16)));
        CUDACHECK(cudaMalloc(&d_y[i], M * sizeof(bf16)));
        CUDACHECK(cudaMalloc(&d_y_partial[i], M * sizeof(float)));
    }
    CUDACHECK(cudaMalloc(&d_y_ref, M * sizeof(bf16)));
    std::cout << "Allocated device memory" << std::endl;

    // Initialize with random values
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<bf16, FillMode::RANDOM>(d_A[i], M * K, seed + i * 100, -1.0f, 1.0f);
        fill<bf16, FillMode::RANDOM>(d_x[i], K, seed + i * 100 + 1, -1.0f, 1.0f);
        fill<bf16, FillMode::CONSTANT>(d_y[i], M, 0.0f);
        CUDACHECK(cudaMemset(d_y_partial[i], 0, M * sizeof(float)));
    }
    fill<bf16, FillMode::CONSTANT>(d_y_ref, M, 0.0f);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Initialized matrices on device" << std::endl;

    // Compute reference
    reference_gemv<bf16>(d_y_ref, d_A[0], d_x[0], M, K);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Computed reference GEMV on device" << std::endl;

    // Kernel launch parameters
    dim3 block(Config::BLOCK_SIZE);
    dim3 grid((M + Config::ROWS_PER_BLOCK - 1) / Config::ROWS_PER_BLOCK, Config::K_SPLITS);
    dim3 convert_block(256);
    dim3 convert_grid((M + 255) / 256);

    int m_blocks = (M + Config::ROWS_PER_BLOCK - 1) / Config::ROWS_PER_BLOCK;
    std::cout << "Grid: (" << m_blocks << ", " << Config::K_SPLITS << ") = "
              << m_blocks * Config::K_SPLITS << " total blocks\n";

    // Number of iterations
    int num_warmups = ncu ? 0 : 500;
    int num_iters = ncu ? 1 : 100;

    // Warmup
    for (int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        CUDACHECK(cudaMemset(d_y_partial[idx], 0, M * sizeof(float)));
        fma_gemv_splitk_kernel<Config><<<grid, block>>>(d_y_partial[idx], d_A[idx], d_x[idx], M, K);
        convert_f32_to_bf16<<<convert_grid, convert_block>>>(d_y[idx], d_y_partial[idx], M);
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));

    for (int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        CUDACHECK(cudaMemsetAsync(d_y_partial[idx], 0, M * sizeof(float)));
        fma_gemv_splitk_kernel<Config><<<grid, block>>>(d_y_partial[idx], d_A[idx], d_x[idx], M, K);
        convert_f32_to_bf16<<<convert_grid, convert_block>>>(d_y[idx], d_y_partial[idx], M);
    }

    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Calculate performance
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0 / num_iters;
    double bytes = double(2) * (double(M) * K + K + M);
    double gb_per_sec = (bytes / microseconds) / 1e3;
    double gflops = (double(2) * M * K / microseconds) / 1e3;

    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Achieved bandwidth: " << gb_per_sec << " GB/s\n";
    std::cout << "Achieved performance: " << gflops << " GFLOPs\n";

    // Verify correctness (run once cleanly for check)
    CUDACHECK(cudaMemset(d_y_partial[0], 0, M * sizeof(float)));
    fma_gemv_splitk_kernel<Config><<<grid, block>>>(d_y_partial[0], d_A[0], d_x[0], M, K);
    convert_f32_to_bf16<<<convert_grid, convert_block>>>(d_y[0], d_y_partial[0], M);
    CUDACHECK(cudaDeviceSynchronize());
    check_correctness(d_y[0], d_y_ref, M);

    // Cleanup
    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_x[i]);
        cudaFree(d_y[i]);
        cudaFree(d_y_partial[i]);
    }
    cudaFree(d_y_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return gb_per_sec;
}

__host__ int main() {
    bool ncu = false;

    // RPW=2 is the best from previous experiment. Sweep K_SPLITS.

    std::cout << "\n========== FMA GEMV: RPW=2, K_CHUNK=2048, K_SPLITS=1 (baseline) ==========\n";
    run_fma_gemv_benchmark<fma_gemv_config<2, 2048, 1>>(4096, 4096, ncu);
    run_fma_gemv_benchmark<fma_gemv_config<2, 2048, 1>>(8192, 8192, ncu);
    run_fma_gemv_benchmark<fma_gemv_config<2, 2048, 1>>(16384, 16384, ncu);
    run_fma_gemv_benchmark<fma_gemv_config<2, 2048, 1>>(32768, 32768, ncu);

    std::cout << "\n========== FMA GEMV: RPW=2, K_CHUNK=2048, K_SPLITS=2 ==========\n";
    run_fma_gemv_benchmark<fma_gemv_config<2, 2048, 2>>(4096, 4096, ncu);
    run_fma_gemv_benchmark<fma_gemv_config<2, 2048, 2>>(8192, 8192, ncu);
    run_fma_gemv_benchmark<fma_gemv_config<2, 2048, 2>>(16384, 16384, ncu);
    run_fma_gemv_benchmark<fma_gemv_config<2, 2048, 2>>(32768, 32768, ncu);

    std::cout << "\n========== FMA GEMV: RPW=2, K_CHUNK=2048, K_SPLITS=4 ==========\n";
    run_fma_gemv_benchmark<fma_gemv_config<2, 2048, 4>>(4096, 4096, ncu);
    run_fma_gemv_benchmark<fma_gemv_config<2, 2048, 4>>(8192, 8192, ncu);
    run_fma_gemv_benchmark<fma_gemv_config<2, 2048, 4>>(16384, 16384, ncu);
    run_fma_gemv_benchmark<fma_gemv_config<2, 2048, 4>>(32768, 32768, ncu);

    std::cout << "\n========== FMA GEMV: RPW=2, K_CHUNK=2048, K_SPLITS=8 ==========\n";
    run_fma_gemv_benchmark<fma_gemv_config<2, 2048, 8>>(4096, 4096, ncu);
    run_fma_gemv_benchmark<fma_gemv_config<2, 2048, 8>>(8192, 8192, ncu);
    run_fma_gemv_benchmark<fma_gemv_config<2, 2048, 8>>(16384, 16384, ncu);
    run_fma_gemv_benchmark<fma_gemv_config<2, 2048, 8>>(32768, 32768, ncu);

    return 0;
}
