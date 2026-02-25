#include "kittens.cuh"
#include "../common.cuh"

using namespace kittens;

/*
 * Dual-strategy FMA GEMV: y = A * x
 *
 * Strategy 1: Block-per-row (small M)
 *   - One block (256 threads) per output row
 *   - All 256 threads collaborate on the K-dimension dot product
 *   - Two-level reduction: warp shuffle (32→1) + shared memory + warp shuffle (8→1)
 *   - M=4096 → 4096 blocks → 21 blocks/SM → excellent occupancy
 *
 * Strategy 2: Warp-per-row (large M)
 *   - Each warp handles ROWS_PER_WARP rows, 8 warps per block → 16 rows/block
 *   - Already enough blocks at large M for full occupancy
 *   - Avoids per-block reduction overhead
 */

// ============================================================================
// Strategy 1: Block-per-row kernel (for small M)
// ============================================================================

template <int _K_CHUNK>
struct block_per_row_config {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int K_CHUNK = _K_CHUNK;
    static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;  // 8
    static constexpr int ROWS_PER_BLOCK = 1;
};

template <typename Config>
__global__ void fma_gemv_block_per_row_kernel(
    bf16* __restrict__ y,
    const bf16* __restrict__ A,
    const bf16* __restrict__ x,
    int M, int K
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    __shared__ bf16 x_smem[Config::K_CHUNK];
    __shared__ float warp_sums[Config::WARPS_PER_BLOCK];

    float acc = 0.0f;
    const bf16* A_row = A + row * K;

    constexpr int elems_per_thread = 8;
    constexpr int elems_per_pass = Config::BLOCK_SIZE * elems_per_thread;  // 2048

    for (int k_base = 0; k_base < K; k_base += Config::K_CHUNK) {
        const int k_remaining = min(Config::K_CHUNK, K - k_base);

        // Collaborative x load: 256 threads × 8 elements = 2048 per pass
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

        // Each thread processes elements at stride 256 (32 lanes × 8 elements)
        // But with block-per-row, ALL 256 threads work on the same row
        // Thread t handles elements: t*8, t*8 + 2048, t*8 + 4096, ...
        for (int k = threadIdx.x * elems_per_thread; k < k_remaining; k += elems_per_pass) {
            float4 x_vec = *reinterpret_cast<const float4*>(&x_smem[k]);
            const bf16* x_ptr = reinterpret_cast<const bf16*>(&x_vec);

            float4 a_vec = *reinterpret_cast<const float4*>(&A_row[k_base + k]);
            const bf16* a_ptr = reinterpret_cast<const bf16*>(&a_vec);

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                acc = fmaf(__bfloat162float(a_ptr[i]), __bfloat162float(x_ptr[i]), acc);
            }
        }
        __syncthreads();
    }

    // Two-level reduction
    // Stage 1: Warp shuffle (32 → 1)
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Stage 2: Block reduction via shared memory (8 warps → 1)
    if (lane_id == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    // First warp reduces the 8 warp sums
    if (warp_id == 0) {
        float val = (lane_id < Config::WARPS_PER_BLOCK) ? warp_sums[lane_id] : 0.0f;

        #pragma unroll
        for (int offset = 4; offset >= 1; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (lane_id == 0) {
            y[row] = __float2bfloat16(val);
        }
    }
}

// ============================================================================
// Strategy 2: Warp-per-row kernel (for large M)
// ============================================================================

template <int _ROWS_PER_WARP, int _K_CHUNK>
struct warp_per_row_config {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int ROWS_PER_WARP = _ROWS_PER_WARP;
    static constexpr int K_CHUNK = _K_CHUNK;
    static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;  // 8
    static constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;
};

template <typename Config>
__global__ void fma_gemv_warp_per_row_kernel(
    bf16* __restrict__ y,
    const bf16* __restrict__ A,
    const bf16* __restrict__ x,
    int M, int K
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int base_row = blockIdx.x * Config::ROWS_PER_BLOCK + warp_id * Config::ROWS_PER_WARP;

    if (base_row >= M) return;

    __shared__ bf16 x_smem[Config::K_CHUNK];

    float acc[Config::ROWS_PER_WARP] = {0.0f};

    const bf16* A_row_ptrs[Config::ROWS_PER_WARP];
    #pragma unroll
    for (int r = 0; r < Config::ROWS_PER_WARP; r++) {
        A_row_ptrs[r] = A + (base_row + r) * K;
    }

    constexpr int elems_per_thread = 8;
    constexpr int elems_per_pass = Config::BLOCK_SIZE * elems_per_thread;
    constexpr int lane_stride = 32 * elems_per_thread;  // 256

    for (int k_base = 0; k_base < K; k_base += Config::K_CHUNK) {
        const int k_remaining = min(Config::K_CHUNK, K - k_base);

        // Collaborative x load
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

        for (int k = lane_id * elems_per_thread; k < k_remaining; k += lane_stride) {
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

    if (lane_id == 0) {
        #pragma unroll
        for (int r = 0; r < Config::ROWS_PER_WARP; r++) {
            if (base_row + r < M) {
                y[base_row + r] = __float2bfloat16(acc[r]);
            }
        }
    }
}

// ============================================================================
// Reference and benchmark infrastructure
// ============================================================================

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

// Benchmark for block-per-row kernel
template <typename Config>
__host__ double run_block_per_row_benchmark(size_t M, size_t K, bool ncu = false) {
    std::cout << "--------------------  M=" << M << " K=" << K << "  --------------------\n";
    std::cout << "Strategy: BLOCK-PER-ROW, K_CHUNK=" << Config::K_CHUNK
              << " BLOCK_SIZE=" << Config::BLOCK_SIZE << "\n";

    sleep_ms(500);

    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = 2 * (size_t(M) * K + K + M);
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    std::vector<bf16*> d_A(arg_group_count);
    std::vector<bf16*> d_x(arg_group_count);
    std::vector<bf16*> d_y(arg_group_count);
    bf16* d_y_ref;

    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaMalloc(&d_A[i], M * K * sizeof(bf16)));
        CUDACHECK(cudaMalloc(&d_x[i], K * sizeof(bf16)));
        CUDACHECK(cudaMalloc(&d_y[i], M * sizeof(bf16)));
    }
    CUDACHECK(cudaMalloc(&d_y_ref, M * sizeof(bf16)));

    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<bf16, FillMode::RANDOM>(d_A[i], M * K, seed + i * 100, -1.0f, 1.0f);
        fill<bf16, FillMode::RANDOM>(d_x[i], K, seed + i * 100 + 1, -1.0f, 1.0f);
        fill<bf16, FillMode::CONSTANT>(d_y[i], M, 0.0f);
    }
    fill<bf16, FillMode::CONSTANT>(d_y_ref, M, 0.0f);
    CUDACHECK(cudaDeviceSynchronize());

    reference_gemv<bf16>(d_y_ref, d_A[0], d_x[0], M, K);
    CUDACHECK(cudaDeviceSynchronize());

    dim3 block(Config::BLOCK_SIZE);
    dim3 grid(M);  // One block per row

    std::cout << "Grid: " << M << " blocks (" << (float)M / 192.0f << " blocks/SM)\n";

    int num_warmups = ncu ? 0 : 500;
    int num_iters = ncu ? 1 : 100;

    for (int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        fma_gemv_block_per_row_kernel<Config><<<grid, block>>>(d_y[idx], d_A[idx], d_x[idx], M, K);
    }
    CUDACHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));

    for (int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        fma_gemv_block_per_row_kernel<Config><<<grid, block>>>(d_y[idx], d_A[idx], d_x[idx], M, K);
    }

    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0 / num_iters;
    double bytes = double(2) * (double(M) * K + K + M);
    double gb_per_sec = (bytes / microseconds) / 1e3;
    double gflops = (double(2) * M * K / microseconds) / 1e3;

    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Achieved bandwidth: " << gb_per_sec << " GB/s\n";
    std::cout << "Achieved performance: " << gflops << " GFLOPs\n";

    check_correctness(d_y[0], d_y_ref, M);

    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_x[i]);
        cudaFree(d_y[i]);
    }
    cudaFree(d_y_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return gb_per_sec;
}

// Benchmark for warp-per-row kernel
template <typename Config>
__host__ double run_warp_per_row_benchmark(size_t M, size_t K, bool ncu = false) {
    std::cout << "--------------------  M=" << M << " K=" << K << "  --------------------\n";
    std::cout << "Strategy: WARP-PER-ROW, RPW=" << Config::ROWS_PER_WARP
              << " K_CHUNK=" << Config::K_CHUNK
              << " BLOCK_SIZE=" << Config::BLOCK_SIZE << "\n";

    sleep_ms(500);

    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = 2 * (size_t(M) * K + K + M);
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    std::vector<bf16*> d_A(arg_group_count);
    std::vector<bf16*> d_x(arg_group_count);
    std::vector<bf16*> d_y(arg_group_count);
    bf16* d_y_ref;

    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaMalloc(&d_A[i], M * K * sizeof(bf16)));
        CUDACHECK(cudaMalloc(&d_x[i], K * sizeof(bf16)));
        CUDACHECK(cudaMalloc(&d_y[i], M * sizeof(bf16)));
    }
    CUDACHECK(cudaMalloc(&d_y_ref, M * sizeof(bf16)));

    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<bf16, FillMode::RANDOM>(d_A[i], M * K, seed + i * 100, -1.0f, 1.0f);
        fill<bf16, FillMode::RANDOM>(d_x[i], K, seed + i * 100 + 1, -1.0f, 1.0f);
        fill<bf16, FillMode::CONSTANT>(d_y[i], M, 0.0f);
    }
    fill<bf16, FillMode::CONSTANT>(d_y_ref, M, 0.0f);
    CUDACHECK(cudaDeviceSynchronize());

    reference_gemv<bf16>(d_y_ref, d_A[0], d_x[0], M, K);
    CUDACHECK(cudaDeviceSynchronize());

    dim3 block(Config::BLOCK_SIZE);
    dim3 grid((M + Config::ROWS_PER_BLOCK - 1) / Config::ROWS_PER_BLOCK);

    int num_blocks = (M + Config::ROWS_PER_BLOCK - 1) / Config::ROWS_PER_BLOCK;
    std::cout << "Grid: " << num_blocks << " blocks (" << (float)num_blocks / 192.0f << " blocks/SM)\n";

    int num_warmups = ncu ? 0 : 500;
    int num_iters = ncu ? 1 : 100;

    for (int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        fma_gemv_warp_per_row_kernel<Config><<<grid, block>>>(d_y[idx], d_A[idx], d_x[idx], M, K);
    }
    CUDACHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));

    for (int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        fma_gemv_warp_per_row_kernel<Config><<<grid, block>>>(d_y[idx], d_A[idx], d_x[idx], M, K);
    }

    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0 / num_iters;
    double bytes = double(2) * (double(M) * K + K + M);
    double gb_per_sec = (bytes / microseconds) / 1e3;
    double gflops = (double(2) * M * K / microseconds) / 1e3;

    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Achieved bandwidth: " << gb_per_sec << " GB/s\n";
    std::cout << "Achieved performance: " << gflops << " GFLOPs\n";

    check_correctness(d_y[0], d_y_ref, M);

    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_x[i]);
        cudaFree(d_y[i]);
    }
    cudaFree(d_y_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return gb_per_sec;
}

__host__ int main() {
    bool ncu = false;

    // ---- Block-per-row: best for small M (high block count) ----
    std::cout << "\n========== BLOCK-PER-ROW, K_CHUNK=2048 ==========\n";
    run_block_per_row_benchmark<block_per_row_config<2048>>(4096, 4096, ncu);
    run_block_per_row_benchmark<block_per_row_config<2048>>(8192, 8192, ncu);
    run_block_per_row_benchmark<block_per_row_config<2048>>(16384, 16384, ncu);
    run_block_per_row_benchmark<block_per_row_config<2048>>(32768, 32768, ncu);

    // ---- Warp-per-row RPW=2: best for large M (efficient per-block) ----
    std::cout << "\n========== WARP-PER-ROW, RPW=2, K_CHUNK=2048 ==========\n";
    run_warp_per_row_benchmark<warp_per_row_config<2, 2048>>(4096, 4096, ncu);
    run_warp_per_row_benchmark<warp_per_row_config<2, 2048>>(8192, 8192, ncu);
    run_warp_per_row_benchmark<warp_per_row_config<2, 2048>>(16384, 16384, ncu);
    run_warp_per_row_benchmark<warp_per_row_config<2, 2048>>(32768, 32768, ncu);

    return 0;
}
