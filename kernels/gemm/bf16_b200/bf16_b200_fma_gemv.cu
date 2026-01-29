#include "kittens.cuh"
#include "../common.cuh"

using namespace kittens;

/*
 * FMA-based GEMV: y = A * x
 *
 * This kernel uses CUDA cores (FMA instructions) instead of tensor cores.
 * Benefits:
 *   - No need to pad x to 16 rows of zeros
 *   - Simpler memory access pattern
 *   - May be faster for small/medium sizes where tensor core setup overhead dominates
 *
 * Strategy:
 *   - Each thread block handles BLOCK_ROWS rows of output
 *   - x vector is loaded into shared memory (reused by all threads)
 *   - Each thread computes one output element by streaming through A and x
 */

// Configuration
template <int _BLOCK_ROWS, int _BLOCK_K>
struct fma_gemv_config {
    static constexpr int BLOCK_ROWS = _BLOCK_ROWS;  // Output rows per block
    static constexpr int BLOCK_K = _BLOCK_K;        // K elements loaded into SMEM per iteration
    static constexpr int NUM_THREADS = BLOCK_ROWS;  // One thread per output row
};

// Kernel: each thread computes one output element
template <typename C>
__global__ void fma_gemv_kernel_v1(
    bf16* __restrict__ y,
    const bf16* __restrict__ A,
    const bf16* __restrict__ x,
    int M, int K
) {
    const int row = blockIdx.x * C::BLOCK_ROWS + threadIdx.x;

    if (row >= M) return;

    // Shared memory for x vector chunk
    __shared__ bf16 x_smem[C::BLOCK_K];

    float acc = 0.0f;

    // Process K in chunks
    for (int k_base = 0; k_base < K; k_base += C::BLOCK_K) {
        // Collaboratively load x chunk into shared memory
        for (int k = threadIdx.x; k < C::BLOCK_K && (k_base + k) < K; k += C::NUM_THREADS) {
            x_smem[k] = x[k_base + k];
        }
        __syncthreads();

        // Compute partial dot product
        const int k_end = min(C::BLOCK_K, K - k_base);
        const bf16* A_row = A + row * K + k_base;

        #pragma unroll 8
        for (int k = 0; k < k_end; k++) {
            float a_val = __bfloat162float(A_row[k]);
            float x_val = __bfloat162float(x_smem[k]);
            acc = fmaf(a_val, x_val, acc);  // FMA: acc += a_val * x_val
        }
        __syncthreads();
    }

    y[row] = __float2bfloat16(acc);
}

// Kernel v2: Multiple rows per thread for better instruction-level parallelism
template <typename C, int ROWS_PER_THREAD = 4>
__global__ void fma_gemv_kernel_v2(
    bf16* __restrict__ y,
    const bf16* __restrict__ A,
    const bf16* __restrict__ x,
    int M, int K
) {
    const int base_row = blockIdx.x * (C::NUM_THREADS * ROWS_PER_THREAD) + threadIdx.x * ROWS_PER_THREAD;

    // Shared memory for x vector chunk
    __shared__ bf16 x_smem[C::BLOCK_K];

    // Accumulators for multiple rows
    float acc[ROWS_PER_THREAD] = {0.0f};

    // Process K in chunks
    for (int k_base = 0; k_base < K; k_base += C::BLOCK_K) {
        // Collaboratively load x chunk into shared memory
        for (int k = threadIdx.x; k < C::BLOCK_K && (k_base + k) < K; k += C::NUM_THREADS) {
            x_smem[k] = x[k_base + k];
        }
        __syncthreads();

        const int k_end = min(C::BLOCK_K, K - k_base);

        // Compute partial dot products for all rows this thread handles
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; r++) {
            const int row = base_row + r;
            if (row < M) {
                const bf16* A_row = A + row * K + k_base;

                #pragma unroll 8
                for (int k = 0; k < k_end; k++) {
                    float a_val = __bfloat162float(A_row[k]);
                    float x_val = __bfloat162float(x_smem[k]);
                    acc[r] = fmaf(a_val, x_val, acc[r]);
                }
            }
        }
        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; r++) {
        const int row = base_row + r;
        if (row < M) {
            y[row] = __float2bfloat16(acc[r]);
        }
    }
}

// Kernel v3: Vectorized loads (load 4 bf16 at once using bf16x2)
template <int BLOCK_SIZE = 256, int K_CHUNK = 1024>
__global__ void fma_gemv_kernel_v3(
    bf16* __restrict__ y,
    const bf16* __restrict__ A,
    const bf16* __restrict__ x,
    int M, int K
) {
    const int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ bf16 x_smem[K_CHUNK];

    float acc = 0.0f;

    for (int k_base = 0; k_base < K; k_base += K_CHUNK) {
        // Load x chunk - vectorized using float (2 bf16s at once)
        const int k_chunk_size = min(K_CHUNK, K - k_base);
        for (int k = threadIdx.x * 2; k < k_chunk_size; k += BLOCK_SIZE * 2) {
            if (k + 1 < k_chunk_size) {
                // Load 2 bf16s as one 32-bit value
                *reinterpret_cast<float*>(&x_smem[k]) =
                    *reinterpret_cast<const float*>(&x[k_base + k]);
            } else if (k < k_chunk_size) {
                x_smem[k] = x[k_base + k];
            }
        }
        __syncthreads();

        if (row < M) {
            const bf16* A_row = A + row * K + k_base;

            // Process 2 elements at a time
            int k = 0;
            for (; k + 1 < k_chunk_size; k += 2) {
                float a0 = __bfloat162float(A_row[k]);
                float a1 = __bfloat162float(A_row[k + 1]);
                float x0 = __bfloat162float(x_smem[k]);
                float x1 = __bfloat162float(x_smem[k + 1]);
                acc = fmaf(a0, x0, acc);
                acc = fmaf(a1, x1, acc);
            }
            // Handle odd element
            if (k < k_chunk_size) {
                float a_val = __bfloat162float(A_row[k]);
                float x_val = __bfloat162float(x_smem[k]);
                acc = fmaf(a_val, x_val, acc);
            }
        }
        __syncthreads();
    }

    if (row < M) {
        y[row] = __float2bfloat16(acc);
    }
}

// Reference GEMV for correctness check
template <typename T>
__global__ void reference_gemv_kernel(T* y, const T* A, const T* x, int M, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            float a_val = __bfloat162float(A[row * K + k]);
            float x_val = __bfloat162float(x[k]);
            acc += a_val * x_val;
        }
        y[row] = __float2bfloat16(acc);
    }
}

template <typename T>
void reference_gemv(T* y, const T* A, const T* x, int M, int K) {
    dim3 block(256);
    dim3 grid((M + 255) / 256);
    reference_gemv_kernel<T><<<grid, block>>>(y, A, x, M, K);
}

template <int KERNEL_VERSION>
__host__ double run_fma_gemv_benchmark(size_t M, size_t K, bool ncu = false) {
    std::cout << "--------------------  M=" << M << " K=" << K << "  --------------------\n";
    std::cout << "FMA GEMV Kernel v" << KERNEL_VERSION << "\n";

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
    bf16* d_y_ref;

    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaMalloc(&d_A[i], M * K * sizeof(bf16)));
        CUDACHECK(cudaMalloc(&d_x[i], K * sizeof(bf16)));
        CUDACHECK(cudaMalloc(&d_y[i], M * sizeof(bf16)));
    }
    CUDACHECK(cudaMalloc(&d_y_ref, M * sizeof(bf16)));
    std::cout << "Allocated device memory" << std::endl;

    // Initialize with random values
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<bf16, FillMode::RANDOM>(d_A[i], M * K, seed + i * 100, -1.0f, 1.0f);
        fill<bf16, FillMode::RANDOM>(d_x[i], K, seed + i * 100 + 1, -1.0f, 1.0f);
        fill<bf16, FillMode::CONSTANT>(d_y[i], M, 0.0f);
    }
    fill<bf16, FillMode::CONSTANT>(d_y_ref, M, 0.0f);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Initialized matrices on device" << std::endl;

    // Compute reference
    reference_gemv<bf16>(d_y_ref, d_A[0], d_x[0], M, K);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Computed reference GEMV on device" << std::endl;

    // Kernel launch parameters
    constexpr int BLOCK_SIZE = 256;
    constexpr int K_CHUNK = 1024;
    dim3 block(BLOCK_SIZE);
    dim3 grid;

    if constexpr (KERNEL_VERSION == 1) {
        grid = dim3((M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    } else if constexpr (KERNEL_VERSION == 2) {
        constexpr int ROWS_PER_THREAD = 4;
        grid = dim3((M + BLOCK_SIZE * ROWS_PER_THREAD - 1) / (BLOCK_SIZE * ROWS_PER_THREAD));
    } else {
        grid = dim3((M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    }

    // Number of iterations
    int num_warmups = ncu ? 0 : 500;
    int num_iters = ncu ? 1 : 100;

    // Warmup
    for (int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        if constexpr (KERNEL_VERSION == 1) {
            using Config = fma_gemv_config<BLOCK_SIZE, K_CHUNK>;
            fma_gemv_kernel_v1<Config><<<grid, block>>>(d_y[idx], d_A[idx], d_x[idx], M, K);
        } else if constexpr (KERNEL_VERSION == 2) {
            using Config = fma_gemv_config<BLOCK_SIZE, K_CHUNK>;
            fma_gemv_kernel_v2<Config, 4><<<grid, block>>>(d_y[idx], d_A[idx], d_x[idx], M, K);
        } else {
            fma_gemv_kernel_v3<BLOCK_SIZE, K_CHUNK><<<grid, block>>>(d_y[idx], d_A[idx], d_x[idx], M, K);
        }
    }
    CUDACHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));

    for (int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        if constexpr (KERNEL_VERSION == 1) {
            using Config = fma_gemv_config<BLOCK_SIZE, K_CHUNK>;
            fma_gemv_kernel_v1<Config><<<grid, block>>>(d_y[idx], d_A[idx], d_x[idx], M, K);
        } else if constexpr (KERNEL_VERSION == 2) {
            using Config = fma_gemv_config<BLOCK_SIZE, K_CHUNK>;
            fma_gemv_kernel_v2<Config, 4><<<grid, block>>>(d_y[idx], d_A[idx], d_x[idx], M, K);
        } else {
            fma_gemv_kernel_v3<BLOCK_SIZE, K_CHUNK><<<grid, block>>>(d_y[idx], d_A[idx], d_x[idx], M, K);
        }
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

    // Verify correctness
    check_correctness(d_y[0], d_y_ref, M);

    // Cleanup
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

    std::cout << "\n========== FMA GEMV Kernel v1 (1 row per thread) ==========\n";
    run_fma_gemv_benchmark<1>(4096, 4096, ncu);
    run_fma_gemv_benchmark<1>(8192, 8192, ncu);
    run_fma_gemv_benchmark<1>(16384, 16384, ncu);
    run_fma_gemv_benchmark<1>(32768, 32768, ncu);

    std::cout << "\n========== FMA GEMV Kernel v2 (4 rows per thread) ==========\n";
    run_fma_gemv_benchmark<2>(4096, 4096, ncu);
    run_fma_gemv_benchmark<2>(8192, 8192, ncu);
    run_fma_gemv_benchmark<2>(16384, 16384, ncu);
    run_fma_gemv_benchmark<2>(32768, 32768, ncu);

    std::cout << "\n========== FMA GEMV Kernel v3 (vectorized loads) ==========\n";
    run_fma_gemv_benchmark<3>(4096, 4096, ncu);
    run_fma_gemv_benchmark<3>(8192, 8192, ncu);
    run_fma_gemv_benchmark<3>(16384, 16384, ncu);
    run_fma_gemv_benchmark<3>(32768, 32768, ncu);

    return 0;
}
