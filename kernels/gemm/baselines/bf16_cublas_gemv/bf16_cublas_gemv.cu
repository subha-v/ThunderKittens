/***************************************************************************************************
 * cuBLAS BF16 GEMV Benchmark
 *
 * y = A * x (no alpha/beta scaling)
 * A: RowMajor (M x K), x: vector (K), y: vector (M)
 * Accumulator: FP32, Output: BF16
 **************************************************************************************************/

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "../../common.cuh"

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error in " << __FILE__ << " line " << __LINE__ << ": " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

static constexpr int warmup_iters = 500;
static constexpr int profiling_iters = 100;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Reference GEMV for correctness check
///////////////////////////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////////////////////////
// cuBLAS GEMV: y = A * x
// Uses cublasGemmEx with N=1 since there's no dedicated BF16 GEMV
// A: RowMajor (M x K), x: vector (K), y: vector (M)
///////////////////////////////////////////////////////////////////////////////////////////////////

void cublas_gemv(
    cublasHandle_t handle,
    __nv_bfloat16 const* A,
    __nv_bfloat16 const* x,
    __nv_bfloat16* y,
    int M, int K) {

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Treat GEMV as GEMM with N=1
    // y = A * x where A is MxK, x is Kx1, y is Mx1
    // In cuBLAS col-major: y' = A'^T * x' where A' is KxM (col-major view of row-major MxK)
    // So we use CUBLAS_OP_T on A
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_T,    // Transpose A (KxM col-major -> MxK effective)
        CUBLAS_OP_N,    // x as-is
        M, 1, K,        // M x 1 output
        &alpha,
        A, CUDA_R_16BF, K,   // A: RowMajor MxK = ColMajor KxM, ld = K
        x, CUDA_R_16BF, K,   // x: vector K, ld = K
        &beta,
        y, CUDA_R_16BF, M,   // y: vector M, ld = M
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Benchmark function
///////////////////////////////////////////////////////////////////////////////////////////////////

void benchmark(int M, int K) {
    // Cooldown between configurations
    sleep_ms(500);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "Problem size: M=" << M << ", K=" << K << std::endl;

    // L2 cache eviction - multiple buffer groups
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = 2 * (size_t(M) * K + K + M);  // bytes for bf16
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    // Allocate buffer groups
    std::vector<__nv_bfloat16*> blocks_A(arg_group_count);
    std::vector<__nv_bfloat16*> blocks_x(arg_group_count);
    std::vector<__nv_bfloat16*> blocks_y(arg_group_count);
    __nv_bfloat16* block_y_ref;

    size_t size_A = size_t(M) * K;
    size_t size_x = size_t(K);
    size_t size_y = size_t(M);

    CHECK_CUDA(cudaMalloc(&block_y_ref, size_y * sizeof(__nv_bfloat16)));

    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; ++i) {
        CHECK_CUDA(cudaMalloc(&blocks_A[i], size_A * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&blocks_x[i], size_x * sizeof(__nv_bfloat16)));
        CHECK_CUDA(cudaMalloc(&blocks_y[i], size_y * sizeof(__nv_bfloat16)));

        // Initialize with uniform random [-1, 1]
        fill<__nv_bfloat16, FillMode::RANDOM>(blocks_A[i], size_A, seed + i * 100, -1.0f, 1.0f);
        fill<__nv_bfloat16, FillMode::RANDOM>(blocks_x[i], size_x, seed + i * 100 + 1, -1.0f, 1.0f);
        fill<__nv_bfloat16, FillMode::CONSTANT>(blocks_y[i], size_y, 0.0f);
    }
    fill<__nv_bfloat16, FillMode::CONSTANT>(block_y_ref, size_y, 0.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Compute reference GEMV
    reference_gemv<__nv_bfloat16>(block_y_ref, blocks_A[0], blocks_x[0], M, K);
    CHECK_CUDA(cudaDeviceSynchronize());

    // cuBLAS Benchmark
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        int idx = i % arg_group_count;
        cublas_gemv(handle, blocks_A[idx], blocks_x[idx], blocks_y[idx], M, K);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < profiling_iters; ++i) {
        int idx = i % arg_group_count;
        cublas_gemv(handle, blocks_A[idx], blocks_x[idx], blocks_y[idx], M, K);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    double runtime_us = static_cast<double>(milliseconds) * 1000.0 / profiling_iters;
    double bytes = double(2) * (double(M) * K + K + M);  // bf16 = 2 bytes
    double gb_per_sec = (bytes / runtime_us) / 1e3;  // GB/s
    double gflops = (double(2) * M * K / runtime_us) / 1e3;  // GFLOPs

    std::cout << "Average runtime: " << runtime_us << " us" << std::endl;
    std::cout << "Achieved bandwidth: " << gb_per_sec << " GB/s" << std::endl;
    std::cout << "Achieved performance: " << gflops << " GFLOPs" << std::endl;

    // Verify correctness
    fill<__nv_bfloat16, FillMode::CONSTANT>(blocks_y[0], size_y, 0.0f);
    cublas_gemv(handle, blocks_A[0], blocks_x[0], blocks_y[0], M, K);
    CHECK_CUDA(cudaDeviceSynchronize());
    check_correctness(blocks_y[0], block_y_ref, size_y);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream));

    for (int i = 0; i < arg_group_count; ++i) {
        CHECK_CUDA(cudaFree(blocks_A[i]));
        CHECK_CUDA(cudaFree(blocks_x[i]));
        CHECK_CUDA(cudaFree(blocks_y[i]));
    }
    CHECK_CUDA(cudaFree(block_y_ref));

    CHECK_CUBLAS(cublasDestroy(handle));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int main() {
    std::cout << "cuBLAS BF16 GEMV Profiler" << std::endl;
    std::cout << "y = A * x, A: RowMajor (MxK), x: vector (K), y: vector (M)" << std::endl;
    std::cout << "Accumulator: FP32, Output: BF16" << std::endl;
    std::cout << "Warmup: " << warmup_iters << ", Profiling: " << profiling_iters << std::endl;

    benchmark(4096, 4096);
    benchmark(8192, 8192);
    benchmark(16384, 16384);
    benchmark(32768, 32768);

    return 0;
}
