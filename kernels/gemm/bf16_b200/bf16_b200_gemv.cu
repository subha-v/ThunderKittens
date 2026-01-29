/**
 * BF16 GEMV Kernel for B200
 *
 * Computes y = A * x where:
 *   - A is M×K (bf16, row-major)
 *   - x is K×1 (bf16)
 *   - y is M×1 (bf16)
 *
 * Approach: Use WGMMA tensor cores by treating GEMV as degenerate GEMM.
 * x is stored as a tile where row 0 = x values, rows 1..N-1 = 0.
 * We compute D = A * x_tile^T using mm_ABt/mma_ABt, then extract column 0 as y.
 */

#include "kittens.cuh"
#include "../common.cuh"

using namespace kittens;

// Configuration for GEMV kernel
template <int _Mb, int _Kb, int _PIPE_DEPTH>
struct gemv_config {
    static constexpr int Mb = _Mb;           // Rows per block (output elements per block)
    static constexpr int Nb = 16;            // Fixed at 16 (minimum tile width for tensor cores)
    static constexpr int Kb = _Kb;           // K tile size
    static constexpr int PIPE_DEPTH = _PIPE_DEPTH;

    static constexpr int NUM_CONSUMERS = 1;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

// Global memory descriptors for GEMV
template <typename C>
struct gemv_globals {
    // A tile: Mb x Kb chunk of the matrix
    using a_tile = st_bf<C::Mb, C::Kb>;
    // x tile: Nb x Kb - row 0 holds x values, rest are zeros
    // Note: For ABt operation, B is Nb x Kb, transposed gives Kb x Nb
    using x_tile = st_bf<C::Nb, C::Kb>;
    // y tile: Mb x Nb output (we only use column 0)
    using y_tile = st_bf<C::Mb, C::Nb>;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    using x_gl = gl<bf16, 1, 1, -1, -1, x_tile>;
    using y_gl = gl<bf16, 1, 1, -1, -1, y_tile>;

    a_gl a;  // Matrix A: M x K
    x_gl x;  // Vector x stored as tiles: (K/Kb) x 1 tiles, each Nb x Kb
    y_gl y;  // Output y stored as tiles: (M/Mb) x 1 tiles, each Mb x Nb

    int M, K;

    __host__ __inline__ dim3 grid() { return dim3(M / C::Mb); }
    __host__ __inline__ dim3 block() { return dim3(C::NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() {
        constexpr size_t mem = sizeof(a_tile) * C::PIPE_DEPTH +
                               sizeof(x_tile) * C::PIPE_DEPTH +
                               sizeof(y_tile) + 1024;
        static_assert(mem <= MAX_SHARED_MEMORY - 1024);
        return mem;
    }
};

template <typename C>
__launch_bounds__(C::NUM_THREADS, 1)
__global__ void gemv_kernel(const __grid_constant__ gemv_globals<C> g) {
    using G = gemv_globals<C>;

    // TMA prefetch
    if (threadIdx.x == 0) {
        g.a.template prefetch_tma<typename G::a_tile>();
        g.x.template prefetch_tma<typename G::x_tile>();
        g.y.template prefetch_tma<typename G::y_tile>();
    }

    const int row_block = blockIdx.x;  // Which Mb-row block we're computing
    const int iters_per_task = g.K / C::Kb;  // Number of K iterations

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    typename G::a_tile (&a_smem)[C::PIPE_DEPTH] = al.allocate<typename G::a_tile, C::PIPE_DEPTH>();
    typename G::x_tile (&x_smem)[C::PIPE_DEPTH] = al.allocate<typename G::x_tile, C::PIPE_DEPTH>();
    typename G::y_tile (&y_smem)                = al.allocate<typename G::y_tile>();

    // Tensor memory allocator for accumulator
    tensor_allocator<1, 1> tm_alloc{};
    using d_tt_t = tt<float, C::Mb, C::Nb>;

    __shared__ semaphore inputs_arrived[C::PIPE_DEPTH], inputs_finished[C::PIPE_DEPTH];
    __shared__ semaphore outputs_arrived, outputs_finished;
    uint32_t bitfield = 0xFFFF0000;

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < C::PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        init_semaphore(outputs_finished, 0, 1);
    }
    __syncthreads();

    int warpgroupid = warpgroup::groupid();

    if (warpgroupid == C::NUM_CONSUMERS) {
        // Producer warpgroup
        warpgroup::decrease_registers<56>();

        if (warpgroup::warpid() == 3 && warp::laneid() == 0) {
            // TMA loader thread
            int input_ring = 0;

            for (int idx = 0; idx < iters_per_task; idx++) {
                wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                update_phasebit<1>(bitfield, input_ring);

                tma::expect(inputs_arrived[input_ring], a_smem[0], x_smem[0]);
                tma::load_async(a_smem[input_ring], g.a, {row_block, idx}, inputs_arrived[input_ring]);
                tma::load_async(x_smem[input_ring], g.x, {0, idx}, inputs_arrived[input_ring]);

                input_ring = ring_advance<C::PIPE_DEPTH>(input_ring);
            }

            // Signal completion
            for (int idx = 0; idx < C::PIPE_DEPTH; idx++) {
                wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                input_ring = ring_advance<C::PIPE_DEPTH>(input_ring);
            }
            arrive(outputs_arrived);
        }
        else if (warpgroup::warpid() == 0 && warp::laneid() == 0) {
            // MMA issuer thread
            d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(0);
            int input_ring = 0;

            // Wait for outputs_finished to ensure tensor memory is ready
            wait(outputs_finished, 1);

            // First iteration: mm (no accumulate)
            wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
            update_phasebit<0>(bitfield, input_ring);
            mm_ABt(d_tt, a_smem[input_ring], x_smem[input_ring], inputs_finished[input_ring]);
            input_ring = ring_advance<C::PIPE_DEPTH>(input_ring);

            // Remaining iterations: mma (accumulate)
            for (int idx = 1; idx < iters_per_task; idx++) {
                wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                update_phasebit<0>(bitfield, input_ring);
                mma_ABt(d_tt, a_smem[input_ring], x_smem[input_ring], inputs_finished[input_ring]);
                input_ring = ring_advance<C::PIPE_DEPTH>(input_ring);
            }
        }
    }
    else {
        // Consumer warpgroup (epilogue)
        warpgroup::increase_registers<224>();

        d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(0);

        // Wait for MMA to complete
        wait(outputs_arrived, 0);

        // Load result from tensor memory
        rt_bf<C::Mb/4, C::Nb> d_reg;
        warpgroup::load_async(d_reg, d_tt);
        tensor_load_wait();

        // Signal that tensor memory is free
        warpgroup::sync(warpgroupid + 1);
        if (warpgroup::laneid() == 0) arrive(outputs_finished);

        // Store to shared memory
        warpgroup::store(y_smem, d_reg);
        warpgroup::sync(warpgroupid + 1);

        // TMA store to global memory
        if (warpgroup::laneid() == 0) {
            tma::store_async(g.y, y_smem, {row_block, 0});
        }
        tma::store_async_read_wait();
    }

    __syncthreads();
}

// Reference GEMV: y = A * x
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

// Prepare x vector as tiles (row 0 = x values, rest = 0)
template <typename C>
__global__ void prepare_x_tiles(bf16* x_tiles, const bf16* x, int K) {
    // x_tiles layout: (K/Kb) tiles, each Nb x Kb
    // Row 0 of each tile = x[k*Kb : (k+1)*Kb]
    // Rows 1..Nb-1 = 0

    int tile_idx = blockIdx.x;  // Which K tile
    int col = threadIdx.x;      // Column within tile (0..Kb-1)

    if (col < C::Kb) {
        int k = tile_idx * C::Kb + col;

        // Row 0: copy x value
        if (k < K) {
            x_tiles[tile_idx * (C::Nb * C::Kb) + col] = x[k];
        }

        // Rows 1..Nb-1: zeros
        for (int row = 1; row < C::Nb; row++) {
            x_tiles[tile_idx * (C::Nb * C::Kb) + row * C::Kb + col] =
                kittens::base_types::convertor<bf16, float>::convert(0.0f);
        }
    }
}

// Extract y from output tiles (column 0)
template <typename C>
__global__ void extract_y(bf16* y, const bf16* y_tiles, int M) {
    // y_tiles layout: (M/Mb) tiles, each Mb x Nb
    // Column 0 of each tile = y[m*Mb : (m+1)*Mb]

    int tile_idx = blockIdx.x;  // Which M tile
    int row = threadIdx.x;      // Row within tile (0..Mb-1)

    if (row < C::Mb) {
        int m = tile_idx * C::Mb + row;
        if (m < M) {
            // Extract column 0
            y[m] = y_tiles[tile_idx * (C::Mb * C::Nb) + row * C::Nb + 0];
        }
    }
}

// Check correctness for GEMV
template <typename T>
void check_gemv_correctness(const T* d_out, const T* d_ref, size_t count) {
    std::vector<T> h_out(count);
    std::vector<T> h_ref(count);

    cudaMemcpy(h_out.data(), d_out, count * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref.data(), d_ref, count * sizeof(T), cudaMemcpyDeviceToHost);

    double abs_sum = 0.0, abs_max = 0.0;
    double err_sum = 0.0, err_max = 0.0;

    for (size_t i = 0; i < count; i++) {
        float val = kittens::base_types::convertor<float, T>::convert(h_out[i]);
        float ref = kittens::base_types::convertor<float, T>::convert(h_ref[i]);
        float err = std::abs(val - ref);

        abs_sum += std::abs(val);
        abs_max = std::max(abs_max, (double)std::abs(val));
        err_sum += err;
        err_max = std::max(err_max, (double)err);
    }

    double abs_mean = abs_sum / count;
    double err_mean = err_sum / count;

    std::cout << "abs mean: " << std::setw(12) << abs_mean << std::endl;
    std::cout << "abs max:  " << std::setw(12) << abs_max << std::endl;
    std::cout << "err mean: " << std::setw(12) << err_mean << std::endl;
    std::cout << "err max:  " << std::setw(12) << err_max << std::endl;
}

template <typename C>
__host__ double run_gemv_benchmark(size_t M, size_t K, bool ncu = false) {
    std::cout << "--------------------  M=" << M << " K=" << K << "  --------------------\n";
    std::cout << "Template: Mb=" << C::Mb << " Nb=" << C::Nb << " Kb=" << C::Kb
              << " PIPE_DEPTH=" << C::PIPE_DEPTH << "\n";

    // Cooldown between configurations
    sleep_ms(500);

    // L2 cache eviction - multiple buffer groups
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0);
    const size_t arg_size = 2 * (size_t(M) * K + K + M);  // bf16 bytes
    const size_t ideal_arg_size = size_t(l2_cache_size) * 3;
    const int arg_group_count = (arg_size > ideal_arg_size) ? 1 : int(ideal_arg_size / arg_size) + 1;

    // Number of K tiles
    int num_k_tiles = K / C::Kb;
    // Number of M tiles
    int num_m_tiles = M / C::Mb;

    // Allocate device memory
    std::vector<bf16*> d_A(arg_group_count);
    std::vector<bf16*> d_x(arg_group_count);        // Original x vector
    std::vector<bf16*> d_x_tiles(arg_group_count);  // x prepared as tiles
    std::vector<bf16*> d_y_tiles(arg_group_count);  // y as tiles (kernel output)
    std::vector<bf16*> d_y(arg_group_count);        // y extracted from tiles
    bf16* d_y_ref;

    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaMalloc(&d_A[i], M * K * sizeof(bf16)));
        CUDACHECK(cudaMalloc(&d_x[i], K * sizeof(bf16)));
        CUDACHECK(cudaMalloc(&d_x_tiles[i], num_k_tiles * C::Nb * C::Kb * sizeof(bf16)));
        CUDACHECK(cudaMalloc(&d_y_tiles[i], num_m_tiles * C::Mb * C::Nb * sizeof(bf16)));
        CUDACHECK(cudaMalloc(&d_y[i], M * sizeof(bf16)));
    }
    CUDACHECK(cudaMalloc(&d_y_ref, M * sizeof(bf16)));
    std::cout << "Allocated device memory" << std::endl;

    // Initialize matrices with random values
    uint64_t seed = 2024;
    for (int i = 0; i < arg_group_count; i++) {
        fill<bf16, FillMode::RANDOM>(d_A[i], M * K, seed + i * 100, -1.0f, 1.0f);
        fill<bf16, FillMode::RANDOM>(d_x[i], K, seed + i * 100 + 1, -1.0f, 1.0f);
        fill<bf16, FillMode::CONSTANT>(d_y[i], M, 0.0f);
        fill<bf16, FillMode::CONSTANT>(d_y_tiles[i], num_m_tiles * C::Mb * C::Nb, 0.0f);

        // Prepare x tiles
        prepare_x_tiles<C><<<num_k_tiles, C::Kb>>>(d_x_tiles[i], d_x[i], K);
    }
    fill<bf16, FillMode::CONSTANT>(d_y_ref, M, 0.0f);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Initialized matrices on device" << std::endl;

    // Compute reference GEMV
    reference_gemv<bf16>(d_y_ref, d_A[0], d_x[0], M, K);
    CUDACHECK(cudaDeviceSynchronize());
    std::cout << "Computed reference GEMV on device" << std::endl;

    // Prepare kernel inputs
    std::vector<gemv_globals<C>> g;
    for (int i = 0; i < arg_group_count; i++) {
        typename gemv_globals<C>::a_gl Ag{d_A[i], nullptr, nullptr, M, K};
        typename gemv_globals<C>::x_gl Xg{d_x_tiles[i], nullptr, nullptr, C::Nb, num_k_tiles * C::Kb};
        typename gemv_globals<C>::y_gl Yg{d_y_tiles[i], nullptr, nullptr, M, C::Nb};
        g.push_back(gemv_globals<C>{Ag, Xg, Yg, (int)M, (int)K});
    }

    // Set kernel attributes
    CUDACHECK(cudaFuncSetAttribute(gemv_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   g[0].dynamic_shared_memory()));

    // Number of iterations
    int num_warmups = ncu ? 0 : 5;
    int num_iters = ncu ? 1 : 10;

    // Warmup
    for (int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        gemv_kernel<C><<<g[idx].grid(), g[idx].block(), g[idx].dynamic_shared_memory()>>>(g[idx]);
    }

    // Benchmark
    cudaEvent_t start, stop;
    CUDACHECK(cudaEventCreate(&start));
    CUDACHECK(cudaEventCreate(&stop));
    CUDACHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        gemv_kernel<C><<<g[idx].grid(), g[idx].block(), g[idx].dynamic_shared_memory()>>>(g[idx]);
    }
    CUDACHECK(cudaEventRecord(stop));
    CUDACHECK(cudaEventSynchronize(stop));

    // Calculate duration and bandwidth
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0 / num_iters;
    double bytes = double(2) * (double(M) * K + K + M);  // bf16 = 2 bytes
    double gb_per_sec = (bytes / microseconds) / 1e3;  // GB/s
    double gflops = (double(2) * M * K / microseconds) / 1e3;  // GFLOPs

    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Achieved bandwidth: " << gb_per_sec << " GB/s\n";
    std::cout << "Achieved performance: " << gflops << " GFLOPs\n";

    // Extract y from tiles and verify correctness
    extract_y<C><<<num_m_tiles, C::Mb>>>(d_y[0], d_y_tiles[0], M);
    CUDACHECK(cudaDeviceSynchronize());

    check_gemv_correctness(d_y[0], d_y_ref, M);

    // Cleanup
    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_x[i]);
        cudaFree(d_x_tiles[i]);
        cudaFree(d_y_tiles[i]);
        cudaFree(d_y[i]);
    }
    cudaFree(d_y_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return gb_per_sec;
}

__host__ int main() {
    int N;
    bool ncu = false;

    // Template parameters: Mb, Kb, PIPE_DEPTH
    N = 4096;
    run_gemv_benchmark<gemv_config<128, 128, 4>>(N, N, ncu);

    N = 8192;
    run_gemv_benchmark<gemv_config<128, 128, 4>>(N, N, ncu);

    N = 16384;
    run_gemv_benchmark<gemv_config<128, 128, 4>>(N, N, ncu);

    N = 32768;
    run_gemv_benchmark<gemv_config<128, 64, 4>>(N, N, ncu);

    return 0;
}
