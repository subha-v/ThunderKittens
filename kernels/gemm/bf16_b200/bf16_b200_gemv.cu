#include "kittens.cuh"
#include "../common.cuh"

using namespace kittens;

// configuration for GEMV
template <int _Mb, int _Kb, int _PIPE_DEPTH>
struct gemv_config {
    static constexpr int Mb = _Mb;           
    static constexpr int Nb = 16;           
    static constexpr int Kb = _Kb;        
    static constexpr int PIPE_DEPTH = _PIPE_DEPTH;

    static constexpr int NUM_CONSUMERS = 1;
    static constexpr int NUM_PRODUCERS = 1;
    static constexpr int NUM_WARPS = (NUM_CONSUMERS + NUM_PRODUCERS) * WARPGROUP_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
};

// global memory descriptors
template <typename C>
struct gemv_globals {
    using a_tile = st_bf<C::Mb, C::Kb>;
    using x_tile = st_bf<C::Nb, C::Kb>;  // Nb×Kb tile in shared memory
    // note we only use column 0
    using y_tile = st_bf<C::Mb, C::Nb>;

    using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
    // x is now stored as a 1D vector (K elements), not a 2D tile
    using x_gl = gl<bf16, 1, 1, 1, -1>;  // 1×K vector
    using y_gl = gl<bf16, 1, 1, -1, -1, y_tile>;

    a_gl a;
    x_gl x;   // Now points to raw x vector (K elements)
    y_gl y;

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

    // TMA prefetch (only A and y use TMA now, x is loaded directly)
    if (threadIdx.x == 0) {
        g.a.template prefetch_tma<typename G::a_tile>();
        // x is loaded as raw vector, no TMA prefetch needed
        g.y.template prefetch_tma<typename G::y_tile>();
    }

    const int row_block = blockIdx.x;   
    const int iters_per_task = g.K / C::Kb;   

    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    typename G::a_tile (&a_smem)[C::PIPE_DEPTH] = al.allocate<typename G::a_tile, C::PIPE_DEPTH>();
    typename G::x_tile (&x_smem)[C::PIPE_DEPTH] = al.allocate<typename G::x_tile, C::PIPE_DEPTH>();
    typename G::y_tile (&y_smem)                = al.allocate<typename G::y_tile>();

    // tmem allocator
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
        // producer wg doesnt need these
        warpgroup::decrease_registers<56>();

        if (warpgroup::warpid() == 3 && warp::laneid() == 0) {
            // TMA loader thread - only loads A tiles now
            int input_ring = 0;

            for (int idx = 0; idx < iters_per_task; idx++) {
                wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                update_phasebit<1>(bitfield, input_ring);

                // Only expect and load A tile via TMA
                // x will be loaded separately by the warpgroup
                tma::expect(inputs_arrived[input_ring], a_smem[0]);
                tma::load_async(a_smem[input_ring], g.a, {row_block, idx}, inputs_arrived[input_ring]);

                input_ring = ring_advance<C::PIPE_DEPTH>(input_ring);
            }

            // Signal completion
            for (int idx = 0; idx < C::PIPE_DEPTH; idx++) {
                wait(inputs_finished[input_ring], get_phasebit<1>(bitfield, input_ring));
                input_ring = ring_advance<C::PIPE_DEPTH>(input_ring);
            }
            arrive(outputs_arrived);
        }
        else if (warpgroup::warpid() == 0) {
            // MMA issuer warp - also helps load x vector
            d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(0);
            int input_ring = 0;
            const int lane = warp::laneid();

            // wait to ensure TMEM is ready
            if (lane == 0) wait(outputs_finished, 1);

            for (int idx = 0; idx < iters_per_task; idx++) {
                // Load x vector (Kb elements) into row 0 of x_smem
                // Each thread in warp loads multiple elements
                const int x_offset = idx * C::Kb;
                bf16* x_row0 = reinterpret_cast<bf16*>(&x_smem[input_ring]);

                #pragma unroll
                for (int k = lane; k < C::Kb; k += 32) {
                    x_row0[k] = *(g.x.raw_ptr + x_offset + k);
                }

                // Zero out rows 1 to Nb-1 of x_smem
                #pragma unroll
                for (int row = 1; row < C::Nb; row++) {
                    bf16* x_row = x_row0 + row * C::Kb;
                    #pragma unroll
                    for (int k = lane; k < C::Kb; k += 32) {
                        x_row[k] = __float2bfloat16(0.0f);
                    }
                }
                __syncwarp();

                // Wait for A tile to arrive
                if (lane == 0) {
                    wait(inputs_arrived[input_ring], get_phasebit<0>(bitfield, input_ring));
                    update_phasebit<0>(bitfield, input_ring);
                }
                __syncwarp();

                // Issue MMA
                if (lane == 0) {
                    if (idx == 0) {
                        mm_ABt(d_tt, a_smem[input_ring], x_smem[input_ring], inputs_finished[input_ring]);
                    } else {
                        mma_ABt(d_tt, a_smem[input_ring], x_smem[input_ring], inputs_finished[input_ring]);
                    }
                }
                input_ring = ring_advance<C::PIPE_DEPTH>(input_ring);
            }
        }
    }
    else {
        // consumer epilogue
        warpgroup::increase_registers<224>();

        d_tt_t d_tt = tm_alloc.allocate<d_tt_t>(0);

        // wait for mma
        wait(outputs_arrived, 0);

        // load result from TMEM
        rt_bf<C::Mb/4, C::Nb> d_reg;
        warpgroup::load_async(d_reg, d_tt);
        tensor_load_wait();

        // signal TMEM is free
        warpgroup::sync(warpgroupid + 1);
        if (warpgroup::laneid() == 0) arrive(outputs_finished);

        warpgroup::store(y_smem, d_reg);
        warpgroup::sync(warpgroupid + 1);

        // go to global memory
        if (warpgroup::laneid() == 0) {
            tma::store_async(g.y, y_smem, {row_block, 0});
        }
        tma::store_async_read_wait();
    }

    __syncthreads();
}

// reference GEMV
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

// prepare_x_tiles is no longer needed - x is loaded directly as a vector in the kernel

// do the same for the y vector as the x vector
template <typename C>
__global__ void extract_y(bf16* y, const bf16* y_tiles, int M, int Nb_stride) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;  

    if (m < M) {
        y[m] = y_tiles[m * Nb_stride];
    }
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

    // Number of M tiles
    int num_m_tiles = M / C::Mb;

    // Allocate device memory
    std::vector<bf16*> d_A(arg_group_count);
    std::vector<bf16*> d_x(arg_group_count);        // x vector (K elements, no padding!)
    std::vector<bf16*> d_y_tiles(arg_group_count);  // y as tiles (kernel output)
    std::vector<bf16*> d_y(arg_group_count);        // y extracted from tiles
    bf16* d_y_ref;

    for (int i = 0; i < arg_group_count; i++) {
        CUDACHECK(cudaMalloc(&d_A[i], M * K * sizeof(bf16)));
        CUDACHECK(cudaMalloc(&d_x[i], K * sizeof(bf16)));  // Just K elements, not Nb*K!
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
        // No more prepare_x_tiles - x is loaded directly in the kernel!
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
        // x_gl is gl<bf16, 1, 1, 1, -1>: b=1(fixed), d=1(fixed), r=1(fixed), c=-1(dynamic)
        // Fixed dimensions get nullptr, dynamic dimensions get the actual value
        typename gemv_globals<C>::x_gl Xg{d_x[i], nullptr, nullptr, nullptr, K};
        typename gemv_globals<C>::y_gl Yg{d_y_tiles[i], nullptr, nullptr, M, C::Nb};
        g.push_back(gemv_globals<C>{Ag, Xg, Yg, (int)M, (int)K});
    }

    // Set kernel attributes
    CUDACHECK(cudaFuncSetAttribute(gemv_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   g[0].dynamic_shared_memory()));

    // Number of iterations
    int num_warmups = ncu ? 0 : 500;
    int num_iters = ncu ? 1 : 100;

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

    // Extract y from tiles (2D row-major array of shape M x Nb)
    extract_y<C><<<(M + 255) / 256, 256>>>(d_y[0], d_y_tiles[0], M, C::Nb);
    CUDACHECK(cudaDeviceSynchronize());

    check_correctness(d_y[0], d_y_ref, M);

    // Cleanup
    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_x[i]);
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
