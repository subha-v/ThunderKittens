# FMA GEMV Kernel Experiments (B200)

## Overview

This document summarizes experiments with CUDA FMA-based (non-tensor-core) GEMV kernels on B200. The goal was to determine whether FMA cores could match or beat cuBLAS for the operation `y = A * x` where A is MxK in bf16.

**Conclusion:** FMA kernels reached up to **92% of cuBLAS at large sizes** but couldn't close the gap at small sizes (76% max). The tensor core TK kernel remains competitive or superior, making further FMA optimization lower priority.

## Reference Baselines

### cuBLAS Performance
| Size (M=K) | Bandwidth (GB/s) | Latency (μs) |
|------------|-------------------|---------------|
| 4096       | 3,548             | 9.46          |
| 8192       | 5,472             | 24.5          |
| 16384      | 6,413             | 83.7          |
| 32768      | 6,468             | 332.0         |

### TK Tensor Core Performance
| Size (M=K) | Bandwidth (GB/s) | Latency (μs) |
|------------|-------------------|---------------|
| 4096       | 2,048             | 16.4          |
| 8192       | 4,248             | 31.6          |
| 16384      | 6,362             | 84.4          |
| 32768      | 5,579             | 385.0         |

B200 peak HBM bandwidth: ~8,000 GB/s.

---

## Experiment 1: Initial Kernel Versions (v1–v6)

Six progressively optimized kernels exploring the design space:

| Version | Strategy | Key Idea |
|---------|----------|----------|
| v1 | 1 thread per row | Simplest. Each thread streams through one row of A. |
| v2 | Multi-row per thread | Each thread handles 4 rows for ILP. |
| v3 | Vectorized loads | Uses `float2` (2 bf16) loads for x and A. |
| v4 | Warp-coalesced | Each warp handles 1 row; 32 lanes share the K-dimension. Coalesced A reads. |
| v5 | Vec + warp-coalesced | v4 plus `float2` vectorized x loads. |
| v6 | Multi-row per warp | Each warp handles 4 rows. x value broadcast, multiple A loads. |

**Result:** v6 was best overall but still far from cuBLAS. v1–v3 suffered from poor memory coalescing (each thread read its own row of A, preventing coalesced access). v4–v6 fixed this with warp-per-row decomposition.

---

## Experiment 2: Optimized Single Kernel with RPW Sweep

Replaced v1–v6 with a single optimized kernel parameterized by `ROWS_PER_WARP` (RPW) and `K_CHUNK`.

### Kernel Design
- **Block:** 256 threads = 8 warps
- **Each warp:** handles `ROWS_PER_WARP` output rows
- **K-loop:** x loaded into `__shared__ bf16 x_smem[K_CHUNK]` via vectorized `float4` loads (8 bf16/thread, all 256 threads)
- **Compute:** each lane loads `float4` from A (8 bf16), unpacks, 8 FMAs per row per vector. Lane stride = 256.
- **Reduction:** `__shfl_down_sync` with offsets {16, 8, 4, 2, 1}
- **Write:** lane 0 stores `__float2bfloat16(acc[r])` with bounds check

### Results (GB/s), K_CHUNK=2048

| Size | RPW=2 | RPW=4 | RPW=8 |
|------|-------|-------|-------|
| 4096 | **1,624** | 909 | 491 |
| 8192 | **3,300** | 1,872 | 1,006 |
| 16384 | **5,712** | 3,731 | 2,004 |
| 32768 | **5,560** | 3,747 | 3,860 |

**Finding:** RPW=2 was the clear winner. Fewer rows per warp = more blocks = better SM occupancy and memory-level parallelism. RPW=2 peaked at **5,712 GB/s (71% of peak HBM)** at 16K.

---

## Experiment 3: Split-K (Unfused — Separate Kernels)

Split the K-dimension across multiple thread block groups to increase parallelism at small M.

### Design
- **Grid:** `dim3(M_blocks, K_SPLITS)` — `blockIdx.y` selects K partition
- **Each block:** computes partial dot product for its K range
- **Accumulation:** `atomicAdd` to pre-zeroed `float32` buffer
- **Conversion:** separate `convert_f32_to_bf16` kernel
- **Per-iteration cost:** `cudaMemsetAsync` (M floats) + main kernel + convert kernel

### Results (GB/s), RPW=2, K_CHUNK=2048

| Size | S=1 | S=2 | S=4 | S=8 |
|------|-----|-----|-----|-----|
| 4096 | 1,323 | 1,935 | 2,172 | **2,198** |
| 8192 | 2,904 | 3,901 | **3,981** | 3,841 |
| 16384 | 5,373 | 5,432 | 5,444 | **5,576** |
| 32768 | 5,483 | 5,542 | **5,971** | 5,955 |

**Finding:** Split-K helped at all sizes. Best gains at small M (+35% at 4K). But the S=1 baseline (1,323) was slower than the non-split kernel (1,624) due to memset + convert kernel overhead (~5 μs total).

---

## Experiment 4: Fused Split-K with Spin-Wait (Failed)

Attempted to eliminate all overhead by fusing into a single kernel:
- `blockIdx.y == 0`: direct store to float32 buffer (no memset needed)
- `blockIdx.y > 0`: spin-wait until split 0 finishes, then `atomicAdd`
- Last split converts to bf16 inline

### Results (GB/s)
| Size | S=1 | S=2 | S=4 |
|------|-----|-----|-----|
| 4096 | 1,628 | **2,032** | 1,818 |
| 8192 | 3,306 | 2,975 | 3,447 |
| 16384 | **5,713** | **4,357** | 4,772 |
| 32768 | **5,580** | **4,787** | 4,721 |

**Finding:** This approach **backfired** for K_SPLITS > 1. The spin-wait serialized execution — `blockIdx.y > 0` blocks occupied SMs while spinning, waiting for `blockIdx.y == 0`. At large M with many m_blocks, this caused severe performance degradation (up to 20% worse than unfused). S=1 correctly matched the original kernel (no spin path taken).

---

## Experiment 5: Hybrid Fused Split-K (No Spin-Wait)

Kept concurrent `atomicAdd` (no ordering) but fused the bf16 conversion into the main kernel:
- `cudaMemsetAsync` zeros y_partial (M floats) and split_done (M_blocks ints)
- All splits `atomicAdd` freely to pre-zeroed buffer
- Last split to arrive (via atomic done counter) converts to bf16 inline

### Results (GB/s)
| Size | S=1 | S=2 | S=4 | S=8 |
|------|-----|-----|-----|-----|
| 4096 | 1,621 | 1,814 | **2,178** | 2,029 |
| 8192 | 3,296 | **3,969** | 3,934 | 3,652 |
| 16384 | **5,711** | 5,432 | 5,305 | 5,325 |
| 32768 | 5,580 | 5,511 | **5,892** | 5,762 |

**Finding:** Performance roughly matched the unfused version (within 1-2%). The convert kernel launch was not the main bottleneck — the `cudaMemsetAsync` and `atomicAdd` cost dominated. Fusing the conversion saved ~2-3 μs per iteration but this was in the noise.

---

## Experiment 6: Block-Per-Row Kernel (Best Small-M Result)

Fundamentally different decomposition for small M: one entire block (256 threads) per output row.

### Design
- **Grid:** `dim3(M)` — one block per row
- **All 256 threads** collaborate on loading x into smem AND computing one dot product
- **Two-level reduction:**
  1. Warp shuffle within each of 8 warps (32→1)
  2. Shared memory `warp_sums[8]` + warp 0 shuffle (8→1)
- **Key advantage:** M=4096 → 4096 blocks → 21.3 blocks/SM (vs 1.3 with warp-per-row)

### Results (GB/s)

| Size | Block/Row | Warp/Row (RPW=2) | Best | vs cuBLAS |
|------|-----------|------------------|------|-----------|
| 4096 | **2,690** | 1,616 | Block/Row | 76% |
| 8192 | **3,848** | 3,289 | Block/Row | 70% |
| 16384 | 4,766 | **5,694** | Warp/Row | 89% |
| 32768 | 5,001 | **5,562** | Warp/Row | 86% |

**Finding:** Block-per-row was a **+66% improvement** at M=4096 over warp-per-row (2,690 vs 1,616). The occupancy theory was validated — 21× more blocks dramatically improved HBM pipeline utilization. Crossover point is between 8K and 16K.

---

## Summary: Best Achievable Performance

Picking the optimal strategy per size:

| Size | Best Strategy | BW (GB/s) | vs cuBLAS | vs TK Tensor Core |
|------|--------------|-----------|-----------|-------------------|
| 4096 | Block-per-row | 2,690 | 76% | 131% (FMA wins) |
| 8192 | Block-per-row / Split-K=4 | 3,848–3,981 | 70-73% | 91-94% |
| 16384 | Warp-per-row RPW=2 | 5,712 | 89% | 90% (roughly tied) |
| 32768 | Warp-per-row + Split-K=4 | 5,971 | 92% | 107% (FMA wins) |

### Key Insights
1. **FMA beats tensor cores at small and very large sizes** (4K, 32K) because tensor cores waste bandwidth padding x to 16 rows of zeros.
2. **Tensor cores match FMA at medium sizes** (8K, 16K) where wgmma throughput compensates for padding overhead.
3. **Neither approach matches cuBLAS**, which likely uses persistent kernels and size-specific kernel selection.
4. **Block count is the #1 determinant of small-M performance** — the block-per-row redesign gave 2× the improvement of split-K.
5. **Split-K has diminishing returns** due to atomicAdd + memset overhead that grows with K_SPLITS.
6. **Spin-wait synchronization is catastrophic on GPUs** — never block threads waiting for other blocks.

### Remaining Gap to cuBLAS
- **Small M (4K–8K):** cuBLAS is ~1.3–1.5× faster. Likely uses persistent kernels or fundamentally different thread mapping.
- **Large M (16K–32K):** Within 8–11% of cuBLAS. Near the practical limit for this approach.
