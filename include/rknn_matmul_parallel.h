/**
 * @file rknn_matmul_parallel.h
 * @brief High-performance parallel matmul wrapper for Rockchip RK3576/RK3588 NPU
 *
 * Key features:
 *   - Dual NPU core utilization via multi-process parallelism
 *   - Persistent worker contexts for minimal overhead
 *   - Zero-copy shared memory IPC
 *
 * Why this exists:
 *   Rockchip's rknn_matmul_run() is synchronous blocking with an internal
 *   global lock. Single-process cannot utilize dual NPU cores. This library
 *   solves it via fork + persistent worker processes.
 */

#ifndef RKNN_MATMUL_PARALLEL_H
#define RKNN_MATMUL_PARALLEL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of parallel workers (RK3576/RK3588 have 2 NPU cores) */
#define RMP_MAX_WORKERS 2

/* Matmul precision types
 * Note: RK3576 NPU does not support FP16→FP16 output.
 * All types produce FP32 output. */
typedef enum {
    RMP_TYPE_FP16_FP16 = 0,    /* FP16 input x FP16 weight -> FP32 output */
    RMP_TYPE_FP16_INT4 = 1,    /* FP16 input x INT4 weight -> FP16 output */
    RMP_TYPE_FP16_INT8 = 2,    /* FP16 input x INT8 weight -> FP16 output */
} RmpMatmulType;

/* Weight layout */
typedef enum {
    RMP_LAYOUT_NORMAL = 0,    /* Row-major, standard layout */
    RMP_LAYOUT_NATIVE = 1,    /* NPU-optimized native layout */
} RmpWeightLayout;

/**
 * Configuration for parallel matmul context
 */
typedef struct {
    int M;                      /* Batch size (typically 1 for autoregressive) */
    int K;                      /* Input dimension */
    int N;                      /* Output dimension */
    RmpMatmulType type;         /* Precision type */
    RmpWeightLayout layout;     /* Weight layout */
    int n_workers;              /* Number of workers (default: 2) */
} RmpConfig;

/**
 * Opaque context handle
 */
typedef struct RmpContext RmpContext;

/**
 * Create a parallel matmul context.
 *
 * This forks worker processes that hold persistent NPU contexts.
 * Call once at startup, reuse for all subsequent matmul operations.
 *
 * @param config    Configuration (M, K, N, type, layout)
 * @param weights   Weight matrix (K x N, format depends on type)
 * @param scales    Quantization scales (N elements, NULL for FP16)
 * @return          Context handle, or NULL on failure
 */
RmpContext* rmp_create(const RmpConfig* config, const void* weights, const float* scales);

/**
 * Execute parallel matmul.
 *
 * Input is split across workers, each computing N/n_workers columns.
 * Workers run in parallel on separate NPU cores.
 *
 * @param ctx       Context from rmp_create()
 * @param input     Input matrix (M x K, FP16)
 * @param output    Output matrix (M x N, FP32)
 * @return          0 on success, negative on error
 */
int rmp_run(RmpContext* ctx, const int16_t* input, float* output);

/**
 * Destroy context and cleanup worker processes.
 *
 * @param ctx       Context to destroy
 */
void rmp_destroy(RmpContext* ctx);

/**
 * Benchmark matmul performance.
 *
 * @param ctx       Context
 * @param n_runs    Number of iterations
 * @return          Average time per run in milliseconds
 */
float rmp_benchmark(RmpContext* ctx, int n_runs);

/**
 * Get the equivalent single-core matmul time for comparison.
 *
 * @param ctx       Context
 * @return          Theoretical single-core time in ms
 */
float rmp_get_single_core_time(RmpContext* ctx);

#ifdef __cplusplus
}
#endif

#endif /* RKNN_MATMUL_PARALLEL_H */