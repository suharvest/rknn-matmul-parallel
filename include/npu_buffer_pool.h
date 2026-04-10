/**
 * Zero-copy buffer pool for Rockchip NPU matmul operations.
 *
 * Eliminates memcpy overhead by reusing NPU-accessible buffers.
 * CPU and NPU share the same physical memory via rknn_tensor_mem.
 *
 * Key insight: rknn_create_mem() allocates memory that is BOTH
 * CPU-accessible (via virt_addr) and NPU-accessible.
 * Buffers can be reused across layers and inference steps.
 *
 * Platform: RK3576/RK3588 (Rockchip NPU)
 */

#ifndef NPU_BUFFER_POOL_H
#define NPU_BUFFER_POOL_H

#include <stdint.h>
#include "rknn_matmul_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Buffer Pool Configuration ─── */

#define POOL_MAX_BUFFERS    32
#define POOL_MAX_SIZE       (4 * 1024 * 1024)  /* 4MB max per buffer */

/* ─── Buffer Pool Handle ─── */

typedef struct {
    /* Pre-allocated buffers */
    rknn_tensor_mem* buffers[POOL_MAX_BUFFERS];
    int buffer_sizes[POOL_MAX_BUFFERS];  /* in bytes */
    int n_buffers;

    /* Availability tracking */
    int buffer_available[POOL_MAX_BUFFERS];

    /* Stats */
    int n_reuses;
    int n_allocs;

} BufferPool;

/* ─── Pool Management ─── */

/**
 * Create buffer pool with pre-allocated buffers.
 *
 * @param ctx    RKNN matmul context (used for allocation)
 * @param sizes  Array of buffer sizes (bytes), can be NULL
 * @param n      Number of pre-allocated buffers
 * @return       Pool handle or NULL on failure
 */
BufferPool* buffer_pool_create(rknn_matmul_ctx ctx, const int* sizes, int n);

/**
 * Destroy buffer pool and free all buffers.
 *
 * @param pool   Buffer pool to destroy
 * @param ctx    RKNN matmul context (used for deallocation)
 */
void buffer_pool_destroy(BufferPool* pool, rknn_matmul_ctx ctx);

/**
 * Get a buffer of at least the requested size.
 * Returns existing buffer if available, otherwise allocates new.
 *
 * @param pool      Buffer pool
 * @param ctx       RKNN matmul context
 * @param min_size  Minimum required size in bytes
 * @return          Buffer or NULL on failure
 */
rknn_tensor_mem* buffer_pool_get(BufferPool* pool, rknn_matmul_ctx ctx, int min_size);

/**
 * Return a buffer to the pool for reuse.
 *
 * @param pool  Buffer pool
 * @param mem   Buffer to return
 */
void buffer_pool_release(BufferPool* pool, rknn_tensor_mem* mem);

/**
 * Reset pool for next inference (marks all buffers as available).
 *
 * @param pool  Buffer pool
 */
void buffer_pool_reset(BufferPool* pool);

/* ─── KV Cache Management ─── */

/**
 * KV cache for transformer decoder.
 * Manages key and value caches for all layers.
 */
typedef struct {
    float* k_cache;    /* [num_layers][num_kv_heads][max_seq_len][head_dim] */
    float* v_cache;    /* [num_layers][num_kv_heads][max_seq_len][head_dim] */

    int num_layers;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;
    int current_len;   /* Current sequence length */

} KVCache;

/**
 * Create KV cache.
 *
 * @param num_layers    Number of transformer layers
 * @param num_kv_heads  Number of KV heads per layer
 * @param head_dim      Dimension per head
 * @param max_seq_len   Maximum sequence length
 * @return              KV cache or NULL on failure
 */
KVCache* kv_cache_create(int num_layers, int num_kv_heads, int head_dim, int max_seq_len);

/**
 * Clear KV cache for new sequence.
 *
 * @param cache  KV cache
 */
void kv_cache_clear(KVCache* cache);

/**
 * Get pointer to K cache for a specific layer and head.
 */
static inline float* kv_cache_get_k(KVCache* cache, int layer, int head) {
    return cache->k_cache +
           (layer * cache->num_kv_heads + head) * cache->max_seq_len * cache->head_dim;
}

/**
 * Get pointer to V cache for a specific layer and head.
 */
static inline float* kv_cache_get_v(KVCache* cache, int layer, int head) {
    return cache->v_cache +
           (layer * cache->num_kv_heads + head) * cache->max_seq_len * cache->head_dim;
}

/**
 * Destroy KV cache.
 */
void kv_cache_destroy(KVCache* cache);

/* ─── Utility Functions ─── */

/**
 * Get CPU-accessible pointer to NPU buffer (zero-copy).
 * CPU can read/write directly without memcpy.
 */
static inline void* buffer_get_cpu_ptr(rknn_tensor_mem* mem) {
    return mem->virt_addr;
}

/**
 * Get physical address (for NPU DMA).
 */
static inline uint64_t buffer_get_phys_addr(rknn_tensor_mem* mem) {
    return mem->phys_addr;
}

#ifdef __cplusplus
}
#endif

#endif /* NPU_BUFFER_POOL_H */