/**
 * Zero-copy buffer pool implementation for Rockchip NPU.
 */

#include "npu_buffer_pool.h"
#include <stdlib.h>
#include <string.h>

/* ─── Pool Management ─── */

BufferPool* buffer_pool_create(rknn_matmul_ctx ctx, const int* sizes, int n) {
    BufferPool* pool = (BufferPool*)calloc(1, sizeof(BufferPool));
    if (!pool) return NULL;

    pool->n_buffers = 0;
    pool->n_reuses = 0;
    pool->n_allocs = 0;

    /* Pre-allocate buffers if sizes provided */
    if (sizes && n > 0) {
        for (int i = 0; i < n && i < POOL_MAX_BUFFERS; i++) {
            pool->buffers[i] = rknn_create_mem(ctx, sizes[i]);
            if (!pool->buffers[i]) {
                /* Clean up on failure */
                for (int j = 0; j < i; j++) {
                    rknn_destroy_mem(ctx, pool->buffers[j]);
                }
                free(pool);
                return NULL;
            }
            pool->buffer_sizes[i] = sizes[i];
            pool->buffer_available[i] = 1;
            pool->n_buffers++;
        }
    }

    return pool;
}

void buffer_pool_destroy(BufferPool* pool, rknn_matmul_ctx ctx) {
    if (!pool) return;

    for (int i = 0; i < pool->n_buffers; i++) {
        if (pool->buffers[i]) {
            rknn_destroy_mem(ctx, pool->buffers[i]);
        }
    }

    free(pool);
}

rknn_tensor_mem* buffer_pool_get(BufferPool* pool, rknn_matmul_ctx ctx, int min_size) {
    /* Find smallest available buffer that fits */
    int best_idx = -1;
    int best_size = POOL_MAX_SIZE + 1;

    for (int i = 0; i < pool->n_buffers; i++) {
        if (pool->buffer_available[i] &&
            pool->buffer_sizes[i] >= min_size &&
            pool->buffer_sizes[i] < best_size) {
            best_idx = i;
            best_size = pool->buffer_sizes[i];
        }
    }

    if (best_idx >= 0) {
        pool->buffer_available[best_idx] = 0;  /* Mark as in use */
        pool->n_reuses++;
        return pool->buffers[best_idx];
    }

    /* Need to allocate new buffer */
    if (pool->n_buffers >= POOL_MAX_BUFFERS) {
        return NULL;  /* Pool full */
    }

    rknn_tensor_mem* mem = rknn_create_mem(ctx, min_size);
    if (mem) {
        int idx = pool->n_buffers++;
        pool->buffers[idx] = mem;
        pool->buffer_sizes[idx] = min_size;
        pool->buffer_available[idx] = 0;  /* Mark as in use */
        pool->n_allocs++;
    }

    return mem;
}

void buffer_pool_release(BufferPool* pool, rknn_tensor_mem* mem) {
    if (!pool || !mem) return;

    /* Find the buffer and mark as available */
    for (int i = 0; i < pool->n_buffers; i++) {
        if (pool->buffers[i] == mem) {
            pool->buffer_available[i] = 1;
            return;
        }
    }
}

void buffer_pool_reset(BufferPool* pool) {
    if (!pool) return;

    /* Mark all buffers as available */
    for (int i = 0; i < pool->n_buffers; i++) {
        pool->buffer_available[i] = 1;
    }
}

/* ─── KV Cache Management ─── */

KVCache* kv_cache_create(int num_layers, int num_kv_heads, int head_dim, int max_seq_len) {
    KVCache* cache = (KVCache*)calloc(1, sizeof(KVCache));
    if (!cache) return NULL;

    cache->num_layers = num_layers;
    cache->num_kv_heads = num_kv_heads;
    cache->head_dim = head_dim;
    cache->max_seq_len = max_seq_len;
    cache->current_len = 0;

    /* Allocate K and V caches */
    size_t cache_size = (size_t)num_layers * num_kv_heads * max_seq_len * head_dim * sizeof(float);
    cache->k_cache = (float*)calloc(cache_size, 1);
    cache->v_cache = (float*)calloc(cache_size, 1);

    if (!cache->k_cache || !cache->v_cache) {
        kv_cache_destroy(cache);
        return NULL;
    }

    return cache;
}

void kv_cache_clear(KVCache* cache) {
    if (!cache) return;

    size_t cache_size = (size_t)cache->num_layers * cache->num_kv_heads *
                        cache->max_seq_len * cache->head_dim * sizeof(float);
    memset(cache->k_cache, 0, cache_size);
    memset(cache->v_cache, 0, cache_size);
    cache->current_len = 0;
}

void kv_cache_destroy(KVCache* cache) {
    if (!cache) return;

    free(cache->k_cache);
    free(cache->v_cache);
    free(cache);
}