/**
 * Generic Matmul Decoder Implementation
 *
 * Implements transformer decoder using RKNN matmul API.
 * Supports both single-core and dual-core execution modes.
 */

#include "matmul_decoder.h"
#include "cpu_ops.h"
#include "rknn_matmul_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>

/* Timing helper */
static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#include <sys/stat.h>

/* ─── Internal Structures ─── */

/**
 * Persistent matmul context for a single projection.
 *
 * In pooled mode (is_pooled=1): ctx, mem_A, mem_C are shared across all
 * layers via a context pool. Only mem_B (weight) is per-projection.
 * Before each run, B is rebound via rknn_matmul_set_io_mem.
 *
 * This reduces NPU handle count from O(layers×7) to O(unique_dims + layers×7_B_only),
 * avoiding the /dev/rknpu per-process handle table limit (~1020).
 */
typedef struct {
    rknn_matmul_ctx ctx;
    rknn_matmul_io_attr io;
    rknn_tensor_mem* mem_A;
    rknn_tensor_mem* mem_B;
    rknn_tensor_mem* mem_C;
    int K, N;
    int initialized;
    int is_pooled;          /* 1 = ctx/A/C are shared, only destroy B */
    int output_is_fp32;     /* 1 = C output is FP32 (FLOAT16_TO_FLOAT32) */
    float* col_scales;      /* Per-column W4A16 scales [N] (NULL for FP16) */
    rknn_matmul_info* pool_info;  /* Points to pool's info or own_info (for B layout conversion) */
    rknn_matmul_info _own_info;   /* Stored copy for dedicated mode (pool_info points here) */
} PersistentMatmul;

/**
 * Context pool entry — one per unique (K, N) dimension pair.
 * Shared by all layers that have the same projection dimensions.
 */
typedef struct {
    rknn_matmul_ctx ctx;
    rknn_matmul_info info;      /* Saved for B layout conversion */
    rknn_matmul_io_attr io;
    rknn_tensor_mem* mem_A;
    rknn_tensor_mem* mem_C;
    int K, N;
    int initialized;
} MatmulPoolEntry;

#define MAX_MATMUL_POOL 8

/**
 * Layer matmul contexts (all projections for one layer).
 */
typedef struct {
    PersistentMatmul q_proj;
    PersistentMatmul k_proj;
    PersistentMatmul v_proj;
    PersistentMatmul o_proj;
    PersistentMatmul gate_proj;
    PersistentMatmul up_proj;
    PersistentMatmul down_proj;

    /* Norm weights (CPU) */
    float* input_norm_w;
    float* post_attn_norm_w;

    /* QK norm weights (optional, [head_dim] each) */
    float* q_norm_w;
    float* k_norm_w;
} LayerMatmulContext;

/**
 * Full decoder context.
 */
struct MatmulDecoderContext {
    MatmulDecoderConfig config;
    QuantizationType quant_type;

    /* Embeddings */
    float* embeddings;          /* [vocab_size, hidden_dim] */

    /* LM head(s) */
    float* lm_head;             /* Single lm_head: [vocab_size, hidden_dim] FP32 */
    int16_t* lm_head_fp16;      /* FP16 copy for fast GEMV (halves bandwidth) */
    float** lm_heads;           /* Multi lm_head: [num_lm_heads][lm_head_vocab_size * hidden_dim] */
    int num_lm_heads;           /* 0 = single head mode */

    /* Layer contexts */
    LayerMatmulContext* layers;

    /* Context pool (reduces NPU handle count for large models) */
    MatmulPoolEntry pool[MAX_MATMUL_POOL];
    int n_pool;

    /* KV Cache */
    MatmulKVCache* kv_cache;

    /* RoPE tables */
    float* cos_table;           /* [max_seq_len, head_dim/2] */
    float* sin_table;

    /* Final norm weight (separate from per-layer norms) */
    float* final_norm_w;        /* [hidden_dim] */

    /* Working buffers */
    float* hidden;              /* [hidden_dim] */
    float* residual;            /* [hidden_dim] — saves hidden for residual connection */
    float* normed;              /* [hidden_dim] — normalized intermediate */
    float* q_out;               /* [num_q_heads * head_dim] */
    float* k_out;               /* [num_kv_heads * head_dim] */
    float* v_out;               /* [num_kv_heads * head_dim] */
    float* attn_out;            /* [num_q_heads * head_dim] */
    float* ffn_gate;            /* [ffn_dim] */
    float* ffn_up;              /* [ffn_dim] */
    float* ffn_down;            /* [hidden_dim] */
    float* logits;              /* [max(vocab_size, lm_head_vocab_size)] */

    /* Execution mode */
    ExecutionMode exec_mode;
    int n_workers;              /* 1 for single-core, 2 for dual-core */

    /* Stats */
    MatmulDecoderStats stats;
};

/* ─── Matmul Context Management ─── */

/* Map quantization type to RKNN matmul type */
static rknn_matmul_type quant_to_rknn_type(QuantizationType quant_type) {
    switch (quant_type) {
        case QUANT_FP16:     return RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
        case QUANT_INT4:
        case QUANT_INT4_G128: return RKNN_FLOAT16_MM_INT4_TO_FLOAT16;
        case QUANT_INT8:     return RKNN_FLOAT16_MM_INT8_TO_FLOAT16;
        default:             return RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
    }
}

/* ─── Context Pool ─── */

/**
 * Find or create a pool entry for the given (K, N) dimensions.
 * Returns pool index, or -1 on error.
 */
static int pool_get_or_create(MatmulPoolEntry* pool, int* n_pool,
                               int K, int N, QuantizationType quant_type,
                               int iommu_domain_id) {
    /* Search existing */
    for (int i = 0; i < *n_pool; i++) {
        if (pool[i].K == K && pool[i].N == N) return i;
    }

    /* Create new */
    if (*n_pool >= MAX_MATMUL_POOL) {
        fprintf(stderr, "[Pool] Exceeded max pool size %d\n", MAX_MATMUL_POOL);
        return -1;
    }

    int idx = (*n_pool)++;
    MatmulPoolEntry* pe = &pool[idx];
    memset(pe, 0, sizeof(*pe));

    rknn_matmul_info info;
    memset(&info, 0, sizeof(info));
    info.M = 1;
    info.K = K;
    info.N = N;
    info.type = quant_to_rknn_type(quant_type);
    info.B_layout = 1;
    info.iommu_domain_id = iommu_domain_id;

    pe->info = info;  /* Save for B layout conversion */

    int ret = rknn_matmul_create(&pe->ctx, &info, &pe->io);
    if (ret != 0) {
        fprintf(stderr, "[Pool] rknn_matmul_create(%d,%d) failed: %d\n", K, N, ret);
        (*n_pool)--;
        return -1;
    }

    pe->mem_A = rknn_create_mem(pe->ctx, pe->io.A.size);
    pe->mem_C = rknn_create_mem(pe->ctx, pe->io.C.size);
    if (!pe->mem_A || !pe->mem_C) {
        fprintf(stderr, "[Pool] rknn_create_mem failed for pool(%d,%d)\n", K, N);
        rknn_matmul_destroy(pe->ctx);
        (*n_pool)--;
        return -1;
    }

    rknn_matmul_set_io_mem(pe->ctx, pe->mem_A, &pe->io.A);
    rknn_matmul_set_io_mem(pe->ctx, pe->mem_C, &pe->io.C);

    pe->K = K;
    pe->N = N;
    pe->initialized = 1;

    printf("[Pool] Created pool[%d]: K=%d N=%d domain=%d (handles: 1 ctx + A + C = 3)\n",
           idx, K, N, iommu_domain_id);
    return idx;
}

/**
 * Allocate a B weight buffer from a pool entry.
 * Sets up PersistentMatmul to share ctx/A/C from pool, own B only.
 */
static int create_pooled_matmul(PersistentMatmul* pm, MatmulPoolEntry* pe) {
    memset(pm, 0, sizeof(PersistentMatmul));

    pm->ctx = pe->ctx;
    pm->io  = pe->io;
    pm->mem_A = pe->mem_A;
    pm->mem_C = pe->mem_C;

    /* Only B is per-projection */
    pm->mem_B = rknn_create_mem(pe->ctx, pe->io.B.size);
    if (!pm->mem_B) {
        fprintf(stderr, "[Pool] rknn_create_mem for B failed\n");
        return MATMUL_DECODER_ERR_MEMORY;
    }

    pm->K = pe->K;
    pm->N = pe->N;
    pm->initialized = 1;
    pm->is_pooled = 1;
    pm->output_is_fp32 = (pe->info.type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32);
    pm->pool_info = &pe->info;

    return MATMUL_DECODER_OK;
}

/**
 * Legacy: create standalone matmul context (for small models / non-pooled mode).
 */
static int create_persistent_matmul(PersistentMatmul* pm, int M, int K, int N,
                                     QuantizationType quant_type, int iommu_domain_id) {
    memset(pm, 0, sizeof(PersistentMatmul));

    rknn_matmul_info info;
    memset(&info, 0, sizeof(info));
    info.M = M;
    info.K = K;
    info.N = N;
    info.type = quant_to_rknn_type(quant_type);
    info.B_layout = 1;
    info.iommu_domain_id = iommu_domain_id;

    int ret = rknn_matmul_create(&pm->ctx, &info, &pm->io);
    if (ret != 0) {
        fprintf(stderr, "[MatmulDecoder] rknn_matmul_create failed: %d\n", ret);
        return MATMUL_DECODER_ERR_RKNN;
    }

    pm->mem_A = rknn_create_mem(pm->ctx, pm->io.A.size);
    pm->mem_B = rknn_create_mem(pm->ctx, pm->io.B.size);
    pm->mem_C = rknn_create_mem(pm->ctx, pm->io.C.size);

    if (!pm->mem_A || !pm->mem_B || !pm->mem_C) {
        fprintf(stderr, "[MatmulDecoder] rknn_create_mem failed\n");
        return MATMUL_DECODER_ERR_MEMORY;
    }

    rknn_matmul_set_io_mem(pm->ctx, pm->mem_A, &pm->io.A);
    rknn_matmul_set_io_mem(pm->ctx, pm->mem_B, &pm->io.B);
    rknn_matmul_set_io_mem(pm->ctx, pm->mem_C, &pm->io.C);

    pm->K = K;
    pm->N = N;
    pm->initialized = 1;
    pm->is_pooled = 0;
    pm->output_is_fp32 = (info.type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32);
    /* Store info for B layout conversion (dedicated mode has no pool) */
    pm->_own_info = info;
    pm->pool_info = &pm->_own_info;

    return MATMUL_DECODER_OK;
}

static void destroy_persistent_matmul(PersistentMatmul* pm) {
    if (!pm->initialized) return;

    free(pm->col_scales);
    if (pm->is_pooled) {
        /* Pooled: B was allocated from pool ctx.
         * rknn_matmul_destroy(pool_ctx) will free all associated mem.
         * Explicitly destroying B here can SIGSEGV if pool ctx state is
         * inconsistent. Skip — pool cleanup handles it. */
    } else {
        /* Standalone: destroy everything */
        if (pm->mem_A) rknn_destroy_mem(pm->ctx, pm->mem_A);
        if (pm->mem_B) rknn_destroy_mem(pm->ctx, pm->mem_B);
        if (pm->mem_C) rknn_destroy_mem(pm->ctx, pm->mem_C);
        if (pm->ctx) rknn_matmul_destroy(pm->ctx);
    }

    memset(pm, 0, sizeof(PersistentMatmul));
}

static void destroy_pool(MatmulPoolEntry* pool, int n_pool) {
    for (int i = 0; i < n_pool; i++) {
        MatmulPoolEntry* pe = &pool[i];
        if (!pe->initialized) continue;
        /* Destroy all B mems first (allocated from this pool ctx),
         * then A/C, then the ctx itself. Order matters — ctx must be last. */
        if (pe->mem_A) rknn_destroy_mem(pe->ctx, pe->mem_A);
        if (pe->mem_C) rknn_destroy_mem(pe->ctx, pe->mem_C);
        /* Note: B mems created from this ctx are NOT tracked here —
         * they leak on destroy. This is acceptable since process exit
         * frees all RKNPU resources. For explicit cleanup, skip destroy
         * entirely and let the kernel driver clean up on fd close. */
        if (pe->ctx) rknn_matmul_destroy(pe->ctx);
        memset(pe, 0, sizeof(*pe));
    }
}

/* Per-step timing accumulators (reset at start of each step) */
static double _acc_rebind_ms = 0;
static double _acc_matmul_ms = 0;
static double _acc_convert_ms = 0;

static int run_persistent_matmul(PersistentMatmul* pm, const float* input_fp32, float* output_fp32) {
    int K = pm->K, N = pm->N;
    double t0, t1;

    /* Pooled mode: rebind B weight before each run */
    if (pm->is_pooled) {
        t0 = now_ms();
        rknn_matmul_set_io_mem(pm->ctx, pm->mem_B, &pm->io.B);
        _acc_rebind_ms += now_ms() - t0;
    }

    /* Convert input to FP16 */
    t0 = now_ms();
    vec_fp32_to_fp16((int16_t*)pm->mem_A->virt_addr, input_fp32, K);
    _acc_convert_ms += now_ms() - t0;

    /* Run matmul */
    t0 = now_ms();
    int ret = rknn_matmul_run(pm->ctx);
    _acc_matmul_ms += now_ms() - t0;
    if (ret != 0) {
        return MATMUL_DECODER_ERR_RKNN;
    }

    /* Read output and apply per-column scales if W4A16 */
    t0 = now_ms();
    if (pm->output_is_fp32) {
        memcpy(output_fp32, pm->mem_C->virt_addr, N * sizeof(float));
    } else if (pm->col_scales) {
        const int16_t* src = (const int16_t*)pm->mem_C->virt_addr;
        const float* scales = pm->col_scales;
        int i;
        for (i = 0; i <= N - 4; i += 4) {
            float16x4_t h = vld1_f16((const __fp16*)(src + i));
            float32x4_t f = vcvt_f32_f16(h);
            float32x4_t s = vld1q_f32(scales + i);
            vst1q_f32(output_fp32 + i, vmulq_f32(f, s));
        }
        for (; i < N; i++) {
            output_fp32[i] = (float)(*(const __fp16*)(src + i)) * scales[i];
        }
    } else {
        vec_fp16_to_fp32(output_fp32, (int16_t*)pm->mem_C->virt_addr, N);
    }
    _acc_convert_ms += now_ms() - t0;

    return MATMUL_DECODER_OK;
}

/* ─── KV Cache Implementation ─── */

struct MatmulKVCache {
    float* k_cache;             /* [num_layers, max_seq_len, num_kv_heads, head_dim] */
    float* v_cache;
    int num_layers;
    int num_kv_heads;
    int head_dim;
    int max_seq_len;
    int seq_len;                /* Current sequence length */
};

MatmulKVCache* matmul_kv_cache_create(int num_layers, int num_kv_heads,
                                       int head_dim, int max_seq_len) {
    MatmulKVCache* cache = calloc(1, sizeof(MatmulKVCache));
    if (!cache) return NULL;

    cache->num_layers = num_layers;
    cache->num_kv_heads = num_kv_heads;
    cache->head_dim = head_dim;
    cache->max_seq_len = max_seq_len;

    size_t cache_size = (size_t)num_layers * max_seq_len * num_kv_heads * head_dim * sizeof(float);
    cache->k_cache = aligned_alloc(64, cache_size);
    cache->v_cache = aligned_alloc(64, cache_size);

    if (!cache->k_cache || !cache->v_cache) {
        free(cache->k_cache);
        free(cache->v_cache);
        free(cache);
        return NULL;
    }

    memset(cache->k_cache, 0, cache_size);
    memset(cache->v_cache, 0, cache_size);

    return cache;
}

void matmul_kv_cache_clear(MatmulKVCache* cache) {
    if (cache) {
        cache->seq_len = 0;
    }
}

int matmul_kv_cache_get_seq_len(const MatmulKVCache* cache) {
    return cache ? cache->seq_len : 0;
}

void matmul_kv_cache_destroy(MatmulKVCache* cache) {
    if (cache) {
        free(cache->k_cache);
        free(cache->v_cache);
        free(cache);
    }
}

/* ─── NEON-optimized GEMV for LM head ─── */

/**
 * Compute logits = x @ W^T where W is [N, K] row-major (each row is one vocab entry).
 * x: [K], W: [N, K], out: [N].
 * Uses NEON vfmaq_f32 for 4-wide FMA, processes 4 output rows at a time.
 */
static void gemv_f32_neon(float* out, const float* x, const float* W, int N, int K) {
    int v = 0;

    /* Process 4 vocab entries at a time */
    for (; v <= N - 4; v += 4) {
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        const float* w0 = W + (v + 0) * K;
        const float* w1 = W + (v + 1) * K;
        const float* w2 = W + (v + 2) * K;
        const float* w3 = W + (v + 3) * K;

        int h = 0;
        for (; h <= K - 4; h += 4) {
            float32x4_t xv = vld1q_f32(x + h);
            acc0 = vfmaq_f32(acc0, xv, vld1q_f32(w0 + h));
            acc1 = vfmaq_f32(acc1, xv, vld1q_f32(w1 + h));
            acc2 = vfmaq_f32(acc2, xv, vld1q_f32(w2 + h));
            acc3 = vfmaq_f32(acc3, xv, vld1q_f32(w3 + h));
        }

        float s0 = vaddvq_f32(acc0);
        float s1 = vaddvq_f32(acc1);
        float s2 = vaddvq_f32(acc2);
        float s3 = vaddvq_f32(acc3);

        /* Scalar tail */
        for (; h < K; h++) {
            s0 += x[h] * w0[h];
            s1 += x[h] * w1[h];
            s2 += x[h] * w2[h];
            s3 += x[h] * w3[h];
        }

        out[v + 0] = s0;
        out[v + 1] = s1;
        out[v + 2] = s2;
        out[v + 3] = s3;
    }

    /* Remaining vocab entries */
    for (; v < N; v++) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        const float* w = W + v * K;
        int h = 0;
        for (; h <= K - 4; h += 4) {
            acc = vfmaq_f32(acc, vld1q_f32(x + h), vld1q_f32(w + h));
        }
        float s = vaddvq_f32(acc);
        for (; h < K; h++) s += x[h] * w[h];
        out[v] = s;
    }
}

/**
 * GEMV with FP16 weight matrix — halves memory bandwidth vs FP32.
 * x: [K] FP32, W: [N, K] FP16 (row-major), out: [N] FP32.
 */
static void gemv_f16_neon(float* out, const float* x, const int16_t* W_fp16, int N, int K) {
    int v = 0;

    for (; v <= N - 4; v += 4) {
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        const __fp16* w0 = (const __fp16*)(W_fp16 + (v + 0) * K);
        const __fp16* w1 = (const __fp16*)(W_fp16 + (v + 1) * K);
        const __fp16* w2 = (const __fp16*)(W_fp16 + (v + 2) * K);
        const __fp16* w3 = (const __fp16*)(W_fp16 + (v + 3) * K);

        int h = 0;
        for (; h <= K - 4; h += 4) {
            float32x4_t xv = vld1q_f32(x + h);
            acc0 = vfmaq_f32(acc0, xv, vcvt_f32_f16(vld1_f16(w0 + h)));
            acc1 = vfmaq_f32(acc1, xv, vcvt_f32_f16(vld1_f16(w1 + h)));
            acc2 = vfmaq_f32(acc2, xv, vcvt_f32_f16(vld1_f16(w2 + h)));
            acc3 = vfmaq_f32(acc3, xv, vcvt_f32_f16(vld1_f16(w3 + h)));
        }

        float s0 = vaddvq_f32(acc0), s1 = vaddvq_f32(acc1);
        float s2 = vaddvq_f32(acc2), s3 = vaddvq_f32(acc3);
        for (; h < K; h++) {
            float xh = x[h];
            s0 += xh * (float)w0[h]; s1 += xh * (float)w1[h];
            s2 += xh * (float)w2[h]; s3 += xh * (float)w3[h];
        }
        out[v] = s0; out[v+1] = s1; out[v+2] = s2; out[v+3] = s3;
    }

    for (; v < N; v++) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        const __fp16* w = (const __fp16*)(W_fp16 + v * K);
        int h = 0;
        for (; h <= K - 4; h += 4) {
            acc = vfmaq_f32(acc, vld1q_f32(x + h), vcvt_f32_f16(vld1_f16(w + h)));
        }
        float s = vaddvq_f32(acc);
        for (; h < K; h++) s += x[h] * (float)w[h];
        out[v] = s;
    }
}

/* ─── RoPE Precompute ─── */

static void precompute_rope_tables(float* cos_table, float* sin_table,
                                    int max_seq_len, int head_dim, float theta) {
    int half = head_dim / 2;
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(theta, (2.0f * i) / head_dim);
            float angle = pos * freq;
            cos_table[pos * half + i] = cosf(angle);
            sin_table[pos * half + i] = sinf(angle);
        }
    }
}

/* ─── Weight Loading Helpers ─── */

static float* load_fp32_file(const char* path, size_t n_elements) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    float* data = malloc(n_elements * sizeof(float));
    if (!data) {
        fclose(f);
        return NULL;
    }

    size_t read = fread(data, sizeof(float), n_elements, f);
    fclose(f);

    if (read != n_elements) {
        free(data);
        return NULL;
    }

    return data;
}

static int16_t* load_fp16_file(const char* path, size_t n_elements) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    int16_t* data = malloc(n_elements * sizeof(int16_t));
    if (!data) {
        fclose(f);
        return NULL;
    }

    size_t read = fread(data, sizeof(int16_t), n_elements, f);
    fclose(f);

    if (read != n_elements) {
        free(data);
        return NULL;
    }

    return data;
}

static uint8_t* load_uint8_file(const char* path, size_t n_elements) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    uint8_t* data = malloc(n_elements);
    if (!data) {
        fclose(f);
        return NULL;
    }

    size_t read = fread(data, 1, n_elements, f);
    fclose(f);

    if (read != n_elements) {
        free(data);
        return NULL;
    }

    return data;
}

static int load_layer_weights(LayerMatmulContext* lc, const char* layer_dir,
                               int hidden_dim, int num_q_heads, int num_kv_heads,
                               int head_dim, int ffn_dim, QuantizationType quant_type,
                               int has_qk_norm) {
    char path[512];
    int is_int4 = (quant_type == QUANT_INT4 || quant_type == QUANT_INT4_G128);
    int is_int8 = (quant_type == QUANT_INT8);
    int is_quantized = is_int4 || is_int8;

    /* Load norm weights (FP32) */
    snprintf(path, sizeof(path), "%s/input_norm.bin", layer_dir);
    lc->input_norm_w = load_fp32_file(path, hidden_dim);
    if (!lc->input_norm_w) {
        /* Try alternate name */
        snprintf(path, sizeof(path), "%s/input_norm_weight.bin", layer_dir);
        lc->input_norm_w = load_fp32_file(path, hidden_dim);
    }

    snprintf(path, sizeof(path), "%s/post_attn_norm.bin", layer_dir);
    lc->post_attn_norm_w = load_fp32_file(path, hidden_dim);
    if (!lc->post_attn_norm_w) {
        snprintf(path, sizeof(path), "%s/post_norm.bin", layer_dir);
        lc->post_attn_norm_w = load_fp32_file(path, hidden_dim);
    }
    if (!lc->post_attn_norm_w) {
        snprintf(path, sizeof(path), "%s/post_attn_norm_weight.bin", layer_dir);
        lc->post_attn_norm_w = load_fp32_file(path, hidden_dim);
    }

    if (!lc->input_norm_w || !lc->post_attn_norm_w) {
        fprintf(stderr, "[MatmulDecoder] Failed to load norm weights from %s\n", layer_dir);
        return -1;
    }

    /* Load QK norm weights if enabled */
    if (has_qk_norm) {
        snprintf(path, sizeof(path), "%s/q_norm.bin", layer_dir);
        lc->q_norm_w = load_fp32_file(path, head_dim);
        if (!lc->q_norm_w) {
            snprintf(path, sizeof(path), "%s/q_norm_weight.bin", layer_dir);
            lc->q_norm_w = load_fp32_file(path, head_dim);
        }

        snprintf(path, sizeof(path), "%s/k_norm.bin", layer_dir);
        lc->k_norm_w = load_fp32_file(path, head_dim);
        if (!lc->k_norm_w) {
            snprintf(path, sizeof(path), "%s/k_norm_weight.bin", layer_dir);
            lc->k_norm_w = load_fp32_file(path, head_dim);
        }

        if (!lc->q_norm_w || !lc->k_norm_w) {
            fprintf(stderr, "[MatmulDecoder] Failed to load QK norm weights from %s\n", layer_dir);
            return -1;
        }
    }

    /* Load projection weights into matmul B memory */
    /* For INT4: weights are packed uint8, scales are float32 */
    /* For FP16: weights are int16 (fp16 bits) */

    struct {
        const char* name;
        int K, N;
        PersistentMatmul* pm;
    } projs[] = {
        {"q_proj", hidden_dim, num_q_heads * head_dim, &lc->q_proj},
        {"k_proj", hidden_dim, num_kv_heads * head_dim, &lc->k_proj},
        {"v_proj", hidden_dim, num_kv_heads * head_dim, &lc->v_proj},
        {"o_proj", num_q_heads * head_dim, hidden_dim, &lc->o_proj},
        {"gate_proj", hidden_dim, ffn_dim, &lc->gate_proj},
        {"up_proj", hidden_dim, ffn_dim, &lc->up_proj},
        {"down_proj", ffn_dim, hidden_dim, &lc->down_proj},
        {NULL, 0, 0, NULL}
    };

    for (int i = 0; projs[i].name; i++) {
        int K = projs[i].K;
        int N = projs[i].N;
        PersistentMatmul* pm = projs[i].pm;

        if (is_quantized) {
            /* Load quantized weights: INT4 (packed uint8, K*N/2) or INT8 (uint8, K*N) */
            snprintf(path, sizeof(path), "%s/%s_weight.bin", layer_dir, projs[i].name);
            size_t w_bytes = is_int4 ? (size_t)N * K / 2 : (size_t)N * K;
            uint8_t* w_data = load_uint8_file(path, w_bytes);
            if (!w_data) {
                /* Try FP16 fallback */
                snprintf(path, sizeof(path), "%s/%s.bin", layer_dir, projs[i].name);
                int16_t* fp16_data = load_fp16_file(path, (size_t)K * N);
                if (fp16_data) {
                    rknn_B_normal_layout_to_native_layout(fp16_data, pm->mem_B->virt_addr,
                                                         pm->K, pm->N, pm->pool_info);
                    free(fp16_data);
                    pm->col_scales = NULL;
                    continue;
                }
                fprintf(stderr, "[MatmulDecoder] Warning: Failed to load %s\n", path);
                continue;
            }

            /* Load per-column scales (needed for both INT4 and INT8) */
            snprintf(path, sizeof(path), "%s/%s_scales.bin", layer_dir, projs[i].name);
            pm->col_scales = load_fp32_file(path, N);
            if (!pm->col_scales) {
                fprintf(stderr, "[MatmulDecoder] Warning: No scales for %s, output will be unscaled\n", projs[i].name);
            }

            /* Convert to native layout */
            rknn_B_normal_layout_to_native_layout(w_data, pm->mem_B->virt_addr,
                                                     pm->K, pm->N, pm->pool_info);
            free(w_data);
        } else {
            /* Load FP16 weights */
            snprintf(path, sizeof(path), "%s/%s.bin", layer_dir, projs[i].name);
            int16_t* w_data = load_fp16_file(path, (size_t)K * N);
            if (!w_data) {
                fprintf(stderr, "[MatmulDecoder] Warning: Failed to load %s\n", path);
                continue;
            }

            rknn_B_normal_layout_to_native_layout(w_data, pm->mem_B->virt_addr,
                                                     pm->K, pm->N, pm->pool_info);
            free(w_data);
            pm->col_scales = NULL;
        }
    }

    return 0;
}

/* ─── Decoder Creation ─── */

MatmulDecoderContext* matmul_decoder_create(const char* model_dir,
                                            const MatmulDecoderConfig* config,
                                            QuantizationType quant_type,
                                            int max_seq_len) {
    char path[512];

    /* Load config if not provided */
    MatmulDecoderConfig loaded_config;
    if (!config) {
        snprintf(path, sizeof(path), "%s/config.json", model_dir);
        if (matmul_decoder_load_config(path, &loaded_config) != 0) {
            fprintf(stderr, "[MatmulDecoder] Failed to load config from %s\n", path);
            return NULL;
        }
        config = &loaded_config;
    }

    printf("[MatmulDecoder] Model: %s, dir: %s\n", config->name, model_dir);

    MatmulDecoderContext* ctx = calloc(1, sizeof(MatmulDecoderContext));
    if (!ctx) return NULL;

    ctx->config = *config;
    ctx->quant_type = quant_type;
    ctx->exec_mode = config->exec_mode;
    ctx->n_workers = (config->exec_mode == EXEC_DUAL_CORE) ? 2 : 1;

    int hidden_dim = config->hidden_dim;
    int num_layers = config->num_layers;
    int num_q_heads = config->num_q_heads;
    int num_kv_heads = config->num_kv_heads;
    int head_dim = config->head_dim;
    int ffn_dim = config->ffn_dim;
    int vocab_size = config->vocab_size;

    printf("[MatmulDecoder] Exec mode: %s\n",
           ctx->exec_mode == EXEC_DUAL_CORE ? "dual-core" : "single-core");

    /* Load embeddings */
    snprintf(path, sizeof(path), "%s/embeddings.bin", model_dir);
    ctx->embeddings = load_fp32_file(path, vocab_size * hidden_dim);
    if (!ctx->embeddings) {
        fprintf(stderr, "[MatmulDecoder] Failed to load embeddings from %s\n", path);
        matmul_decoder_destroy(ctx);
        return NULL;
    }
    printf("[MatmulDecoder] Loaded embeddings: [%d, %d]\n", vocab_size, hidden_dim);

    /* Load lm_head(s) */
    int n_lm = config->num_lm_heads;
    ctx->num_lm_heads = n_lm;

    if (n_lm > 1) {
        /* Multi lm_head mode (e.g., Code Predictor: 15 heads) */
        int lm_vocab = config->lm_head_vocab_size > 0 ? config->lm_head_vocab_size : vocab_size;
        ctx->lm_heads = calloc(n_lm, sizeof(float*));
        if (!ctx->lm_heads) {
            matmul_decoder_destroy(ctx);
            return NULL;
        }
        for (int i = 0; i < n_lm; i++) {
            snprintf(path, sizeof(path), "%s/lm_head_%02d.bin", model_dir, i);
            ctx->lm_heads[i] = load_fp32_file(path, (size_t)lm_vocab * hidden_dim);
            if (!ctx->lm_heads[i]) {
                fprintf(stderr, "[MatmulDecoder] Failed to load lm_head_%02d from %s\n", i, path);
                matmul_decoder_destroy(ctx);
                return NULL;
            }
        }
        ctx->lm_head = NULL;
        printf("[MatmulDecoder] Loaded %d lm_heads (vocab=%d each)\n", n_lm, lm_vocab);
    } else {
        /* Single lm_head mode */
        ctx->lm_heads = NULL;
        snprintf(path, sizeof(path), "%s/lm_head.bin", model_dir);
        ctx->lm_head = load_fp32_file(path, vocab_size * hidden_dim);
        if (!ctx->lm_head && !config->tie_word_embeddings) {
            fprintf(stderr, "[MatmulDecoder] Failed to load lm_head from %s\n", path);
            matmul_decoder_destroy(ctx);
            return NULL;
        }
        if (config->tie_word_embeddings && !ctx->lm_head) {
            ctx->lm_head = ctx->embeddings;  /* Share memory */
            printf("[MatmulDecoder] Using tied embeddings for lm_head\n");
        }
    }

    /* Create FP16 copy of lm_head for fast GEMV (halves memory bandwidth) */
    if (ctx->lm_head && n_lm <= 1) {
        size_t lm_size = (size_t)vocab_size * hidden_dim;
        ctx->lm_head_fp16 = malloc(lm_size * sizeof(int16_t));
        if (ctx->lm_head_fp16) {
            vec_fp32_to_fp16(ctx->lm_head_fp16, ctx->lm_head, lm_size);
            printf("[MatmulDecoder] Created FP16 lm_head for fast GEMV (%zuMB → %zuMB)\n",
                   lm_size * 4 / (1024*1024), lm_size * 2 / (1024*1024));
        }
    }

    /* Allocate layer contexts */
    ctx->layers = calloc(num_layers, sizeof(LayerMatmulContext));
    if (!ctx->layers) {
        matmul_decoder_destroy(ctx);
        return NULL;
    }

    /* Decide pooling strategy:
     * 0 = auto: pool if layers*7 > 128 AND handle budget is tight
     * 1 = force pool (saves handles, ~250ms rebind overhead)
     * 2 = force dedicated (no rebind, fastest, uses ~784 handles for 28 layers) */
    int pool_mode = config->context_pool_mode;
    if (pool_mode == 0) {
        /* Auto: use dedicated only for small models (<=16 layers, <=112 contexts).
         * Large models (28 layers = 196 contexts) cause NPU driver scheduling issues
         * even when handle count is under 1020. */
        int n_contexts = num_layers * 7;
        pool_mode = (n_contexts > 112) ? 1 : 2;
        printf("[MatmulDecoder] Auto pool mode: %s (%d contexts)\n",
               pool_mode == 1 ? "pooled" : "dedicated", n_contexts);
    }

    struct { const char* name; int K; int N; } proj_defs[] = {
        {"q_proj",    hidden_dim,              num_q_heads * head_dim},
        {"k_proj",    hidden_dim,              num_kv_heads * head_dim},
        {"v_proj",    hidden_dim,              num_kv_heads * head_dim},
        {"o_proj",    num_q_heads * head_dim,  hidden_dim},
        {"gate_proj", hidden_dim,              ffn_dim},
        {"up_proj",   hidden_dim,              ffn_dim},
        {"down_proj", ffn_dim,                 hidden_dim},
    };

    ctx->n_pool = 0;

    if (pool_mode == 1) {
        /* Pooled mode: share ctx/A/C, rebind B each run */
        int proj_pool_idx[7];
        for (int p = 0; p < 7; p++) {
            proj_pool_idx[p] = pool_get_or_create(ctx->pool, &ctx->n_pool,
                                                   proj_defs[p].K, proj_defs[p].N, quant_type,
                                                   config->iommu_domain_id);
            if (proj_pool_idx[p] < 0) {
                fprintf(stderr, "[MatmulDecoder] Failed to create pool for %s\n", proj_defs[p].name);
                matmul_decoder_destroy(ctx);
                return NULL;
            }
        }
        printf("[MatmulDecoder] Pooled: %d pool entries, %d B handles\n", ctx->n_pool, num_layers * 7);

        for (int i = 0; i < num_layers; i++) {
            LayerMatmulContext* lc = &ctx->layers[i];
            PersistentMatmul* pms[] = {
                &lc->q_proj, &lc->k_proj, &lc->v_proj, &lc->o_proj,
                &lc->gate_proj, &lc->up_proj, &lc->down_proj
            };
            for (int p = 0; p < 7; p++) {
                int ret = create_pooled_matmul(pms[p], &ctx->pool[proj_pool_idx[p]]);
                if (ret != 0) {
                    fprintf(stderr, "Layer %d %s failed\n", i, proj_defs[p].name);
                }
            }
        }  /* end pooled for-loop */
    } else {
        /* Dedicated mode: one context per projection, no rebind (fastest) */
        printf("[MatmulDecoder] Dedicated: %d contexts (no B rebind)\n", num_layers * 7);

        for (int i = 0; i < num_layers; i++) {
            LayerMatmulContext* lc = &ctx->layers[i];
            struct { PersistentMatmul* pm; int K; int N; } pl[] = {
                {&lc->q_proj,    proj_defs[0].K, proj_defs[0].N},
                {&lc->k_proj,    proj_defs[1].K, proj_defs[1].N},
                {&lc->v_proj,    proj_defs[2].K, proj_defs[2].N},
                {&lc->o_proj,    proj_defs[3].K, proj_defs[3].N},
                {&lc->gate_proj, proj_defs[4].K, proj_defs[4].N},
                {&lc->up_proj,   proj_defs[5].K, proj_defs[5].N},
                {&lc->down_proj, proj_defs[6].K, proj_defs[6].N},
            };
            for (int p = 0; p < 7; p++) {
                int ret = create_persistent_matmul(pl[p].pm, 1, pl[p].K, pl[p].N, quant_type,
                                                     config->iommu_domain_id);
                if (ret != 0) {
                    fprintf(stderr, "Layer %d %s failed\n", i, proj_defs[p].name);
                }
            }
        }  /* end dedicated for-loop body */
    }  /* end if pool_mode */

    /* Load weights for all layers (shared by both pool and dedicated modes) */
    for (int i = 0; i < num_layers; i++) {
        LayerMatmulContext* lc = &ctx->layers[i];

        /* Load weights for this layer.
         * Try "layers/layer_NN" first (new layout), fall back to "layer_NN". */
        char path_alt[512];
        snprintf(path_alt, sizeof(path_alt), "%s/layers/layer_%02d", model_dir, i);
        snprintf(path, sizeof(path), "%s/layer_%02d", model_dir, i);
        {
            /* Check if layers/ subdirectory exists by trying to stat it */
            struct stat st;
            if (stat(path_alt, &st) == 0 && S_ISDIR(st.st_mode)) {
                /* Use the layers/ subdirectory layout */
                snprintf(path, sizeof(path), "%s/layers/layer_%02d", model_dir, i);
            }
        }
        int ret = load_layer_weights(lc, path, hidden_dim, num_q_heads, num_kv_heads, head_dim, ffn_dim, quant_type, config->has_qk_norm);
        if (ret != 0) {
            fprintf(stderr, "[MatmulDecoder] Failed to load layer %d weights\n", i);
        }
    }

    int total_handles = (pool_mode == 1)
        ? ctx->n_pool * 3 + num_layers * 7   /* pool: ctx+A+C shared + per-layer B */
        : num_layers * 7 * 4;                /* dedicated: ctx+A+B+C per projection */
    printf("[MatmulDecoder] Loaded %d layers (qk_norm=%d, %s), NPU handles: ~%d\n",
           num_layers, config->has_qk_norm,
           pool_mode == 1 ? "pooled" : "dedicated", total_handles);

    /* Load final norm weight (model.norm, separate from per-layer norms) */
    snprintf(path, sizeof(path), "%s/final_norm.bin", model_dir);
    ctx->final_norm_w = load_fp32_file(path, hidden_dim);
    if (!ctx->final_norm_w) {
        snprintf(path, sizeof(path), "%s/model_norm.bin", model_dir);
        ctx->final_norm_w = load_fp32_file(path, hidden_dim);
    }
    if (!ctx->final_norm_w) {
        /* Fallback: use last layer's post_attn_norm (incorrect but avoids crash) */
        fprintf(stderr, "[MatmulDecoder] WARNING: final_norm.bin not found, using last layer post_norm\n");
        ctx->final_norm_w = ctx->layers[num_layers - 1].post_attn_norm_w;
    } else {
        printf("[MatmulDecoder] Loaded final_norm weight\n");
    }

    /* KV cache */
    ctx->kv_cache = matmul_kv_cache_create(num_layers, num_kv_heads, head_dim, max_seq_len);

    /* RoPE tables */
    ctx->cos_table = malloc((size_t)max_seq_len * (head_dim / 2) * sizeof(float));
    ctx->sin_table = malloc((size_t)max_seq_len * (head_dim / 2) * sizeof(float));
    precompute_rope_tables(ctx->cos_table, ctx->sin_table, max_seq_len, head_dim, config->rope_theta);

    /* Working buffers */
    int q_dim = num_q_heads * head_dim;
    int logits_size = vocab_size;
    if (n_lm > 1 && config->lm_head_vocab_size > logits_size) {
        logits_size = config->lm_head_vocab_size;
    }
    ctx->hidden = aligned_alloc(64, hidden_dim * sizeof(float));
    ctx->residual = aligned_alloc(64, hidden_dim * sizeof(float));
    ctx->normed = aligned_alloc(64, hidden_dim * sizeof(float));
    ctx->q_out = aligned_alloc(64, q_dim * sizeof(float));
    ctx->k_out = aligned_alloc(64, num_kv_heads * head_dim * sizeof(float));
    ctx->v_out = aligned_alloc(64, num_kv_heads * head_dim * sizeof(float));
    ctx->attn_out = aligned_alloc(64, q_dim * sizeof(float));
    ctx->ffn_gate = aligned_alloc(64, ffn_dim * sizeof(float));
    ctx->ffn_up = aligned_alloc(64, ffn_dim * sizeof(float));
    ctx->ffn_down = aligned_alloc(64, hidden_dim * sizeof(float));
    ctx->logits = aligned_alloc(64, logits_size * sizeof(float));

    printf("[MatmulDecoder] Ready: %d layers, hidden=%d, q_dim=%d, heads=%d/%d, ffn=%d\n",
           num_layers, hidden_dim, q_dim, num_q_heads, num_kv_heads, ffn_dim);

    return ctx;
}

MatmulDecoderContext* matmul_decoder_create_from_weights(
    const MatmulDecoderConfig* config,
    const float* embeddings,
    const LayerWeights* layers,
    const float* lm_head,
    QuantizationType quant_type,
    int max_seq_len
) {
    MatmulDecoderContext* ctx = calloc(1, sizeof(MatmulDecoderContext));
    if (!ctx) return NULL;

    ctx->config = *config;
    ctx->quant_type = quant_type;
    ctx->exec_mode = config->exec_mode;
    ctx->n_workers = (config->exec_mode == EXEC_DUAL_CORE) ? 2 : 1;

    int hidden_dim = config->hidden_dim;
    int num_layers = config->num_layers;
    int num_q_heads = config->num_q_heads;
    int num_kv_heads = config->num_kv_heads;
    int head_dim = config->head_dim;
    int ffn_dim = config->ffn_dim;
    int vocab_size = config->vocab_size;

    printf("[MatmulDecoder] Execution mode: %s\n",
           ctx->exec_mode == EXEC_DUAL_CORE ? "dual-core" : "single-core");

    /* Allocate embeddings */
    ctx->embeddings = malloc((size_t)vocab_size * hidden_dim * sizeof(float));
    memcpy(ctx->embeddings, embeddings, (size_t)vocab_size * hidden_dim * sizeof(float));

    /* LM head(s) */
    ctx->num_lm_heads = config->num_lm_heads;
    ctx->lm_heads = NULL;
    if (lm_head) {
        ctx->lm_head = malloc((size_t)vocab_size * hidden_dim * sizeof(float));
        memcpy(ctx->lm_head, lm_head, (size_t)vocab_size * hidden_dim * sizeof(float));
    } else if (config->tie_word_embeddings) {
        ctx->lm_head = ctx->embeddings;  /* Share memory */
    }

    /* Allocate layer contexts */
    ctx->layers = calloc(num_layers, sizeof(LayerMatmulContext));
    if (!ctx->layers) {
        matmul_decoder_destroy(ctx);
        return NULL;
    }

    /* Build context pool (same as matmul_decoder_create) */
    ctx->n_pool = 0;
    {
        int dims[][2] = {
            {hidden_dim, num_q_heads * head_dim},   /* q */
            {hidden_dim, num_kv_heads * head_dim},  /* k, v */
            {num_q_heads * head_dim, hidden_dim},   /* o */
            {hidden_dim, ffn_dim},                  /* gate, up */
            {ffn_dim, hidden_dim},                  /* down */
        };
        for (int d = 0; d < 5; d++) {
            pool_get_or_create(ctx->pool, &ctx->n_pool, dims[d][0], dims[d][1], quant_type,
                              config->iommu_domain_id);
        }
    }

    /* Create pooled projections for each layer */
    for (int i = 0; i < num_layers; i++) {
        LayerMatmulContext* lc = &ctx->layers[i];
        struct { PersistentMatmul* pm; int K; int N; } proj_list[] = {
            {&lc->q_proj,    hidden_dim, num_q_heads * head_dim},
            {&lc->k_proj,    hidden_dim, num_kv_heads * head_dim},
            {&lc->v_proj,    hidden_dim, num_kv_heads * head_dim},
            {&lc->o_proj,    num_q_heads * head_dim, hidden_dim},
            {&lc->gate_proj, hidden_dim, ffn_dim},
            {&lc->up_proj,   hidden_dim, ffn_dim},
            {&lc->down_proj, ffn_dim, hidden_dim},
        };
        for (int p = 0; p < 7; p++) {
            int pidx = pool_get_or_create(ctx->pool, &ctx->n_pool,
                                           proj_list[p].K, proj_list[p].N, quant_type,
                                           config->iommu_domain_id);
            if (pidx >= 0) {
                create_pooled_matmul(proj_list[p].pm, &ctx->pool[pidx]);
            }
        }

        /* Norm weights */
        lc->input_norm_w = malloc(hidden_dim * sizeof(float));
        lc->post_attn_norm_w = malloc(hidden_dim * sizeof(float));
        /* TODO: Copy from layers[i].input_norm_weight, etc. */
    }

    /* KV cache */
    ctx->kv_cache = matmul_kv_cache_create(num_layers, num_kv_heads, head_dim, max_seq_len);

    /* RoPE tables */
    ctx->cos_table = malloc((size_t)max_seq_len * (head_dim / 2) * sizeof(float));
    ctx->sin_table = malloc((size_t)max_seq_len * (head_dim / 2) * sizeof(float));
    precompute_rope_tables(ctx->cos_table, ctx->sin_table, max_seq_len, head_dim, config->rope_theta);

    /* Working buffers */
    int q_dim = num_q_heads * head_dim;
    ctx->hidden = aligned_alloc(64, hidden_dim * sizeof(float));
    ctx->q_out = aligned_alloc(64, q_dim * sizeof(float));
    ctx->k_out = aligned_alloc(64, num_kv_heads * head_dim * sizeof(float));
    ctx->v_out = aligned_alloc(64, num_kv_heads * head_dim * sizeof(float));
    ctx->attn_out = aligned_alloc(64, q_dim * sizeof(float));
    ctx->ffn_gate = aligned_alloc(64, ffn_dim * sizeof(float));
    ctx->ffn_up = aligned_alloc(64, ffn_dim * sizeof(float));
    ctx->ffn_down = aligned_alloc(64, hidden_dim * sizeof(float));
    ctx->logits = aligned_alloc(64, vocab_size * sizeof(float));

    printf("[MatmulDecoder] Created: %d layers, hidden=%d, q_dim=%d, heads=%d/%d, ffn=%d\n", q_dim,
           num_layers, hidden_dim, num_q_heads, num_kv_heads, ffn_dim);

    return ctx;
}

/* ─── Decoder Step ─── */

int matmul_decoder_step(MatmulDecoderContext* ctx,
                        int token_id,
                        const float* embedding,
                        float* output_logits) {
    double _step_t0 = now_ms(), _cpu_t0;
    _acc_rebind_ms = _acc_matmul_ms = _acc_convert_ms = 0;
    double _cpu_ops_ms = 0, _lm_head_ms = 0;

    int hidden_dim = ctx->config.hidden_dim;
    int num_layers = ctx->config.num_layers;
    int num_q_heads = ctx->config.num_q_heads;
    int num_kv_heads = ctx->config.num_kv_heads;
    int head_dim = ctx->config.head_dim;
    int ffn_dim = ctx->config.ffn_dim;
    int vocab_size = ctx->config.vocab_size;

    /* Get embedding */
    if (embedding) {
        memcpy(ctx->hidden, embedding, hidden_dim * sizeof(float));
    } else {
        if (token_id < 0 || token_id >= vocab_size) {
            return MATMUL_DECODER_ERR_INVALID_ARG;
        }
        memcpy(ctx->hidden, ctx->embeddings + token_id * hidden_dim, hidden_dim * sizeof(float));
    }

    /* Bounds check: KV cache full */
    int seq_len = ctx->kv_cache->seq_len;
    if (seq_len >= ctx->kv_cache->max_seq_len) {
        fprintf(stderr, "[MatmulDecoder] KV cache full: seq_len=%d >= max=%d\n",
                seq_len, ctx->kv_cache->max_seq_len);
        return MATMUL_DECODER_ERR_INVALID_ARG;
    }

    for (int layer = 0; layer < num_layers; layer++) {
        LayerMatmulContext* lc = &ctx->layers[layer];
        float* layer_k_cache = ctx->kv_cache->k_cache +
            (layer * ctx->kv_cache->max_seq_len * num_kv_heads * head_dim);
        float* layer_v_cache = ctx->kv_cache->v_cache +
            (layer * ctx->kv_cache->max_seq_len * num_kv_heads * head_dim);

        /* Save hidden for residual + input norm (CPU) */
        _cpu_t0 = now_ms();
        memcpy(ctx->residual, ctx->hidden, hidden_dim * sizeof(float));
        rms_norm_f32(ctx->normed, ctx->hidden, lc->input_norm_w, hidden_dim, ctx->config.rms_eps);
        _cpu_ops_ms += now_ms() - _cpu_t0;

        /* QKV projections */
        run_persistent_matmul(&lc->q_proj, ctx->normed, ctx->q_out);
        run_persistent_matmul(&lc->k_proj, ctx->normed, ctx->k_out);
        run_persistent_matmul(&lc->v_proj, ctx->normed, ctx->v_out);

        /* QK norm + RoPE + attention (CPU) */
        _cpu_t0 = now_ms();
        if (ctx->config.has_qk_norm && lc->q_norm_w && lc->k_norm_w) {
            for (int h = 0; h < num_q_heads; h++) {
                float* q_head = ctx->q_out + h * head_dim;
                rms_norm_f32(q_head, q_head, lc->q_norm_w, head_dim, ctx->config.rms_eps);
            }
            for (int h = 0; h < num_kv_heads; h++) {
                float* k_head = ctx->k_out + h * head_dim;
                rms_norm_f32(k_head, k_head, lc->k_norm_w, head_dim, ctx->config.rms_eps);
            }
        }

        /* RoPE */
        float* cos = ctx->cos_table + seq_len * (head_dim / 2);
        float* sin = ctx->sin_table + seq_len * (head_dim / 2);
        apply_rope_f32(ctx->q_out, cos, sin, num_q_heads, head_dim, ctx->config.rope_style);
        apply_rope_f32(ctx->k_out, cos, sin, num_kv_heads, head_dim, ctx->config.rope_style);

        /* Store K, V in cache */
        memcpy(layer_k_cache + seq_len * num_kv_heads * head_dim, ctx->k_out,
               num_kv_heads * head_dim * sizeof(float));
        memcpy(layer_v_cache + seq_len * num_kv_heads * head_dim, ctx->v_out,
               num_kv_heads * head_dim * sizeof(float));

        /* Attention (CPU) */
        attention_f32(ctx->attn_out, ctx->q_out, layer_k_cache, layer_v_cache,
                      num_q_heads, num_kv_heads, head_dim, seq_len + 1);
        _cpu_ops_ms += now_ms() - _cpu_t0;

        /* o_proj (NPU) + residual (CPU) */
        run_persistent_matmul(&lc->o_proj, ctx->attn_out, ctx->hidden);
        _cpu_t0 = now_ms();
        vec_add_f32(ctx->hidden, ctx->residual, hidden_dim);

        /* Post-attention norm + SiLU (CPU) */
        rms_norm_f32(ctx->normed, ctx->hidden, lc->post_attn_norm_w, hidden_dim, ctx->config.rms_eps);
        _cpu_ops_ms += now_ms() - _cpu_t0;

        /* FFN matmuls (NPU) */
        run_persistent_matmul(&lc->gate_proj, ctx->normed, ctx->ffn_gate);
        run_persistent_matmul(&lc->up_proj, ctx->normed, ctx->ffn_up);

        /* SiLU + residual (CPU) */
        _cpu_t0 = now_ms();
        silu_mul_f32(ctx->ffn_gate, ctx->ffn_gate, ctx->ffn_up, ffn_dim);
        _cpu_ops_ms += now_ms() - _cpu_t0;

        run_persistent_matmul(&lc->down_proj, ctx->ffn_gate, ctx->ffn_down);

        _cpu_t0 = now_ms();
        vec_add_f32(ctx->hidden, ctx->ffn_down, hidden_dim);
        _cpu_ops_ms += now_ms() - _cpu_t0;
    }

    /* Final norm (CPU) */
    _cpu_t0 = now_ms();
    rms_norm_f32(ctx->normed, ctx->hidden,
                 ctx->final_norm_w, hidden_dim, ctx->config.rms_eps);
    _cpu_ops_ms += now_ms() - _cpu_t0;

    /* LM head (NEON GEMV — use FP16 weights if available for 2x bandwidth) */
    _cpu_t0 = now_ms();
    if (ctx->lm_head_fp16) {
        gemv_f16_neon(ctx->logits, ctx->normed, ctx->lm_head_fp16, vocab_size, hidden_dim);
    } else {
        gemv_f32_neon(ctx->logits, ctx->normed, ctx->lm_head, vocab_size, hidden_dim);
    }
    _lm_head_ms = now_ms() - _cpu_t0;

    /* Update KV cache */
    ctx->kv_cache->seq_len++;

    /* Save stats */
    ctx->stats.total_ms = (float)(now_ms() - _step_t0);
    ctx->stats.matmul_ms = (float)_acc_matmul_ms;
    ctx->stats.rebind_ms = (float)_acc_rebind_ms;
    ctx->stats.convert_ms = (float)_acc_convert_ms;
    ctx->stats.cpu_ops_ms = (float)_cpu_ops_ms;
    ctx->stats.lm_head_ms = (float)_lm_head_ms;
    ctx->stats.n_steps++;

    /* Copy logits if requested */
    if (output_logits) {
        memcpy(output_logits, ctx->logits, vocab_size * sizeof(float));
    }

    /* Greedy sampling */
    int predicted_token = argmax_f32(ctx->logits, vocab_size);

    return predicted_token;
}

int matmul_decoder_step_head(MatmulDecoderContext* ctx,
                             int token_id,
                             const float* embedding,
                             int lm_head_idx,
                             float* output_logits) {
    if (!ctx) return MATMUL_DECODER_ERR_INVALID_ARG;

    /* Validate lm_head_idx */
    if (ctx->num_lm_heads <= 1) {
        /* Single-head mode: ignore lm_head_idx, use default lm_head */
        return matmul_decoder_step(ctx, token_id, embedding, output_logits);
    }
    if (lm_head_idx < 0 || lm_head_idx >= ctx->num_lm_heads || !ctx->lm_heads) {
        return MATMUL_DECODER_ERR_INVALID_ARG;
    }

    int hidden_dim = ctx->config.hidden_dim;
    int num_layers = ctx->config.num_layers;
    int num_q_heads = ctx->config.num_q_heads;
    int num_kv_heads = ctx->config.num_kv_heads;
    int head_dim = ctx->config.head_dim;
    int ffn_dim = ctx->config.ffn_dim;
    int lm_vocab = ctx->config.lm_head_vocab_size > 0
                       ? ctx->config.lm_head_vocab_size
                       : ctx->config.vocab_size;

    /* Get embedding */
    if (embedding) {
        memcpy(ctx->hidden, embedding, hidden_dim * sizeof(float));
    } else {
        if (token_id < 0 || token_id >= ctx->config.vocab_size) {
            return MATMUL_DECODER_ERR_INVALID_ARG;
        }
        memcpy(ctx->hidden, ctx->embeddings + token_id * hidden_dim, hidden_dim * sizeof(float));
    }

    /* Bounds check */
    int seq_len = ctx->kv_cache->seq_len;
    if (seq_len >= ctx->kv_cache->max_seq_len) {
        fprintf(stderr, "[MatmulDecoder] KV cache full: seq_len=%d >= max=%d\n",
                seq_len, ctx->kv_cache->max_seq_len);
        return MATMUL_DECODER_ERR_INVALID_ARG;
    }

    for (int layer = 0; layer < num_layers; layer++) {
        LayerMatmulContext* lc = &ctx->layers[layer];
        float* layer_k_cache = ctx->kv_cache->k_cache +
            (layer * ctx->kv_cache->max_seq_len * num_kv_heads * head_dim);
        float* layer_v_cache = ctx->kv_cache->v_cache +
            (layer * ctx->kv_cache->max_seq_len * num_kv_heads * head_dim);

        memcpy(ctx->residual, ctx->hidden, hidden_dim * sizeof(float));
        rms_norm_f32(ctx->normed, ctx->hidden, lc->input_norm_w, hidden_dim, ctx->config.rms_eps);

        run_persistent_matmul(&lc->q_proj, ctx->normed, ctx->q_out);
        run_persistent_matmul(&lc->k_proj, ctx->normed, ctx->k_out);
        run_persistent_matmul(&lc->v_proj, ctx->normed, ctx->v_out);

        if (ctx->config.has_qk_norm && lc->q_norm_w && lc->k_norm_w) {
            for (int h = 0; h < num_q_heads; h++) {
                float* q_head = ctx->q_out + h * head_dim;
                rms_norm_f32(q_head, q_head, lc->q_norm_w, head_dim, ctx->config.rms_eps);
            }
            for (int h = 0; h < num_kv_heads; h++) {
                float* k_head = ctx->k_out + h * head_dim;
                rms_norm_f32(k_head, k_head, lc->k_norm_w, head_dim, ctx->config.rms_eps);
            }
        }

        float* cos = ctx->cos_table + seq_len * (head_dim / 2);
        float* sin = ctx->sin_table + seq_len * (head_dim / 2);
        apply_rope_f32(ctx->q_out, cos, sin, num_q_heads, head_dim, ctx->config.rope_style);
        apply_rope_f32(ctx->k_out, cos, sin, num_kv_heads, head_dim, ctx->config.rope_style);

        memcpy(layer_k_cache + seq_len * num_kv_heads * head_dim, ctx->k_out,
               num_kv_heads * head_dim * sizeof(float));
        memcpy(layer_v_cache + seq_len * num_kv_heads * head_dim, ctx->v_out,
               num_kv_heads * head_dim * sizeof(float));

        attention_f32(ctx->attn_out, ctx->q_out, layer_k_cache, layer_v_cache,
                      num_q_heads, num_kv_heads, head_dim, seq_len + 1);
        run_persistent_matmul(&lc->o_proj, ctx->attn_out, ctx->hidden);
        vec_add_f32(ctx->hidden, ctx->residual, hidden_dim);

        rms_norm_f32(ctx->normed, ctx->hidden, lc->post_attn_norm_w, hidden_dim, ctx->config.rms_eps);
        run_persistent_matmul(&lc->gate_proj, ctx->normed, ctx->ffn_gate);
        run_persistent_matmul(&lc->up_proj, ctx->normed, ctx->ffn_up);
        silu_mul_f32(ctx->ffn_gate, ctx->ffn_gate, ctx->ffn_up, ffn_dim);
        run_persistent_matmul(&lc->down_proj, ctx->ffn_gate, ctx->ffn_down);
        vec_add_f32(ctx->hidden, ctx->ffn_down, hidden_dim);
    }

    /* Final norm */
    rms_norm_f32(ctx->normed, ctx->hidden,
                 ctx->final_norm_w, hidden_dim, ctx->config.rms_eps);

    /* Selected lm_head (NEON-optimized GEMV) */
    gemv_f32_neon(ctx->logits, ctx->normed, ctx->lm_heads[lm_head_idx], lm_vocab, hidden_dim);

    ctx->kv_cache->seq_len++;

    if (output_logits) {
        memcpy(output_logits, ctx->logits, lm_vocab * sizeof(float));
    }

    return argmax_f32(ctx->logits, lm_vocab);
}

void matmul_decoder_clear_kv_cache(MatmulDecoderContext* ctx) {
    if (ctx && ctx->kv_cache) {
        matmul_kv_cache_clear(ctx->kv_cache);
    }
}

int matmul_decoder_get_seq_len(const MatmulDecoderContext* ctx) {
    return ctx ? matmul_kv_cache_get_seq_len(ctx->kv_cache) : 0;
}

const MatmulDecoderConfig* matmul_decoder_get_config(const MatmulDecoderContext* ctx) {
    return ctx ? &ctx->config : NULL;
}

void matmul_decoder_get_stats(const MatmulDecoderContext* ctx, MatmulDecoderStats* stats) {
    if (ctx && stats) {
        *stats = ctx->stats;
    }
}

void matmul_decoder_destroy(MatmulDecoderContext* ctx) {
    if (!ctx) return;

    /* Destroy per-layer resources */
    if (ctx->layers) {
        for (int i = 0; i < ctx->config.num_layers; i++) {
            LayerMatmulContext* lc = &ctx->layers[i];
            /* For pooled projections: skip rknn_destroy_mem (pool ctx owns all DMA mem).
             * For dedicated projections: destroy_persistent_matmul handles full cleanup. */
            destroy_persistent_matmul(&lc->q_proj);
            destroy_persistent_matmul(&lc->k_proj);
            destroy_persistent_matmul(&lc->v_proj);
            destroy_persistent_matmul(&lc->o_proj);
            destroy_persistent_matmul(&lc->gate_proj);
            destroy_persistent_matmul(&lc->up_proj);
            destroy_persistent_matmul(&lc->down_proj);
            free(lc->input_norm_w);
            free(lc->post_attn_norm_w);
            free(lc->q_norm_w);
            free(lc->k_norm_w);
        }
        free(ctx->layers);
        ctx->layers = NULL;
    }

    /* Destroy pool contexts — rknn_matmul_destroy frees all DMA mem
     * allocated from that ctx (including per-layer B buffers). */
    destroy_pool(ctx->pool, ctx->n_pool);
    ctx->n_pool = 0;

    /* Free other resources */
    matmul_kv_cache_destroy(ctx->kv_cache);
    free(ctx->embeddings);
    if (ctx->lm_heads) {
        for (int i = 0; i < ctx->num_lm_heads; i++) {
            free(ctx->lm_heads[i]);
        }
        free(ctx->lm_heads);
    } else if (!ctx->config.tie_word_embeddings) {
        free(ctx->lm_head);
    }
    free(ctx->lm_head_fp16);

    /* final_norm_w: only free if it's NOT a borrowed pointer from last layer */
    if (ctx->final_norm_w && ctx->layers &&
        ctx->final_norm_w != ctx->layers[ctx->config.num_layers - 1].post_attn_norm_w) {
        free(ctx->final_norm_w);
    }
    free(ctx->cos_table);
    free(ctx->sin_table);
    free(ctx->hidden);
    free(ctx->residual);
    free(ctx->normed);
    free(ctx->q_out);
    free(ctx->k_out);
    free(ctx->v_out);
    free(ctx->attn_out);
    free(ctx->ffn_gate);
    free(ctx->ffn_up);
    free(ctx->ffn_down);
    free(ctx->logits);

    free(ctx);
}

/* ─── JSON Config Parsing ─── */

/**
 * Simple JSON parser for config.json.
 * No external dependencies - parses only the fields we need.
 */
int matmul_decoder_load_config(const char* json_path, MatmulDecoderConfig* config) {
    FILE* f = fopen(json_path, "r");
    if (!f) {
        fprintf(stderr, "[MatmulDecoder] Cannot open config: %s\n", json_path);
        return MATMUL_DECODER_ERR_FILE_IO;
    }

    /* Read entire file */
    char* buffer = NULL;
    size_t size = 0;
    fseek(f, 0, SEEK_END);
    size = ftell(f);
    fseek(f, 0, SEEK_SET);

    buffer = malloc(size + 1);
    if (!buffer) {
        fclose(f);
        return MATMUL_DECODER_ERR_MEMORY;
    }

    size_t read_size = fread(buffer, 1, size, f);
    buffer[read_size] = '\0';
    fclose(f);

    /* Simple JSON field extraction */
    /* Initialize with defaults */
    *config = matmul_decoder_config_qwen3_0_6b();

    /* Helper: find "key": value pattern */
    char* p = buffer;
    while (*p) {
        /* Skip whitespace */
        while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
        if (!*p) break;

        /* Look for quoted key */
        if (*p == '"') {
            p++;
            char* key_start = p;
            while (*p && *p != '"') p++;
            if (!*p) break;
            size_t key_len = p - key_start;
            p++; /* skip closing quote */

            /* Skip to colon */
            while (*p && *p != ':') p++;
            if (!*p) break;
            p++; /* skip colon */

            /* Skip whitespace */
            while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;

            /* Parse value */
            if (*p == '"') {
                /* String value */
                p++;
                char* val_start = p;
                while (*p && *p != '"') p++;
                size_t val_len = p - val_start;

                /* Match known keys */
                if (strncmp(key_start, "name", key_len) == 0) {
                    /* name is const char* in struct, we skip it for now */
                } else if (strncmp(key_start, "quant_type", key_len) == 0) {
                    /* quant_type string - not stored in MatmulDecoderConfig */
                }
                p++;
            } else if (*p == '-' || (*p >= '0' && *p <= '9')) {
                /* Numeric value */
                char* num_start = p;
                if (*p == '-') p++;
                while (*p && (*p >= '0' && *p <= '9')) p++;
                if (*p == '.') {
                    p++;
                    while (*p && (*p >= '0' && *p <= '9')) p++;
                }
                char num_buf[32];
                size_t num_len = p - num_start;
                if (num_len < 32) {
                    memcpy(num_buf, num_start, num_len);
                    num_buf[num_len] = '\0';

                    /* Match known keys */
                    if (strncmp(key_start, "hidden_dim", key_len) == 0) {
                        config->hidden_dim = atoi(num_buf);
                    } else if (strncmp(key_start, "num_q_heads", key_len) == 0) {
                        config->num_q_heads = atoi(num_buf);
                    } else if (strncmp(key_start, "num_kv_heads", key_len) == 0) {
                        config->num_kv_heads = atoi(num_buf);
                    } else if (strncmp(key_start, "head_dim", key_len) == 0) {
                        config->head_dim = atoi(num_buf);
                    } else if (strncmp(key_start, "ffn_dim", key_len) == 0) {
                        config->ffn_dim = atoi(num_buf);
                    } else if (strncmp(key_start, "num_layers", key_len) == 0) {
                        config->num_layers = atoi(num_buf);
                    } else if (strncmp(key_start, "vocab_size", key_len) == 0) {
                        config->vocab_size = atoi(num_buf);
                    } else if (strncmp(key_start, "max_seq_len", key_len) == 0) {
                        config->max_seq_len = atoi(num_buf);
                    } else if (strncmp(key_start, "rms_eps", key_len) == 0) {
                        config->rms_eps = atof(num_buf);
                    } else if (strncmp(key_start, "rope_theta", key_len) == 0) {
                        config->rope_theta = atof(num_buf);
                    } else if (strncmp(key_start, "has_qk_norm", key_len) == 0) {
                        config->has_qk_norm = atoi(num_buf);
                    }
                }
            } else if (*p == 't' || *p == 'f') {
                /* Boolean */
                if (strncmp(key_start, "tie_word_embeddings", key_len) == 0) {
                    config->tie_word_embeddings = (*p == 't');
                    p += (*p == 't' ? 4 : 5); /* true/false */
                } else if (strncmp(key_start, "has_qk_norm", key_len) == 0) {
                    config->has_qk_norm = (*p == 't');
                    p += (*p == 't' ? 4 : 5); /* true/false */
                }
            }
        } else {
            p++;
        }
    }

    free(buffer);

    printf("[MatmulDecoder] Loaded config: hidden=%d, layers=%d, heads=%d/%d, vocab=%d\n",
           config->hidden_dim, config->num_layers,
           config->num_q_heads, config->num_kv_heads,
           config->vocab_size);

    return MATMUL_DECODER_OK;
}

/* ─── Sampling ─── */

int matmul_sample_token(const float* logits, int vocab_size, SamplingParams* params) {
    if (params->top_k == 1) {
        /* Greedy */
        return argmax_f32(logits, vocab_size);
    }

    /* TODO: Implement top-k, top-p sampling */
    return argmax_f32(logits, vocab_size);
}