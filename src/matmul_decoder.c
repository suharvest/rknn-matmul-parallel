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
#include <arm_neon.h>

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
} PersistentMatmul;

/**
 * Context pool entry — one per unique (K, N) dimension pair.
 * Shared by all layers that have the same projection dimensions.
 */
typedef struct {
    rknn_matmul_ctx ctx;
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
    float* lm_head;             /* Single lm_head: [vocab_size, hidden_dim] */
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

    /* Working buffers */
    float* hidden;              /* [hidden_dim] */
    float* q_out;               /* [num_q_heads, head_dim] */
    float* k_out;               /* [num_kv_heads, head_dim] */
    float* v_out;               /* [num_kv_heads, head_dim] */
    float* attn_out;            /* [hidden_dim] */
    float* ffn_gate;            /* [ffn_dim] */
    float* ffn_up;              /* [ffn_dim] */
    float* ffn_down;            /* [hidden_dim] */
    float* logits;              /* [vocab_size] */

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
        case QUANT_FP16:     return RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
        case QUANT_INT4:
        case QUANT_INT4_G128: return RKNN_FLOAT16_MM_INT4_TO_FLOAT16;
        case QUANT_INT8:     return RKNN_FLOAT16_MM_INT8_TO_FLOAT16;
        default:             return RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
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

    return MATMUL_DECODER_OK;
}

/**
 * Legacy: create standalone matmul context (for small models / non-pooled mode).
 */
static int create_persistent_matmul(PersistentMatmul* pm, int M, int K, int N,
                                     QuantizationType quant_type) {
    memset(pm, 0, sizeof(PersistentMatmul));

    rknn_matmul_info info;
    memset(&info, 0, sizeof(info));
    info.M = M;
    info.K = K;
    info.N = N;
    info.type = quant_to_rknn_type(quant_type);
    info.B_layout = 1;

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

    return MATMUL_DECODER_OK;
}

static void destroy_persistent_matmul(PersistentMatmul* pm) {
    if (!pm->initialized) return;

    if (pm->is_pooled) {
        /* Pooled: only destroy B (ctx/A/C are shared via pool) */
        if (pm->mem_B) rknn_destroy_mem(pm->ctx, pm->mem_B);
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
        if (pe->mem_A) rknn_destroy_mem(pe->ctx, pe->mem_A);
        if (pe->mem_C) rknn_destroy_mem(pe->ctx, pe->mem_C);
        if (pe->ctx) rknn_matmul_destroy(pe->ctx);
        memset(pe, 0, sizeof(*pe));
    }
}

static int run_persistent_matmul(PersistentMatmul* pm, const float* input_fp32, float* output_fp32) {
    int K = pm->K, N = pm->N;

    /* Pooled mode: rebind B weight before each run (~10μs ioctl) */
    if (pm->is_pooled) {
        rknn_matmul_set_io_mem(pm->ctx, pm->mem_B, &pm->io.B);
    }

    /* Convert input to FP16 */
    vec_fp32_to_fp16((int16_t*)pm->mem_A->virt_addr, input_fp32, K);

    /* Run matmul */
    int ret = rknn_matmul_run(pm->ctx);
    if (ret != 0) {
        return MATMUL_DECODER_ERR_RKNN;
    }

    /* Convert output to FP32 */
    vec_fp16_to_fp32(output_fp32, (int16_t*)pm->mem_C->virt_addr, N);

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

    /* Load norm weights (FP32) */
    snprintf(path, sizeof(path), "%s/input_norm.bin", layer_dir);
    lc->input_norm_w = load_fp32_file(path, hidden_dim);
    if (!lc->input_norm_w) {
        /* Try alternate name */
        snprintf(path, sizeof(path), "%s/input_norm_weight.bin", layer_dir);
        lc->input_norm_w = load_fp32_file(path, hidden_dim);
    }

    snprintf(path, sizeof(path), "%s/post_norm.bin", layer_dir);
    lc->post_attn_norm_w = load_fp32_file(path, hidden_dim);
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

        if (is_int4) {
            /* Load INT4 packed weights (uint8) */
            snprintf(path, sizeof(path), "%s/%s_weight.bin", layer_dir, projs[i].name);
            uint8_t* w_data = load_uint8_file(path, (size_t)N * K / 2);  /* Packed: N*K bits = N*K/2 bytes */
            if (!w_data) {
                /* Try FP16 fallback */
                snprintf(path, sizeof(path), "%s/%s.bin", layer_dir, projs[i].name);
                int16_t* fp16_data = load_fp16_file(path, (size_t)K * N);
                if (fp16_data) {
                    memcpy(pm->mem_B->virt_addr, fp16_data, K * N * sizeof(int16_t));
                    free(fp16_data);
                    continue;
                }
                fprintf(stderr, "[MatmulDecoder] Warning: Failed to load %s\n", path);
                continue;
            }

            /* Load scales */
            snprintf(path, sizeof(path), "%s/%s_scales.bin", layer_dir, projs[i].name);
            float* scales = load_fp32_file(path, K);  /* Per-column scales */
            if (!scales) {
                fprintf(stderr, "[MatmulDecoder] Warning: Failed to load scales for %s\n", projs[i].name);
            }

            /* Copy packed weights to B memory */
            /* RKNN expects B in [K, N] packed format */
            memcpy(pm->mem_B->virt_addr, w_data, (size_t)N * K / 2);

            /* Store scales pointer for later use (in custom quantization) */
            /* For now, RKNN handles INT4 internally if we pass the right type */

            free(w_data);
            free(scales);
        } else {
            /* Load FP16 weights */
            snprintf(path, sizeof(path), "%s/%s.bin", layer_dir, projs[i].name);
            int16_t* w_data = load_fp16_file(path, (size_t)K * N);
            if (!w_data) {
                fprintf(stderr, "[MatmulDecoder] Warning: Failed to load %s\n", path);
                continue;
            }

            memcpy(pm->mem_B->virt_addr, w_data, K * N * sizeof(int16_t));
            free(w_data);
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

    /* Allocate layer contexts */
    ctx->layers = calloc(num_layers, sizeof(LayerMatmulContext));
    if (!ctx->layers) {
        matmul_decoder_destroy(ctx);
        return NULL;
    }

    /* Build context pool — one entry per unique (K, N) dimension pair.
     * For Qwen3-0.6B: 5 unique pairs instead of 196 standalone contexts.
     * Saves ~573 NPU handles (from 784 down to ~211). */
    ctx->n_pool = 0;
    struct { const char* name; int K; int N; } proj_defs[] = {
        {"q_proj",    hidden_dim,              num_q_heads * head_dim},
        {"k_proj",    hidden_dim,              num_kv_heads * head_dim},
        {"v_proj",    hidden_dim,              num_kv_heads * head_dim},
        {"o_proj",    num_q_heads * head_dim,  hidden_dim},
        {"gate_proj", hidden_dim,              ffn_dim},
        {"up_proj",   hidden_dim,              ffn_dim},
        {"down_proj", ffn_dim,                 hidden_dim},
    };
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
    printf("[MatmulDecoder] Context pool: %d entries (from 7 projection types)\n", ctx->n_pool);

    /* Create per-layer projections — only B weight is per-projection */
    printf("[MatmulDecoder] Creating %d layers × 7 projections (pooled, B-only handles)...\n", num_layers);

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

        /* Load weights for this layer */
        snprintf(path, sizeof(path), "%s/layer_%02d", model_dir, i);
        int ret = load_layer_weights(lc, path, hidden_dim, num_q_heads, num_kv_heads, head_dim, ffn_dim, quant_type, config->has_qk_norm);
        if (ret != 0) {
            fprintf(stderr, "[MatmulDecoder] Failed to load layer %d weights\n", i);
        }
    }

    int total_handles = ctx->n_pool * 3 + num_layers * 7;  /* pool(ctx+A+C) + per-layer B */
    printf("[MatmulDecoder] Loaded %d layers (qk_norm=%d), NPU handles: %d (pool=%d + B=%d)\n",
           num_layers, config->has_qk_norm, total_handles, ctx->n_pool * 3, num_layers * 7);

    /* KV cache */
    ctx->kv_cache = matmul_kv_cache_create(num_layers, num_kv_heads, head_dim, max_seq_len);

    /* RoPE tables */
    ctx->cos_table = malloc((size_t)max_seq_len * (head_dim / 2) * sizeof(float));
    ctx->sin_table = malloc((size_t)max_seq_len * (head_dim / 2) * sizeof(float));
    precompute_rope_tables(ctx->cos_table, ctx->sin_table, max_seq_len, head_dim, config->rope_theta);

    /* Working buffers */
    ctx->hidden = aligned_alloc(64, hidden_dim * sizeof(float));
    ctx->q_out = aligned_alloc(64, num_q_heads * head_dim * sizeof(float));
    ctx->k_out = aligned_alloc(64, num_kv_heads * head_dim * sizeof(float));
    ctx->v_out = aligned_alloc(64, num_kv_heads * head_dim * sizeof(float));
    ctx->attn_out = aligned_alloc(64, hidden_dim * sizeof(float));
    ctx->ffn_gate = aligned_alloc(64, ffn_dim * sizeof(float));
    ctx->ffn_up = aligned_alloc(64, ffn_dim * sizeof(float));
    ctx->ffn_down = aligned_alloc(64, hidden_dim * sizeof(float));
    ctx->logits = aligned_alloc(64, vocab_size * sizeof(float));

    printf("[MatmulDecoder] Ready: %d layers, hidden=%d, heads=%d/%d, ffn=%d\n",
           num_layers, hidden_dim, num_q_heads, num_kv_heads, ffn_dim);

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
    ctx->hidden = aligned_alloc(64, hidden_dim * sizeof(float));
    ctx->q_out = aligned_alloc(64, num_q_heads * head_dim * sizeof(float));
    ctx->k_out = aligned_alloc(64, num_kv_heads * head_dim * sizeof(float));
    ctx->v_out = aligned_alloc(64, num_kv_heads * head_dim * sizeof(float));
    ctx->attn_out = aligned_alloc(64, hidden_dim * sizeof(float));
    ctx->ffn_gate = aligned_alloc(64, ffn_dim * sizeof(float));
    ctx->ffn_up = aligned_alloc(64, ffn_dim * sizeof(float));
    ctx->ffn_down = aligned_alloc(64, hidden_dim * sizeof(float));
    ctx->logits = aligned_alloc(64, vocab_size * sizeof(float));

    printf("[MatmulDecoder] Created: %d layers, hidden=%d, heads=%d/%d, ffn=%d\n",
           num_layers, hidden_dim, num_q_heads, num_kv_heads, ffn_dim);

    return ctx;
}

/* ─── Decoder Step ─── */

int matmul_decoder_step(MatmulDecoderContext* ctx,
                        int token_id,
                        const float* embedding,
                        float* output_logits) {
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
        /* Lookup embedding */
        if (token_id < 0 || token_id >= vocab_size) {
            return MATMUL_DECODER_ERR_INVALID_ARG;
        }
        memcpy(ctx->hidden, ctx->embeddings + token_id * hidden_dim, hidden_dim * sizeof(float));
    }

    /* Process each layer */
    int seq_len = ctx->kv_cache->seq_len;

    for (int layer = 0; layer < num_layers; layer++) {
        LayerMatmulContext* lc = &ctx->layers[layer];
        float* layer_k_cache = ctx->kv_cache->k_cache +
            (layer * ctx->kv_cache->max_seq_len * num_kv_heads * head_dim);
        float* layer_v_cache = ctx->kv_cache->v_cache +
            (layer * ctx->kv_cache->max_seq_len * num_kv_heads * head_dim);

        /* Input norm */
        float normed[hidden_dim];
        rms_norm_f32(normed, ctx->hidden, lc->input_norm_w, hidden_dim, ctx->config.rms_eps);

        /* QKV projections */
        run_persistent_matmul(&lc->q_proj, normed, ctx->q_out);
        run_persistent_matmul(&lc->k_proj, normed, ctx->k_out);
        run_persistent_matmul(&lc->v_proj, normed, ctx->v_out);

        /* QK norm: per-head RMSNorm on Q and K before RoPE */
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

        /* Apply RoPE */
        float* cos = ctx->cos_table + seq_len * (head_dim / 2);
        float* sin = ctx->sin_table + seq_len * (head_dim / 2);
        apply_rope_f32(ctx->q_out, cos, sin, num_q_heads, head_dim);
        apply_rope_f32(ctx->k_out, cos, sin, num_kv_heads, head_dim);

        /* Store K, V in cache */
        memcpy(layer_k_cache + seq_len * num_kv_heads * head_dim, ctx->k_out,
               num_kv_heads * head_dim * sizeof(float));
        memcpy(layer_v_cache + seq_len * num_kv_heads * head_dim, ctx->v_out,
               num_kv_heads * head_dim * sizeof(float));

        /* Attention */
        attention_f32(ctx->attn_out, ctx->q_out, layer_k_cache, layer_v_cache,
                      num_q_heads, num_kv_heads, head_dim, seq_len + 1);

        /* Output projection */
        run_persistent_matmul(&lc->o_proj, ctx->attn_out, ctx->hidden);

        /* Residual */
        vec_add_f32(ctx->hidden, ctx->attn_out, hidden_dim);

        /* Post-attention norm */
        rms_norm_f32(normed, ctx->hidden, lc->post_attn_norm_w, hidden_dim, ctx->config.rms_eps);

        /* FFN */
        run_persistent_matmul(&lc->gate_proj, normed, ctx->ffn_gate);
        run_persistent_matmul(&lc->up_proj, normed, ctx->ffn_up);
        silu_mul_f32(ctx->ffn_gate, ctx->ffn_gate, ctx->ffn_up, ffn_dim);
        run_persistent_matmul(&lc->down_proj, ctx->ffn_gate, ctx->ffn_down);

        /* Residual */
        vec_add_f32(ctx->hidden, ctx->ffn_down, hidden_dim);
    }

    /* Final norm (use last layer's post_attn_norm as final norm) */
    float final_normed[hidden_dim];
    rms_norm_f32(final_normed, ctx->hidden,
                 ctx->layers[num_layers - 1].post_attn_norm_w, hidden_dim, ctx->config.rms_eps);

    /* LM head (simple matmul: [hidden_dim] @ [hidden_dim, vocab_size] */
    /* TODO: Use persistent matmul for lm_head */
    for (int v = 0; v < vocab_size; v++) {
        float sum = 0.0f;
        for (int h = 0; h < hidden_dim; h++) {
            sum += final_normed[h] * ctx->lm_head[v * hidden_dim + h];
        }
        ctx->logits[v] = sum;
    }

    /* Update KV cache */
    ctx->kv_cache->seq_len++;

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

    /* Process each layer (identical to matmul_decoder_step) */
    int seq_len = ctx->kv_cache->seq_len;

    for (int layer = 0; layer < num_layers; layer++) {
        LayerMatmulContext* lc = &ctx->layers[layer];
        float* layer_k_cache = ctx->kv_cache->k_cache +
            (layer * ctx->kv_cache->max_seq_len * num_kv_heads * head_dim);
        float* layer_v_cache = ctx->kv_cache->v_cache +
            (layer * ctx->kv_cache->max_seq_len * num_kv_heads * head_dim);

        float normed[hidden_dim];
        rms_norm_f32(normed, ctx->hidden, lc->input_norm_w, hidden_dim, ctx->config.rms_eps);

        run_persistent_matmul(&lc->q_proj, normed, ctx->q_out);
        run_persistent_matmul(&lc->k_proj, normed, ctx->k_out);
        run_persistent_matmul(&lc->v_proj, normed, ctx->v_out);

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
        apply_rope_f32(ctx->q_out, cos, sin, num_q_heads, head_dim);
        apply_rope_f32(ctx->k_out, cos, sin, num_kv_heads, head_dim);

        memcpy(layer_k_cache + seq_len * num_kv_heads * head_dim, ctx->k_out,
               num_kv_heads * head_dim * sizeof(float));
        memcpy(layer_v_cache + seq_len * num_kv_heads * head_dim, ctx->v_out,
               num_kv_heads * head_dim * sizeof(float));

        attention_f32(ctx->attn_out, ctx->q_out, layer_k_cache, layer_v_cache,
                      num_q_heads, num_kv_heads, head_dim, seq_len + 1);

        run_persistent_matmul(&lc->o_proj, ctx->attn_out, ctx->hidden);
        vec_add_f32(ctx->hidden, ctx->attn_out, hidden_dim);

        rms_norm_f32(normed, ctx->hidden, lc->post_attn_norm_w, hidden_dim, ctx->config.rms_eps);

        run_persistent_matmul(&lc->gate_proj, normed, ctx->ffn_gate);
        run_persistent_matmul(&lc->up_proj, normed, ctx->ffn_up);
        silu_mul_f32(ctx->ffn_gate, ctx->ffn_gate, ctx->ffn_up, ffn_dim);
        run_persistent_matmul(&lc->down_proj, ctx->ffn_gate, ctx->ffn_down);

        vec_add_f32(ctx->hidden, ctx->ffn_down, hidden_dim);
    }

    /* Final norm */
    float final_normed[hidden_dim];
    rms_norm_f32(final_normed, ctx->hidden,
                 ctx->layers[num_layers - 1].post_attn_norm_w, hidden_dim, ctx->config.rms_eps);

    /* Selected lm_head */
    const float* head_w = ctx->lm_heads[lm_head_idx];
    for (int v = 0; v < lm_vocab; v++) {
        float sum = 0.0f;
        for (int h = 0; h < hidden_dim; h++) {
            sum += final_normed[h] * head_w[v * hidden_dim + h];
        }
        ctx->logits[v] = sum;
    }

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

    /* Destroy per-layer B weight handles (must come before pool destroy) */
    if (ctx->layers) {
        for (int i = 0; i < ctx->config.num_layers; i++) {
            LayerMatmulContext* lc = &ctx->layers[i];
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
    }

    /* Destroy shared pool contexts (ctx + A + C buffers) */
    destroy_pool(ctx->pool, ctx->n_pool);

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
    free(ctx->cos_table);
    free(ctx->sin_table);
    free(ctx->hidden);
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