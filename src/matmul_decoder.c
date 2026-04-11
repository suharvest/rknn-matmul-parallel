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
 * For single-core mode, uses simple rknn_matmul context.
 * For dual-core mode, may use batch_matmul for parallel execution.
 */
typedef struct {
    rknn_matmul_ctx ctx;
    rknn_matmul_io_attr io;
    rknn_tensor_mem* mem_A;
    rknn_tensor_mem* mem_B;
    rknn_tensor_mem* mem_C;
    int K, N;
    int initialized;
} PersistentMatmul;

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
    float* lm_head;             /* [vocab_size, hidden_dim] transposed for matmul */

    /* Layer contexts */
    LayerMatmulContext* layers;

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

static int create_persistent_matmul(PersistentMatmul* pm, int M, int K, int N,
                                     QuantizationType quant_type) {
    memset(pm, 0, sizeof(PersistentMatmul));

    rknn_matmul_info info;
    memset(&info, 0, sizeof(info));
    info.M = M;
    info.K = K;
    info.N = N;

    /* Map quantization type to RKNN type */
    switch (quant_type) {
        case QUANT_FP16:
            info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
            break;
        case QUANT_INT4:
        case QUANT_INT4_G128:
            info.type = RKNN_FLOAT16_MM_INT4_TO_FLOAT16;
            break;
        case QUANT_INT8:
            info.type = RKNN_FLOAT16_MM_INT8_TO_FLOAT16;
            break;
        default:
            return MATMUL_DECODER_ERR_UNSUPPORTED;
    }

    info.B_layout = 1;  /* Native layout for better performance */

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

    return MATMUL_DECODER_OK;
}

static void destroy_persistent_matmul(PersistentMatmul* pm) {
    if (!pm->initialized) return;

    if (pm->mem_A) rknn_destroy_mem(pm->ctx, pm->mem_A);
    if (pm->mem_B) rknn_destroy_mem(pm->ctx, pm->mem_B);
    if (pm->mem_C) rknn_destroy_mem(pm->ctx, pm->mem_C);
    if (pm->ctx) rknn_matmul_destroy(pm->ctx);

    memset(pm, 0, sizeof(PersistentMatmul));
}

static int run_persistent_matmul(PersistentMatmul* pm, const float* input_fp32, float* output_fp32) {
    /* Convert input to FP16 */
    int K = pm->K, N = pm->N;
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

    /* Load lm_head (optional) */
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

    /* Allocate layer contexts */
    ctx->layers = calloc(num_layers, sizeof(LayerMatmulContext));
    if (!ctx->layers) {
        matmul_decoder_destroy(ctx);
        return NULL;
    }

    /* Create persistent matmul contexts for each layer */
    printf("[MatmulDecoder] Creating %d layer matmul contexts...\n", num_layers);

    for (int i = 0; i < num_layers; i++) {
        LayerMatmulContext* lc = &ctx->layers[i];
        int ret;

        /* Attention projections */
        ret = create_persistent_matmul(&lc->q_proj, 1, hidden_dim, num_q_heads * head_dim, quant_type);
        if (ret != 0) { fprintf(stderr, "Layer %d q_proj failed\n", i); }

        ret = create_persistent_matmul(&lc->k_proj, 1, hidden_dim, num_kv_heads * head_dim, quant_type);
        if (ret != 0) { fprintf(stderr, "Layer %d k_proj failed\n", i); }

        ret = create_persistent_matmul(&lc->v_proj, 1, hidden_dim, num_kv_heads * head_dim, quant_type);
        if (ret != 0) { fprintf(stderr, "Layer %d v_proj failed\n", i); }

        ret = create_persistent_matmul(&lc->o_proj, 1, num_q_heads * head_dim, hidden_dim, quant_type);
        if (ret != 0) { fprintf(stderr, "Layer %d o_proj failed\n", i); }

        /* FFN projections */
        ret = create_persistent_matmul(&lc->gate_proj, 1, hidden_dim, ffn_dim, quant_type);
        if (ret != 0) { fprintf(stderr, "Layer %d gate_proj failed\n", i); }

        ret = create_persistent_matmul(&lc->up_proj, 1, hidden_dim, ffn_dim, quant_type);
        if (ret != 0) { fprintf(stderr, "Layer %d up_proj failed\n", i); }

        ret = create_persistent_matmul(&lc->down_proj, 1, ffn_dim, hidden_dim, quant_type);
        if (ret != 0) { fprintf(stderr, "Layer %d down_proj failed\n", i); }

        /* Load weights for this layer */
        snprintf(path, sizeof(path), "%s/layer_%02d", model_dir, i);
        ret = load_layer_weights(lc, path, hidden_dim, num_q_heads, num_kv_heads, head_dim, ffn_dim, quant_type, config->has_qk_norm);
        if (ret != 0) {
            fprintf(stderr, "[MatmulDecoder] Failed to load layer %d weights\n", i);
        }
    }

    printf("[MatmulDecoder] Loaded %d layer weights (qk_norm=%d)\n", num_layers, config->has_qk_norm);

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

    /* LM head */
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

    /* Create persistent matmul contexts for each layer */
    printf("[MatmulDecoder] Creating %d layer matmul contexts...\n", num_layers);

    for (int i = 0; i < num_layers; i++) {
        LayerMatmulContext* lc = &ctx->layers[i];
        int ret;

        /* Attention projections */
        ret = create_persistent_matmul(&lc->q_proj, 1, hidden_dim, num_q_heads * head_dim, quant_type);
        if (ret != 0) { fprintf(stderr, "Layer %d q_proj failed\n", i); }

        ret = create_persistent_matmul(&lc->k_proj, 1, hidden_dim, num_kv_heads * head_dim, quant_type);
        if (ret != 0) { fprintf(stderr, "Layer %d k_proj failed\n", i); }

        ret = create_persistent_matmul(&lc->v_proj, 1, hidden_dim, num_kv_heads * head_dim, quant_type);
        if (ret != 0) { fprintf(stderr, "Layer %d v_proj failed\n", i); }

        ret = create_persistent_matmul(&lc->o_proj, 1, num_q_heads * head_dim, hidden_dim, quant_type);
        if (ret != 0) { fprintf(stderr, "Layer %d o_proj failed\n", i); }

        /* FFN projections */
        ret = create_persistent_matmul(&lc->gate_proj, 1, hidden_dim, ffn_dim, quant_type);
        if (ret != 0) { fprintf(stderr, "Layer %d gate_proj failed\n", i); }

        ret = create_persistent_matmul(&lc->up_proj, 1, hidden_dim, ffn_dim, quant_type);
        if (ret != 0) { fprintf(stderr, "Layer %d up_proj failed\n", i); }

        ret = create_persistent_matmul(&lc->down_proj, 1, ffn_dim, hidden_dim, quant_type);
        if (ret != 0) { fprintf(stderr, "Layer %d down_proj failed\n", i); }

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

    /* Destroy layer matmul contexts */
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

    /* Free other resources */
    matmul_kv_cache_destroy(ctx->kv_cache);
    free(ctx->embeddings);
    if (!ctx->config.tie_word_embeddings) {
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