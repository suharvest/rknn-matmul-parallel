/**
 * Generic Matmul Decoder API
 *
 * A model-agnostic transformer decoder using RKNN matmul API.
 * Designed to replace RKLLM for avoiding RKNN/RKLLM conflicts.
 *
 * Features:
 *   - Dual-core NPU parallelism via persistent matmul contexts
 *   - NEON-optimized CPU operators
 *   - Configurable model architecture
 *   - INT4 quantization support
 */

#ifndef MATMUL_DECODER_H
#define MATMUL_DECODER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Version ─── */
#define MATMUL_DECODER_VERSION_MAJOR 1
#define MATMUL_DECODER_VERSION_MINOR 0
#define MATMUL_DECODER_VERSION_PATCH 0

/* ─── Error Codes ─── */
typedef enum {
    MATMUL_DECODER_OK = 0,
    MATMUL_DECODER_ERR_INVALID_ARG = -1,
    MATMUL_DECODER_ERR_MEMORY = -2,
    MATMUL_DECODER_ERR_FILE_IO = -3,
    MATMUL_DECODER_ERR_RKNN = -4,
    MATMUL_DECODER_ERR_UNSUPPORTED = -5,
} MatmulDecoderError;

/* ─── Execution Mode ─── */

typedef enum {
    EXEC_SINGLE_CORE = 0,       /* Single NPU core (simpler, no fork) */
    EXEC_DUAL_CORE = 1,         /* Dual NPU core via fork + shared memory */
} ExecutionMode;

/* ─── Model Configuration ─── */

/**
 * Model architecture configuration.
 * Supports Qwen, Llama, Mistral-style decoders.
 */
typedef struct {
    const char* name;           /* Model name for logging */
    int hidden_dim;             /* Hidden dimension (d_model) */
    int num_q_heads;            /* Number of query attention heads */
    int num_kv_heads;           /* Number of KV heads (GQA: num_kv_heads <= num_q_heads) */
    int head_dim;               /* Dimension per attention head */
    int ffn_dim;                /* FFN intermediate dimension */
    int num_layers;             /* Number of transformer layers */
    int vocab_size;             /* Vocabulary size */
    int max_seq_len;            /* Maximum sequence length for KV cache */
    float rms_eps;              /* RMSNorm epsilon (typically 1e-6) */
    float rope_theta;           /* RoPE theta (Qwen: 1e6, Llama: 1e4) */
    int tie_word_embeddings;    /* Whether embed and lm_head share weights */

    /* Optional: rope scaling for long context models */
    float rope_scaling_factor;
    int rope_scaling_type;      /* 0=none, 1=linear, 2=dynamic */

    /* Optional: normalization type */
    int norm_type;              /* 0=RMSNorm, 1=LayerNorm */

    /* RoPE pairing style */
    int rope_style;             /* 0=interleaved (Qwen3/LLaMA), 1=split-half (GPT-NeoX) */

    /* Optional: activation type */
    int ffn_act_type;           /* 0=SwiGLU, 1=GeGLU, 2=ReLU */

    /* Optional: QK norm (per-head RMSNorm on Q and K after projection) */
    int has_qk_norm;            /* 0=disabled, 1=enabled (Qwen3 uses this) */

    /* Multi lm_head support (e.g., Code Predictor: 15 heads, each vocab=2048) */
    int num_lm_heads;           /* Number of lm_heads (0 or 1 = single head, >1 = multi) */
    int lm_head_vocab_size;     /* Per-head vocab size (only used when num_lm_heads > 1) */

    /* Context pooling strategy.
     * 0 = auto (pool if num_layers * 7 > 128, else dedicated)
     * 1 = force pool (share ctx/A/C, rebind B each run — saves handles, ~250ms overhead)
     * 2 = force dedicated (one context per projection — no rebind, fastest, uses more handles)
     * For 28-layer models: dedicated uses ~784 handles (under 1020 limit if no RKNN coexistence) */
    int context_pool_mode;

    /* IOMMU domain isolation (SDK >= V2.0.0-beta0).
     * RKNN models (rknn_init) default to domain 0.
     * Set this to 1+ so matmul contexts use a separate domain,
     * avoiding IOMMU address space contention and 6s timeout EINVAL.
     * 0 = default (same domain as RKNN models, may conflict).
     * Valid range: 0-15 (RKNPU_MAX_IOMMU_DOMAIN_NUM=16). */
    int iommu_domain_id;

    /* Execution mode */
    ExecutionMode exec_mode;    /* Single or dual NPU core */
} MatmulDecoderConfig;

/**
 * Default config for Qwen3-ASR-0.6B.
 */
static inline MatmulDecoderConfig matmul_decoder_config_qwen3_0_6b(void) {
    return (MatmulDecoderConfig){
        .name = "qwen3-asr-0.6b",
        .hidden_dim = 1024,
        .num_q_heads = 16,
        .num_kv_heads = 8,
        .head_dim = 128,
        .ffn_dim = 3072,
        .num_layers = 28,
        .vocab_size = 151936,
        .max_seq_len = 4096,
        .rms_eps = 1e-6f,
        .rope_theta = 1000000.0f,
        .tie_word_embeddings = 1,
        .rope_scaling_factor = 1.0f,
        .rope_scaling_type = 0,
        .norm_type = 0,  /* RMSNorm */
        .ffn_act_type = 0, /* SwiGLU */
        .has_qk_norm = 1,  /* Qwen3 uses per-head QK norm */
        .num_lm_heads = 0, /* Single lm_head (standard LLM) */
        .lm_head_vocab_size = 0,
        .iommu_domain_id = 1,  /* Separate from RKNN models on domain 0 */
        .exec_mode = EXEC_DUAL_CORE,  /* Use dual NPU core by default */
    };
}

/**
 * Default config for single-core execution (simpler, no fork).
 */
static inline MatmulDecoderConfig matmul_decoder_config_qwen3_0_6b_single_core(void) {
    MatmulDecoderConfig config = matmul_decoder_config_qwen3_0_6b();
    config.exec_mode = EXEC_SINGLE_CORE;
    return config;
}

/**
 * Default config for Qwen3-TTS Code Predictor (15 lm_heads, vocab=2048 each).
 * 5-layer small transformer, autoregressive 15-step decode.
 */
static inline MatmulDecoderConfig matmul_decoder_config_qwen3_tts_cp(void) {
    return (MatmulDecoderConfig){
        .name = "qwen3-tts-cp",
        .hidden_dim = 1024,
        .num_q_heads = 16,
        .num_kv_heads = 8,
        .head_dim = 128,
        .ffn_dim = 3072,
        .num_layers = 5,
        .vocab_size = 2048,     /* Per-head vocab size */
        .max_seq_len = 20,      /* 2 prefill + 15 decode + margin */
        .rms_eps = 1e-6f,
        .rope_theta = 1000000.0f,
        .tie_word_embeddings = 0,
        .rope_scaling_factor = 1.0f,
        .rope_scaling_type = 0,
        .norm_type = 0,
        .ffn_act_type = 0,
        .has_qk_norm = 1,
        .num_lm_heads = 15,
        .lm_head_vocab_size = 2048,
        .iommu_domain_id = 1,
        .exec_mode = EXEC_DUAL_CORE,
    };
}

/* ─── Quantization Type ─── */

typedef enum {
    QUANT_FP16 = 0,             /* FP16 weights (no quantization) */
    QUANT_INT8 = 1,             /* INT8 per-row quantization */
    QUANT_INT4 = 2,             /* INT4 per-column quantization (W4A16) */
    QUANT_INT4_G128 = 3,        /* INT4 group-128 quantization */
} QuantizationType;

/* ─── Weight Format ─── */

/**
 * Single projection weight (e.g., q_proj).
 * Supports both FP16 and quantized formats.
 */
typedef struct {
    void* data;                 /* Weight data (FP16 or quantized) */
    float* scales;              /* Quantization scales (NULL for FP16) */
    int K;                      /* Input dimension */
    int N;                      /* Output dimension */
    QuantizationType quant_type;
} ProjectionWeight;

/**
 * Full layer weights.
 */
typedef struct {
    /* Attention */
    ProjectionWeight q_proj;
    ProjectionWeight k_proj;
    ProjectionWeight v_proj;
    ProjectionWeight o_proj;

    /* FFN */
    ProjectionWeight gate_proj;
    ProjectionWeight up_proj;
    ProjectionWeight down_proj;

    /* Normalization */
    float* input_norm_weight;   /* [hidden_dim] */
    float* post_attn_norm_weight; /* [hidden_dim] */

    /* QK norm (optional, only if has_qk_norm=1) */
    float* q_norm_weight;       /* [head_dim] */
    float* k_norm_weight;       /* [head_dim] */
} LayerWeights;

/* ─── KV Cache ─── */

typedef struct MatmulKVCache MatmulKVCache;

/**
 * Create KV cache.
 *
 * @param num_layers    Number of layers
 * @param num_kv_heads  Number of KV heads per layer
 * @param head_dim      Dimension per head
 * @param max_seq_len   Maximum sequence length
 * @return              KV cache or NULL on error
 */
MatmulKVCache* matmul_kv_cache_create(int num_layers, int num_kv_heads,
                                       int head_dim, int max_seq_len);

/**
 * Clear KV cache for new sequence.
 */
void matmul_kv_cache_clear(MatmulKVCache* cache);

/**
 * Get current sequence length.
 */
int matmul_kv_cache_get_seq_len(const MatmulKVCache* cache);

/**
 * Free KV cache.
 */
void matmul_kv_cache_destroy(MatmulKVCache* cache);

/* ─── Decoder Context ─── */

typedef struct MatmulDecoderContext MatmulDecoderContext;

/**
 * Create decoder context from weights directory.
 *
 * Directory layout:
 *   model_dir/
 *   ├── config.json           # Model configuration
 *   ├── embeddings.npy        # [vocab_size, hidden_dim] FP32
 *   ├── lm_head.npy           # [hidden_dim, vocab_size] FP32 (optional if tied)
 *   └── layers/
 *       ├── layer_00.npz
 *       └── ...
 *
 * @param model_dir     Model directory path
 * @param config        Model config (NULL to load from config.json)
 * @param quant_type    Quantization type for matmul
 * @param max_seq_len   Maximum KV cache size
 * @return              Decoder context or NULL on error
 */
MatmulDecoderContext* matmul_decoder_create(const char* model_dir,
                                            const MatmulDecoderConfig* config,
                                            QuantizationType quant_type,
                                            int max_seq_len);

/**
 * Create decoder from in-memory weights (for Python binding).
 */
MatmulDecoderContext* matmul_decoder_create_from_weights(
    const MatmulDecoderConfig* config,
    const float* embeddings,        /* [vocab_size, hidden_dim] */
    const LayerWeights* layers,     /* [num_layers] */
    const float* lm_head,           /* [hidden_dim, vocab_size] or NULL if tied */
    QuantizationType quant_type,
    int max_seq_len
);

/**
 * Run one decoding step.
 *
 * @param ctx           Decoder context
 * @param token_id      Input token ID (ignored if embedding != NULL)
 * @param embedding     Input embedding [hidden_dim] (NULL to use token_id)
 * @param output_logits Output logits [vocab_size] (can be NULL)
 * @return              Sampled token ID (greedy) or error code
 */
int matmul_decoder_step(MatmulDecoderContext* ctx,
                        int token_id,
                        const float* embedding,
                        float* output_logits);

/**
 * Run one step and return logits (for custom sampling).
 */
int matmul_decoder_step_get_logits(MatmulDecoderContext* ctx,
                                    int token_id,
                                    const float* embedding,
                                    float* logits_out);

/**
 * Run one step with a specific lm_head (for multi-head models like Code Predictor).
 *
 * @param ctx           Decoder context
 * @param token_id      Input token ID (ignored if embedding != NULL)
 * @param embedding     Input embedding [hidden_dim] (NULL to use token_id)
 * @param lm_head_idx   Which lm_head to use (0 to num_lm_heads-1)
 * @param output_logits Output logits [lm_head_vocab_size] (can be NULL)
 * @return              Sampled token ID (greedy) or error code
 */
int matmul_decoder_step_head(MatmulDecoderContext* ctx,
                             int token_id,
                             const float* embedding,
                             int lm_head_idx,
                             float* output_logits);

/**
 * Clear KV cache for new sequence.
 */
void matmul_decoder_clear_kv_cache(MatmulDecoderContext* ctx);

/**
 * Get current sequence length.
 */
int matmul_decoder_get_seq_len(const MatmulDecoderContext* ctx);

/**
 * Get model configuration.
 */
const MatmulDecoderConfig* matmul_decoder_get_config(const MatmulDecoderContext* ctx);

/**
 * Get performance stats.
 */
typedef struct {
    float total_ms;
    float matmul_ms;
    float cpu_ops_ms;
    int n_steps;
} MatmulDecoderStats;

void matmul_decoder_get_stats(const MatmulDecoderContext* ctx, MatmulDecoderStats* stats);

/**
 * Destroy decoder context.
 */
void matmul_decoder_destroy(MatmulDecoderContext* ctx);

/* ─── Sampling Parameters ─── */

typedef struct {
    int top_k;                   /* Top-K sampling (1 = greedy) */
    float top_p;                 /* Nucleus sampling threshold */
    float temperature;           /* Sampling temperature */
    float repeat_penalty;        /* Repetition penalty */
    const int* repeat_tokens;    /* Tokens to penalize (recent tokens) */
    int n_repeat_tokens;         /* Number of repeat tokens */
} SamplingParams;

/**
 * Default sampling params for ASR (greedy).
 */
static inline SamplingParams sampling_params_greedy(void) {
    return (SamplingParams){
        .top_k = 1,
        .top_p = 1.0f,
        .temperature = 1.0f,
        .repeat_penalty = 1.0f,
        .repeat_tokens = NULL,
        .n_repeat_tokens = 0,
    };
}

/**
 * Sample token from logits.
 *
 * @param logits        Logits array [vocab_size]
 * @param vocab_size    Vocabulary size
 * @param params        Sampling parameters
 * @return              Sampled token ID
 */
int matmul_sample_token(const float* logits, int vocab_size, SamplingParams* params);

/* ─── Batch Generation ─── */

/**
 * Callback for streaming generation.
 * Called after each token is generated.
 *
 * @param token_id      Generated token ID
 * @param token_text    Decoded token text (if tokenizer provided)
 * @param user_data     User-provided callback data
 */
typedef void (*GenerateCallback)(int token_id, const char* token_text, void* user_data);

/**
 * Generate tokens autoregressively.
 *
 * @param ctx           Decoder context
 * @param prompt_tokens Prompt token IDs
 * @param n_prompt      Number of prompt tokens
 * @param max_new_tokens Maximum tokens to generate
 * @param params        Sampling parameters
 * @param callback      Optional callback for streaming
 * @param user_data     User data for callback
 * @return              Number of tokens generated (negative on error)
 */
int matmul_decoder_generate(MatmulDecoderContext* ctx,
                            const int* prompt_tokens,
                            int n_prompt,
                            int max_new_tokens,
                            SamplingParams* params,
                            GenerateCallback callback,
                            void* user_data);

/* ─── Utility Functions ─── */

/**
 * Load config from JSON file.
 */
int matmul_decoder_load_config(const char* json_path, MatmulDecoderConfig* config);

/**
 * Export HF model to matmul decoder format.
 * Python script wrapper.
 */
int matmul_decoder_export_hf_model(const char* hf_model_path,
                                   const char* output_dir,
                                   QuantizationType quant_type);

#ifdef __cplusplus
}
#endif

#endif /* MATMUL_DECODER_H */