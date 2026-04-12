/**
 * ARM NEON-optimized CPU operators for transformer decoder.
 *
 * Optimized for ARM Cortex-A72/A78 (RK3576/RK3588, Raspberry Pi 4/5, etc.).
 *
 * All functions are parameterized - no model-specific constants.
 * Model config (hidden_dim, num_heads, etc.) should be passed as parameters.
 */

#ifndef CPU_OPS_H
#define CPU_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── RMSNorm ─── */

/**
 * RMS normalization (Llama/Qwen style).
 * out = (x * rsqrt(mean(x^2) + eps)) * weight
 *
 * @param out     Output buffer [dim]
 * @param x       Input buffer [dim]
 * @param weight  Normalization weight [dim]
 * @param dim     Dimension
 * @param eps     Epsilon (typically 1e-6)
 */
void rms_norm_f32(float* out, const float* x, const float* weight, int dim, float eps);

/**
 * RMSNorm with FP16 input/output (avoids conversion overhead).
 */
void rms_norm_fp16(int16_t* out, const int16_t* x, const float* weight, int dim, float eps);

/* ─── LayerNorm ─── */

/**
 * Layer normalization (BERT/GPT style).
 * out = ((x - mean) / sqrt(var + eps)) * weight + bias
 */
void layer_norm_f32(float* out, const float* x, const float* weight, const float* bias,
                    int dim, float eps);

/* ─── RoPE (Rotary Position Embedding) ─── */

/**
 * Precompute RoPE cos/sin tables.
 * Must call once before using apply_rope.
 *
 * @param cos_tab     Output cos table [max_seq_len * head_dim / 2]
 * @param sin_tab     Output sin table [max_seq_len * head_dim / 2]
 * @param max_seq_len Maximum sequence length
 * @param head_dim    Dimension per attention head
 * @param theta       RoPE base frequency (Llama: 10000, Qwen: 1000000)
 */
void rope_precompute(float* cos_tab, float* sin_tab, int max_seq_len, int head_dim, float theta);

/**
 * Apply RoPE in-place to Q or K tensor.
 *
 * @param x           [num_heads, head_dim] - modified in-place
 * @param cos_tab     Precomputed cos table [head_dim / 2]
 * @param sin_tab     Precomputed sin table [head_dim / 2]
 * @param num_heads   Number of attention heads
 * @param head_dim    Dimension per head
 * @param rope_style  0=interleaved (Qwen3/LLaMA: pair 2i,2i+1)
 *                    1=split-half (GPT-NeoX: pair i,i+half)
 */
void apply_rope_f32(float* x, const float* cos_tab, const float* sin_tab,
                    int num_heads, int head_dim, int rope_style);

/**
 * Apply RoPE to FP16 tensor.
 */
void apply_rope_fp16(int16_t* x, const float* cos_tab, const float* sin_tab,
                     int num_heads, int head_dim, int rope_style);

/* ─── Softmax ─── */

/**
 * Softmax for attention scores.
 * Uses online softmax trick for numerical stability.
 *
 * @param scores   [seq_len] - modified in-place
 * @param seq_len  Number of elements
 */
void softmax_f32(float* scores, int seq_len);

/**
 * Softmax with temperature scaling.
 */
void softmax_with_temp_f32(float* scores, int seq_len, float temperature);

/* ─── Attention ─── */

/**
 * Fused scaled dot-product attention for autoregressive decoding.
 * Computes: softmax(Q @ K^T / sqrt(d)) @ V
 *
 * Optimized for small batch (M=1) and growing KV cache.
 *
 * @param out         Output [num_q_heads, head_dim]
 * @param q           Query [num_q_heads, head_dim]
 * @param k_cache     Key cache [num_kv_heads, max_seq_len, head_dim]
 * @param v_cache     Value cache [num_kv_heads, max_seq_len, head_dim]
 * @param num_q_heads Number of query heads
 * @param num_kv_heads Number of KV heads (GQA support)
 * @param head_dim    Dimension per head
 * @param seq_len     Current KV cache length (1 to max_seq_len)
 */
void attention_f32(float* out, const float* q,
                   const float* k_cache, const float* v_cache,
                   int num_q_heads, int num_kv_heads, int head_dim, int seq_len);

/**
 * Attention with FP16 Q, FP32 cache.
 */
void attention_fp16_f32(float* out, const int16_t* q,
                        const float* k_cache, const float* v_cache,
                        int num_q_heads, int num_kv_heads, int head_dim, int seq_len);

/* ─── Activation Functions ─── */

/**
 * SiLU (Swish) activation: x * sigmoid(x)
 */
void silu_f32(float* x, int dim);

/**
 * SiLU with multiplication (for SwiGLU FFN).
 * out = silu(gate) * up
 */
void silu_mul_f32(float* out, const float* gate, const float* up, int dim);

/**
 * SiLU with FP16 input/output.
 */
void silu_mul_fp16(int16_t* out, const int16_t* gate, const int16_t* up, int dim);

/**
 * GELU approximation (fast version).
 * Uses 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
void gelu_f32(float* x, int dim);

/* ─── Vector Operations ─── */

/**
 * Element-wise add: dst += src
 */
void vec_add_f32(float* dst, const float* src, int n);

/**
 * Element-wise add with residual: out = a + b
 */
void vec_add_residual_f32(float* out, const float* a, const float* b, int n);

/**
 * Scale vector: x *= scale
 */
void vec_scale_f32(float* x, float scale, int n);

/**
 * Copy with FP32 to FP16 conversion.
 */
void vec_fp32_to_fp16(int16_t* dst, const float* src, int n);

/**
 * Copy with FP16 to FP32 conversion.
 */
void vec_fp16_to_fp32(float* dst, const int16_t* src, int n);

/* ─── Batch Operations (for batch prefill) ─── */

/**
 * Batch RMSNorm: process M vectors.
 * Calls rms_norm_f32() for each of M rows.
 *
 * @param out     Output buffer [M * dim]
 * @param x       Input buffer [M * dim]
 * @param weight  Normalization weight [dim]
 * @param M       Number of vectors
 * @param dim     Dimension per vector
 * @param eps     Epsilon (typically 1e-6)
 */
void rms_norm_batch_f32(float* out, const float* x, const float* weight,
                         int M, int dim, float eps);

/**
 * Batch RoPE: each token t gets position start_pos + t.
 *
 * @param x          [M * num_heads * head_dim] — modified in-place
 * @param M          Number of tokens
 * @param start_pos  Position of first token in sequence
 * @param num_heads  Number of attention heads
 * @param head_dim   Dimension per head
 * @param rope_theta RoPE base frequency
 * @param rope_style 0=interleaved (Qwen3/LLaMA), 1=split-half (GPT-NeoX)
 */
void apply_rope_batch_f32(float* x, int M, int start_pos,
                           int num_heads, int head_dim, float rope_theta, int rope_style);

/**
 * Batch SiLU * element-wise multiply.
 * out[t] = silu(gate[t]) * up[t] for t in [0, M).
 *
 * @param out   Output buffer [M * dim]
 * @param gate  Gate input [M * dim]
 * @param up    Up input [M * dim]
 * @param M     Number of vectors
 * @param dim   Dimension per vector
 */
void silu_mul_batch_f32(float* out, const float* gate, const float* up, int M, int dim);

/**
 * Batch vector add: dst[t] += src[t] for t in [0, M).
 *
 * @param dst   Destination buffer [M * dim] — modified in-place
 * @param src   Source buffer [M * dim]
 * @param M     Number of vectors
 * @param dim   Dimension per vector
 */
void vec_add_batch_f32(float* dst, const float* src, int M, int dim);

/**
 * Batch causal attention for prefill.
 * For each token t in [0,M), computes attention over the combined
 * KV cache (cache_seq_len entries) + batch tokens [0..t] (causal mask).
 *
 * KV cache layout: [seq_len, num_kv_heads, head_dim] (interleaved per position).
 * Batch Q/K/V layout: [M, num_heads, head_dim] contiguous.
 *
 * @param out            Output [M * num_q_heads * head_dim]
 * @param q_batch        Query batch [M * num_q_heads * head_dim]
 * @param k_batch        Key batch [M * num_kv_heads * head_dim]
 * @param v_batch        Value batch [M * num_kv_heads * head_dim]
 * @param k_cache        Key cache [max_seq * num_kv_heads * head_dim]
 * @param v_cache        Value cache [max_seq * num_kv_heads * head_dim]
 * @param M              Number of tokens in batch
 * @param num_q_heads    Number of query heads
 * @param num_kv_heads   Number of KV heads (GQA support)
 * @param head_dim       Dimension per head
 * @param cache_seq_len  Entries already in KV cache before this batch
 */
void attention_batch_causal_f32(
    float* out,
    const float* q_batch,
    const float* k_batch,
    const float* v_batch,
    const float* k_cache,
    const float* v_cache,
    int M, int num_q_heads, int num_kv_heads, int head_dim,
    int cache_seq_len);

/* ─── Sampling ─── */

/**
 * Greedy sampling: argmax.
 *
 * @param x  Logits array [vocab_size]
 * @param n  Vocabulary size
 * @return   Index of maximum value
 */
int argmax_f32(const float* x, int n);

/**
 * Softmax then argmax (for probability-based greedy).
 */
int softmax_argmax_f32(float* x, int n);

/**
 * Top-k sampling.
 *
 * @param x           Logits array [vocab_size] - modified in-place
 * @param vocab_size  Vocabulary size
 * @param k           Number of top candidates
 * @param temperature Sampling temperature
 * @return            Sampled token index
 */
int top_k_sample_f32(float* x, int vocab_size, int k, float temperature);

/**
 * Top-p (nucleus) sampling.
 *
 * @param x           Logits array [vocab_size] - modified in-place
 * @param vocab_size  Vocabulary size
 * @param p           Cumulative probability threshold
 * @param temperature Sampling temperature
 * @return            Sampled token index
 */
int top_p_sample_f32(float* x, int vocab_size, float p, float temperature);

#ifdef __cplusplus
}
#endif

#endif /* CPU_OPS_H */