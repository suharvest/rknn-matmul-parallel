/**
 * ARM NEON-optimized CPU operators implementation.
 *
 * Optimized for ARM Cortex-A72/A78 (RK3576/RK3588, Raspberry Pi 4/5, etc.).
 */

#include "cpu_ops.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <arm_neon.h>

/* ─── Helper macros ─── */
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/* ─── RMSNorm ─── */

void rms_norm_f32(float* out, const float* x, const float* weight, int dim, float eps) {
    /* Compute sum of squares */
    float sum_sq = 0.0f;
    int i;

    /* NEON: process 4 elements at a time */
    float32x4_t v_sum = vdupq_n_f32(0.0f);
    for (i = 0; i <= dim - 8; i += 8) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        v_sum = vaddq_f32(v_sum, vmulq_f32(v0, v0));
        v_sum = vaddq_f32(v_sum, vmulq_f32(v1, v1));
    }
    sum_sq = vaddvq_f32(v_sum);

    /* Tail */
    for (; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }

    /* Compute RMS: 1 / sqrt(mean(x^2) + eps) */
    float rms = 1.0f / sqrtf(sum_sq / dim + eps);
    float32x4_t v_rms = vdupq_n_f32(rms);

    /* Normalize and scale by weight */
    for (i = 0; i <= dim - 8; i += 8) {
        float32x4_t x0 = vld1q_f32(x + i);
        float32x4_t x1 = vld1q_f32(x + i + 4);
        float32x4_t w0 = vld1q_f32(weight + i);
        float32x4_t w1 = vld1q_f32(weight + i + 4);

        vst1q_f32(out + i,     vmulq_f32(vmulq_f32(x0, v_rms), w0));
        vst1q_f32(out + i + 4, vmulq_f32(vmulq_f32(x1, v_rms), w1));
    }
    for (; i < dim; i++) {
        out[i] = x[i] * rms * weight[i];
    }
}

void rms_norm_fp16(int16_t* out, const int16_t* x, const float* weight, int dim, float eps) {
    /* FP16 -> FP32, normalize, FP32 -> FP16 */
    float* x_f32 = (float*)malloc(dim * sizeof(float));
    float* out_f32 = (float*)malloc(dim * sizeof(float));

    vec_fp16_to_fp32(x_f32, x, dim);
    rms_norm_f32(out_f32, x_f32, weight, dim, eps);
    vec_fp32_to_fp16(out, out_f32, dim);

    free(x_f32);
    free(out_f32);
}

/* ─── LayerNorm ─── */

void layer_norm_f32(float* out, const float* x, const float* weight, const float* bias,
                    int dim, float eps) {
    /* Compute mean */
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) {
        mean += x[i];
    }
    mean /= dim;

    /* Compute variance */
    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= dim;

    /* Normalize and scale */
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < dim; i++) {
        out[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

/* ─── RoPE ─── */

void rope_precompute(float* cos_tab, float* sin_tab, int max_seq_len, int head_dim, float theta) {
    int half = head_dim / 2;
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half; i++) {
            float freq = pos / powf(theta, (2.0f * i) / head_dim);
            cos_tab[pos * half + i] = cosf(freq);
            sin_tab[pos * half + i] = sinf(freq);
        }
    }
}

void apply_rope_f32(float* x, const float* cos_tab, const float* sin_tab,
                    int num_heads, int head_dim) {
    int half = head_dim / 2;

    for (int h = 0; h < num_heads; h++) {
        float* hx = x + h * head_dim;

        /* Process 4 pairs at a time */
        for (int i = 0; i <= half - 4; i += 4) {
            float32x4_t x0 = vld1q_f32(hx + i);          /* x[..., 0:half] */
            float32x4_t x1 = vld1q_f32(hx + half + i);   /* x[..., half:] */
            float32x4_t c  = vld1q_f32(cos_tab + i);
            float32x4_t s  = vld1q_f32(sin_tab + i);

            /* out0 = x0 * cos - x1 * sin */
            /* out1 = x1 * cos + x0 * sin */
            vst1q_f32(hx + i,        vmlsq_f32(vmulq_f32(x0, c), x1, s));
            vst1q_f32(hx + half + i, vmlaq_f32(vmulq_f32(x1, c), x0, s));
        }

        /* Tail */
        for (int i = half - (half % 4); i < half; i++) {
            float x0 = hx[i];
            float x1 = hx[half + i];
            float c = cos_tab[i];
            float s = sin_tab[i];
            hx[i] = x0 * c - x1 * s;
            hx[half + i] = x1 * c + x0 * s;
        }
    }
}

void apply_rope_fp16(int16_t* x, const float* cos_tab, const float* sin_tab,
                     int num_heads, int head_dim) {
    /* Convert to FP32, apply, convert back */
    int total = num_heads * head_dim;
    float* x_f32 = (float*)malloc(total * sizeof(float));
    vec_fp16_to_fp32(x_f32, x, total);
    apply_rope_f32(x_f32, cos_tab, sin_tab, num_heads, head_dim);
    vec_fp32_to_fp16(x, x_f32, total);
    free(x_f32);
}

/* ─── Softmax ─── */

void softmax_f32(float* scores, int seq_len) {
    /* Find max for numerical stability */
    float max_val = scores[0];
    for (int i = 1; i < seq_len; i++) {
        if (scores[i] > max_val) max_val = scores[i];
    }

    /* Subtract max and compute exp */
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }

    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < seq_len; i++) {
        scores[i] *= inv_sum;
    }
}

void softmax_with_temp_f32(float* scores, int seq_len, float temperature) {
    /* Scale by temperature first */
    if (temperature != 1.0f) {
        for (int i = 0; i < seq_len; i++) {
            scores[i] /= temperature;
        }
    }
    softmax_f32(scores, seq_len);
}

/* ─── Attention ─── */

void attention_f32(float* out, const float* q,
                   const float* k_cache, const float* v_cache,
                   int num_q_heads, int num_kv_heads, int head_dim, int seq_len) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int groups = num_q_heads / num_kv_heads;

    /* Allocate scores buffer dynamically */
    float* scores = (float*)malloc(seq_len * sizeof(float));

    /* KV cache layout: [seq_len, num_kv_heads, head_dim] (interleaved per position) */
    int kv_stride = num_kv_heads * head_dim;  /* stride between positions */

    for (int h = 0; h < num_q_heads; h++) {
        int kv_h = h / groups;
        const float* qh = q + h * head_dim;

        /* Compute attention scores: Q @ K^T */
        float max_score = -1e30f;

        for (int s = 0; s < seq_len; s++) {
            const float* ks = k_cache + s * kv_stride + kv_h * head_dim;

            /* Dot product with NEON */
            float dot = 0.0f;
            for (int d = 0; d <= head_dim - 4; d += 4) {
                float32x4_t qv = vld1q_f32(qh + d);
                float32x4_t kv = vld1q_f32(ks + d);
                dot += vaddvq_f32(vmulq_f32(qv, kv));
            }

            scores[s] = dot * scale;
            if (scores[s] > max_score) max_score = scores[s];
        }

        /* Softmax */
        float sum = 0.0f;
        for (int s = 0; s < seq_len; s++) {
            scores[s] = expf(scores[s] - max_score);
            sum += scores[s];
        }
        float inv_sum = 1.0f / sum;
        for (int s = 0; s < seq_len; s++) {
            scores[s] *= inv_sum;
        }

        /* Compute output: scores @ V */
        float* oh = out + h * head_dim;
        memset(oh, 0, head_dim * sizeof(float));

        for (int s = 0; s < seq_len; s++) {
            const float* vs = v_cache + s * kv_stride + kv_h * head_dim;
            float sc = scores[s];

            for (int d = 0; d <= head_dim - 4; d += 4) {
                float32x4_t acc = vld1q_f32(oh + d);
                float32x4_t vv = vld1q_f32(vs + d);
                vst1q_f32(oh + d, vmlaq_n_f32(acc, vv, sc));
            }
        }
    }

    free(scores);
}

void attention_fp16_f32(float* out, const int16_t* q,
                        const float* k_cache, const float* v_cache,
                        int num_q_heads, int num_kv_heads, int head_dim, int seq_len) {
    /* Convert Q to FP32 */
    int q_size = num_q_heads * head_dim;
    float* q_f32 = (float*)malloc(q_size * sizeof(float));
    vec_fp16_to_fp32(q_f32, q, q_size);

    attention_f32(out, q_f32, k_cache, v_cache, num_q_heads, num_kv_heads, head_dim, seq_len);

    free(q_f32);
}

/* ─── Activation Functions ─── */

void silu_f32(float* x, int dim) {
    for (int i = 0; i <= dim - 4; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        /* sigmoid(x) = 1 / (1 + exp(-x)) */
        /* silu(x) = x * sigmoid(x) */
        /* Approximation: silu(x) ≈ x / (1 + exp(-x)) */
        float vals[4];
        vst1q_f32(vals, v);
        for (int j = 0; j < 4; j++) {
            vals[j] = vals[j] / (1.0f + expf(-vals[j]));
        }
        vst1q_f32(x + i, vld1q_f32(vals));
    }
}

void silu_mul_f32(float* out, const float* gate, const float* up, int dim) {
    for (int i = 0; i <= dim - 4; i += 4) {
        float32x4_t g = vld1q_f32(gate + i);
        float32x4_t u = vld1q_f32(up + i);

        /* Compute silu gate by gate */
        float gv[4], uv[4];
        vst1q_f32(gv, g);
        vst1q_f32(uv, u);

        float outv[4];
        for (int j = 0; j < 4; j++) {
            float sigmoid = 1.0f / (1.0f + expf(-gv[j]));
            outv[j] = gv[j] * sigmoid * uv[j];
        }
        vst1q_f32(out + i, vld1q_f32(outv));
    }

    /* Tail */
    for (int i = dim - (dim % 4); i < dim; i++) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

void silu_mul_fp16(int16_t* out, const int16_t* gate, const int16_t* up, int dim) {
    float* g_f32 = (float*)malloc(dim * sizeof(float));
    float* u_f32 = (float*)malloc(dim * sizeof(float));
    float* o_f32 = (float*)malloc(dim * sizeof(float));

    vec_fp16_to_fp32(g_f32, gate, dim);
    vec_fp16_to_fp32(u_f32, up, dim);
    silu_mul_f32(o_f32, g_f32, u_f32, dim);
    vec_fp32_to_fp16(out, o_f32, dim);

    free(g_f32);
    free(u_f32);
    free(o_f32);
}

void gelu_f32(float* x, int dim) {
    /* GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
    const float SQRT_2_PI = 0.7978845608f;  /* sqrt(2/pi) */
    const float COEFF = 0.044715f;

    for (int i = 0; i < dim; i++) {
        float v = x[i];
        float inner = SQRT_2_PI * (v + COEFF * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

/* ─── Vector Operations ─── */

void vec_add_f32(float* dst, const float* src, int n) {
    for (int i = 0; i <= n - 4; i += 4) {
        float32x4_t a = vld1q_f32(dst + i);
        float32x4_t b = vld1q_f32(src + i);
        vst1q_f32(dst + i, vaddq_f32(a, b));
    }
    for (int i = n - (n % 4); i < n; i++) {
        dst[i] += src[i];
    }
}

void vec_add_residual_f32(float* out, const float* a, const float* b, int n) {
    for (int i = 0; i <= n - 4; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(va, vb));
    }
    for (int i = n - (n % 4); i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void vec_scale_f32(float* x, float scale, int n) {
    float32x4_t v_scale = vdupq_n_f32(scale);
    for (int i = 0; i <= n - 4; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        vst1q_f32(x + i, vmulq_f32(v, v_scale));
    }
    for (int i = n - (n % 4); i < n; i++) {
        x[i] *= scale;
    }
}

void vec_fp32_to_fp16(int16_t* dst, const float* src, int n) {
    for (int i = 0; i <= n - 4; i += 4) {
        float32x4_t f = vld1q_f32(src + i);
        float16x4_t h = vcvt_f16_f32(f);
        vst1_f16((__fp16*)(dst + i), h);
    }
    for (int i = n - (n % 4); i < n; i++) {
        __fp16 tmp = (__fp16)src[i];
        memcpy(dst + i, &tmp, sizeof(__fp16));
    }
}

void vec_fp16_to_fp32(float* dst, const int16_t* src, int n) {
    for (int i = 0; i <= n - 4; i += 4) {
        float16x4_t h = vld1_f16((const __fp16*)(src + i));
        float32x4_t f = vcvt_f32_f16(h);
        vst1q_f32(dst + i, f);
    }
    for (int i = n - (n % 4); i < n; i++) {
        __fp16 tmp;
        memcpy(&tmp, src + i, sizeof(__fp16));
        dst[i] = (float)tmp;
    }
}

/* ─── Sampling ─── */

int argmax_f32(const float* x, int n) {
    float max_val = x[0];
    int max_idx = 0;

    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int softmax_argmax_f32(float* x, int n) {
    softmax_f32(x, n);
    return argmax_f32(x, n);
}

int top_k_sample_f32(float* x, int vocab_size, int k, float temperature) {
    /* Simple implementation: find top-k indices, apply softmax, sample */
    /* TODO: optimize with partial sort */

    /* Apply temperature */
    if (temperature != 1.0f) {
        for (int i = 0; i < vocab_size; i++) {
            x[i] /= temperature;
        }
    }

    /* Find top-k by sorting (inefficient but correct) */
    int* indices = (int*)malloc(k * sizeof(int));
    float* values = (float*)malloc(k * sizeof(float));

    for (int i = 0; i < k; i++) {
        values[i] = -1e30f;
        indices[i] = 0;
    }

    for (int i = 0; i < vocab_size; i++) {
        if (x[i] > values[k - 1]) {
            /* Insert into sorted position */
            int j = k - 1;
            while (j > 0 && x[i] > values[j - 1]) {
                values[j] = values[j - 1];
                indices[j] = indices[j - 1];
                j--;
            }
            values[j] = x[i];
            indices[j] = i;
        }
    }

    /* Softmax over top-k */
    float max_val = values[0];
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        values[i] = expf(values[i] - max_val);
        sum += values[i];
    }

    /* Sample */
    float r = (float)rand() / RAND_MAX * sum;
    float cumsum = 0.0f;
    for (int i = 0; i < k; i++) {
        cumsum += values[i];
        if (r < cumsum) {
            int result = indices[i];
            free(indices);
            free(values);
            return result;
        }
    }

    int result = indices[k - 1];
    free(indices);
    free(values);
    return result;
}

int top_p_sample_f32(float* x, int vocab_size, float p, float temperature) {
    /* Apply temperature and softmax */
    softmax_with_temp_f32(x, vocab_size, temperature);

    /* Sort by probability (descending) - simplified */
    /* TODO: implement properly with argsort */

    /* Cumulative sum until threshold */
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += x[i];
        if (cumsum >= p) {
            /* Sample from top candidates */
            return argmax_f32(x, vocab_size);  /* Simplified: just return max */
        }
    }

    return argmax_f32(x, vocab_size);
}