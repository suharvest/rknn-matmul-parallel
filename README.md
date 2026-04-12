# RKNN Matmul Decoder

<p align="center">
  <strong>Open-source transformer decoder for Rockchip NPU — a transparent RKLLM alternative.</strong>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/topics/rk3588"><img src="https://img.shields.io/badge/Platform-RK3588%20%7C%20RK3576-blue" alt="Platform"></a>
  <a href="https://github.com/airockchip/rknn-llm"><img src="https://img.shields.io/badge/Compared%20to-RKLLM-green" alt="RKLLM Alternative"></a>
</p>

Run transformer decoders on Rockchip NPU with full source code, conflict-free RKNN coexistence, and per-step profiling.

## Why This Exists

Rockchip's RKLLM is fast but closed-source. This library provides a fully transparent alternative using the RKNN matmul API, with native IOMMU domain isolation for conflict-free coexistence with RKNN models (encoder, vocoder, etc.).

> **Note:** Since v1.2.1, RKLLM also supports `base_domain_id` for IOMMU domain isolation (see [Coexistence Guide](docs/rkllm-coexistence.md)). If RKLLM's speed is critical and you only need domain isolation, a one-line change (`base_domain_id = 1`) may be sufficient. This library remains valuable for full source transparency, custom ops, and environments where RKLLM is unavailable.

## Matmul Decoder vs RKLLM

| Aspect | RKLLM | This Project |
|--------|-------|--------------|
| **Decode speed** | 43 ms/token | 225 ms/token (5.2x slower) |
| **Prefill speed** | 108ms / 60 tokens | 1,047ms / 60 tokens (batch) |
| **Open source** | Closed `librkllmrt.so` | Full C code |
| **RKNN coexistence** | `base_domain_id=1` (v1.2.1+) | `iommu_domain_id=1` (always) |
| **Custom ops** | Black box | Full control |
| **Profiling** | No visibility | Per-component timing |
| **Debuggability** | Opaque | Regression test suite |
| **INT4 quantization** | Works | Broken on librknnrt 2.3.x (NPU bug) |

### When to use which?

| Scenario | Recommendation |
|----------|---------------|
| **Real-time dialogue** (V2V < 1s) | RKLLM (`base_domain_id=1`) |
| **Offline transcription** (RTF < 3) | Either — matmul decoder is sufficient |
| **Full source control required** | This project |
| **RKLLM unavailable** (licensing, platform) | This project |
| **Custom decoder architecture** | This project |
| **Research / profiling** | This project |

### V2V Latency Comparison (2s audio, streaming dialogue)

```
                    RKLLM          Matmul Decoder
Encoder (RKNN):     250ms           250ms          (same)
Decoder prefill:    108ms         1,047ms          (9.7x)
Decoder generate:   645ms         3,375ms          (5.2x)
TTS first chunk:    100ms           100ms          (same)
────────────────────────────────────────────────────
V2V total:         ~0.8s          ~2.5s
```

The decode speed gap (5.2x) is architectural: RKLLM compiles the entire transformer into one fused NPU graph with SRAM caching, while this library dispatches 196 individual `rknn_matmul_run` calls per token with CPU-side attention/RoPE/RMSNorm.

## Performance

### Measured on RK3576 (Qwen3-ASR-0.6B, 28 layers, d=1024, librknnrt 2.3.2)

| Configuration | ms/token | Breakdown |
|---------------|----------|-----------|
| **FP16 + INT8 NPU lm_head** | **225** | matmul 91 + rebind 91 + lm_head 35 + cpu 8 |
| INT8 layers + INT8 lm_head | 146 | matmul 49 + rebind 49 + lm_head 34 + cpu 8 |
| RKLLM W4A16 (closed) | 43 | compiled NPU graph |

### Batch Prefill (11.6x speedup)

| Mode | 60 tokens | Per token |
|------|-----------|-----------|
| Sequential (M=1) | 12,117ms | 202ms |
| **Batch (M=60)** | **1,047ms** | **17ms** |

Batch prefill processes all embeddings in one pass through all 28 layers, sharing B weight buffers between M=1 decode and M=batch prefill contexts.

### Optimization Journey

<details>
<summary>Full optimization history (461ms → 225ms FP16)</summary>

#### Phase 1: Make It Work — Infrastructure

| Step | What | Impact |
|------|------|--------|
| Context pooling | 196 per-layer NPU contexts → 5 shared pools with B rebind | Handle count 784→211 |
| IOMMU domain isolation | `iommu_domain_id=1` separates matmul from RKNN models | Eliminates domain contention |
| B native layout | `rknn_B_normal_layout_to_native_layout` with separate buffers | In-place corrupts data |
| FP16→FP32 output | RK3576 NPU doesn't support `FLOAT16_TO_FLOAT16` | Changed to `FLOAT16_TO_FLOAT32` |
| W4A16 per-column scales | NPU INT4 per-layer; CPU applies per-column post-matmul | NEON-fused FP16→FP32 × scale |

#### Phase 2: Make It Correct — 8 Critical Bugs Fixed

| Bug | Impact |
|-----|--------|
| attn_out buffer overflow | 2x heap overflow (hidden_dim vs q_dim) |
| Residual connection | o_proj destroyed hidden before residual add |
| GQA attention indexing | All Q heads attended to KV head 0 only |
| RoPE pairing style | Wrong pairing (split-half vs interleaved), added configurable `rope_style` |
| KV cache bounds | Silent overflow on long sequences |
| VLA stack overflow | Replaced with heap-allocated buffers |

#### Phase 3: Make It Fast — Performance

| Step | ms/token | Technique |
|------|----------|-----------|
| Baseline | 461 | Naive C loops, CPU scalar lm_head |
| + NEON GEMV | 335 | ARM NEON 4-wide FMA |
| + FP16 GEMV weights | 259 | Halve memory bandwidth |
| + INT8 NPU lm_head | **225** | 38 tiles × (1024,4096), per-column scales |
| + Batch prefill | **17ms/tok prefill** | M=60 batch matmul + causal attention |

</details>

## Features

- **IOMMU domain isolation** — runs on separate domain from RKNN models
- **Batch prefill** — 11.6x faster prefill via M>1 matmul + causal attention
- **Context pooling** — 28-layer decoder uses only ~211 NPU handles
- **NPU-tiled lm_head** — 151936-vocab projection via INT8 tiled matmul (35ms)
- **Per-step profiling** — `decoder.stats` returns matmul_ms, rebind_ms, lm_head_ms, cpu_ops_ms
- **Regression test suite** — `test_precision.py` (cosine vs numpy ref), `benchmark.py` (per-component timing)
- **Complete decoder stack**: KV cache, RoPE (interleaved + split-half), RMSNorm, GQA attention, FFN, sampling
- **Multi lm_head support**: multiple output heads per step (e.g., Code Predictor with 15 heads)
- **NEON-optimized CPU operators** and FP16 GEMV
- **Python bindings** via pybind11

## Quick Start

### Python

```python
import matmul_decoder as md
import numpy as np

# Create decoder
decoder = md.MatmulDecoder(
    model_dir="/path/to/weights",
    max_seq_len=4096,
    quant_type="fp16",           # "fp16" or "int8" (INT4 disabled on 2.3.x)
    exec_mode="single_core"
)

# Batch prefill (fast — 17ms/token)
embeddings = np.random.randn(60, 1024).astype(np.float32)
decoder.prefill_batch(embeddings)

# Decode (autoregressive)
logits = decoder.step(token_id=151644)
predicted = np.argmax(logits)

# Per-step profiling
stats = decoder.stats
print(f"total={stats['total_ms']:.1f}ms, matmul={stats['matmul_ms']:.1f}ms")
```

### C API

```c
#include "matmul_decoder.h"

MatmulDecoderConfig config = matmul_decoder_config_qwen3_0_6b();
MatmulDecoderContext* ctx = matmul_decoder_create(
    "/path/to/weights", &config, QUANT_FP16, 4096);

// Batch prefill
float embeddings[60 * 1024];  // 60 tokens × hidden_dim
matmul_decoder_prefill_batch(ctx, embeddings, 60);

// Decode
float logits[151936];
int token = matmul_decoder_step(ctx, 151644, NULL, logits);

matmul_decoder_destroy(ctx);
```

## Installation

```bash
# Prerequisites: RKNN SDK (RKNPU2 runtime), GCC, Python 3.9+

export RKNN_SDK_PATH=/path/to/rknn-toolkit2/rknpu2/runtime/Linux
make python  # builds C library + Python binding
```

## Testing

```bash
# Precision regression (compares all quant types against numpy reference)
python3 tests/test_precision.py
#   FP16: PASS (cos=0.9999, top100=99/100)
#   INT8: POOR (cos=0.7754) — expected for 8-bit quantized layers

# Performance benchmark (auto-discovers all model variants)
python3 tests/benchmark.py --steps 50

# Generate numpy reference (one-time, pure numpy, no NPU needed)
python3 tests/generate_reference.py --model-dir /path/to/weights
```

## Configuration

### Quantization

| Type | matmul type | Speed | Accuracy | Status |
|------|------------|-------|----------|--------|
| **FP16** | type=1 (FP16→FP32) | 225 ms/tok | Best (cos=0.9999) | Recommended |
| **INT8** | type=5 (FP16×INT8→FP32) | 146 ms/tok | Good (cos=0.775) | Supported |
| INT4 | type=7/8 | - | - | Disabled (NPU bug on 2.3.x) |

### Context Pool Mode

| Mode | Value | NPU Handles | Best For |
|------|-------|-------------|----------|
| **Auto** | `0` | adaptive | Default — pools for ≥16 layers |
| **Pool** | `1` | ~211 (28L) | Running alongside RKNN models |
| **Dedicated** | `2` | ~784 (28L) | Small models (≤16 layers) |

### RoPE Style

| Style | Value | Models |
|-------|-------|--------|
| Interleaved | `0` | Qwen3, LLaMA, Mistral (default) |
| Split-half | `1` | GPT-NeoX, ChatGLM |

## Supported Platforms

| SoC | NPU Cores | Status | Notes |
|-----|-----------|--------|-------|
| RK3576 | 2 | Tested | Primary development platform |
| RK3588 | 3 | Compatible | Expected faster (more NPU cores) |

## Known Limitations

- **INT4 quantization disabled on librknnrt 2.3.x**: The RKNN NPU INT4 matmul kernel (type 7/8) has a confirmed bug — positive INT4 values lose ~50% magnitude. See [airockchip/rknn-toolkit2#412](https://github.com/airockchip/rknn-toolkit2/pull/412). Use `quant_type="fp16"` or `"int8"`.
- **Decode speed**: 225ms/token (FP16) vs RKLLM 43ms. The gap is architectural (per-op dispatch vs compiled graph) and cannot be closed via code optimization.
- **B rebind overhead**: Context pooling requires rebinding B weights per matmul (~90ms/token for 28 layers FP16).
- **IOMMU memory budget**: ~226 DMA handles for 28 layers + batch pool + lm_head tiles.

## License

MIT License.

## Acknowledgments

- Inspired by [qwen3asr_rk](https://github.com/qzxyz/qwen3asr_rk) for ASR decoder integration
- Architecture informed by Rockchip's [rknn-llm](https://github.com/airockchip/rknn-llm)
