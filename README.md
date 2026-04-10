# RKNN Matmul Decoder

<p align="center">
  <img src="media/hero.png" alt="Dual-core NPU parallelism visualization" width="80%">
</p>

<p align="center">
  <strong>Open-source RKLLM alternative — ~16ms/token with full transparency.</strong>
</p>

<p align="center">
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/topics/rk3588"><img src="https://img.shields.io/badge/Platform-RK3588%20%7C%20RK3576-blue" alt="Platform"></a>
  <a href="https://github.com/airockchip/rknn-llm"><img src="https://img.shields.io/badge/Compared%20to-RKLLM-green" alt="RKLLM Alternative"></a>
</p>

Run transformer decoders on Rockchip NPU with performance matching RKLLM, but fully open-source and conflict-free with RKNN models.

## Why This Exists

Rockchip's RKLLM achieves ~16ms/token on Qwen3-ASR-0.6B, but:

| Aspect | RKLLM | This Project |
|--------|-------|--------------|
| **Performance** | ~16 ms/token | ~16 ms/token |
| **Open source** | ❌ Closed `librkllmrt.so` | ✅ Full code |
| **RKNN coexistence** | ❌ Conflicts with RKNN models | ✅ No conflict |
| **Custom ops** | ❌ Black box | ✅ Full control |
| **Debuggability** | ❌ Opaque | ✅ Transparent |

**RKLLM has known conflicts when running alongside RKNN models** (e.g., TTS vocoder, ASR encoder). This library provides a conflict-free alternative with equivalent performance.

## Performance

| Implementation | ms/token | Notes |
|----------------|----------|-------|
| **This project (dual-core)** | **~16** | Open source, W4A16 quantization |
| RKLLM (official) | ~16 | Closed runtime |
| Matmul single-core | ~27 | Baseline |

*Measured on RK3576 with Qwen3-ASR-0.6B decoder (28 layers, d=1024).*

### Single Matmul Operation

| Dimensions | Single-core | Dual-core | Speedup |
|------------|-------------|-----------|---------|
| 1×1024×1024 | 0.64 ms | 0.47 ms | **1.37×** |
| 1×1024×3072 | 1.79 ms | 1.31 ms | **1.37×** |

## Features

- **Dual NPU core parallelism** via fork + persistent worker processes
- **Complete decoder stack**: KV cache, RoPE, RMSNorm, attention, FFN, sampling
- **Python bindings**: `pip install` and use like RKLLM
- **Model-agnostic config**: adapt to any decoder-only transformer
- **INT4/INT8/FP16 quantization** support
- **NEON-optimized CPU operators**

## Quick Start

### Python (Recommended)

```python
from matmul_decoder import MatmulDecoder

# Create decoder with model weights
decoder = MatmulDecoder(
    model_path="/path/to/qwen3-asr-weights",
    exec_mode="dual_core"
)

# Run inference
result = decoder.run_embed(embeddings, n_tokens=10)
print(result["text"])
```

### C API

```c
#include <rknn_matmul_parallel.h>

int main() {
    RmpConfig config = {
        .M = 1, .K = 1024, .N = 1024,
        .type = RMP_TYPE_FP16_FP16,
        .n_workers = 2,
    };

    RmpContext* ctx = rmp_create(&config, weights, NULL);
    rmp_run(ctx, input, output);
    rmp_destroy(ctx);
}
```

## Installation

### Prerequisites

- RKNN SDK (RKNPU2 runtime)
- CMake or Make
- Python 3.9+ (for Python bindings)

### Build from Source

```bash
# Set RKNN SDK path
export RKNN_SDK_PATH=/path/to/rknn-toolkit2/rknpu2/runtime/Linux

# Build C library
make

# Build Python bindings
cd python && pip install .
```

## Architecture

```
┌─────────────────────────────────────────────┐
│  matmul_decoder.py (Python wrapper)         │  ← User API
├─────────────────────────────────────────────┤
│  pybind_matmul_decoder.cpp                  │  ← Python binding
├─────────────────────────────────────────────┤
│  matmul_decoder.c (Complete decoder)        │
│    ├─ KV cache management                   │
│    ├─ CPU ops (RoPE/RMSNorm/attention)      │
│    ├─ Sampling (top-k/top-p/greedy)         │
│    └─ rmp_run() (parallel matmul)           │
├─────────────────────────────────────────────┤
│  rknn_matmul_parallel.c (Core library)      │
│    ├─ Fork + persistent workers             │
│    ├─ Shared memory IPC                     │
│    └─ Weight splitting (N/2 per worker)     │
└─────────────────────────────────────────────┘
```

**How dual-core parallelism works:**

```
Parent Process
    │
    │  input (M × K)
    ▼
┌─────────────────┐
│  Shared Memory  │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│Worker │ │Worker │
│ NPU#0 │ │ NPU#1 │
│ N/2   │ │ N/2   │
│ cols  │ │ cols  │
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         ▼
    output (M × N)
```

Workers split the weight matrix columns, compute independently on separate NPU cores, then outputs are gathered.

## Real-World Usage

This library powers **Qwen3-ASR on RK3576**:

> [qwen3asr_rk](https://github.com/qzxyz/qwen3asr_rk) — 52-language streaming ASR
> - Encoder: RKNN FP16 merged model (431ms/4s chunk)
> - **Decoder: This matmul library (~16ms/token)**
> - End-to-end RTF: 0.44

## Model Adaptation

This is a **model-agnostic** decoder. To adapt a new model:

1. Export weights in required format:
   ```
   model_dir/
   ├── config.json           # Model configuration
   ├── embeddings.npy        # [vocab_size, hidden_dim]
   └── layers/
       ├── layer_00.npz
       └── ...
   ```

2. Create config with your model's architecture:
   ```python
   config = {
       "hidden_dim": 1024,
       "num_q_heads": 16,
       "num_kv_heads": 8,
       "head_dim": 64,
       "ffn_dim": 3072,
       "num_layers": 28,
       "vocab_size": 151936,
       "rope_theta": 1000000.0,
   }
   ```

3. (Optional) Write export script for your model format

See `examples/export_qwen3_asr.py` for reference.

## API Reference

### C API

| Function | Description |
|----------|-------------|
| `rmp_create(config, weights, scales)` | Create context, fork workers |
| `rmp_run(ctx, input, output)` | Execute parallel matmul |
| `rmp_destroy(ctx)` | Cleanup workers |
| `rmp_benchmark(ctx, n_runs)` | Measure performance |

### Python API

| Class/Method | Description |
|--------------|-------------|
| `MatmulDecoder(model_path, exec_mode)` | Create decoder instance |
| `decoder.run_embed(embeddings, n_tokens)` | Run inference with embeddings |
| `decoder.clear_kv_cache()` | Clear KV cache |
| `decoder.release()` | Release resources |

## Supported Platforms

| SoC | NPU Cores | Status |
|-----|-----------|--------|
| RK3576 | 2 | Tested |
| RK3588/RK3588S | 2 | Compatible |

## Why Not Just Use RKLLM?

| Aspect | This Library | RKLLM |
|--------|--------------|-------|
| **Performance** | ~16 ms/token | ~16 ms/token |
| **Open source** | ✅ Full code | ❌ Closed `librkllmrt.so` |
| **RKNN compatibility** | ✅ No conflict | ⚠️ Conflicts with RKNN models |
| **Custom ops** | ✅ Full control | ❌ Black box |
| **Debuggability** | ✅ Transparent | ❌ Opaque |

RKLLM has known conflicts when running alongside RKNN models (e.g., TTS vocoder, ASR encoder). This library provides a conflict-free alternative with equivalent performance.

## Contributing

Contributions welcome! Areas of interest:

- **New model support**: Export scripts for Llama, Mistral, Phi, etc.
- **CPU op optimizations**: SIMD optimizations for other ARM variants
- **Documentation**: Tutorials, integration guides
- **Testing**: More model/SoC combinations

Please open an issue or PR on GitHub.

## License

MIT License. Use freely for any purpose.

## Acknowledgments

- Inspired by [qwen3asr_rk](https://github.com/qzxyz/qwen3asr_rk) for ASR decoder integration
- Architecture informed by Rockchip's [rknn-llm](https://github.com/airockchip/rknn-llm) documentation