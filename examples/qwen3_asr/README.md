# Qwen3-ASR with Matmul Decoder

Complete speech recognition example running Qwen3-ASR on RK3576 using the matmul decoder from rknn-matmul-parallel. No RKLLM dependency needed -- avoids the NPU handle conflict between RKLLM and RKNN on RK3576.

## Architecture

```
Audio -> Mel Spectrogram -> RKNN Encoder (NPU) -> Embeddings
    -> Matmul Decoder (NPU, dual-core) -> Token IDs -> Text
```

- **Encoder**: RKNN model running on NPU (FP16 or INT8)
- **Decoder**: Autoregressive transformer using `matmul_decoder` C extension
  - Uses RKNN matmul API for matrix multiplications on NPU
  - Dual-core parallelism for ~16ms/token throughput
  - INT4/INT8/FP16 weight quantization

## Prerequisites

### Hardware
- RK3576 or RK3588 board (tested on RK3576)

### Software
- RKNN Toolkit Lite (`rknnlite`)
- `tokenizers` Python package
- `soundfile` or `pydub` (for audio loading)
- `numpy`

### Build the C extension

From the rknn-matmul-parallel project root:

```bash
make python
```

This builds `matmul_decoder.cpython-*.so` in the project root.

### Model files

You need the following model directory structure:

```
models/qwen3-asr/
├── encoder/
│   └── rk3576/
│       └── *.rknn            # Encoder model(s)
├── decoder/
│   └── matmul/               # Matmul decoder weights
│       ├── config.json
│       ├── embeddings.bin
│       └── layer_XX/
│           ├── input_norm.bin
│           ├── q_proj.bin
│           └── ...
├── mel_filters.npy
├── embed_tokens.npy
└── tokenizer.json
```

## Export decoder weights

On a machine with the HuggingFace model:

```bash
pip install safetensors transformers

python export_weights.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --output-dir ./qwen3-matmul

# Copy to device
rsync -avP ./qwen3-matmul/ device:/home/cat/models/qwen3-asr/decoder/matmul/
```

## Run

```bash
cd examples/qwen3_asr

# Basic usage
python run_asr.py /path/to/audio.wav

# Specify model directory and language
python run_asr.py audio.wav --model-dir /home/cat/models/qwen3-asr --language English

# Auto-detect language
python run_asr.py audio.wav --language auto

# Quiet mode (only print transcription)
python run_asr.py audio.wav --quiet

# Single-core mode (if dual-core causes issues)
python run_asr.py audio.wav --exec-mode single_core
```

### Environment variables

- `MATMUL_WEIGHTS_DIR` -- Override matmul weights directory
- `MATMUL_QUANT_TYPE` -- Override quantization type (int4, int8, fp16)

## Expected performance (RK3576)

| Metric | Value |
|---|---|
| Encoder (5s audio, merged) | ~120ms |
| Decoder prefill (per token) | ~4ms |
| Decoder generation (per token) | ~16ms (dual-core) |
| RTF (5s chunk) | ~0.44 |
| RTF (30s batch) | ~0.35 |

## Files

| File | Description |
|---|---|
| `run_asr.py` | CLI entry point |
| `engine.py` | ASR engine (encoder + decoder orchestration) |
| `encoder.py` | RKNN encoder wrapper (merged/split auto-detect) |
| `matmul_decoder.py` | Matmul decoder wrapper (C extension interface) |
| `mel.py` | Pure NumPy mel spectrogram extractor |
| `stream.py` | Streaming session with sliding window |
| `export_weights.py` | HuggingFace model weight exporter |
