# Qwen3-ASR 部署指南 (RK3576)

在 RK3576 上部署 Qwen3-ASR 语音识别服务，使用 RKLLM 解码器（43ms/token）+ RKNN 编码器，支持与 TTS 模型共存。

## 架构概览

```
麦克风 → [16kHz PCM]
  → VAD (Silero, CPU)              ~5ms/frame
  → Encoder (RKNN, NPU core 1)    ~500ms/4s chunk
  → Decoder (RKLLM, NPU domain 1) ~43ms/token
  → Tokenizer → 文本输出

同时加载（不冲突）：
  TTS vocoder (RKNN, NPU domain 0)
```

| 组件 | 运行位置 | IOMMU Domain |
|------|---------|-------------|
| ASR Encoder (Zipformer) | RKNN, NPU core 1 | 0 (固定) |
| ASR Decoder (Qwen3-0.6B) | RKLLM | **1** (`base_domain_id=1`) |
| TTS Vocoder (Matcha/Vocos) | RKNN, NPU core 0 | 0 (固定) |

## 前置条件

### 硬件
- RK3576 开发板，8GB RAM
- 存储 ≥ 4GB 可用（模型文件）

### 软件
- librknnrt.so 2.3.0 或 2.3.2（`/usr/lib/librknnrt.so`）
- librkllmrt.so（RKLLM 运行时）
- Python 3.10+
- rknn-toolkit-lite2

## 模型文件

```
/opt/asr/models/                     # ASR_MODEL_DIR
├── encoder/rk3576/
│   └── qwen3_asr_encoder_merged.fp16.4s.rk3576.rknn
├── decoder/
│   └── decoder_hf.w4a16.rk3576.rkllm
├── embed_tokens.npy                 # 622MB, Qwen3 词嵌入表
├── mel_filters.npy                  # MEL 滤波器组
├── tokenizer.json                   # Qwen3 BPE 分词器
└── vad/
    └── silero_vad.onnx              # 可选，VAD 模型
```

## 快速启动

### 方式一：Docker（推荐）

```bash
# docker-compose.yml 关键配置
services:
  speech:
    environment:
      - ASR_BACKEND=qwen3_asr_rk
      - ASR_DECODER_TYPE=rkllm         # 使用 RKLLM（快）
      - ASR_MODEL_DIR=/opt/asr/models
    volumes:
      - /path/to/models:/opt/asr/models
    devices:
      - /dev/rknpu
    ports:
      - "8621:8621"

docker-compose up
```

### 方式二：直接运行

```bash
export ASR_BACKEND=qwen3_asr_rk
export ASR_DECODER_TYPE=rkllm
export ASR_MODEL_DIR=/home/cat/qwen3-asr-models

cd rk3576/app
python3 -m uvicorn main:app --host 0.0.0.0 --port 8621
```

### 测试

```bash
# 转写音频文件
curl -X POST http://localhost:8621/asr \
  -F "file=@test.wav" \
  -F "language=auto"
# 返回: {"text": "你好世界", "language": "zh", ...}

# 健康检查
curl http://localhost:8621/health
```

## 性能指标

### 实测数据 (RK3576, Qwen3-ASR-0.6B)

| 指标 | RKLLM 方案 | matmul 方案 |
|------|-----------|------------|
| Decode 速度 | **43 ms/token** | 225 ms/token |
| Prefill (60 token) | **108 ms** | 1,047 ms (batch) |
| Encoder (4s chunk) | 500 ms | 500 ms |
| RTF (2s 音频) | **0.65** | 2.5 |
| V2V 延迟（流式） | **~0.8s** | ~2.5s |

### V2V 延迟拆解（RKLLM，流式）

```
用户说话结束
  ├─ VAD 尾部检测:    300ms
  ├─ Encoder (2s):    250ms
  ├─ RKLLM prefill:   108ms
  ├─ RKLLM 首 token:   43ms
  ├─ TTS 首 chunk:    100ms
  └─ 开始播放:       ~800ms
```

## 性能优化指南

### 已实现的优化

| 优化 | 效果 | 说明 |
|------|------|------|
| RKLLM domain 隔离 | RKNN+RKLLM 共存 | `base_domain_id=1` |
| Batch prefill | prefill 11.6x 加速 | M=60 batch matmul（matmul 方案） |
| INT8 NPU lm_head | lm_head 3x 加速 | 107ms→35ms（matmul 方案） |

### 进一步优化方向

| 优化 | 预估收益 | 难度 | 说明 |
|------|---------|------|------|
| **投机执行** | -200ms V2V | 中 | VAD 确认前就开始 encoder |
| **Pipeline TTS** | -100ms V2V | 低 | RKLLM 流式输出 3-5 token 后启动 TTS |
| **KV Cache 复用** | -50ms V2V | 低 | System prompt 跨轮缓存 |
| **更短 chunk** | -125ms V2V | 低 | 2s→1s chunk（精度有损） |
| **Streaming encoder** | -200ms V2V | 高 | 逐帧编码替代 chunk 编码 |

### 全部优化后预估

```
当前 RKLLM V2V:     ~800ms
+ 投机执行:          ~600ms
+ Pipeline TTS:      ~500ms
+ KV Cache 复用:     ~450ms
理论极限:            ~400ms
```

## 切换解码器

通过环境变量切换，代码不需要改：

```bash
# RKLLM 方案（快，推荐生产环境）
export ASR_DECODER_TYPE=rkllm

# matmul 方案（开源可控，调试/开发用）
export ASR_DECODER_TYPE=matmul
export MATMUL_QUANT_TYPE=fp16    # fp16 或 int8（INT4 在 2.3.x 上不可用）
export PYTHONPATH=/opt/rknn-matmul-parallel
```

## 常见问题

### RKNN 在 RKLLM 运行后卡死

**原因：** RKLLM `base_domain_id` 没有设为 1，跟 RKNN 冲突。

**修复：** 确认 `decoder.py` 中 `param.extend_param.base_domain_id = 1`。

### NPU 进入 soft reset 循环

**原因：** 之前的 NPU 操作崩溃后 handle 没释放。

**修复：** 重启设备（`sudo reboot`）。代码中已加 `atexit` 防御。

### INT4 量化输出错误

**原因：** librknnrt 2.3.x 的 INT4 NPU kernel bug（正值丢失 50% 量级）。

**修复：** 使用 `quant_type="fp16"` 或 `"int8"`。INT4 已在代码中禁用。

### 模型加载报错 "weights not found"

**检查：**
```bash
ls $ASR_MODEL_DIR/encoder/rk3576/*.rknn
ls $ASR_MODEL_DIR/decoder/*.rkllm
ls $ASR_MODEL_DIR/embed_tokens.npy
ls $ASR_MODEL_DIR/tokenizer.json
```

## 参考

- [rknn-matmul-parallel](https://github.com/suharvest/rknn-matmul-parallel) — 开源 matmul decoder
- [airockchip/rknn-llm](https://github.com/airockchip/rknn-llm) — RKLLM 官方
- [RKLLM+RKNN 共存指南](rkllm-coexistence.md) — Domain 隔离详解
