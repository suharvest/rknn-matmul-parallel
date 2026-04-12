# RKLLM + RKNN Coexistence Guide

How to run RKLLM (decoder) and RKNN (encoder, vocoder) on the same RK3576/RK3588 device without conflicts.

## The Problem

RKLLM and RKNN both use the NPU but default to the same IOMMU domain 0. When both are loaded, they contend for the 4GB DMA address space, causing:
- RKNN models stuck after RKLLM inference (`rknn_run` hangs indefinitely)
- NPU soft reset loops in dmesg
- Device requires reboot to recover

## The Solution: IOMMU Domain Isolation

Put RKLLM on domain 1, keep RKNN on domain 0. Each domain has its own 4GB address space.

```
Domain 0: RKNN encoder + RKNN TTS vocoder (default, cannot be changed)
Domain 1: RKLLM decoder (configurable via base_domain_id)
```

### How to set it

In your RKLLM initialization code:

```python
# Python (ctypes wrapper)
param.extend_param.base_domain_id = 1  # Move RKLLM to domain 1
```

```c
// C
RKLLMParam param = rkllm_createDefaultParam();
param.extend_param.base_domain_id = 1;
rkllm_init(&handle, &param, callback);
```

That's it — one line.

### Verified Results (RK3576, 2026-04-12)

| Test | Result |
|------|--------|
| RKNN encoder inference (baseline) | PASS — 502ms |
| RKLLM load (domain 1) | PASS — 2.4s |
| RKNN after RKLLM load | **PASS — output bit-identical to baseline** |
| RKLLM inference (prefill + generate) | PASS — 108ms + 309ms |
| RKNN after RKLLM inference | **PASS — output bit-identical to baseline** |

## IOMMU Domain Support by API

| API | Domain configurable | How |
|-----|-------------------|-----|
| **RKLLM** (`rkllm.h`) | Yes | `param.extend_param.base_domain_id = N` |
| **RKNN matmul** (`rknn_matmul_api.h`) | Yes | `info.iommu_domain_id = N` |
| **RKNN model** (`rknn_api.h`) | **No** | Fixed domain 0 ([requested: #287](https://github.com/airockchip/rknn-toolkit2/issues/287)) |

Since RKNN models are locked to domain 0, the only option is to move other components (RKLLM, matmul) to domain 1+.

## Deployment Configurations

### Config A: RKLLM decoder (fastest, recommended for real-time)

```
Domain 0: RKNN ASR encoder + RKNN TTS vocoder
Domain 1: RKLLM decoder (43ms/token)
```

V2V latency: **~0.8s** (streaming)

### Config B: Matmul decoder (open source, no RKLLM dependency)

```
Domain 0: RKNN ASR encoder + RKNN TTS vocoder
Domain 1: matmul decoder (225ms/token FP16)
```

V2V latency: **~2.5s** (streaming)

### Config C: Mixed (RKLLM primary, matmul fallback)

Use RKLLM for production speed, matmul decoder for debugging/profiling:

```python
import os
decoder_type = os.environ.get("ASR_DECODER_TYPE", "rkllm")  # "rkllm" or "matmul"
```

## Constraints

- Each IOMMU domain has a **4GB address space** limit
- Total model memory must fit in physical RAM (8GB on typical RK3576 boards)
- Domain isolation solves address space contention, but both still share NPU compute
- Sequential execution (ASR then TTS) is safe; truly concurrent NPU access may still have scheduling issues

## References

- [airockchip/rknn-llm#437](https://github.com/airockchip/rknn-llm/issues/437) — Official confirmation of `base_domain_id` usage
- [airockchip/rknn-toolkit2#287](https://github.com/airockchip/rknn-toolkit2/issues/287) — Request for RKNN model domain support (open)
