/**
 * @file rknn_matmul_api.h
 * @brief RKNN Matmul API type definitions
 *
 * Struct layouts must match Rockchip RKNN SDK v2.x exactly.
 * If a real SDK header is available at compile time (via RKNN_INCLUDE_PATH),
 * it takes priority because it appears later in the include search path.
 * This file is used only when no real SDK header is present.
 */

#ifndef _RKNN_MATMUL_API_H
#define _RKNN_MATMUL_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/* ── Context handle ───────────────────────────────────────────────── */

typedef uint64_t rknn_context;
typedef rknn_context rknn_matmul_ctx;

/* ── Tensor type (must match rknn_api.h) ─────────────────────────── */

typedef enum _rknn_tensor_type {
    RKNN_TENSOR_FLOAT32  = 0,
    RKNN_TENSOR_FLOAT16  = 1,
    RKNN_TENSOR_INT8     = 2,
    RKNN_TENSOR_UINT8    = 3,
    RKNN_TENSOR_INT16    = 4,
    RKNN_TENSOR_UINT16   = 5,
    RKNN_TENSOR_INT32    = 6,
    RKNN_TENSOR_UINT32   = 7,
    RKNN_TENSOR_INT64    = 8,
    RKNN_TENSOR_BOOL     = 9,
    RKNN_TENSOR_INT4     = 10,
    RKNN_TENSOR_BFLOAT16 = 11,
    RKNN_TENSOR_TYPE_MAX
} rknn_tensor_type;

/* ── Core mask ───────────────────────────────────────────────────── */

typedef enum _rknn_core_mask {
    RKNN_NPU_CORE_AUTO    = 0,
    RKNN_NPU_CORE_0       = 1,
    RKNN_NPU_CORE_1       = 2,
    RKNN_NPU_CORE_2       = 4,
    RKNN_NPU_CORE_0_1     = 3,
    RKNN_NPU_CORE_0_1_2   = 7,
    RKNN_NPU_CORE_ALL     = 0xffff,
    RKNN_NPU_CORE_UNDEFINED,
} rknn_core_mask;

/* ── Tensor memory (must match rknn_api.h _rknn_tensor_memory) ───── */

typedef struct _rknn_tensor_memory {
    void*     virt_addr;   /* virtual address of tensor buffer */
    uint64_t  phys_addr;   /* physical address for NPU DMA */
    int32_t   fd;          /* file descriptor of tensor buffer */
    int32_t   offset;      /* memory offset */
    uint32_t  size;        /* size of tensor buffer in bytes */
    uint32_t  flags;       /* reserved flags */
    void*     priv_data;   /* private data */
} rknn_tensor_mem;

/* ── Matmul precision types ──────────────────────────────────────── */

typedef enum _rknn_matmul_type {
    RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32 = 1,
    RKNN_INT8_MM_INT8_TO_INT32         = 2,
    RKNN_INT8_MM_INT8_TO_INT8          = 3,
    RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16 = 4,
    RKNN_FLOAT16_MM_INT8_TO_FLOAT32    = 5,
    RKNN_FLOAT16_MM_INT8_TO_FLOAT16    = 6,
    RKNN_FLOAT16_MM_INT4_TO_FLOAT32    = 7,
    RKNN_FLOAT16_MM_INT4_TO_FLOAT16    = 8,
    RKNN_INT8_MM_INT8_TO_FLOAT32       = 9,
    RKNN_INT4_MM_INT4_TO_INT16         = 10,
    RKNN_INT8_MM_INT4_TO_INT32         = 11,
    RKNN_FLOAT16_MM_INT4_TO_BFLOAT16   = 12,
    RKNN_INT8_MM_INT4_TO_FLOAT16       = 15,
} rknn_matmul_type;

/* ── Matmul quant type ───────────────────────────────────────────── */

typedef enum _rknn_matmul_quant_type {
    RKNN_QUANT_TYPE_PER_LAYER_SYM    = 0,
    RKNN_QUANT_TYPE_PER_LAYER_ASYM   = 1,
    RKNN_QUANT_TYPE_PER_CHANNEL_SYM  = 2,
    RKNN_QUANT_TYPE_PER_CHANNEL_ASYM = 3,
    RKNN_QUANT_TYPE_PER_GROUP_SYM    = 4,
    RKNN_QUANT_TYPE_PER_GROUP_ASYM   = 5,
} rknn_matmul_quant_type;

/* ── Matmul info (must match _rknn_matmul_info_t exactly) ────────── */

typedef struct _rknn_matmul_info_t {
    int32_t M;
    int32_t K;
    int32_t N;
    rknn_matmul_type type;

    int16_t B_layout;       /* 0=normal, 1=native */
    int16_t B_quant_type;   /* 0=per-layer, 1=per-channel, 2=per-group */
    int16_t AC_layout;      /* 0=normal, 1=native */
    int16_t AC_quant_type;  /* reserved, must be 0 */

    int32_t iommu_domain_id;

    int16_t group_size;     /* valid when B_quant_type==2 */
    int8_t  reserved[34];
} rknn_matmul_info;

/* ── Matmul tensor attribute ─────────────────────────────────────── */

#define RKNN_MAX_NAME_LEN  256
#define RKNN_MAX_DIMS      16

typedef struct _rknn_matmul_tensor_attr {
    char     name[RKNN_MAX_NAME_LEN]; /* tensor name */
    uint32_t n_dims;
    uint32_t dims[RKNN_MAX_DIMS];
    uint32_t size;                     /* byte size of the tensor buffer */
    rknn_tensor_type type;
} rknn_matmul_tensor_attr;

/* ── Matmul I/O attribute ────────────────────────────────────────── */

typedef struct _rknn_matmul_io_attr {
    rknn_matmul_tensor_attr A;
    rknn_matmul_tensor_attr B;
    rknn_matmul_tensor_attr C;
} rknn_matmul_io_attr;

/* ── Quant params ────────────────────────────────────────────────── */

typedef struct _rknn_quant_params {
    char     name[RKNN_MAX_NAME_LEN];
    float*   scale;
    int32_t  scale_len;
    int32_t* zp;
    int32_t  zp_len;
} rknn_quant_params;

/* ── API ─────────────────────────────────────────────────────────── */

int rknn_matmul_create(rknn_matmul_ctx* ctx, rknn_matmul_info* info,
                       rknn_matmul_io_attr* io_attr);
int rknn_matmul_destroy(rknn_matmul_ctx ctx);
int rknn_matmul_run(rknn_matmul_ctx ctx);
int rknn_matmul_set_core_mask(rknn_matmul_ctx ctx, rknn_core_mask core_mask);
int rknn_matmul_set_io_mem(rknn_matmul_ctx ctx, rknn_tensor_mem* mem,
                           rknn_matmul_tensor_attr* attr);
int rknn_matmul_set_quant_params(rknn_matmul_ctx ctx, rknn_quant_params* params);
int rknn_matmul_get_quant_params(rknn_matmul_ctx ctx, rknn_quant_params* params,
                                 int n_params);

rknn_tensor_mem* rknn_create_mem(rknn_matmul_ctx ctx, uint32_t size);
int rknn_destroy_mem(rknn_matmul_ctx ctx, rknn_tensor_mem* mem);

int rknn_B_normal_layout_to_native_layout(void* B_input, void* B_output,
                                          int K, int N, rknn_matmul_info* info);

#ifdef __cplusplus
}
#endif

#endif /* _RKNN_MATMUL_API_H */
