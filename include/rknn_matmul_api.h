/**
 * @file rknn_matmul_api.h
 * @brief RKNN Matmul API type definitions (stub for build)
 *
 * Real definitions come from Rockchip RKNN SDK.
 * This is a minimal stub for documentation purposes.
 */

#ifndef RKNN_MATMUL_API_H
#define RKNN_MATMUL_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Matmul context handle */
typedef void* rknn_matmul_ctx;

/* Matmul precision types */
typedef enum {
    RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16 = 0,
    RKNN_FLOAT16_MM_INT4_TO_FLOAT16    = 1,
    RKNN_FLOAT16_MM_INT8_TO_FLOAT16    = 2,
} rknn_matmul_type;

/* Matmul info structure */
typedef struct {
    int M;
    int K;
    int N;
    rknn_matmul_type type;
    int B_layout;
    int B_quant_type;
} rknn_matmul_info;

/* I/O attribute structure */
typedef struct {
    struct { size_t size; } A;
    struct { size_t size; } B;
    struct { size_t size; } C;
} rknn_matmul_io_attr;

/* Tensor memory handle */
typedef struct rknn_tensor_mem {
    void* virt_addr;
    uint64_t phys_addr;  /* Physical address for NPU DMA */
    size_t size;
} rknn_tensor_mem;

/* API functions */
int rknn_matmul_create(rknn_matmul_ctx* ctx, rknn_matmul_info* info, rknn_matmul_io_attr* io_attr);
int rknn_matmul_destroy(rknn_matmul_ctx ctx);
int rknn_matmul_run(rknn_matmul_ctx ctx);
int rknn_matmul_set_io_mem(rknn_matmul_ctx ctx, rknn_tensor_mem* mem, void* attr);

rknn_tensor_mem* rknn_create_mem(rknn_matmul_ctx ctx, size_t size);
int rknn_destroy_mem(rknn_matmul_ctx ctx, rknn_tensor_mem* mem);

int rknn_B_normal_layout_to_native_layout(void* src, void* dst, int K, int N, rknn_matmul_info* info);

#ifdef __cplusplus
}
#endif

#endif /* RKNN_MATMUL_API_H */