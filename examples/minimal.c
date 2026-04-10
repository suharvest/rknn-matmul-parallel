/**
 * @file minimal.c
 * @brief Minimal example of parallel matmul
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rknn_matmul_parallel.h"

int main() {
    // Configuration for a simple 1x1024x1024 matmul
    RmpConfig config = {
        .M = 1,
        .K = 1024,
        .N = 1024,
        .type = RMP_TYPE_FP16_FP16,
        .layout = RMP_LAYOUT_NORMAL,
        .n_workers = 2,  // Use both NPU cores
    };

    // Generate random FP16 weights
    int16_t* weights = malloc(config.K * config.N * sizeof(int16_t));
    for (int i = 0; i < config.K * config.N; i++) {
        float v = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
        __fp16 h = (__fp16)v;
        memcpy(&weights[i], &h, sizeof(__fp16));
    }

    // Create parallel context
    printf("Creating parallel matmul context...\n");
    RmpContext* ctx = rmp_create(&config, weights, NULL);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }
    printf("Context created (workers forked)\n\n");

    // Prepare input
    int16_t* input = malloc(config.M * config.K * sizeof(int16_t));
    int16_t* output = malloc(config.M * config.N * sizeof(int16_t));
    for (int i = 0; i < config.M * config.K; i++) {
        float v = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
        __fp16 h = (__fp16)v;
        memcpy(&input[i], &h, sizeof(__fp16));
    }

    // Run matmul
    printf("Running parallel matmul...\n");
    int ret = rmp_run(ctx, input, output);
    if (ret != 0) {
        fprintf(stderr, "Matmul failed\n");
        return 1;
    }
    printf("Done!\n\n");

    // Benchmark
    printf("Benchmarking...\n");
    float avg_ms = rmp_benchmark(ctx, 20);
    printf("Average time: %.3f ms\n", avg_ms);
    printf("Single-core equivalent: %.3f ms\n", rmp_get_single_core_time(ctx));
    printf("Speedup: %.2fx\n", rmp_get_single_core_time(ctx) / avg_ms);

    // Cleanup
    rmp_destroy(ctx);
    free(weights);
    free(input);
    free(output);

    return 0;
}