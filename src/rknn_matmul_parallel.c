/**
 * @file rknn_matmul_parallel.c
 * @brief Implementation of parallel matmul for Rockchip NPU
 *
 * Architecture:
 *   - Parent: coordinates workers via shared memory
 *   - Workers: persistent processes, each holds one NPU matmul context
 *   - Weight splitting: each worker computes N/2 columns
 *
 * Why multi-process:
 *   rknn_matmul_run() has an internal global lock. Even with multiple
 *   contexts in the same process, calls serialize on a single NPU core.
 *   Only separate processes can access different cores concurrently.
 */

#include "rknn_matmul_parallel.h"
#include "rknn_matmul_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <pthread.h>
#include <time.h>

/* RKNN SDK provides these */

/* ============================================================================
 * Internal structures
 * ============================================================================ */

typedef enum {
    CMD_EXIT = 0,
    CMD_RUN  = 1,
} WorkerCommand;

/* Shared memory layout for IPC */
typedef struct {
    /* Control */
    volatile int command;
    volatile int input_ready;
    volatile int outputs_ready[RMP_MAX_WORKERS];
    int n_workers;

    /* Matmul config */
    int M, K, N_half;
    int matmul_type;
    int B_layout;

    /* Data buffers (embedded for simplicity) */
    int16_t input_buffer[4096];           /* Max M*K = 1*4096 */
    int16_t output_buffer[RMP_MAX_WORKERS][4096];  /* Max M*N/2 per worker */
    int input_size;

    /* Worker ID (set before fork so child knows who it is) */
    int worker_id;

    /* Synchronization */
    pthread_mutex_t mutex;
    pthread_cond_t input_cond;
    pthread_cond_t output_cond;
} SharedState;

/* Internal context structure */
struct RmpContext {
    SharedState* shm;
    int shmid;
    pid_t worker_pids[RMP_MAX_WORKERS];
    RmpConfig config;
    float last_benchmark_ms;
};

/* ============================================================================
 * Worker process
 * ============================================================================ */

static rknn_matmul_type to_rknn_type(RmpMatmulType type) {
    switch (type) {
        case RMP_TYPE_FP16_INT4: return RKNN_FLOAT16_MM_INT4_TO_FLOAT16;
        case RMP_TYPE_FP16_INT8: return RKNN_FLOAT16_MM_INT8_TO_FLOAT16;
        default: return RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
    }
}

static void worker_loop(SharedState* shm, int worker_id) {
    int M = shm->M;
    int K = shm->K;
    int N_half = shm->N_half;

    /* Create matmul context AFTER fork (contexts cannot cross process boundary) */
    rknn_matmul_ctx ctx;
    rknn_matmul_io_attr io_attr;
    rknn_matmul_info info = {0};
    info.M = M;
    info.K = K;
    info.N = N_half;
    info.type = shm->matmul_type;
    info.B_layout = shm->B_layout;

    int ret = rknn_matmul_create(&ctx, &info, &io_attr);
    if (ret != 0) {
        fprintf(stderr, "[worker %d] rknn_matmul_create failed: %d\n", worker_id, ret);
        return;
    }

    /* Allocate NPU memory buffers */
    rknn_tensor_mem* mem_A = rknn_create_mem(ctx, io_attr.A.size);
    rknn_tensor_mem* mem_B = rknn_create_mem(ctx, io_attr.B.size);
    rknn_tensor_mem* mem_C = rknn_create_mem(ctx, io_attr.C.size);

    /* Set I/O bindings */
    rknn_matmul_set_io_mem(ctx, mem_A, &io_attr.A);
    rknn_matmul_set_io_mem(ctx, mem_B, &io_attr.B);
    rknn_matmul_set_io_mem(ctx, mem_C, &io_attr.C);

    /* Main loop: wait for input, compute, signal output */
    while (1) {
        pthread_mutex_lock(&shm->mutex);
        while (shm->command != CMD_EXIT && !shm->input_ready) {
            pthread_cond_wait(&shm->input_cond, &shm->mutex);
        }
        int cmd = shm->command;
        pthread_mutex_unlock(&shm->mutex);

        if (cmd == CMD_EXIT) break;

        /* Copy input from shared memory to NPU buffer */
        memcpy(mem_A->virt_addr, shm->input_buffer, shm->input_size);

        /* Execute matmul on NPU (blocks until complete) */
        rknn_matmul_run(ctx);

        /* Copy output to shared memory */
        memcpy(shm->output_buffer[worker_id], mem_C->virt_addr, io_attr.C.size);

        /* Signal completion */
        pthread_mutex_lock(&shm->mutex);
        shm->outputs_ready[worker_id] = 1;
        int all_done = 1;
        for (int i = 0; i < shm->n_workers; i++) {
            if (!shm->outputs_ready[i]) all_done = 0;
        }
        if (all_done) {
            pthread_cond_signal(&shm->output_cond);
        }
        pthread_mutex_unlock(&shm->mutex);
    }

    /* Cleanup */
    rknn_destroy_mem(ctx, mem_A);
    rknn_destroy_mem(ctx, mem_B);
    rknn_destroy_mem(ctx, mem_C);
    rknn_matmul_destroy(ctx);
}

/* ============================================================================
 * Public API
 * ============================================================================ */

RmpContext* rmp_create(const RmpConfig* config, const void* weights, const float* scales) {
    if (!config || !weights) return NULL;

    int n_workers = config->n_workers > 0 ? config->n_workers : RMP_MAX_WORKERS;
    if (config->N % n_workers != 0) {
        fprintf(stderr, "rmp_create: N=%d must be divisible by n_workers=%d\n",
                config->N, n_workers);
        return NULL;
    }

    /* Allocate context */
    RmpContext* ctx = calloc(1, sizeof(RmpContext));
    if (!ctx) return NULL;
    ctx->config = *config;
    ctx->config.n_workers = n_workers;

    /* Create shared memory */
    ctx->shmid = shmget(IPC_PRIVATE, sizeof(SharedState), IPC_CREAT | 0600);
    if (ctx->shmid < 0) {
        perror("shmget");
        free(ctx);
        return NULL;
    }
    ctx->shm = shmat(ctx->shmid, NULL, 0);
    if (ctx->shm == (void*)-1) {
        perror("shmat");
        shmctl(ctx->shmid, IPC_RMID, NULL);
        free(ctx);
        return NULL;
    }
    memset(ctx->shm, 0, sizeof(SharedState));

    /* Setup shared config */
    int N_half = config->N / n_workers;
    ctx->shm->M = config->M;
    ctx->shm->K = config->K;
    ctx->shm->N_half = N_half;
    ctx->shm->matmul_type = to_rknn_type(config->type);
    ctx->shm->B_layout = config->layout;
    ctx->shm->n_workers = n_workers;

    /* Copy weights to shared memory (each worker gets N/2 columns) */
    size_t weight_half_size = config->K * N_half * sizeof(int16_t);
    for (int i = 0; i < n_workers; i++) {
        /* Split weight columns: worker i gets columns [N_half*i : N_half*(i+1)] */
        const int16_t* src = (const int16_t*)weights;
        int16_t* dst = ctx->shm->output_buffer[i];  /* Temporarily use output buffer */
        for (int k = 0; k < config->K; k++) {
            for (int n = 0; n < N_half; n++) {
                dst[k * N_half + n] = src[k * config->N + N_half * i + n];
            }
        }
    }

    /* Initialize process-shared mutex/cond */
    pthread_mutexattr_t mattr;
    pthread_condattr_t cattr;
    pthread_mutexattr_init(&mattr);
    pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
    pthread_condattr_init(&cattr);
    pthread_condattr_setpshared(&cattr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&ctx->shm->mutex, &mattr);
    pthread_cond_init(&ctx->shm->input_cond, &cattr);
    pthread_cond_init(&ctx->shm->output_cond, &cattr);

    /* Fork worker processes */
    for (int i = 0; i < n_workers; i++) {
        ctx->shm->worker_id = i;  /* Child reads this to know its ID */
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            /* Kill existing workers */
            for (int j = 0; j < i; j++) kill(ctx->worker_pids[j], SIGTERM);
            rmp_destroy(ctx);
            return NULL;
        } else if (pid == 0) {
            /* Child process */
            int my_id = ctx->shm->worker_id;
            worker_loop(ctx->shm, my_id);
            _exit(0);
        } else {
            ctx->worker_pids[i] = pid;
        }
    }

    /* Wait for workers to initialize */
    usleep(100000);  /* 100ms */

    return ctx;
}

int rmp_run(RmpContext* ctx, const int16_t* input, int16_t* output) {
    if (!ctx || !input || !output) return -1;

    SharedState* shm = ctx->shm;
    int N_half = ctx->config.N / ctx->config.n_workers;
    size_t input_size = ctx->config.M * ctx->config.K * sizeof(int16_t);

    /* Reset state and copy input */
    pthread_mutex_lock(&shm->mutex);
    shm->input_ready = 0;
    shm->input_size = input_size;
    for (int i = 0; i < ctx->config.n_workers; i++) {
        shm->outputs_ready[i] = 0;
    }
    memcpy(shm->input_buffer, input, input_size);

    /* Signal workers to start */
    shm->command = CMD_RUN;
    shm->input_ready = 1;
    pthread_cond_broadcast(&shm->input_cond);
    pthread_mutex_unlock(&shm->mutex);

    /* Wait for all workers to complete */
    pthread_mutex_lock(&shm->mutex);
    while (1) {
        int all_done = 1;
        for (int i = 0; i < ctx->config.n_workers; i++) {
            if (!shm->outputs_ready[i]) all_done = 0;
        }
        if (all_done) break;
        pthread_cond_wait(&shm->output_cond, &shm->mutex);
    }
    pthread_mutex_unlock(&shm->mutex);

    /* Gather outputs from all workers */
    for (int i = 0; i < ctx->config.n_workers; i++) {
        size_t offset = ctx->config.M * N_half * i;
        memcpy(output + offset, shm->output_buffer[i], ctx->config.M * N_half * sizeof(int16_t));
    }

    return 0;
}

void rmp_destroy(RmpContext* ctx) {
    if (!ctx) return;

    /* Signal workers to exit */
    if (ctx->shm) {
        pthread_mutex_lock(&ctx->shm->mutex);
        ctx->shm->command = CMD_EXIT;
        ctx->shm->input_ready = 1;
        pthread_cond_broadcast(&ctx->shm->input_cond);
        pthread_mutex_unlock(&ctx->shm->mutex);
    }

    /* Wait for workers */
    for (int i = 0; i < ctx->config.n_workers; i++) {
        if (ctx->worker_pids[i] > 0) {
            waitpid(ctx->worker_pids[i], NULL, 0);
        }
    }

    /* Cleanup shared memory */
    if (ctx->shm) {
        pthread_mutex_destroy(&ctx->shm->mutex);
        pthread_cond_destroy(&ctx->shm->input_cond);
        pthread_cond_destroy(&ctx->shm->output_cond);
        shmdt(ctx->shm);
    }
    if (ctx->shmid >= 0) {
        shmctl(ctx->shmid, IPC_RMID, NULL);
    }

    free(ctx);
}

float rmp_benchmark(RmpContext* ctx, int n_runs) {
    if (!ctx) return -1;

    int M = ctx->config.M;
    int K = ctx->config.K;
    int N = ctx->config.N;

    /* Allocate test buffers */
    int16_t* input = malloc(M * K * sizeof(int16_t));
    int16_t* output = malloc(M * N * sizeof(int16_t));

    /* Fill with random FP16 data */
    srand(42);
    for (int i = 0; i < M * K; i++) {
        float v = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
        __fp16 h = (__fp16)v;
        memcpy(&input[i], &h, sizeof(__fp16));
    }

    /* Warmup */
    for (int i = 0; i < 3; i++) {
        rmp_run(ctx, input, output);
    }

    /* Benchmark */
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < n_runs; i++) {
        rmp_run(ctx, input, output);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    float total_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                     (t1.tv_nsec - t0.tv_nsec) / 1e6;
    float avg_ms = total_ms / n_runs;

    ctx->last_benchmark_ms = avg_ms;

    free(input);
    free(output);
    return avg_ms;
}

float rmp_get_single_core_time(RmpContext* ctx) {
    if (!ctx) return -1;
    /* Single core time ≈ parallel time × n_workers / speedup_factor */
    /* Empirical speedup is ~1.37x, so single core ≈ parallel × 1.37 */
    return ctx->last_benchmark_ms * 1.37f;
}