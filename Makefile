# RKNN Matmul Parallel - Open-source RKLLM alternative
#
# Usage:
#   make              # Build core matmul library only
#   make decoder      # Build complete decoder (C library)
#   make python       # Build Python binding
#   make all          # Build everything
#   make benchmark    # Build and run benchmark

# Compiler settings
CC ?= gcc
CXX ?= g++
CFLAGS = -O3 -Wall -Wextra -fPIC
CXXFLAGS = -O3 -Wall -Wextra -fPIC -std=c++17

# RKNN SDK path (set externally or use default)
RKNN_SDK_PATH ?= /opt/rknn-toolkit2/rknpu2/runtime/Linux
# RKNN_INCLUDE_PATH: override to use real SDK headers (e.g., from rk3576-tts-build/engine/include)
RKNN_INCLUDE_PATH ?= $(RKNN_SDK_PATH)/include
INCLUDES = -I./include -I$(RKNN_INCLUDE_PATH)
LDFLAGS = -L$(RKNN_SDK_PATH)/lib -lrknnrt -lpthread -lm

# Python binding
PYTHON = python3
PYBIND_INCLUDE = $(shell $(PYTHON) -c "import pybind11; print(pybind11.get_include())")
PYTHON_INCLUDE = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")

# Source files
MATMUL_SRC = src/rknn_matmul_parallel.c
DECODER_SRC = src/matmul_decoder.c src/cpu_ops.c src/npu_buffer_pool.c
PYBIND_SRC = src/pybind_matmul_decoder.cpp

# Output files
BUILD_DIR = build
LIB_DIR = lib
MATMUL_LIB = $(LIB_DIR)/librmp.a
DECODER_LIB = $(LIB_DIR)/libmatmul_decoder.a
PYTHON_LIB = matmul_decoder$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

.PHONY: all clean matmul decoder python benchmark

# Default: build core matmul library
all: matmul

# Create directories
$(BUILD_DIR) $(LIB_DIR):
	mkdir -p $@

# ============================================================================
# Core matmul library (rmp_create/rmp_run/rmp_destroy)
# ============================================================================
matmul: $(MATMUL_LIB)

$(BUILD_DIR)/rknn_matmul_parallel.o: $(MATMUL_SRC) | $(BUILD_DIR) $(LIB_DIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(MATMUL_LIB): $(BUILD_DIR)/rknn_matmul_parallel.o
	ar rcs $@ $<

# ============================================================================
# Complete decoder (includes KV cache, CPU ops, sampling)
# ============================================================================
decoder: $(DECODER_LIB)

$(BUILD_DIR)/matmul_decoder.o: src/matmul_decoder.c include/matmul_decoder.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I./include $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/cpu_ops.o: src/cpu_ops.c include/cpu_ops.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I./include $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/npu_buffer_pool.o: src/npu_buffer_pool.c include/npu_buffer_pool.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I./include $(INCLUDES) -c $< -o $@

$(DECODER_LIB): $(BUILD_DIR)/matmul_decoder.o $(BUILD_DIR)/cpu_ops.o $(BUILD_DIR)/npu_buffer_pool.o
	ar rcs $@ $^

# ============================================================================
# Python binding (pybind11)
# ============================================================================
python: $(PYTHON_LIB)

$(PYTHON_LIB): $(PYBIND_SRC) $(DECODER_LIB) $(MATMUL_LIB) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(PYBIND_INCLUDE) -I$(PYTHON_INCLUDE) $(INCLUDES) \
		-shared -fPIC $< \
		-L$(LIB_DIR) -lmatmul_decoder -lrmp \
		$(LDFLAGS) \
		-o $@

# ============================================================================
# Benchmark
# ============================================================================
benchmark: $(MATMUL_LIB) benchmarks/benchmark.c
	$(CC) $(CFLAGS) $(INCLUDES) benchmarks/benchmark.c -L./lib -lrmp $(LDFLAGS) -o benchmark
	./benchmark

# ============================================================================
# Clean
# ============================================================================
clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR) benchmark $(PYTHON_LIB)