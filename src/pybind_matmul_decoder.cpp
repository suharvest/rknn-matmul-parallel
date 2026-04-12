/**
 * Python binding for Generic Matmul Decoder.
 *
 * Build:
 *   c++ -O3 -shared -fPIC -I../include -I/usr/include/python3.10 \
 *       pybind_matmul_decoder.cpp ../src/matmul_decoder.c ../src/cpu_ops.c \
 *       -lrknnrt -o matmul_decoder.cpython-310-aarch64-linux-gnu.so
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "matmul_decoder.h"

namespace py = pybind11;


class PyMatmulDecoder {
public:
    PyMatmulDecoder(const std::string& model_dir,
                    int max_seq_len = 4096,
                    const std::string& quant_type = "int4",
                    const std::string& exec_mode = "dual_core",
                    bool disable_npu_lm_head = false)
        : ctx_(nullptr), max_seq_len_(max_seq_len) {

        QuantizationType quant = QUANT_INT4;
        if (quant_type == "fp16") quant = QUANT_FP16;
        else if (quant_type == "int8") quant = QUANT_INT8;
        else if (quant_type == "int4_g128") quant = QUANT_INT4_G128;

        ExecutionMode exec = EXEC_DUAL_CORE;
        if (exec_mode == "single_core") exec = EXEC_SINGLE_CORE;

        // Load config from model_dir/config.json
        MatmulDecoderConfig config;
        if (matmul_decoder_load_config((model_dir + "/config.json").c_str(), &config) != 0) {
            throw std::runtime_error("Failed to load config.json from " + model_dir);
        }
        config.exec_mode = exec;
        config.max_seq_len = max_seq_len;
        config.disable_npu_lm_head = disable_npu_lm_head ? 1 : 0;

        ctx_ = matmul_decoder_create(model_dir.c_str(), &config, quant, max_seq_len);
        if (!ctx_) {
            throw std::runtime_error("Failed to create MatmulDecoder from " + model_dir);
        }

        config_ = config;
    }

    PyMatmulDecoder(py::dict config_dict,
                    py::array_t<float> embeddings,
                    py::list layers,
                    py::object lm_head,
                    const std::string& quant_type = "int4",
                    const std::string& exec_mode = "dual_core",
                    int max_seq_len = 4096)
        : ctx_(nullptr), max_seq_len_(max_seq_len) {

        // Build config from dict
        MatmulDecoderConfig config = {};
        config.name = py::str(config_dict["name"]).cast<std::string>().c_str();
        config.hidden_dim = config_dict["hidden_dim"].cast<int>();
        config.num_q_heads = config_dict["num_q_heads"].cast<int>();
        config.num_kv_heads = config_dict["num_kv_heads"].cast<int>();
        config.head_dim = config_dict["head_dim"].cast<int>();
        config.ffn_dim = config_dict["ffn_dim"].cast<int>();
        config.num_layers = config_dict["num_layers"].cast<int>();
        config.vocab_size = config_dict["vocab_size"].cast<int>();
        config.max_seq_len = max_seq_len;
        config.rms_eps = config_dict["rms_eps"].cast<float>();
        config.rope_theta = config_dict["rope_theta"].cast<float>();
        config.tie_word_embeddings = config_dict["tie_word_embeddings"].cast<bool>();
        config.exec_mode = (exec_mode == "dual_core") ? EXEC_DUAL_CORE : EXEC_SINGLE_CORE;

        QuantizationType quant = QUANT_INT4;
        if (quant_type == "fp16") quant = QUANT_FP16;
        else if (quant_type == "int8") quant = QUANT_INT8;
        else if (quant_type == "int4_g128") quant = QUANT_INT4_G128;

        // Get embeddings
        auto emb_buf = embeddings.request();
        float* emb_ptr = static_cast<float*>(emb_buf.ptr);

        // Get LM head
        float* lm_head_ptr = nullptr;
        py::array_t<float> lm_head_arr;
        if (!lm_head.is_none() && !config.tie_word_embeddings) {
            lm_head_arr = lm_head.cast<py::array_t<float>>();
            auto lm_buf = lm_head_arr.request();
            lm_head_ptr = static_cast<float*>(lm_buf.ptr);
        }

        // TODO: Convert layers list to LayerWeights array
        // For now, this is a placeholder
        throw std::runtime_error("Direct weight initialization not yet implemented; use model_dir instead");
    }

    ~PyMatmulDecoder() {
        if (ctx_) {
            matmul_decoder_destroy(ctx_);
        }
    }

    py::array_t<float> step(int token_id, py::object embedding = py::none()) {
        if (!ctx_) {
            throw std::runtime_error("Decoder not initialized");
        }

        float* emb_ptr = nullptr;
        py::array_t<float> emb_arr;

        if (!embedding.is_none()) {
            emb_arr = embedding.cast<py::array_t<float>>();
            auto buf = emb_arr.request();
            emb_ptr = static_cast<float*>(buf.ptr);
        }

        // Allocate output logits
        py::array_t<float> logits({config_.vocab_size});
        auto logits_buf = logits.request();
        float* logits_ptr = static_cast<float*>(logits_buf.ptr);

        matmul_decoder_step(ctx_, token_id, emb_ptr, logits_ptr);

        return logits;
    }

    void prefill_batch(py::array_t<float> embeddings) {
        if (!ctx_) {
            throw std::runtime_error("Decoder not initialized");
        }

        auto buf = embeddings.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("prefill_batch: embeddings must be 2D [M, hidden_dim]");
        }
        int M = buf.shape[0];
        int ret = matmul_decoder_prefill_batch(ctx_, static_cast<float*>(buf.ptr), M);
        if (ret != 0) {
            throw std::runtime_error("prefill_batch failed with error code " + std::to_string(ret));
        }
    }

    void prefill(int token_id, py::object embedding = py::none()) {
        if (!ctx_) {
            throw std::runtime_error("Decoder not initialized");
        }

        float* emb_ptr = nullptr;
        py::array_t<float> emb_arr;

        if (!embedding.is_none()) {
            emb_arr = embedding.cast<py::array_t<float>>();
            auto buf = emb_arr.request();

            // Auto-detect 2D input: route to batch prefill
            if (buf.ndim == 2) {
                prefill_batch(emb_arr);
                return;
            }

            emb_ptr = static_cast<float*>(buf.ptr);
        }

        // Pass NULL for output_logits → skips final_norm + lm_head
        matmul_decoder_step(ctx_, token_id, emb_ptr, nullptr);
    }

    int step_get_token(int token_id, py::object embedding = py::none()) {
        if (!ctx_) {
            throw std::runtime_error("Decoder not initialized");
        }

        float* emb_ptr = nullptr;
        py::array_t<float> emb_arr;

        if (!embedding.is_none()) {
            emb_arr = embedding.cast<py::array_t<float>>();
            auto buf = emb_arr.request();
            emb_ptr = static_cast<float*>(buf.ptr);
        }

        // Must pass non-NULL logits buffer so C code runs lm_head + argmax.
        // NULL logits triggers prefill-only mode (skips lm_head, returns 0).
        std::vector<float> logits(config_.vocab_size);
        int token = matmul_decoder_step(ctx_, token_id, emb_ptr, logits.data());
        return token;
    }

    void clear_kv_cache() {
        if (ctx_) {
            matmul_decoder_clear_kv_cache(ctx_);
        }
    }

    int get_seq_len() const {
        return ctx_ ? matmul_decoder_get_seq_len(ctx_) : 0;
    }

    py::dict get_config() const {
        return py::dict(
            py::arg("name") = std::string(config_.name ? config_.name : ""),
            py::arg("hidden_dim") = config_.hidden_dim,
            py::arg("num_q_heads") = config_.num_q_heads,
            py::arg("num_kv_heads") = config_.num_kv_heads,
            py::arg("head_dim") = config_.head_dim,
            py::arg("ffn_dim") = config_.ffn_dim,
            py::arg("num_layers") = config_.num_layers,
            py::arg("vocab_size") = config_.vocab_size,
            py::arg("max_seq_len") = config_.max_seq_len,
            py::arg("rms_eps") = config_.rms_eps,
            py::arg("rope_theta") = config_.rope_theta,
            py::arg("tie_word_embeddings") = config_.tie_word_embeddings,
            py::arg("has_qk_norm") = config_.has_qk_norm,
            py::arg("exec_mode") = (config_.exec_mode == EXEC_DUAL_CORE) ? "dual_core" : "single_core"
        );
    }

    py::dict get_stats() const {
        MatmulDecoderStats stats;
        if (ctx_) {
            matmul_decoder_get_stats(ctx_, &stats);
        }
        return py::dict(
            py::arg("total_ms") = stats.total_ms,
            py::arg("matmul_ms") = stats.matmul_ms,
            py::arg("rebind_ms") = stats.rebind_ms,
            py::arg("convert_ms") = stats.convert_ms,
            py::arg("cpu_ops_ms") = stats.cpu_ops_ms,
            py::arg("lm_head_ms") = stats.lm_head_ms,
            py::arg("n_steps") = stats.n_steps
        );
    }

    std::vector<int> generate(const std::vector<int>& prompt_tokens,
                               int max_new_tokens = 128,
                               int top_k = 1,
                               float top_p = 1.0f,
                               float temperature = 1.0f,
                               float repeat_penalty = 1.0f,
                               py::object callback = py::none()) {
        if (!ctx_) {
            throw std::runtime_error("Decoder not initialized");
        }

        std::vector<int> output_tokens;

        SamplingParams params = sampling_params_greedy();
        params.top_k = top_k;
        params.top_p = top_p;
        params.temperature = temperature;
        params.repeat_penalty = repeat_penalty;

        // Feed prompt tokens
        for (int tok : prompt_tokens) {
            matmul_decoder_step(ctx_, tok, nullptr, nullptr);
        }

        // Generate new tokens
        for (int i = 0; i < max_new_tokens; i++) {
            float logits[config_.vocab_size];
            int token = matmul_decoder_step(ctx_, -1, nullptr, logits);

            // Apply sampling (for non-greedy)
            if (top_k > 1) {
                token = matmul_sample_token(logits, config_.vocab_size, &params);
            }

            output_tokens.push_back(token);

            // Check for EOS (implementation-specific)
            // TODO: Get EOS token ID from config

            // Callback
            if (!callback.is_none()) {
                callback(token);
            }
        }

        return output_tokens;
    }

private:
    MatmulDecoderContext* ctx_;
    MatmulDecoderConfig config_;
    int max_seq_len_;
};


PYBIND11_MODULE(matmul_decoder, m) {
    m.doc() = "Generic Matmul Decoder for RK3576 NPU";

    py::class_<PyMatmulDecoder>(m, "MatmulDecoder")
        .def(py::init<const std::string&, int, const std::string&, const std::string&, bool>(),
             py::arg("model_dir"),
             py::arg("max_seq_len") = 4096,
             py::arg("quant_type") = "int4",
             py::arg("exec_mode") = "dual_core",
             py::arg("disable_npu_lm_head") = false)

        .def("step", &PyMatmulDecoder::step,
             py::arg("token_id"),
             py::arg("embedding") = py::none(),
             "Run one decoding step, return logits")

        .def("prefill", &PyMatmulDecoder::prefill,
             py::arg("token_id"),
             py::arg("embedding") = py::none(),
             "Run one prefill step (fills KV cache, skips lm_head). "
             "If embedding is 2D [M, hidden_dim], routes to batch prefill.")

        .def("prefill_batch", &PyMatmulDecoder::prefill_batch,
             py::arg("embeddings"),
             "Batch prefill M tokens. embeddings must be 2D [M, hidden_dim].")

        .def("step_get_token", &PyMatmulDecoder::step_get_token,
             py::arg("token_id"),
             py::arg("embedding") = py::none(),
             "Run one decoding step, return sampled token (greedy)")

        .def("clear_kv_cache", &PyMatmulDecoder::clear_kv_cache,
             "Clear KV cache for new sequence")

        .def("generate", &PyMatmulDecoder::generate,
             py::arg("prompt_tokens"),
             py::arg("max_new_tokens") = 128,
             py::arg("top_k") = 1,
             py::arg("top_p") = 1.0f,
             py::arg("temperature") = 1.0f,
             py::arg("repeat_penalty") = 1.0f,
             py::arg("callback") = py::none(),
             "Generate tokens autoregressively")

        .def_property_readonly("seq_len", &PyMatmulDecoder::get_seq_len,
                               "Current sequence length")

        .def_property_readonly("config", &PyMatmulDecoder::get_config,
                               "Model configuration")

        .def_property_readonly("stats", &PyMatmulDecoder::get_stats,
                               "Performance statistics")

        .def("__repr__", [](const PyMatmulDecoder& d) {
            return "<MatmulDecoder>";
        });

    // Export default configs
    m.def("config_qwen3_0_6b", []() {
        MatmulDecoderConfig c = matmul_decoder_config_qwen3_0_6b();
        return py::dict(
            py::arg("name") = std::string(c.name),
            py::arg("hidden_dim") = c.hidden_dim,
            py::arg("num_q_heads") = c.num_q_heads,
            py::arg("num_kv_heads") = c.num_kv_heads,
            py::arg("head_dim") = c.head_dim,
            py::arg("ffn_dim") = c.ffn_dim,
            py::arg("num_layers") = c.num_layers,
            py::arg("vocab_size") = c.vocab_size,
            py::arg("has_qk_norm") = c.has_qk_norm,
            py::arg("exec_mode") = "dual_core"
        );
    }, "Default config for Qwen3-0.6B (dual-core)");

    m.def("config_qwen3_0_6b_single_core", []() {
        MatmulDecoderConfig c = matmul_decoder_config_qwen3_0_6b_single_core();
        return py::dict(
            py::arg("name") = std::string(c.name),
            py::arg("hidden_dim") = c.hidden_dim,
            py::arg("num_q_heads") = c.num_q_heads,
            py::arg("num_kv_heads") = c.num_kv_heads,
            py::arg("head_dim") = c.head_dim,
            py::arg("ffn_dim") = c.ffn_dim,
            py::arg("num_layers") = c.num_layers,
            py::arg("vocab_size") = c.vocab_size,
            py::arg("has_qk_norm") = c.has_qk_norm,
            py::arg("exec_mode") = "single_core"
        );
    }, "Default config for Qwen3-0.6B (single-core)");

    // Version
    m.attr("__version__") = "1.0.0";
}