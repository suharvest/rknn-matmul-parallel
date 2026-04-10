#!/usr/bin/env python3
"""
Minimal example of using MatmulDecoder for text generation.

This demonstrates the Python API matching RKLLM's interface.
"""

import numpy as np
from matmul_decoder import MatmulDecoder


def main():
    # Model configuration (Qwen3-ASR-0.6B example)
    model_config = {
        "hidden_dim": 1024,
        "num_q_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 64,
        "ffn_dim": 3072,
        "num_layers": 28,
        "vocab_size": 151936,
        "rms_eps": 1e-6,
        "rope_theta": 1000000.0,
    }

    # Create decoder
    print("Creating MatmulDecoder (dual-core mode)...")
    decoder = MatmulDecoder(
        model_path="/path/to/exported/weights",
        exec_mode="dual_core",
    )

    # Create dummy input embedding (in practice, comes from encoder)
    hidden_dim = model_config["hidden_dim"]
    embeddings = np.random.randn(5, hidden_dim).astype(np.float32) * 0.1

    # Run generation
    print("Running generation...")
    result = decoder.run_embed(
        embed_array=embeddings,
        n_tokens=len(embeddings),
        keep_history=0,
    )

    print(f"Generated {result['n_tokens_generated']} tokens")
    print(f"Text: {result['text']}")
    print(f"Performance: {result['perf']}")

    # Cleanup
    decoder.release()


if __name__ == "__main__":
    main()